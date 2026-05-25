use std::path::Path;
use std::sync::Arc;

use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::backend::Backend;
use burn::tensor::module::{conv2d, interpolate, max_pool2d};
use burn::tensor::ops::{InterpolateMode, InterpolateOptions, PaddedConvOptions};
use burn::tensor::{Int, IntDType, Tensor, TensorData, Transaction};
use burn_vision::{Nms, NmsOptions, VisionBackend};
use half::f16;

use crate::ambar::{GridAndStride, detections_from_xyxy, generate_grids_and_stride};
use crate::ncnn::{
    NcnnLayer, NcnnModel, NcnnParam, NcnnWeightBlob, NcnnWeightKind, NcnnWeightStorage, NcnnWeights,
};
use crate::types::{
    AmbarError, Config, Detections, InferenceBackend, ModelOutput, PreprocessedImage, Result,
};

// ── Weight decoding ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub(crate) struct DecodedLayerWeights {
    pub weight: Arc<[f32]>,
    pub bias: Arc<[f32]>,
}

fn decode_weight_blob_to_f32(weights: &NcnnWeights, blob: &NcnnWeightBlob) -> Result<Arc<[f32]>> {
    let bytes = weights.blob_bytes(blob);
    let values: Vec<f32> = match blob.storage {
        NcnnWeightStorage::Float32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect(),
        NcnnWeightStorage::Float16 => bytes
            .chunks_exact(2)
            .map(|chunk| f16::from_le_bytes(chunk.try_into().unwrap()).to_f32())
            .collect(),
        NcnnWeightStorage::Int8 => bytes.iter().map(|&value| value as i8 as f32).collect(),
        NcnnWeightStorage::Raw(_) => {
            return Err(AmbarError::InvalidNcnnWeights {
                offset: blob.offset,
                message: "raw weight storage cannot be decoded to f32".to_owned(),
            });
        }
    };
    Ok(Arc::from(values))
}

pub(crate) fn decode_model_weights(model: &NcnnModel) -> Result<Vec<Option<DecodedLayerWeights>>> {
    model
        .weights
        .layers
        .iter()
        .map(|layer_weights| {
            let Some(layer_weights) = layer_weights else {
                return Ok(None);
            };
            let mut weight = Arc::<[f32]>::from([]);
            let mut bias = Arc::<[f32]>::from([]);

            for blob in &layer_weights.blobs {
                let decoded = decode_weight_blob_to_f32(&model.weights, blob)?;
                match blob.kind {
                    NcnnWeightKind::Weight => weight = decoded,
                    NcnnWeightKind::Bias => bias = decoded,
                    _ => {}
                }
            }

            Ok(Some(DecodedLayerWeights { weight, bias }))
        })
        .collect()
}

// ── Burn-specific weight and plan types ────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct BurnLayerWeights<B: Backend> {
    pub weight: Option<Tensor<B, 4>>,
    pub bias: Option<Tensor<B, 1>>,
}

#[derive(Clone, Debug)]
pub(crate) struct NcnnExecutionPlan {
    pub layers: Vec<NcnnExecutionLayer>,
    pub output_blob_index: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct NcnnExecutionLayer {
    pub bottoms: Vec<usize>,
    pub tops: Vec<usize>,
}

#[derive(Debug)]
pub(crate) struct BurnTensor<B: Backend> {
    pub w: usize,
    pub h: usize,
    pub c: usize,
    pub tensor: Tensor<B, 4>,
}

impl<B: Backend> Clone for BurnTensor<B> {
    fn clone(&self) -> Self {
        Self {
            w: self.w,
            h: self.h,
            c: self.c,
            tensor: self.tensor.clone(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct BurnFocusIndices<B: Backend> {
    pub input_w: usize,
    pub input_h: usize,
    pub h_even: Tensor<B, 1, Int>,
    pub h_odd: Tensor<B, 1, Int>,
    pub w_even: Tensor<B, 1, Int>,
    pub w_odd: Tensor<B, 1, Int>,
}

impl<B: Backend> BurnFocusIndices<B> {
    pub fn new(input_w: usize, input_h: usize, device: &B::Device) -> Self {
        Self {
            input_w,
            input_h,
            h_even: burn_index_tensor::<B>(input_h, 0, device),
            h_odd: burn_index_tensor::<B>(input_h, 1, device),
            w_even: burn_index_tensor::<B>(input_w, 0, device),
            w_odd: burn_index_tensor::<B>(input_w, 1, device),
        }
    }

    fn matches(&self, input: &BurnTensor<B>) -> bool {
        self.input_w == input.w && self.input_h == input.h
    }
}

#[derive(Debug)]
pub(crate) struct BurnDecodeGrids<B: Backend> {
    pub x: Tensor<B, 1>,
    pub y: Tensor<B, 1>,
    pub stride: Tensor<B, 1>,
    pub len: usize,
}

impl<B: Backend> BurnDecodeGrids<B> {
    pub fn new(input_size: u32, strides: &[u32], device: &B::Device) -> Self {
        let grids: Vec<GridAndStride> = generate_grids_and_stride(input_size, strides);
        let len = grids.len();
        let mut x = Vec::with_capacity(len);
        let mut y = Vec::with_capacity(len);
        let mut stride = Vec::with_capacity(len);
        for grid in grids {
            x.push(grid.x as f32);
            y.push(grid.y as f32);
            stride.push(grid.stride as f32);
        }

        Self {
            x: Tensor::<B, 1>::from_data(TensorData::new(x, [len]), device),
            y: Tensor::<B, 1>::from_data(TensorData::new(y, [len]), device),
            stride: Tensor::<B, 1>::from_data(TensorData::new(stride, [len]), device),
            len,
        }
    }
}

// ── Backend structs ────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct BurnNcnnBackend<B: Backend> {
    pub model: NcnnModel,
    pub(crate) decoded_weights: Vec<Option<BurnLayerWeights<B>>>,
    pub(crate) execution_plan: NcnnExecutionPlan,
    pub(crate) focus_indices: Option<BurnFocusIndices<B>>,
    pub(crate) max_candidates_before_nms: usize,
    pub(crate) materialize_layout_ops: bool,
    pub(crate) device: B::Device,
}

impl<B: Backend> BurnNcnnBackend<B> {
    pub fn from_files(
        param_path: impl AsRef<Path>,
        bin_path: impl AsRef<Path>,
        device: B::Device,
        input_size: usize,
        max_candidates_before_nms: usize,
        materialize_layout_ops: bool,
    ) -> Result<Self> {
        let model = NcnnModel::from_files(param_path, bin_path)?;
        let cpu_weights = decode_model_weights(&model)?;
        let decoded_weights = decode_burn_model_weights::<B>(&model, &cpu_weights, &device)?;
        let execution_plan = build_execution_plan(&model.param)?;
        let focus_indices = model
            .param
            .layers
            .iter()
            .any(|layer| layer.layer_type == "YoloV5Focus")
            .then(|| BurnFocusIndices::new(input_size, input_size, &device));
        Ok(Self {
            model,
            decoded_weights,
            execution_plan,
            focus_indices,
            max_candidates_before_nms,
            materialize_layout_ops,
            device,
        })
    }
}

impl<B: Backend> InferenceBackend for BurnNcnnBackend<B> {
    fn infer(&self, input: &PreprocessedImage) -> Result<ModelOutput> {
        let output = execute_burn_ncnn_model(
            &self.model.param,
            &self.decoded_weights,
            &self.execution_plan,
            self.focus_indices.as_ref(),
            self.materialize_layout_ops,
            &self.device,
            input,
        )?;
        burn_to_model_output(output, self.max_candidates_before_nms)
    }
}

#[derive(Debug)]
pub struct BurnVisionNcnnBackend<B: Backend + VisionBackend> {
    pub(crate) inner: BurnNcnnBackend<B>,
    pub(crate) decode_grids: BurnDecodeGrids<B>,
}

impl<B: Backend + VisionBackend> BurnVisionNcnnBackend<B> {
    pub fn from_files(
        param_path: impl AsRef<Path>,
        bin_path: impl AsRef<Path>,
        device: B::Device,
        input_size: usize,
        strides: &[u32],
        max_candidates_before_nms: usize,
        materialize_layout_ops: bool,
    ) -> Result<Self> {
        let inner = BurnNcnnBackend::<B>::from_files(
            param_path,
            bin_path,
            device.clone(),
            input_size,
            max_candidates_before_nms,
            materialize_layout_ops,
        )?;
        let decode_grids = BurnDecodeGrids::new(input_size as u32, strides, &device);
        Ok(Self {
            inner,
            decode_grids,
        })
    }
}

impl<B: Backend + VisionBackend> InferenceBackend for BurnVisionNcnnBackend<B> {
    fn infer(&self, input: &PreprocessedImage) -> Result<ModelOutput> {
        self.inner.infer(input)
    }

    fn infer_detections(
        &self,
        input: &PreprocessedImage,
        config: &Config,
    ) -> Result<Option<Detections>> {
        if config.multi_label {
            return Ok(None);
        }

        let output = execute_burn_ncnn_model(
            &self.inner.model.param,
            &self.inner.decoded_weights,
            &self.inner.execution_plan,
            self.inner.focus_indices.as_ref(),
            self.inner.materialize_layout_ops,
            &self.inner.device,
            input,
        )?;
        burn_decode_detections(
            output,
            &self.decode_grids,
            input,
            config,
            self.inner.max_candidates_before_nms,
        )
        .map(Some)
    }
}

// ── Weight decoding into Burn tensors ──────────────────────────────────────────

pub(crate) fn decode_burn_model_weights<B: Backend>(
    model: &NcnnModel,
    cpu_weights: &[Option<DecodedLayerWeights>],
    device: &B::Device,
) -> Result<Vec<Option<BurnLayerWeights<B>>>> {
    model
        .param
        .layers
        .iter()
        .zip(cpu_weights.iter())
        .map(|(layer, weights)| {
            let Some(weights) = weights else {
                return Ok(None);
            };

            let weight = match layer.layer_type.as_str() {
                "Convolution" | "ConvolutionDepthWise" => {
                    let out_c = layer.param_usize(0).unwrap_or(0);
                    let kernel_w = layer.param_usize(1).unwrap_or(1);
                    let kernel_h = layer.param_usize(11).unwrap_or(kernel_w);
                    let groups = layer.param_usize(7).unwrap_or(1).max(1);
                    let in_per_group = weights.weight.len() / (out_c * kernel_w * kernel_h).max(1);
                    let expected = out_c * in_per_group * kernel_h * kernel_w;
                    if weights.weight.len() != expected {
                        return Err(AmbarError::InvalidNcnnShape {
                            name: layer.name.clone(),
                            message: format!(
                                "Burn weight shape expects {expected} values, got {} (groups={groups})",
                                weights.weight.len()
                            ),
                        });
                    }
                    Some(Tensor::<B, 4>::from_data(
                        TensorData::new(
                            weights.weight.to_vec(),
                            [out_c, in_per_group, kernel_h, kernel_w],
                        ),
                        device,
                    ))
                }
                _ => None,
            };

            let bias = (!weights.bias.is_empty()).then(|| {
                Tensor::<B, 1>::from_data(
                    TensorData::new(weights.bias.to_vec(), [weights.bias.len()]),
                    device,
                )
            });

            Ok(Some(BurnLayerWeights { weight, bias }))
        })
        .collect()
}

// ── Execution plan ─────────────────────────────────────────────────────────────

pub(crate) fn build_execution_plan(param: &NcnnParam) -> Result<NcnnExecutionPlan> {
    let layers = param
        .layers
        .iter()
        .map(|layer| {
            let bottoms = layer
                .bottoms
                .iter()
                .map(|name| {
                    param
                        .blob_index(name)
                        .ok_or_else(|| AmbarError::MissingNcnnBlob(name.clone()))
                })
                .collect::<Result<Vec<_>>>()?;
            let tops = layer
                .tops
                .iter()
                .map(|name| {
                    param
                        .blob_index(name)
                        .ok_or_else(|| AmbarError::MissingNcnnBlob(name.clone()))
                })
                .collect::<Result<Vec<_>>>()?;

            Ok(NcnnExecutionLayer { bottoms, tops })
        })
        .collect::<Result<Vec<_>>>()?;
    let output_blob_index = param
        .blob_index("output")
        .ok_or_else(|| AmbarError::MissingNcnnBlob("output".to_owned()))?;

    Ok(NcnnExecutionPlan {
        layers,
        output_blob_index,
    })
}

// ── Burn op helpers ────────────────────────────────────────────────────────────

pub(crate) fn fused_swish_target(layers: &[NcnnLayer], layer_index: usize) -> Option<&NcnnLayer> {
    if std::env::var_os("AMBAR_DISABLE_FUSED_SWISH").is_some() {
        return None;
    }

    let layer = layers.get(layer_index)?;
    let next = layers.get(layer_index + 1)?;
    if next.layer_type == "Swish"
        && layer.tops.len() == 1
        && next.bottoms.len() == 1
        && next.tops.len() == 1
        && layer.tops[0] == next.bottoms[0]
    {
        Some(next)
    } else {
        None
    }
}

pub(crate) fn fused_unary_target<'a>(
    layers: &'a [NcnnLayer],
    layer_index: usize,
    next_type: &str,
) -> Option<&'a NcnnLayer> {
    let layer = layers.get(layer_index)?;
    let next = layers.get(layer_index + 1)?;
    if next.layer_type == next_type
        && layer.tops.len() == 1
        && next.bottoms.len() == 1
        && next.tops.len() == 1
        && layer.tops[0] == next.bottoms[0]
    {
        Some(next)
    } else {
        None
    }
}

fn burn_index_tensor<B: Backend>(
    len: usize,
    start: usize,
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    let values: Vec<i64> = (start..len).step_by(2).map(|value| value as i64).collect();
    let count = values.len();
    Tensor::<B, 1, Int>::from_data(TensorData::new(values, [count]), device)
}

fn get_burn_bottom<B: Backend>(
    blobs: &[Option<BurnTensor<B>>],
    layer: &NcnnLayer,
    layer_plan: &NcnnExecutionLayer,
    index: usize,
) -> Result<BurnTensor<B>> {
    let blob_index =
        layer_plan
            .bottoms
            .get(index)
            .copied()
            .ok_or_else(|| AmbarError::InvalidNcnnShape {
                name: layer.name.clone(),
                message: format!("missing bottom index {index}"),
            })?;
    blobs
        .get(blob_index)
        .and_then(Option::as_ref)
        .cloned()
        .ok_or_else(|| {
            AmbarError::MissingNcnnBlob(layer.bottoms.get(index).cloned().unwrap_or_default())
        })
}

fn insert_burn_single_top<B: Backend>(
    blobs: &mut [Option<BurnTensor<B>>],
    layer: &NcnnLayer,
    layer_plan: &NcnnExecutionLayer,
    tensor: BurnTensor<B>,
) -> Result<()> {
    maybe_debug_burn_tensor(layer, &tensor);
    let top = layer_plan
        .tops
        .first()
        .copied()
        .ok_or_else(|| AmbarError::InvalidNcnnShape {
            name: layer.name.clone(),
            message: "layer has no top blob".to_owned(),
        })?;
    blobs[top] = Some(tensor);
    Ok(())
}

fn maybe_debug_burn_tensor<B: Backend>(layer: &NcnnLayer, tensor: &BurnTensor<B>) {
    let Ok(filter) = std::env::var("AMBAR_DEBUG_LAYERS") else {
        return;
    };
    if filter.is_empty() || !debug_filter_matches(layer, &filter) {
        return;
    }

    let values = match tensor.tensor.clone().into_data().into_vec::<f32>() {
        Ok(values) => values,
        Err(error) => {
            eprintln!(
                "Layer {} ({}) debug readback failed: {error}",
                layer.name, layer.layer_type
            );
            return;
        }
    };

    let mut finite = 0usize;
    let mut nan = 0usize;
    let mut min_value = f32::INFINITY;
    let mut max_value = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut l2 = 0.0f64;
    for &value in &values {
        if value.is_finite() {
            finite += 1;
            min_value = min_value.min(value);
            max_value = max_value.max(value);
            sum += value as f64;
            sum_abs += value.abs() as f64;
            l2 += (value as f64) * (value as f64);
        } else if value.is_nan() {
            nan += 1;
        }
    }
    let mean = if finite == 0 {
        f32::NAN
    } else {
        (sum / finite as f64) as f32
    };
    let rms = if finite == 0 {
        f32::NAN
    } else {
        (l2 / finite as f64).sqrt() as f32
    };
    let sample_stride = (values.len() / 1024).max(1);
    let checksum = values
        .iter()
        .step_by(sample_stride)
        .enumerate()
        .fold(0.0f64, |acc, (index, value)| {
            acc + (*value as f64) * ((index % 17 + 1) as f64)
        });
    let samples = debug_samples(&values);

    let mut best = None;
    if tensor.w > 5 {
        for (row_index, row) in values.chunks_exact(tensor.w).enumerate() {
            let objectness = row[4];
            if let Some((class_id, class_score)) = row[5..]
                .iter()
                .copied()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
            {
                let score = objectness * class_score;
                if best.is_none_or(|(best_score, _, _, _, _)| score > best_score) {
                    best = Some((score, row_index, class_id, objectness, class_score));
                }
            }
        }
    }

    eprintln!(
        "Layer {} ({}) shape=1x{}x{}x{} values={} finite={} nan={} min={:.6} max={:.6} mean={:.6} abs_mean={:.6} rms={:.6} checksum={:.6}",
        layer.name,
        layer.layer_type,
        tensor.c,
        tensor.h,
        tensor.w,
        values.len(),
        finite,
        nan,
        min_value,
        max_value,
        mean,
        sum_abs / finite.max(1) as f64,
        rms,
        checksum
    );
    eprintln!("  samples {samples}");
    if let Some((score, row, class_id, objectness, class_score)) = best {
        eprintln!(
            "  best score={score:.6} row={row} class={class_id} objectness={objectness:.6} class_score={class_score:.6}"
        );
    }
}

fn debug_samples(values: &[f32]) -> String {
    if values.is_empty() {
        return "[]".to_owned();
    }
    let last = values.len() - 1;
    let positions = [
        0,
        1.min(last),
        2.min(last),
        3.min(last),
        4.min(last),
        5.min(last),
        (values.len() / 2).min(last),
        last,
    ];
    let mut output = String::from("[");
    for (index, position) in positions.into_iter().enumerate() {
        if index > 0 {
            output.push_str(", ");
        }
        output.push_str(&format!("{position}:{:.6}", values[position]));
    }
    output.push(']');
    output
}

fn debug_filter_matches(layer: &NcnnLayer, filter: &str) -> bool {
    filter == "1"
        || filter.split(',').map(str::trim).any(|needle| {
            !needle.is_empty()
                && (needle == layer.name
                    || needle == layer.layer_type
                    || layer.tops.iter().any(|top| top == needle)
                    || layer.bottoms.iter().any(|bottom| bottom == needle))
        })
}

// ── Burn inference primitives ──────────────────────────────────────────────────

fn burn_from_preprocessed<B: Backend>(
    input: &PreprocessedImage,
    device: &B::Device,
) -> Result<BurnTensor<B>> {
    let input_size = input.input_size as usize;
    let burn = Tensor::<B, 4>::from_data(
        TensorData::new(input.rgb_chw.clone(), [1, 3, input_size, input_size]),
        device,
    );
    Ok(BurnTensor {
        w: input_size,
        h: input_size,
        c: 3,
        tensor: burn,
    })
}

pub(crate) fn burn_to_model_output<B: Backend>(
    output: BurnTensor<B>,
    max_candidates_before_nms: usize,
) -> Result<ModelOutput> {
    let row_len = output.w;
    let rows = output.h * output.c;

    if max_candidates_before_nms == 0 || max_candidates_before_nms >= rows || row_len <= 5 {
        let data = output
            .tensor
            .into_data()
            .into_vec::<f32>()
            .map_err(|error| AmbarError::InvalidNcnnShape {
                name: "burn_to_model_output".to_owned(),
                message: error.to_string(),
            })?;
        return Ok(ModelOutput::new(data, row_len));
    }

    let k = max_candidates_before_nms;
    let output = output.tensor.reshape([rows, row_len]);
    let objectness = output.clone().narrow(1, 4, 1);
    let class_score = output.clone().narrow(1, 5, row_len - 5).max_dim(1);
    let confidence = (objectness * class_score).reshape([rows]);
    let (_, row_indices) = confidence.topk_with_indices(k, 0);
    let gather_indices = row_indices.clone().reshape([k, 1]).repeat_dim(1, row_len);
    let selected = output.gather(0, gather_indices);

    let mut tensors = Transaction::default()
        .register(selected)
        .register(row_indices.cast(IntDType::I64))
        .try_execute()
        .map_err(|error| AmbarError::InvalidNcnnShape {
            name: "burn_to_model_output".to_owned(),
            message: error.to_string(),
        })?;
    if tensors.len() != 2 {
        return Err(AmbarError::InvalidNcnnShape {
            name: "burn_to_model_output".to_owned(),
            message: format!("expected 2 tensors from transaction, got {}", tensors.len()),
        });
    }

    let row_indices = tensors
        .pop()
        .unwrap()
        .into_vec::<i64>()
        .map_err(|error| AmbarError::InvalidNcnnShape {
            name: "burn_to_model_output_indices".to_owned(),
            message: error.to_string(),
        })?
        .into_iter()
        .map(|index| index as usize)
        .collect();
    let data =
        tensors
            .pop()
            .unwrap()
            .into_vec::<f32>()
            .map_err(|error| AmbarError::InvalidNcnnShape {
                name: "burn_to_model_output".to_owned(),
                message: error.to_string(),
            })?;

    Ok(ModelOutput::with_row_indices(data, row_len, row_indices))
}

fn burn_decode_detections<B: Backend + VisionBackend>(
    output: BurnTensor<B>,
    grids: &BurnDecodeGrids<B>,
    input: &PreprocessedImage,
    config: &Config,
    max_candidates_before_nms: usize,
) -> Result<Detections> {
    let row_len = output.w;
    let rows = output.h * output.c;
    if row_len <= 5 {
        return Err(AmbarError::InvalidOutputRowLength(row_len));
    }
    if rows != grids.len {
        return Err(AmbarError::AnchorCountMismatch {
            anchors: rows,
            expected: grids.len,
        });
    }

    let output = output.tensor.reshape([rows, row_len]);
    let device = output.device();
    let objectness = output.clone().narrow(1, 4, 1);
    let class_scores = output.clone().narrow(1, 5, row_len - 5);
    let (best_class_scores, class_ids) = class_scores.max_dim_with_indices(1);
    let confidence = (objectness * best_class_scores).reshape([rows]);

    let (candidate_scores, row_indices, candidate_count) =
        if max_candidates_before_nms == 0 || max_candidates_before_nms >= rows {
            (
                confidence,
                Tensor::<B, 1, Int>::arange(0..rows as i64, &device),
                rows,
            )
        } else {
            let k = max_candidates_before_nms;
            let (scores, indices) = confidence.topk_with_indices(k, 0);
            (scores, indices, k)
        };

    let gather_rows = row_indices
        .clone()
        .reshape([candidate_count, 1])
        .repeat_dim(1, row_len);
    let selected = output.gather(0, gather_rows);
    let selected_class_ids = class_ids
        .reshape([rows])
        .select(0, row_indices.clone())
        .cast(IntDType::I64);

    let stride = grids.stride.clone().select(0, row_indices.clone());
    let grid_x = grids.x.clone().select(0, row_indices.clone());
    let grid_y = grids.y.clone().select(0, row_indices);

    let tx = selected.clone().narrow(1, 0, 1).reshape([candidate_count]);
    let ty = selected.clone().narrow(1, 1, 1).reshape([candidate_count]);
    let tw = selected.clone().narrow(1, 2, 1).reshape([candidate_count]);
    let th = selected.narrow(1, 3, 1).reshape([candidate_count]);

    let x_center = (tx + grid_x) * stride.clone();
    let y_center = (ty + grid_y) * stride.clone();
    let width = tw.exp() * stride.clone();
    let height = th.exp() * stride;
    let x0 = x_center.clone() - width.clone() * 0.5;
    let y0 = y_center.clone() - height.clone() * 0.5;
    let x1 = x_center + width * 0.5;
    let y1 = y_center + height * 0.5;
    let boxes = Tensor::cat(
        vec![
            x0.reshape([candidate_count, 1]),
            y0.reshape([candidate_count, 1]),
            x1.reshape([candidate_count, 1]),
            y1.reshape([candidate_count, 1]),
        ],
        1,
    );

    let keep = boxes.clone().nms(
        candidate_scores.clone(),
        NmsOptions {
            iou_threshold: config.nms_threshold,
            score_threshold: config.prob_threshold,
            max_output_boxes: config.max_detections,
        },
    );
    let keep_count = keep.shape().dims::<1>()[0];
    if keep_count == 0 {
        return Ok(Detections::new(Vec::new()));
    }

    let kept_boxes = boxes.gather(0, keep.clone().reshape([keep_count, 1]).repeat_dim(1, 4));
    let kept_scores = candidate_scores.select(0, keep.clone());
    let kept_class_ids = selected_class_ids.select(0, keep);

    let mut tensors = Transaction::default()
        .register(kept_boxes)
        .register(kept_scores)
        .register(kept_class_ids)
        .try_execute()
        .map_err(|error| AmbarError::InvalidNcnnShape {
            name: "burn_decode_detections".to_owned(),
            message: error.to_string(),
        })?;
    if tensors.len() != 3 {
        return Err(AmbarError::InvalidNcnnShape {
            name: "burn_decode_detections".to_owned(),
            message: format!("expected 3 tensors from transaction, got {}", tensors.len()),
        });
    }

    let class_ids =
        tensors
            .pop()
            .unwrap()
            .into_vec::<i64>()
            .map_err(|error| AmbarError::InvalidNcnnShape {
                name: "burn_decode_class_ids".to_owned(),
                message: error.to_string(),
            })?;
    let scores =
        tensors
            .pop()
            .unwrap()
            .into_vec::<f32>()
            .map_err(|error| AmbarError::InvalidNcnnShape {
                name: "burn_decode_scores".to_owned(),
                message: error.to_string(),
            })?;
    let boxes =
        tensors
            .pop()
            .unwrap()
            .into_vec::<f32>()
            .map_err(|error| AmbarError::InvalidNcnnShape {
                name: "burn_decode_boxes".to_owned(),
                message: error.to_string(),
            })?;

    Ok(detections_from_xyxy(
        input,
        &config.model.class_names,
        boxes,
        scores,
        class_ids,
    ))
}

// ── Burn op implementations ────────────────────────────────────────────────────

fn burn_yolo_v5_focus<B: Backend>(
    input: BurnTensor<B>,
    cached_indices: Option<&BurnFocusIndices<B>>,
    device: &B::Device,
) -> Result<BurnTensor<B>> {
    let local_indices;
    let indices = match cached_indices {
        Some(indices) if indices.matches(&input) => indices,
        _ => {
            local_indices = BurnFocusIndices::new(input.w, input.h, device);
            &local_indices
        }
    };

    let top_left = input
        .tensor
        .clone()
        .select(2, indices.h_even.clone())
        .select(3, indices.w_even.clone());
    let top_right = input
        .tensor
        .clone()
        .select(2, indices.h_even.clone())
        .select(3, indices.w_odd.clone());
    let bottom_left = input
        .tensor
        .clone()
        .select(2, indices.h_odd.clone())
        .select(3, indices.w_even.clone());
    let bottom_right = input
        .tensor
        .select(2, indices.h_odd.clone())
        .select(3, indices.w_odd.clone());

    Ok(BurnTensor {
        w: input.w / 2,
        h: input.h / 2,
        c: input.c * 4,
        tensor: Tensor::cat(vec![top_left, bottom_left, top_right, bottom_right], 1),
    })
}

fn burn_convolution<B: Backend>(
    input: BurnTensor<B>,
    layer: &NcnnLayer,
    weights: &BurnLayerWeights<B>,
    swish_activation: bool,
    prefer_elementwise_1x1: bool,
) -> Result<BurnTensor<B>> {
    let out_c = layer.param_usize(0).unwrap_or(0);
    let kernel_w = layer.param_usize(1).unwrap_or(1);
    let kernel_h = layer.param_usize(11).unwrap_or(kernel_w);
    let dilation_w = layer.param_usize(2).unwrap_or(1);
    let dilation_h = layer.param_usize(12).unwrap_or(dilation_w);
    let stride_w = layer.param_usize(3).unwrap_or(1);
    let stride_h = layer.param_usize(13).unwrap_or(stride_w);
    let pad_left = layer.param_i32(4).unwrap_or(0).max(0) as usize;
    let pad_right = layer.param_i32(14).unwrap_or(pad_left as i32).max(0) as usize;
    let pad_top = layer.param_i32(15).unwrap_or(pad_left as i32).max(0) as usize;
    let pad_bottom = layer.param_i32(16).unwrap_or(pad_top as i32).max(0) as usize;
    let groups = layer.param_usize(7).unwrap_or(1).max(1);
    let activation = layer.param_i32(9).unwrap_or(0);

    let weight = weights
        .weight
        .as_ref()
        .ok_or_else(|| AmbarError::MissingNcnnWeights(layer.name.clone()))?
        .clone();
    let use_elementwise_1x1 = prefer_elementwise_1x1
        && layer.name == "Conv_47"
        && kernel_w == 1
        && kernel_h == 1
        && stride_w == 1
        && stride_h == 1
        && dilation_w == 1
        && dilation_h == 1
        && pad_left == 0
        && pad_right == 0
        && pad_top == 0
        && pad_bottom == 0
        && groups == 1;

    let mut output = if use_elementwise_1x1 {
        burn_convolution_1x1_elementwise(input.tensor, weight, weights.bias.as_ref(), out_c)
    } else {
        let options = PaddedConvOptions::asymmetric(
            [stride_h, stride_w],
            [pad_top, pad_left],
            [pad_bottom, pad_right],
            [dilation_h, dilation_w],
            groups,
        );
        conv2d(input.tensor, weight, weights.bias.clone(), options)
    };
    if swish_activation {
        output = sigmoid(output.clone()) * output;
    } else {
        output = burn_apply_activation(output, activation);
    }

    let kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    let kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    let out_w = ((input.w + pad_left + pad_right).saturating_sub(kernel_extent_w)) / stride_w + 1;
    let out_h = ((input.h + pad_top + pad_bottom).saturating_sub(kernel_extent_h)) / stride_h + 1;
    Ok(BurnTensor {
        w: out_w,
        h: out_h,
        c: out_c,
        tensor: output,
    })
}

fn burn_convolution_1x1_elementwise<B: Backend>(
    input: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Option<&Tensor<B, 1>>,
    out_c: usize,
) -> Tensor<B, 4> {
    let mut outputs = Vec::with_capacity(out_c);
    for out_channel in 0..out_c {
        let weight = weight.clone().narrow(0, out_channel, 1);
        let mut output = (input.clone() * weight).sum_dim(1);
        if let Some(bias) = bias {
            output = output + bias.clone().narrow(0, out_channel, 1).reshape([1, 1, 1, 1]);
        }
        outputs.push(output);
    }
    Tensor::cat(outputs, 1)
}

fn burn_apply_activation<B: Backend>(tensor: Tensor<B, 4>, activation: i32) -> Tensor<B, 4> {
    match activation {
        1 => relu(tensor),
        4 => sigmoid(tensor),
        _ => tensor,
    }
}

fn burn_swish<B: Backend>(input: BurnTensor<B>) -> Result<BurnTensor<B>> {
    Ok(BurnTensor {
        w: input.w,
        h: input.h,
        c: input.c,
        tensor: sigmoid(input.tensor.clone()) * input.tensor,
    })
}

fn burn_add<B: Backend>(a: BurnTensor<B>, b: BurnTensor<B>, name: &str) -> Result<BurnTensor<B>> {
    if (a.w, a.h, a.c) != (b.w, b.h, b.c) {
        return Err(AmbarError::InvalidNcnnShape {
            name: name.to_owned(),
            message: "Burn add shape mismatch".to_owned(),
        });
    }
    Ok(BurnTensor {
        w: a.w,
        h: a.h,
        c: a.c,
        tensor: a.tensor + b.tensor,
    })
}

fn burn_concat<B: Backend>(
    inputs: Vec<BurnTensor<B>>,
    axis: i32,
    name: &str,
) -> Result<BurnTensor<B>> {
    let first = inputs.first().ok_or_else(|| AmbarError::InvalidNcnnShape {
        name: name.to_owned(),
        message: "concat has no inputs".to_owned(),
    })?;
    let burn_dim = match axis {
        0 => 1,
        1 => 3,
        2 => 2,
        _ => {
            return Err(AmbarError::InvalidNcnnShape {
                name: name.to_owned(),
                message: format!("unsupported concat axis {axis}"),
            });
        }
    };
    let w = if axis == 1 {
        inputs.iter().map(|tensor| tensor.w).sum()
    } else {
        first.w
    };
    let h = if axis == 2 {
        inputs.iter().map(|tensor| tensor.h).sum()
    } else {
        first.h
    };
    let c = if axis == 0 {
        inputs.iter().map(|tensor| tensor.c).sum()
    } else {
        first.c
    };
    Ok(BurnTensor {
        w,
        h,
        c,
        tensor: Tensor::cat(
            inputs.into_iter().map(|tensor| tensor.tensor).collect(),
            burn_dim,
        ),
    })
}

fn burn_max_pool<B: Backend>(input: BurnTensor<B>, layer: &NcnnLayer) -> Result<BurnTensor<B>> {
    let kernel_w = layer.param_usize(1).unwrap_or(1);
    let kernel_h = layer.param_usize(11).unwrap_or(kernel_w);
    let stride_w = layer.param_usize(2).unwrap_or(1);
    let stride_h = layer.param_usize(12).unwrap_or(stride_w);
    let pad_left = layer.param_usize(3).unwrap_or(0);
    let pad_top = layer.param_usize(13).unwrap_or(pad_left);
    let pad_right = layer.param_usize(14).unwrap_or(pad_left);
    let pad_bottom = layer.param_usize(15).unwrap_or(pad_top);
    let out_w = (input.w + pad_left + pad_right).saturating_sub(kernel_w) / stride_w + 1;
    let out_h = (input.h + pad_top + pad_bottom).saturating_sub(kernel_h) / stride_h + 1;
    Ok(BurnTensor {
        w: out_w,
        h: out_h,
        c: input.c,
        tensor: max_pool2d(
            input.tensor,
            [kernel_h, kernel_w],
            [stride_h, stride_w],
            [pad_top, pad_left],
            [1, 1],
            false,
        ),
    })
}

fn burn_interp_nearest<B: Backend>(
    input: BurnTensor<B>,
    layer: &NcnnLayer,
) -> Result<BurnTensor<B>> {
    let height_scale = layer.param_f32(1).unwrap_or(1.0);
    let width_scale = layer.param_f32(2).unwrap_or(height_scale);
    let out_w = ((input.w as f32) * width_scale).round().max(1.0) as usize;
    let out_h = ((input.h as f32) * height_scale).round().max(1.0) as usize;
    Ok(BurnTensor {
        w: out_w,
        h: out_h,
        c: input.c,
        tensor: interpolate(
            input.tensor,
            [out_h, out_w],
            InterpolateOptions::new(InterpolateMode::Nearest),
        ),
    })
}

fn materialize_burn_tensor<B: Backend>(
    input: BurnTensor<B>,
    device: &B::Device,
    name: &str,
) -> Result<BurnTensor<B>> {
    let data = input
        .tensor
        .into_data()
        .into_vec::<f32>()
        .map_err(|error| AmbarError::InvalidNcnnShape {
            name: name.to_owned(),
            message: error.to_string(),
        })?;
    Ok(BurnTensor {
        w: input.w,
        h: input.h,
        c: input.c,
        tensor: Tensor::<B, 4>::from_data(
            TensorData::new(data, [1, input.c, input.h, input.w]),
            device,
        ),
    })
}

fn burn_reshape<B: Backend>(
    input: BurnTensor<B>,
    layer: &NcnnLayer,
    device: &B::Device,
    materialize_layout_ops: bool,
) -> Result<BurnTensor<B>> {
    let total = input.w * input.h * input.c;
    let mut w = layer.param_i32(0).unwrap_or(input.w as i32);
    let mut h = layer.param_i32(1).unwrap_or(input.h as i32);
    let mut c = layer.param_i32(2).unwrap_or(1);
    let known = [w, h, c]
        .into_iter()
        .filter(|&dim| dim > 0)
        .fold(1usize, |acc, dim| acc * dim as usize);
    let inferred = total / known.max(1);
    if w == -1 {
        w = inferred as i32;
    }
    if h == -1 {
        h = inferred as i32;
    }
    if c == -1 {
        c = inferred as i32;
    }
    if w <= 0 || h <= 0 || c <= 0 || w as usize * h as usize * c as usize != total {
        return Err(AmbarError::InvalidNcnnShape {
            name: layer.name.clone(),
            message: "invalid Burn reshape".to_owned(),
        });
    }
    let input = if materialize_layout_ops {
        materialize_burn_tensor(input, device, &layer.name)?
    } else {
        input
    };

    Ok(BurnTensor {
        w: w as usize,
        h: h as usize,
        c: c as usize,
        tensor: input
            .tensor
            .reshape([1, c as usize, h as usize, w as usize]),
    })
}

fn burn_permute<B: Backend>(
    input: BurnTensor<B>,
    layer: &NcnnLayer,
    device: &B::Device,
    materialize_layout_ops: bool,
) -> Result<BurnTensor<B>> {
    let order_type = layer.param_i32(0).unwrap_or(0);
    match order_type {
        1 => {
            let new_w = input.h;
            let new_h = input.w;
            let c = input.c;
            if materialize_layout_ops {
                let data = input
                    .tensor
                    .into_data()
                    .into_vec::<f32>()
                    .map_err(|error| AmbarError::InvalidNcnnShape {
                        name: layer.name.clone(),
                        message: error.to_string(),
                    })?;
                let mut transposed = vec![0.0; data.len()];
                for channel in 0..c {
                    for y in 0..input.h {
                        for x in 0..input.w {
                            let src = (channel * input.h + y) * input.w + x;
                            let dst = (channel * new_h + x) * new_w + y;
                            transposed[dst] = data[src];
                        }
                    }
                }
                return Ok(BurnTensor {
                    w: new_w,
                    h: new_h,
                    c,
                    tensor: Tensor::<B, 4>::from_data(
                        TensorData::new(transposed, [1, c, new_h, new_w]),
                        device,
                    ),
                });
            }

            Ok(BurnTensor {
                w: new_w,
                h: new_h,
                c,
                tensor: input.tensor.swap_dims(2, 3),
            })
        }
        _ => Err(AmbarError::InvalidNcnnShape {
            name: layer.name.clone(),
            message: format!("unsupported Burn permute order type {order_type}"),
        }),
    }
}

// ── Main execution loop ────────────────────────────────────────────────────────

pub(crate) fn execute_burn_ncnn_model<B: Backend>(
    param: &NcnnParam,
    decoded_weights: &[Option<BurnLayerWeights<B>>],
    execution_plan: &NcnnExecutionPlan,
    focus_indices: Option<&BurnFocusIndices<B>>,
    materialize_layout_ops: bool,
    device: &B::Device,
    input: &PreprocessedImage,
) -> Result<BurnTensor<B>> {
    let mut blobs = Vec::with_capacity(param.blob_count);
    blobs.resize_with(param.blob_count, || None);
    let mut layer_index = 0;

    while layer_index < param.layers.len() {
        let layer = &param.layers[layer_index];
        let layer_plan = &execution_plan.layers[layer_index];
        match layer.layer_type.as_str() {
            "Input" => {
                insert_burn_single_top(
                    &mut blobs,
                    layer,
                    layer_plan,
                    burn_from_preprocessed(input, device)?,
                )?;
            }
            "YoloV5Focus" => {
                let bottom = get_burn_bottom(&blobs, layer, layer_plan, 0)?;
                insert_burn_single_top(
                    &mut blobs,
                    layer,
                    layer_plan,
                    burn_yolo_v5_focus(bottom, focus_indices, device)?,
                )?;
            }
            "Convolution" | "ConvolutionDepthWise" => {
                let bottom = get_burn_bottom(&blobs, layer, layer_plan, 0)?;
                let weights = decoded_weights
                    .get(layer_index)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| AmbarError::MissingNcnnWeights(layer.name.clone()))?;
                if !materialize_layout_ops
                    && let Some(next) = fused_swish_target(&param.layers, layer_index)
                {
                    let tensor =
                        burn_convolution(bottom, layer, weights, true, materialize_layout_ops)?;
                    insert_burn_single_top(
                        &mut blobs,
                        next,
                        &execution_plan.layers[layer_index + 1],
                        tensor,
                    )?;
                    layer_index += 2;
                    continue;
                }
                let tensor =
                    burn_convolution(bottom, layer, weights, false, materialize_layout_ops)?;
                insert_burn_single_top(&mut blobs, layer, layer_plan, tensor)?;
            }
            "Swish" => {
                let bottom = get_burn_bottom(&blobs, layer, layer_plan, 0)?;
                insert_burn_single_top(&mut blobs, layer, layer_plan, burn_swish(bottom)?)?;
            }
            "Split" => {
                let bottom = get_burn_bottom(&blobs, layer, layer_plan, 0)?;
                for &top in &layer_plan.tops {
                    blobs[top] = Some(bottom.clone());
                }
            }
            "BinaryOp" => {
                let a = get_burn_bottom(&blobs, layer, layer_plan, 0)?;
                let b = get_burn_bottom(&blobs, layer, layer_plan, 1)?;
                insert_burn_single_top(
                    &mut blobs,
                    layer,
                    layer_plan,
                    burn_add(a, b, &layer.name)?,
                )?;
            }
            "Concat" => {
                let inputs = layer_plan
                    .bottoms
                    .iter()
                    .enumerate()
                    .map(|(index, &blob_index)| {
                        blobs
                            .get(blob_index)
                            .and_then(Option::as_ref)
                            .cloned()
                            .ok_or_else(|| {
                                AmbarError::MissingNcnnBlob(
                                    layer.bottoms.get(index).cloned().unwrap_or_default(),
                                )
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let axis = layer.param_i32(0).unwrap_or(0);
                insert_burn_single_top(
                    &mut blobs,
                    layer,
                    layer_plan,
                    burn_concat(inputs, axis, &layer.name)?,
                )?;
            }
            "Pooling" => {
                let bottom = get_burn_bottom(&blobs, layer, layer_plan, 0)?;
                insert_burn_single_top(
                    &mut blobs,
                    layer,
                    layer_plan,
                    burn_max_pool(bottom, layer)?,
                )?;
            }
            "Interp" => {
                let bottom = get_burn_bottom(&blobs, layer, layer_plan, 0)?;
                insert_burn_single_top(
                    &mut blobs,
                    layer,
                    layer_plan,
                    burn_interp_nearest(bottom, layer)?,
                )?;
            }
            "Reshape" => {
                let bottom = get_burn_bottom(&blobs, layer, layer_plan, 0)?;
                if let Some(next) = fused_unary_target(&param.layers, layer_index, "Permute") {
                    let tensor = burn_permute(
                        burn_reshape(bottom, layer, device, materialize_layout_ops)?,
                        next,
                        device,
                        materialize_layout_ops,
                    )?;
                    insert_burn_single_top(
                        &mut blobs,
                        next,
                        &execution_plan.layers[layer_index + 1],
                        tensor,
                    )?;
                    layer_index += 2;
                    continue;
                }
                insert_burn_single_top(
                    &mut blobs,
                    layer,
                    layer_plan,
                    burn_reshape(bottom, layer, device, materialize_layout_ops)?,
                )?;
            }
            "Permute" => {
                let bottom = get_burn_bottom(&blobs, layer, layer_plan, 0)?;
                insert_burn_single_top(
                    &mut blobs,
                    layer,
                    layer_plan,
                    burn_permute(bottom, layer, device, materialize_layout_ops)?,
                )?;
            }
            other => {
                return Err(AmbarError::UnsupportedNcnnLayer {
                    layer_type: other.to_owned(),
                    name: layer.name.clone(),
                });
            }
        }
        layer_index += 1;
    }

    let output = blobs
        .get_mut(execution_plan.output_blob_index)
        .and_then(Option::take)
        .ok_or_else(|| AmbarError::MissingNcnnBlob("output".to_owned()))?;
    Ok(output)
}
