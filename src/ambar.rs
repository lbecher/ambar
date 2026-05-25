use std::path::Path;

use image::{DynamicImage, GenericImageView};

use crate::burn_backend::{BurnNcnnBackend, BurnVisionNcnnBackend};
use crate::types::{
    AmbarError, BackendConfig, BoundingBox, BurnDevice, Config, Detection, Detections,
    InferenceBackend, ModelOutput, PreprocessedImage, Result, UnavailableBackend,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct GridAndStride {
    pub x: u32,
    pub y: u32,
    pub stride: u32,
}

pub(crate) fn generate_grids_and_stride(target_size: u32, strides: &[u32]) -> Vec<GridAndStride> {
    let capacity = strides
        .iter()
        .map(|stride| {
            let grid = target_size / stride;
            (grid * grid) as usize
        })
        .sum();
    let mut grid_strides = Vec::with_capacity(capacity);

    for &stride in strides {
        let grid = target_size / stride;
        for y in 0..grid {
            for x in 0..grid {
                grid_strides.push(GridAndStride { x, y, stride });
            }
        }
    }

    grid_strides
}

#[inline]
pub(crate) fn decode_yolox_bbox(row: &[f32], grid: &GridAndStride) -> BoundingBox {
    let stride = grid.stride as f32;
    let x_center = (row[0] + grid.x as f32) * stride;
    let y_center = (row[1] + grid.y as f32) * stride;
    let width = row[2].exp() * stride;
    let height = row[3].exp() * stride;
    BoundingBox {
        x: x_center - width * 0.5,
        y: y_center - height * 0.5,
        width,
        height,
    }
}

pub(crate) fn nms_sorted(
    proposals: Vec<Detection>,
    nms_threshold: f32,
    max_detections: usize,
) -> Vec<Detection> {
    let mut picked: Vec<usize> = Vec::with_capacity(proposals.len().min(max_detections));
    let areas: Vec<f32> = proposals
        .iter()
        .map(|detection| detection.bbox.area())
        .collect();

    'candidate: for (idx, candidate) in proposals.iter().enumerate() {
        for &picked_idx in &picked {
            let inter = candidate.bbox.intersection(proposals[picked_idx].bbox);
            let union = areas[idx] + areas[picked_idx] - inter;
            if union > f32::EPSILON && inter / union > nms_threshold {
                continue 'candidate;
            }
        }

        picked.push(idx);
        if picked.len() == max_detections {
            break;
        }
    }

    picked
        .into_iter()
        .map(|idx| proposals[idx].clone())
        .collect()
}

pub(crate) fn detections_from_xyxy(
    input: &PreprocessedImage,
    class_names: &[String],
    boxes: Vec<f32>,
    scores: Vec<f32>,
    class_ids: Vec<i64>,
) -> Detections {
    let count = scores.len().min(class_ids.len()).min(boxes.len() / 4);
    let max_x = input.width.saturating_sub(1) as f32;
    let max_y = input.height.saturating_sub(1) as f32;
    let mut detections = Vec::with_capacity(count);

    for index in 0..count {
        let base = index * 4;
        let x0 = (boxes[base] / input.scale).clamp(0.0, max_x);
        let y0 = (boxes[base + 1] / input.scale).clamp(0.0, max_y);
        let x1 = (boxes[base + 2] / input.scale).clamp(0.0, max_x);
        let y1 = (boxes[base + 3] / input.scale).clamp(0.0, max_y);
        let class_id = class_ids[index].max(0) as usize;
        detections.push(Detection {
            bbox: BoundingBox {
                x: x0,
                y: y0,
                width: (x1 - x0).max(0.0),
                height: (y1 - y0).max(0.0),
            },
            class_id,
            class_name: class_names.get(class_id).cloned(),
            confidence: scores[index],
        });
    }

    Detections::new(detections)
}

pub fn preprocess(
    image: &DynamicImage,
    input_size: u32,
    fill_value: u8,
) -> Result<PreprocessedImage> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err(AmbarError::EmptyImage);
    }

    let scale = if width > height {
        input_size as f32 / width as f32
    } else {
        input_size as f32 / height as f32
    };
    let resized_width = ((width as f32 * scale).floor() as u32).max(1);
    let resized_height = ((height as f32 * scale).floor() as u32).max(1);
    let resized = image.resize_exact(
        resized_width,
        resized_height,
        image::imageops::FilterType::Triangle,
    );
    let resized = resized.to_rgb8();

    let pixels = input_size as usize * input_size as usize;
    let mut rgb_chw = vec![fill_value as f32; pixels * 3];
    let plane_g = pixels;
    let plane_b = pixels * 2;

    for y in 0..resized_height as usize {
        let dst_row = y * input_size as usize;
        let src_row = y * resized_width as usize * 3;
        let src = resized.as_raw();
        for x in 0..resized_width as usize {
            let src_offset = src_row + x * 3;
            let dst_offset = dst_row + x;
            rgb_chw[dst_offset] = src[src_offset + 2] as f32;
            rgb_chw[plane_g + dst_offset] = src[src_offset + 1] as f32;
            rgb_chw[plane_b + dst_offset] = src[src_offset] as f32;
        }
    }

    Ok(PreprocessedImage {
        width,
        height,
        input_size,
        resized_width,
        resized_height,
        scale,
        rgb_chw,
    })
}

#[derive(Debug)]
pub struct Ambar {
    config: Config,
    backend: Box<dyn InferenceBackend>,
    grid_strides: Vec<GridAndStride>,
}

impl Ambar {
    pub fn new(config: Config) -> Self {
        let backend = UnavailableBackend {
            config: config.backend.clone(),
        };
        Self::with_backend(config, backend)
    }

    pub fn with_backend(config: Config, backend: impl InferenceBackend + 'static) -> Self {
        let grid_strides = generate_grids_and_stride(config.model.input_size, &config.strides);
        Self {
            config,
            backend: Box::new(backend),
            grid_strides,
        }
    }

    pub fn from_ncnn_files(
        config: Config,
        param_path: impl AsRef<Path>,
        bin_path: impl AsRef<Path>,
    ) -> Result<Self> {
        match config.backend.clone() {
            BackendConfig::Burn(burn_config) => match burn_config.device {
                BurnDevice::Cpu => {
                    type B = burn::backend::ndarray::NdArray<f32>;
                    let backend = BurnNcnnBackend::<B>::from_files(
                        &param_path,
                        &bin_path,
                        Default::default(),
                        config.model.input_size as usize,
                        config.max_candidates_before_nms,
                    )?;
                    Ok(Self::with_backend(config, backend))
                }
                BurnDevice::Flex => {
                    type B = burn_flex::Flex;
                    let backend = BurnVisionNcnnBackend::<B>::from_files(
                        &param_path,
                        &bin_path,
                        Default::default(),
                        config.model.input_size as usize,
                        &config.strides,
                        config.max_candidates_before_nms,
                    )?;
                    Ok(Self::with_backend(config, backend))
                }
                BurnDevice::Wgpu => {
                    type B = burn::backend::Wgpu<f32>;
                    // GPU runs the forward pass; NMS/decode runs on CPU via BurnNcnnBackend
                    // (burn_vision NMS/top-k on Wgpu is unreliable — produces zero detections)
                    let backend = BurnNcnnBackend::<B>::from_files(
                        &param_path,
                        &bin_path,
                        Default::default(),
                        config.model.input_size as usize,
                        0,
                    )?;
                    Ok(Self::with_backend(config, backend))
                }
                BurnDevice::Metal => {
                    type B = burn::backend::Metal<f32>;
                    // GPU runs the forward pass; NMS/decode runs on CPU via BurnNcnnBackend
                    // (burn_vision NMS/top-k on Metal is unreliable — produces zero detections)
                    let backend = BurnNcnnBackend::<B>::from_files(
                        &param_path,
                        &bin_path,
                        Default::default(),
                        config.model.input_size as usize,
                        0,
                    )?;
                    Ok(Self::with_backend(config, backend))
                }
                BurnDevice::WgpuRaw => {
                    type B = burn::backend::wgpu::CubeBackend<
                        burn::backend::wgpu::WgpuRuntime,
                        f32,
                        i32,
                        u32,
                    >;
                    let device = Default::default();
                    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::AutoGraphicsApi>(
                        &device,
                        Default::default(),
                    );
                    let backend = BurnNcnnBackend::<B>::from_files(
                        &param_path,
                        &bin_path,
                        device,
                        config.model.input_size as usize,
                        0,
                    )?;
                    Ok(Self::with_backend(config, backend))
                }
                BurnDevice::MetalRaw => {
                    type B = burn::backend::wgpu::CubeBackend<
                        burn::backend::wgpu::WgpuRuntime,
                        f32,
                        i32,
                        u8,
                    >;
                    let device = Default::default();
                    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
                        &device,
                        Default::default(),
                    );
                    let backend = BurnNcnnBackend::<B>::from_files(
                        &param_path,
                        &bin_path,
                        device,
                        config.model.input_size as usize,
                        0,
                    )?;
                    Ok(Self::with_backend(config, backend))
                }
                BurnDevice::Cuda => Err(AmbarError::BackendUnavailable(
                    "Burn CUDA backend selection is not wired on this build".to_owned(),
                )),
            },
            _ => Err(AmbarError::BackendUnavailable(
                "Auto/External backend requires explicit backend setup via with_backend(...)"
                    .to_owned(),
            )),
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn infer(&self, image: &DynamicImage) -> Result<Detections> {
        let input = self.preprocess(image)?;
        self.infer_preprocessed(&input)
    }

    pub fn infer_preprocessed(&self, input: &PreprocessedImage) -> Result<Detections> {
        if let Some(detections) = self.backend.infer_detections(input, &self.config)? {
            return Ok(detections);
        }
        let output = self.backend.infer(input)?;
        self.decode(input, output)
    }

    pub fn infer_raw_preprocessed(&self, input: &PreprocessedImage) -> Result<ModelOutput> {
        self.backend.infer(input)
    }

    pub fn preprocess(&self, image: &DynamicImage) -> Result<PreprocessedImage> {
        preprocess(image, self.config.model.input_size, self.config.fill_value)
    }

    pub fn decode(&self, input: &PreprocessedImage, output: ModelOutput) -> Result<Detections> {
        let rows = output.rows()?;
        match &output.row_indices {
            Some(indices) if indices.len() != rows.len() => {
                return Err(AmbarError::InvalidOutputShape {
                    len: output.data.len(),
                    row_len: output.row_len,
                });
            }
            Some(_) => {}
            None => {
                if rows.len() != self.grid_strides.len() {
                    return Err(AmbarError::AnchorCountMismatch {
                        anchors: rows.len(),
                        expected: self.grid_strides.len(),
                    });
                }
            }
        }

        let mut proposals = Vec::with_capacity(128);
        for (row_index, row) in output.rows()?.enumerate() {
            let grid_index = output
                .row_indices
                .as_ref()
                .and_then(|indices| indices.get(row_index).copied())
                .unwrap_or(row_index);
            let grid =
                self.grid_strides
                    .get(grid_index)
                    .ok_or(AmbarError::AnchorCountMismatch {
                        anchors: grid_index + 1,
                        expected: self.grid_strides.len(),
                    })?;
            let objectness = row[4];

            if self.config.multi_label {
                let mut bbox = None;
                for (class_id, class_score) in row[5..].iter().copied().enumerate() {
                    let confidence = objectness * class_score;
                    if confidence > self.config.prob_threshold {
                        let bbox = *bbox.get_or_insert_with(|| decode_yolox_bbox(row, grid));
                        proposals.push(Detection {
                            bbox,
                            class_id,
                            class_name: self.config.model.class_names.get(class_id).cloned(),
                            confidence,
                        });
                    }
                }
            } else if let Some((class_id, class_score)) = row[5..]
                .iter()
                .copied()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
            {
                let confidence = objectness * class_score;
                if confidence > self.config.prob_threshold {
                    let bbox = decode_yolox_bbox(row, grid);
                    proposals.push(Detection {
                        bbox,
                        class_id,
                        class_name: self.config.model.class_names.get(class_id).cloned(),
                        confidence,
                    });
                }
            }
        }

        proposals.sort_unstable_by(|a, b| {
            b.confidence
                .total_cmp(&a.confidence)
                .then_with(|| a.class_id.cmp(&b.class_id))
        });

        let mut picked = nms_sorted(
            proposals,
            self.config.nms_threshold,
            self.config.max_detections,
        );
        let max_x = input.width.saturating_sub(1) as f32;
        let max_y = input.height.saturating_sub(1) as f32;
        for detection in &mut picked {
            let x0 = (detection.bbox.x / input.scale).clamp(0.0, max_x);
            let y0 = (detection.bbox.y / input.scale).clamp(0.0, max_y);
            let x1 = (detection.bbox.x2() / input.scale).clamp(0.0, max_x);
            let y1 = (detection.bbox.y2() / input.scale).clamp(0.0, max_y);
            detection.bbox = BoundingBox {
                x: x0,
                y: y0,
                width: (x1 - x0).max(0.0),
                height: (y1 - y0).max(0.0),
            };
        }

        Ok(Detections::new(picked))
    }
}

impl Default for Ambar {
    fn default() -> Self {
        Self::new(Config::default())
    }
}
