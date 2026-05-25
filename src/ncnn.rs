use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};

use crate::types::{AmbarError, Result};

#[derive(Debug)]
pub struct NcnnModel {
    pub param: NcnnParam,
    pub weights: NcnnWeights,
}

impl NcnnModel {
    pub fn from_files(param_path: impl AsRef<Path>, bin_path: impl AsRef<Path>) -> Result<Self> {
        let param = NcnnParam::from_file(param_path)?;
        let weights = NcnnWeights::from_file_for_param(bin_path, &param)?;
        Ok(Self { param, weights })
    }
}

#[derive(Clone, Debug)]
pub struct NcnnParam {
    pub layer_count: usize,
    pub blob_count: usize,
    pub layers: Vec<NcnnLayer>,
    layer_by_name: HashMap<String, usize>,
    blob_by_name: HashMap<String, usize>,
}

impl NcnnParam {
    pub const MAGIC: u32 = 7_767_517;

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut contents = String::new();
        File::open(path)
            .and_then(|mut file| file.read_to_string(&mut contents))
            .map_err(|source| AmbarError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        Self::parse(&contents)
    }

    pub fn parse(contents: &str) -> Result<Self> {
        let mut lines = contents.lines().enumerate().filter_map(|(idx, line)| {
            let line = line.trim();
            (!line.is_empty() && !line.starts_with('#')).then_some((idx + 1, line))
        });

        let (magic_line, magic) = lines.next().ok_or_else(|| AmbarError::InvalidNcnnParam {
            line: 1,
            message: "missing magic header".to_owned(),
        })?;
        let magic = magic
            .parse::<u32>()
            .map_err(|_| AmbarError::InvalidNcnnParam {
                line: magic_line,
                message: "magic header is not an unsigned integer".to_owned(),
            })?;
        if magic != Self::MAGIC {
            return Err(AmbarError::InvalidNcnnMagic { found: magic });
        }

        let (count_line, counts) = lines.next().ok_or_else(|| AmbarError::InvalidNcnnParam {
            line: magic_line + 1,
            message: "missing layer/blob counts".to_owned(),
        })?;
        let mut count_parts = counts.split_whitespace();
        let layer_count = parse_usize_token(count_parts.next(), count_line, "layer count")?;
        let blob_count = parse_usize_token(count_parts.next(), count_line, "blob count")?;

        let mut layers = Vec::with_capacity(layer_count);
        let mut layer_by_name = HashMap::with_capacity(layer_count);
        let mut blob_by_name = HashMap::with_capacity(blob_count);

        for (line_number, line) in lines {
            let mut tokens = line.split_whitespace();
            let layer_type = parse_string_token(tokens.next(), line_number, "layer type")?;
            let name = parse_string_token(tokens.next(), line_number, "layer name")?;
            let bottom_count = parse_usize_token(tokens.next(), line_number, "bottom count")?;
            let top_count = parse_usize_token(tokens.next(), line_number, "top count")?;

            let mut bottoms = Vec::with_capacity(bottom_count);
            for _ in 0..bottom_count {
                bottoms.push(parse_string_token(
                    tokens.next(),
                    line_number,
                    "bottom blob",
                )?);
            }

            let mut tops = Vec::with_capacity(top_count);
            for _ in 0..top_count {
                let top = parse_string_token(tokens.next(), line_number, "top blob")?;
                let next_index = blob_by_name.len();
                blob_by_name.entry(top.clone()).or_insert(next_index);
                tops.push(top);
            }

            let mut params = Vec::new();
            for token in tokens {
                params.push(NcnnParamEntry::parse(token, line_number)?);
            }

            let layer = NcnnLayer {
                layer_type,
                name,
                bottoms,
                tops,
                params,
            };
            layer_by_name.insert(layer.name.clone(), layers.len());
            layers.push(layer);
        }

        if layers.len() != layer_count {
            return Err(AmbarError::InvalidNcnnParam {
                line: count_line,
                message: format!("declared {layer_count} layers, parsed {}", layers.len()),
            });
        }

        Ok(Self {
            layer_count,
            blob_count,
            layers,
            layer_by_name,
            blob_by_name,
        })
    }

    pub fn layer(&self, name: &str) -> Option<&NcnnLayer> {
        self.layer_by_name
            .get(name)
            .and_then(|&index| self.layers.get(index))
    }

    pub fn blob_index(&self, name: &str) -> Option<usize> {
        self.blob_by_name.get(name).copied()
    }
}

#[derive(Clone, Debug)]
pub struct NcnnLayer {
    pub layer_type: String,
    pub name: String,
    pub bottoms: Vec<String>,
    pub tops: Vec<String>,
    pub params: Vec<NcnnParamEntry>,
}

impl NcnnLayer {
    pub fn param(&self, id: i32) -> Option<&NcnnParamValue> {
        self.params
            .iter()
            .find(|entry| entry.id == id)
            .map(|entry| &entry.value)
    }

    pub fn param_i32(&self, id: i32) -> Option<i32> {
        self.param(id).and_then(NcnnParamValue::as_i32)
    }

    pub fn param_f32(&self, id: i32) -> Option<f32> {
        self.param(id).and_then(NcnnParamValue::as_f32)
    }

    pub fn param_usize(&self, id: i32) -> Option<usize> {
        self.param_i32(id)
            .and_then(|value| (value >= 0).then_some(value as usize))
    }
}

#[derive(Clone, Debug)]
pub struct NcnnParamEntry {
    pub id: i32,
    pub value: NcnnParamValue,
}

impl NcnnParamEntry {
    pub(crate) fn parse(token: &str, line: usize) -> Result<Self> {
        let (key, value) = token
            .split_once('=')
            .ok_or_else(|| AmbarError::InvalidNcnnParam {
                line,
                message: format!("parameter '{token}' is missing '='"),
            })?;
        let raw_id = key
            .parse::<i32>()
            .map_err(|_| AmbarError::InvalidNcnnParam {
                line,
                message: format!("parameter id '{key}' is not an integer"),
            })?;
        let id = if raw_id <= -23300 {
            -raw_id - 23300
        } else {
            raw_id
        };
        let value = if raw_id <= -23300 {
            NcnnParamValue::parse_array(value)
        } else {
            NcnnParamValue::parse_scalar(value)
        };

        Ok(Self { id, value })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum NcnnParamValue {
    Int(i32),
    Float(f32),
    String(String),
    IntArray(Vec<i32>),
    FloatArray(Vec<f32>),
    StringArray(Vec<String>),
}

impl NcnnParamValue {
    fn parse_scalar(value: &str) -> Self {
        if let Ok(value) = value.parse::<i32>() {
            Self::Int(value)
        } else if let Ok(value) = value.parse::<f32>() {
            Self::Float(value)
        } else {
            Self::String(value.to_owned())
        }
    }

    fn parse_array(value: &str) -> Self {
        let values: Vec<_> = value.split(',').filter(|item| !item.is_empty()).collect();
        if values.iter().all(|item| item.parse::<i32>().is_ok()) {
            Self::IntArray(
                values
                    .into_iter()
                    .filter_map(|item| item.parse::<i32>().ok())
                    .collect(),
            )
        } else if values.iter().all(|item| item.parse::<f32>().is_ok()) {
            Self::FloatArray(
                values
                    .into_iter()
                    .filter_map(|item| item.parse::<f32>().ok())
                    .collect(),
            )
        } else {
            Self::StringArray(values.into_iter().map(ToString::to_string).collect())
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Self::Int(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Int(value) => Some(*value as f32),
            Self::Float(value) => Some(*value),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct NcnnWeights {
    data: NcnnWeightData,
    pub layers: Vec<Option<NcnnLayerWeights>>,
    pub bytes_consumed: usize,
}

impl NcnnWeights {
    pub fn from_file_for_param(path: impl AsRef<Path>, param: &NcnnParam) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|source| AmbarError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let data = if file.metadata().map(|meta| meta.len()).unwrap_or(0) == 0 {
            NcnnWeightData::Owned(Arc::<[u8]>::from([]))
        } else {
            // The file is opened read-only and the mapping is held immutably for the model lifetime.
            let mmap =
                unsafe { MmapOptions::new().map(&file) }.map_err(|source| AmbarError::Io {
                    path: path.to_path_buf(),
                    source,
                })?;
            NcnnWeightData::Mmap(mmap)
        };
        Self::from_data_for_param(data, param)
    }

    pub fn from_bytes_for_param(bytes: impl Into<Arc<[u8]>>, param: &NcnnParam) -> Result<Self> {
        Self::from_data_for_param(NcnnWeightData::Owned(bytes.into()), param)
    }

    fn from_data_for_param(data: NcnnWeightData, param: &NcnnParam) -> Result<Self> {
        let mut offset = 0;
        let mut layers = Vec::with_capacity(param.layers.len());
        let bytes = data.as_slice();

        for (layer_index, layer) in param.layers.iter().enumerate() {
            let blobs = scan_layer_weight_blobs(bytes, &mut offset, layer)?;
            layers.push((!blobs.is_empty()).then(|| NcnnLayerWeights {
                layer_index,
                layer_name: layer.name.clone(),
                blobs,
            }));
        }

        Ok(Self {
            data,
            layers,
            bytes_consumed: offset,
        })
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_slice()
    }

    pub fn layer(&self, layer_index: usize) -> Option<&NcnnLayerWeights> {
        self.layers.get(layer_index).and_then(Option::as_ref)
    }

    pub fn blob_bytes(&self, blob: &NcnnWeightBlob) -> &[u8] {
        &self.as_bytes()[blob.offset..blob.offset + blob.byte_len]
    }
}

#[derive(Debug)]
enum NcnnWeightData {
    Mmap(Mmap),
    Owned(Arc<[u8]>),
}

impl NcnnWeightData {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mmap(data) => data,
            Self::Owned(data) => data,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NcnnLayerWeights {
    pub layer_index: usize,
    pub layer_name: String,
    pub blobs: Vec<NcnnWeightBlob>,
}

#[derive(Clone, Debug)]
pub struct NcnnWeightBlob {
    pub kind: NcnnWeightKind,
    pub offset: usize,
    pub byte_len: usize,
    pub elem_count: usize,
    pub storage: NcnnWeightStorage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NcnnWeightKind {
    Weight,
    Bias,
    Scale,
    Mean,
    Variance,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NcnnWeightStorage {
    Float32,
    Float16,
    Int8,
    Raw(u32),
}

impl NcnnWeightStorage {
    pub fn elem_size(self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float16 => 2,
            Self::Int8 => 1,
            Self::Raw(_) => 1,
        }
    }
}

fn parse_string_token(token: Option<&str>, line: usize, field: &str) -> Result<String> {
    token
        .map(ToString::to_string)
        .ok_or_else(|| AmbarError::InvalidNcnnParam {
            line,
            message: format!("missing {field}"),
        })
}

fn parse_usize_token(token: Option<&str>, line: usize, field: &str) -> Result<usize> {
    let token = token.ok_or_else(|| AmbarError::InvalidNcnnParam {
        line,
        message: format!("missing {field}"),
    })?;
    token
        .parse::<usize>()
        .map_err(|_| AmbarError::InvalidNcnnParam {
            line,
            message: format!("{field} '{token}' is not an unsigned integer"),
        })
}

fn scan_layer_weight_blobs(
    data: &[u8],
    offset: &mut usize,
    layer: &NcnnLayer,
) -> Result<Vec<NcnnWeightBlob>> {
    match layer.layer_type.as_str() {
        "Convolution" | "ConvolutionDepthWise" | "Deconvolution" | "DeconvolutionDepthWise" => {
            let weight_data_size = layer.param_usize(6).unwrap_or(0);
            let num_output = layer.param_usize(0).unwrap_or(0);
            let bias_term = layer.param_i32(5).unwrap_or(0) != 0;
            let mut blobs = Vec::with_capacity(usize::from(bias_term) + 1);

            if weight_data_size > 0 {
                blobs.push(consume_tagged_blob(
                    data,
                    offset,
                    NcnnWeightKind::Weight,
                    weight_data_size,
                )?);
            }
            if bias_term && num_output > 0 {
                blobs.push(consume_plain_f32_blob(
                    data,
                    offset,
                    NcnnWeightKind::Bias,
                    num_output,
                )?);
            }

            Ok(blobs)
        }
        "InnerProduct" => {
            let weight_data_size = layer.param_usize(2).unwrap_or(0);
            let num_output = layer.param_usize(0).unwrap_or(0);
            let bias_term = layer.param_i32(1).unwrap_or(0) != 0;
            let mut blobs = Vec::with_capacity(usize::from(bias_term) + 1);

            if weight_data_size > 0 {
                blobs.push(consume_tagged_blob(
                    data,
                    offset,
                    NcnnWeightKind::Weight,
                    weight_data_size,
                )?);
            }
            if bias_term && num_output > 0 {
                blobs.push(consume_plain_f32_blob(
                    data,
                    offset,
                    NcnnWeightKind::Bias,
                    num_output,
                )?);
            }

            Ok(blobs)
        }
        "BatchNorm" => {
            let channels = layer.param_usize(0).unwrap_or(0);
            let mut blobs = Vec::with_capacity(4);
            for kind in [
                NcnnWeightKind::Scale,
                NcnnWeightKind::Mean,
                NcnnWeightKind::Variance,
                NcnnWeightKind::Bias,
            ] {
                blobs.push(consume_plain_f32_blob(data, offset, kind, channels)?);
            }
            Ok(blobs)
        }
        "Scale" => {
            let channels = layer.param_usize(0).unwrap_or(0);
            let bias_term = layer.param_i32(1).unwrap_or(0) != 0;
            let mut blobs = Vec::with_capacity(usize::from(bias_term) + 1);
            blobs.push(consume_plain_f32_blob(
                data,
                offset,
                NcnnWeightKind::Scale,
                channels,
            )?);
            if bias_term {
                blobs.push(consume_plain_f32_blob(
                    data,
                    offset,
                    NcnnWeightKind::Bias,
                    channels,
                )?);
            }
            Ok(blobs)
        }
        "Bias" => {
            let channels = layer.param_usize(0).unwrap_or(0);
            Ok(vec![consume_plain_f32_blob(
                data,
                offset,
                NcnnWeightKind::Bias,
                channels,
            )?])
        }
        _ => Ok(Vec::new()),
    }
}

fn consume_tagged_blob(
    data: &[u8],
    offset: &mut usize,
    kind: NcnnWeightKind,
    elem_count: usize,
) -> Result<NcnnWeightBlob> {
    let flag_offset = *offset;
    let flag = read_u32_le(data, offset)?;
    let storage = match flag {
        0 => NcnnWeightStorage::Float32,
        0x0130_6b47 => NcnnWeightStorage::Float16,
        0x000d_4b38 => NcnnWeightStorage::Int8,
        other => {
            return Err(AmbarError::UnsupportedNcnnWeightFlag {
                flag: other,
                offset: flag_offset,
            });
        }
    };
    let byte_len = elem_count.checked_mul(storage.elem_size()).ok_or_else(|| {
        AmbarError::InvalidNcnnWeights {
            offset: *offset,
            message: "weight blob byte size overflow".to_owned(),
        }
    })?;
    let data_offset = *offset;
    advance_weight_offset(data, offset, byte_len)?;

    Ok(NcnnWeightBlob {
        kind,
        offset: data_offset,
        byte_len,
        elem_count,
        storage,
    })
}

fn consume_plain_f32_blob(
    data: &[u8],
    offset: &mut usize,
    kind: NcnnWeightKind,
    elem_count: usize,
) -> Result<NcnnWeightBlob> {
    let byte_len = elem_count
        .checked_mul(4)
        .ok_or_else(|| AmbarError::InvalidNcnnWeights {
            offset: *offset,
            message: "f32 blob byte size overflow".to_owned(),
        })?;
    let data_offset = *offset;
    advance_weight_offset(data, offset, byte_len)?;

    Ok(NcnnWeightBlob {
        kind,
        offset: data_offset,
        byte_len,
        elem_count,
        storage: NcnnWeightStorage::Float32,
    })
}

fn read_u32_le(data: &[u8], offset: &mut usize) -> Result<u32> {
    let bytes = data
        .get(*offset..*offset + 4)
        .ok_or_else(|| AmbarError::InvalidNcnnWeights {
            offset: *offset,
            message: "unexpected end of file while reading blob flag".to_owned(),
        })?;
    *offset += 4;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn advance_weight_offset(data: &[u8], offset: &mut usize, byte_len: usize) -> Result<()> {
    let end = offset
        .checked_add(byte_len)
        .ok_or_else(|| AmbarError::InvalidNcnnWeights {
            offset: *offset,
            message: "weight offset overflow".to_owned(),
        })?;
    if end > data.len() {
        return Err(AmbarError::InvalidNcnnWeights {
            offset: *offset,
            message: format!(
                "blob needs {byte_len} bytes, but only {} remain",
                data.len().saturating_sub(*offset)
            ),
        });
    }
    *offset = end;
    Ok(())
}
