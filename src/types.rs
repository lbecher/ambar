use std::fmt;
use std::path::PathBuf;

use thiserror::Error;

pub const COCO_CLASSES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

#[derive(Debug, Error)]
pub enum AmbarError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("image has zero width or height")]
    EmptyImage,
    #[error("the configured backend is not available: {0}")]
    BackendUnavailable(String),
    #[error("model output row length must be at least 6, got {0}")]
    InvalidOutputRowLength(usize),
    #[error("model output length {len} is not divisible by row length {row_len}")]
    InvalidOutputShape { len: usize, row_len: usize },
    #[error("model output has {anchors} anchors, but decoder expects {expected}")]
    AnchorCountMismatch { anchors: usize, expected: usize },
    #[error("invalid ncnn .param at line {line}: {message}")]
    InvalidNcnnParam { line: usize, message: String },
    #[error("invalid ncnn magic {found}; expected 7767517")]
    InvalidNcnnMagic { found: u32 },
    #[error("invalid ncnn .bin at byte {offset}: {message}")]
    InvalidNcnnWeights { offset: usize, message: String },
    #[error("unsupported ncnn weight flag 0x{flag:08x} at byte {offset}")]
    UnsupportedNcnnWeightFlag { flag: u32, offset: usize },
    #[error("unsupported ncnn layer '{layer_type}' ({name})")]
    UnsupportedNcnnLayer { layer_type: String, name: String },
    #[error("missing ncnn blob '{0}'")]
    MissingNcnnBlob(String),
    #[error("missing ncnn weights for layer {0}")]
    MissingNcnnWeights(String),
    #[error("invalid ncnn tensor shape for {name}: {message}")]
    InvalidNcnnShape { name: String, message: String },
}

pub type Result<T> = std::result::Result<T, AmbarError>;

#[derive(Clone, Debug)]
pub struct Config {
    pub model: ModelConfig,
    pub backend: BackendConfig,
    pub prob_threshold: f32,
    pub nms_threshold: f32,
    pub multi_label: bool,
    pub max_candidates_before_nms: usize,
    pub strides: Vec<u32>,
    pub max_detections: usize,
    pub fill_value: u8,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: ModelConfig::preset(ModelPreset::YoloXSmall),
            backend: BackendConfig::Auto,
            prob_threshold: 0.25,
            nms_threshold: 0.45,
            multi_label: false,
            max_candidates_before_nms: 1000,
            strides: vec![8, 16, 32],
            max_detections: 300,
            fill_value: 114,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub name: String,
    pub input_size: u32,
    pub class_names: Vec<String>,
}

impl ModelConfig {
    pub fn preset(preset: ModelPreset) -> Self {
        let (name, input_size) = match preset {
            ModelPreset::YoloXNano => ("yolox-nano", 416),
            ModelPreset::YoloXTiny => ("yolox-tiny", 416),
            ModelPreset::YoloXSmall => ("yolox-small", 640),
        };

        Self {
            name: name.to_owned(),
            input_size,
            class_names: COCO_CLASSES.iter().map(ToString::to_string).collect(),
        }
    }

    pub fn custom(name: impl Into<String>, input_size: u32, class_names: Vec<String>) -> Self {
        Self {
            name: name.into(),
            input_size,
            class_names,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ModelPreset {
    YoloXNano,
    YoloXTiny,
    YoloXSmall,
}

#[derive(Clone, Debug)]
pub enum BackendConfig {
    Auto,
    Burn(BurnBackendConfig),
    External(String),
}

#[derive(Clone, Debug)]
pub struct BurnBackendConfig {
    pub device: BurnDevice,
    pub model_path: Option<String>,
}

impl Default for BurnBackendConfig {
    fn default() -> Self {
        Self {
            device: BurnDevice::Flex,
            model_path: None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum BurnDevice {
    Cpu,
    Flex,
    Wgpu,
    Metal,
    WgpuRaw,
    MetalRaw,
    Cuda,
}

#[derive(Clone, Debug)]
pub struct Detection {
    pub bbox: BoundingBox,
    pub class_id: usize,
    pub class_name: Option<String>,
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl BoundingBox {
    #[inline]
    pub fn x2(self) -> f32 {
        self.x + self.width
    }

    #[inline]
    pub fn y2(self) -> f32 {
        self.y + self.height
    }

    #[inline]
    pub fn area(self) -> f32 {
        self.width.max(0.0) * self.height.max(0.0)
    }

    #[inline]
    pub fn intersection(self, other: Self) -> f32 {
        let left = self.x.max(other.x);
        let top = self.y.max(other.y);
        let right = self.x2().min(other.x2());
        let bottom = self.y2().min(other.y2());

        (right - left).max(0.0) * (bottom - top).max(0.0)
    }

    #[inline]
    pub fn iou(self, other: Self) -> f32 {
        let inter = self.intersection(other);
        let union = self.area() + other.area() - inter;
        if union <= f32::EPSILON {
            0.0
        } else {
            inter / union
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Detections {
    objects: Vec<Detection>,
}

impl Detections {
    pub fn new(objects: Vec<Detection>) -> Self {
        Self { objects }
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Detection> {
        self.objects.iter()
    }

    pub fn into_vec(self) -> Vec<Detection> {
        self.objects
    }
}

impl AsRef<[Detection]> for Detections {
    fn as_ref(&self) -> &[Detection] {
        &self.objects
    }
}

impl IntoIterator for Detections {
    type IntoIter = std::vec::IntoIter<Detection>;
    type Item = Detection;

    fn into_iter(self) -> Self::IntoIter {
        self.objects.into_iter()
    }
}

impl<'a> IntoIterator for &'a Detections {
    type IntoIter = std::slice::Iter<'a, Detection>;
    type Item = &'a Detection;

    fn into_iter(self) -> Self::IntoIter {
        self.objects.iter()
    }
}

#[derive(Clone, Debug)]
pub struct PreprocessedImage {
    pub width: u32,
    pub height: u32,
    pub input_size: u32,
    pub resized_width: u32,
    pub resized_height: u32,
    pub scale: f32,
    pub rgb_chw: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct ModelOutput {
    pub data: Vec<f32>,
    pub row_len: usize,
    pub(crate) row_indices: Option<Vec<usize>>,
}

impl ModelOutput {
    pub fn new(data: Vec<f32>, row_len: usize) -> Self {
        Self {
            data,
            row_len,
            row_indices: None,
        }
    }

    pub(crate) fn with_row_indices(
        data: Vec<f32>,
        row_len: usize,
        row_indices: Vec<usize>,
    ) -> Self {
        Self {
            data,
            row_len,
            row_indices: Some(row_indices),
        }
    }

    pub(crate) fn rows(&self) -> Result<impl ExactSizeIterator<Item = &[f32]>> {
        if self.row_len < 6 {
            return Err(AmbarError::InvalidOutputRowLength(self.row_len));
        }
        if self.data.len() % self.row_len != 0 {
            return Err(AmbarError::InvalidOutputShape {
                len: self.data.len(),
                row_len: self.row_len,
            });
        }

        Ok(self.data.chunks_exact(self.row_len))
    }
}

pub trait InferenceBackend: Send + Sync + fmt::Debug {
    fn infer(&self, input: &PreprocessedImage) -> Result<ModelOutput>;

    fn infer_detections(
        &self,
        _input: &PreprocessedImage,
        _config: &Config,
    ) -> Result<Option<Detections>> {
        Ok(None)
    }
}

#[derive(Debug)]
pub(crate) struct UnavailableBackend {
    pub config: BackendConfig,
}

impl InferenceBackend for UnavailableBackend {
    fn infer(&self, _input: &PreprocessedImage) -> Result<ModelOutput> {
        Err(AmbarError::BackendUnavailable(match &self.config {
            BackendConfig::Auto => {
                "no runtime has been attached; use Ambar::with_backend(...)".to_owned()
            }
            BackendConfig::Burn(config) => format!(
                "Burn {:?} backend was selected, but no Burn YOLOX module/weights were attached",
                config.device
            ),
            BackendConfig::External(name) => {
                format!(
                    "external backend '{name}' was selected, but no implementation was attached"
                )
            }
        }))
    }
}
