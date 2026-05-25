mod ambar;
mod burn_backend;
mod draw;
mod ncnn;
mod types;

pub use ambar::Ambar;
pub use burn_backend::{BurnNcnnBackend, BurnVisionNcnnBackend};
pub use draw::DrawBoxesExt;
pub use ncnn::{NcnnModel, NcnnParam, NcnnWeights};
pub use types::{
    AmbarError, BackendConfig, BoundingBox, BurnBackendConfig, BurnDevice, COCO_CLASSES, Config,
    Detection, Detections, InferenceBackend, ModelConfig, ModelOutput, ModelPreset,
    PreprocessedImage, Result,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct StaticBackend(ModelOutput);

    impl InferenceBackend for StaticBackend {
        fn infer(&self, _input: &PreprocessedImage) -> Result<ModelOutput> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn bbox_iou_matches_expected_overlap() {
        let a = BoundingBox {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
        };
        let b = BoundingBox {
            x: 5.0,
            y: 5.0,
            width: 10.0,
            height: 10.0,
        };

        assert!((a.iou(b) - 25.0 / 175.0).abs() < 1e-6);
    }

    #[test]
    fn decodes_yolox_output_and_applies_nms() {
        let mut config = Config::default();
        config.model.input_size = 8;
        config.strides = vec![8];
        config.model.class_names = vec!["object".to_owned()];
        config.max_detections = 10;

        let output = ModelOutput::new(vec![0.5, 0.5, 0.0, 0.0, 0.9, 0.9], 6);
        let ambar = Ambar::with_backend(config, StaticBackend(output));
        let image = image::DynamicImage::new_rgb8(16, 16);
        let detections = ambar.infer(&image).unwrap();

        assert_eq!(detections.len(), 1);
        let detection = detections.iter().next().unwrap();
        assert_eq!(detection.class_id, 0);
        assert!((detection.confidence - 0.81).abs() < 1e-6);
    }

    #[test]
    fn unavailable_backend_reports_a_clear_error() {
        let ambar = Ambar::new(Config::default());
        let image = image::DynamicImage::new_rgb8(8, 8);
        let error = ambar.infer(&image).unwrap_err();

        assert!(matches!(error, AmbarError::BackendUnavailable(_)));
    }

    #[test]
    fn parses_ncnn_param_and_indexes_names() {
        let param = NcnnParam::parse(
            r#"
7767517
3 4
Input images 0 1 images
Convolution conv 1 1 images conv_out 0=16 1=3 5=1 6=27
Split split 1 2 conv_out a b
"#,
        )
        .unwrap();

        assert_eq!(param.layer_count, 3);
        assert_eq!(param.blob_index("conv_out"), Some(1));
        assert_eq!(param.layer("conv").unwrap().param_i32(6), Some(27));
    }

    #[test]
    fn indexes_ncnn_bin_without_copying_weights() {
        use std::sync::Arc;
        let param = NcnnParam::parse(
            r#"
7767517
2 2
Input images 0 1 images
Convolution conv 1 1 images conv_out 0=2 1=1 5=1 6=3
"#,
        )
        .unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x0130_6b47_u32.to_le_bytes());
        bytes.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());

        let weights = NcnnWeights::from_bytes_for_param(Arc::<[u8]>::from(bytes), &param).unwrap();
        let layer = weights.layer(1).unwrap();

        assert_eq!(weights.bytes_consumed, 18);
        assert_eq!(layer.blobs.len(), 2);
        assert_eq!(layer.blobs[0].storage, ncnn::NcnnWeightStorage::Float16);
        assert_eq!(weights.blob_bytes(&layer.blobs[0]), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn loads_local_yolox_ncnn_fixture_when_available() {
        use std::path::Path;
        let param_path = Path::new("../YoloX-ncnn-Raspberry-Pi-4/yoloxN.param");
        let bin_path = Path::new("../YoloX-ncnn-Raspberry-Pi-4/yoloxN.bin");
        if !param_path.exists() || !bin_path.exists() {
            return;
        }

        let model = NcnnModel::from_files(param_path, bin_path).unwrap();
        assert_eq!(model.param.layers.len(), 280);
        assert_eq!(model.weights.bytes_consumed, model.weights.as_bytes().len());
        assert_eq!(
            model.param.layer("Conv_41").unwrap().param_i32(6),
            Some(1728)
        );
    }
}
