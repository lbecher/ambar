use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;

use ambar::{
    Ambar, BackendConfig, BurnBackendConfig, BurnDevice, Config, DrawBoxesExt, ModelConfig,
    ModelPreset,
};

/// Run YOLOX object detection on an image using Burn backends.
#[derive(Debug, Parser)]
#[command(name = "ambar-infer", version)]
struct Cli {
    /// Input image path
    image: PathBuf,

    /// Output image path
    #[arg(short, long, default_value = "output.jpg")]
    output: PathBuf,

    /// Model preset: nano, tiny, small
    #[arg(short, long, default_value = "small")]
    model: ModelArg,

    /// Custom model .param file (requires --bin)
    #[arg(long, requires = "bin")]
    param: Option<PathBuf>,

    /// Custom model .bin file (requires --param)
    #[arg(long, requires = "param")]
    bin: Option<PathBuf>,

    /// Burn backend to use: ndarray, flex, wgpu, metal, wgpu-raw, metal-raw
    #[arg(short, long, default_value = "flex")]
    backend: BackendArg,

    /// Confidence threshold for detections
    #[arg(long, default_value_t = 0.25)]
    threshold: f32,

    /// Max candidates to consider before NMS (CPU backends; GPU uses full readback for correctness)
    #[arg(long, default_value_t = 1000)]
    candidates: usize,

    /// Enable multi-label mode (one box may produce multiple class detections)
    #[arg(long)]
    multi_label: bool,

    /// Number of warmup inference runs (not timed)
    #[arg(short, long, default_value_t = 0)]
    warmup: usize,

    /// Number of timed inference runs
    #[arg(short, long, alias = "test-runs", default_value_t = 1)]
    runs: usize,

    /// CSV output path for per-run benchmark results
    #[arg(long)]
    test_csv: Option<PathBuf>,

    /// Print raw model output statistics before decoding
    #[arg(long)]
    debug_output: bool,

    /// Print selected backend layer statistics. Use comma-separated names, "early", "checkpoints", or "heads".
    #[arg(long)]
    debug_layers: Option<String>,

    /// Disable manual Conv+Swish fusion, useful when debugging backend divergence.
    #[arg(long)]
    disable_fused_swish: bool,
}

#[derive(Clone, Debug, clap::ValueEnum)]
enum ModelArg {
    Nano,
    Tiny,
    Small,
}

#[derive(Clone, Debug, clap::ValueEnum)]
enum BackendArg {
    Ndarray,
    Flex,
    Wgpu,
    Metal,
    WgpuRaw,
    MetalRaw,
}

impl BackendArg {
    fn uses_gpu(&self) -> bool {
        match self {
            Self::Ndarray | Self::Flex => false,
            Self::Wgpu | Self::Metal | Self::WgpuRaw | Self::MetalRaw => true,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    if cli.runs == 0 {
        eprintln!("error: --runs must be greater than zero");
        std::process::exit(2);
    }

    if let Some(debug_layers) = cli.debug_layers.as_deref() {
        let layers = match debug_layers {
            "early" => "focus,Conv_41,Mul_43,Conv_44,Mul_46",
            "block1" => {
                "Conv_47,Mul_49,Conv_50,Mul_52,Conv_53,Mul_55,Conv_56,Mul_58,Conv_59,Mul_61,Conv_62,Mul_64,Add_65,Concat_66,Conv_67,Mul_69"
            }
            "mid1" => "Mul_69,Mul_115,Mul_161",
            "mid2" => "Concat_193,Mul_196,Concat_218,Mul_221,Concat_243,Mul_246",
            "mid3" => "Concat_269,Mul_272,Concat_295,Mul_298",
            "checkpoints" => {
                "Mul_46,Mul_69,Mul_115,Mul_161,Concat_193,Mul_196,Concat_218,Mul_221,Concat_243,Mul_246,Concat_269,Mul_272,Concat_295,Mul_298,Concat_331,Concat_364,Concat_397,Transpose_423"
            }
            "heads" => "Reshape_405,Reshape_413,Reshape_421,Concat_422,Transpose_423",
            layers => layers,
        };
        // Set before Burn initializes GPU worker threads.
        unsafe {
            std::env::set_var("AMBAR_DEBUG_LAYERS", layers);
        }
    }

    if cli.disable_fused_swish {
        // Set before Burn initializes GPU worker threads.
        unsafe {
            std::env::set_var("AMBAR_DISABLE_FUSED_SWISH", "1");
        }
    }

    let preset = match cli.model {
        ModelArg::Nano => ModelPreset::YoloXNano,
        ModelArg::Tiny => ModelPreset::YoloXTiny,
        ModelArg::Small => ModelPreset::YoloXSmall,
    };

    let gpu = usize::from(cli.backend.uses_gpu());
    let burn_device = match cli.backend {
        BackendArg::Ndarray => BurnDevice::Cpu,
        BackendArg::Flex => BurnDevice::Flex,
        BackendArg::Wgpu => BurnDevice::Wgpu,
        BackendArg::Metal => BurnDevice::Metal,
        BackendArg::WgpuRaw => BurnDevice::WgpuRaw,
        BackendArg::MetalRaw => BurnDevice::MetalRaw,
    };

    let model = ModelConfig::preset(preset);
    let target_size = model.input_size;

    let config = Config {
        model,
        backend: BackendConfig::Burn(BurnBackendConfig {
            device: burn_device,
            model_path: None,
        }),
        prob_threshold: cli.threshold,
        max_candidates_before_nms: cli.candidates,
        multi_label: cli.multi_label,
        ..Config::default()
    };

    let (param_path, bin_path) = match (cli.param, cli.bin) {
        (Some(param), Some(bin)) => (param, bin),
        _ => (PathBuf::from("yoloxN.param"), PathBuf::from("yoloxN.bin")),
    };
    let ambar = Ambar::from_ncnn_files(config, &param_path, &bin_path)?;

    let image = image::open(&cli.image)?;
    let input = ambar.preprocess(&image)?;

    if cli.debug_output {
        let output = ambar.infer_raw_preprocessed(&input)?;
        print_output_debug(&output);
    }

    if cli.warmup > 0 {
        for _ in 0..cli.warmup {
            let _ = ambar.infer_preprocessed(&input)?;
        }
    }

    let mut csv = match cli.test_csv.as_deref() {
        Some(path) => Some(create_test_csv(path)?),
        None => None,
    };
    if let Some(path) = cli.test_csv.as_deref() {
        println!(
            "Running test mode with {} repetitions. CSV: {}",
            cli.runs,
            path.display()
        );
    }

    let mut result = None;
    let mut elapsed_ms: Vec<f64> = Vec::with_capacity(cli.runs);

    for run in 1..=cli.runs {
        let start = Instant::now();
        result = Some(ambar.infer_preprocessed(&input)?);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        elapsed_ms.push(elapsed);

        if let Some(csv) = csv.as_mut() {
            let detections = result.as_ref().expect("result was just set");
            writeln!(
                csv,
                "{run},{elapsed},{objects_detected},{image_path},{param_path},{bin_path},{target_size},0,{gpu}",
                objects_detected = detections.len(),
                image_path = csv_escape_path(&cli.image),
                param_path = csv_escape_path(&param_path),
                bin_path = csv_escape_path(&bin_path),
            )?;
        }
    }
    let result = result.expect("runs >= 1, so result is always set");

    if let Some(mut csv) = csv {
        csv.flush()?;
        if let Some(path) = cli.test_csv.as_deref() {
            println!("Test results saved to {}", path.display());
        }
    }

    // Print detections
    let labels: Vec<String> = result
        .iter()
        .map(|d| {
            let name = d.class_name.as_deref().unwrap_or("?");
            format!("[{} {:.2}]", name, d.confidence)
        })
        .collect();
    if labels.is_empty() {
        println!("No detections.");
    } else {
        println!("{}", labels.join(" "));
    }

    // Timing report
    if cli.warmup > 0 {
        println!("Warmup runs: {}", cli.warmup);
    }
    if cli.runs == 1 {
        println!("Inference time: {:.2} ms", elapsed_ms[0]);
    } else {
        let total: f64 = elapsed_ms.iter().sum();
        let avg = total / cli.runs as f64;
        let min = elapsed_ms.iter().copied().fold(f64::INFINITY, f64::min);
        let max = elapsed_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        println!("Inference runs: {}", cli.runs);
        println!("Average: {avg:.2} ms  Min: {min:.2} ms  Max: {max:.2} ms");
    }

    // Save annotated image
    let mut output = image.clone();
    output.draw_boxes(&result);
    output.save(&cli.output)?;
    println!("Saved to {}", cli.output.display());

    Ok(())
}

fn create_test_csv(path: &Path) -> std::io::Result<BufWriter<File>> {
    let mut csv = BufWriter::new(File::create(path)?);
    writeln!(
        csv,
        "run,elapsed_ms,objects_detected,image_path,param_path,bin_path,target_size,threads,gpu"
    )?;
    Ok(csv)
}

fn csv_escape_path(path: &Path) -> String {
    csv_escape(&path.display().to_string())
}

fn csv_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for c in value.chars() {
        if c == '"' {
            escaped.push('"');
            escaped.push('"');
        } else {
            escaped.push(c);
        }
    }
    escaped.push('"');
    escaped
}

fn print_output_debug(output: &ambar::ModelOutput) {
    let row_len = output.row_len;
    let rows = if row_len == 0 {
        0
    } else {
        output.data.len() / row_len
    };
    let mut min_value = f32::INFINITY;
    let mut max_value = f32::NEG_INFINITY;
    let mut finite = 0usize;
    let mut nan = 0usize;
    let mut best = (f32::NEG_INFINITY, 0usize, 0usize, 0.0f32, 0.0f32);

    for (row_index, row) in output.data.chunks_exact(row_len).enumerate() {
        for &value in row {
            if value.is_finite() {
                finite += 1;
                min_value = min_value.min(value);
                max_value = max_value.max(value);
            } else if value.is_nan() {
                nan += 1;
            }
        }

        if row_len > 5 {
            let objectness = row[4];
            if let Some((class_id, class_score)) = row[5..]
                .iter()
                .copied()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
            {
                let score = objectness * class_score;
                if score > best.0 {
                    best = (score, row_index, class_id, objectness, class_score);
                }
            }
        }
    }

    println!(
        "Raw output: rows={rows} row_len={row_len} values={} finite={finite} nan={nan} min={min_value:.6} max={max_value:.6}",
        output.data.len()
    );
    if row_len > 5 {
        println!(
            "Best candidate: score={:.6} row={} class={} objectness={:.6} class_score={:.6}",
            best.0, best.1, best.2, best.3, best.4
        );
    }
}
