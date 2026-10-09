//! Native runner for vs1-core. Frames are decoded with ffmpeg at the same
//! timestamps and size the web page uses, so results match the browser build.
//!
//!   vs1 evaluate <video> --questions q.json [--frames 16] [--head h.safetensors]
//!   vs1 embed-video <video>... --out-dir DIR [--frames 16]     # [T, 512] f32 .npy per video
//!   vs1 embed-questions questions.jsonl --out-dir DIR           # deduplicated text embeddings
//!   vs1 check-head head.safetensors parity.json                 # Rust vs Python head outputs
//!
//! Common: --models DIR (default: models/vs1 in the repo)

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{bail, ensure, Context};
use vs1_core::api::{Question, Request};
use vs1_core::head::Head;
use vs1_core::{Frame, ModelFiles, SystemOne};

const DEFAULT_MODELS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/vs1");
const MAX_ANALYSIS_WIDTH: usize = 640;

const USAGE: &str = "usage:
  vs1 evaluate <video> --questions q.json [--frames 16] [--head h.safetensors] [--models DIR]
  vs1 embed-video <video>... --out-dir DIR [--frames 16] [--models DIR]
  vs1 embed-questions <questions.jsonl> --out-dir DIR [--models DIR]
  vs1 check-head <head.safetensors> <parity.json>";

struct Args {
    command: String,
    positional: Vec<PathBuf>,
    questions: Option<PathBuf>,
    frames: usize,
    head: Option<PathBuf>,
    models: PathBuf,
    out_dir: Option<PathBuf>,
}

fn parse_args() -> anyhow::Result<Args> {
    let mut it = std::env::args().skip(1);
    let command = it.next().unwrap_or_default();
    if command.is_empty() || command == "-h" || command == "--help" {
        println!("{USAGE}");
        std::process::exit(0);
    }
    let mut args = Args {
        command,
        positional: Vec::new(),
        questions: None,
        frames: 16,
        head: None,
        models: PathBuf::from(DEFAULT_MODELS),
        out_dir: None,
    };
    while let Some(a) = it.next() {
        let mut value = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--questions" => args.questions = Some(value()?.into()),
            "--frames" => args.frames = value()?.parse()?,
            "--head" => args.head = Some(value()?.into()),
            "--models" => args.models = value()?.into(),
            "--out-dir" => args.out_dir = Some(value()?.into()),
            _ if a.starts_with("--") => bail!("unknown flag {a}\n{USAGE}"),
            _ => args.positional.push(a.into()),
        }
    }
    ensure!(args.frames > 0, "--frames must be positive");
    Ok(args)
}

fn load_model(models: &Path, head: Option<&Path>) -> anyhow::Result<SystemOne> {
    let read = |name: &str| std::fs::read(models.join(name)).with_context(|| format!("reading {}", models.join(name).display()));
    let head = head.map(|p| std::fs::read(p).with_context(|| format!("reading {}", p.display()))).transpose()?;
    let start = Instant::now();
    let model = SystemOne::new(ModelFiles {
        vision_onnx: &read("vision.onnx")?,
        text_onnx: &read("text.onnx")?,
        tokenizer_json: &read("tokenizer.json")?,
        head: head.as_deref(),
    })?;
    eprintln!("loaded {} in {:.0} ms", model.model_name(), start.elapsed().as_secs_f64() * 1e3);
    Ok(model)
}

/// Uniformly sampled RGBA frames: T frames at the midpoints of T equal spans.
fn sample_frames(video: &Path, count: usize) -> anyhow::Result<(Vec<Vec<u8>>, usize, usize)> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration"])
        .args(["-of", "json"])
        .arg(video)
        .output()
        .context("running ffprobe (is ffmpeg installed?)")?;
    ensure!(out.status.success(), "ffprobe failed on {}: {}", video.display(), String::from_utf8_lossy(&out.stderr));
    let info: serde_json::Value = serde_json::from_slice(&out.stdout)?;
    let src_w = info["streams"][0]["width"].as_u64().context("no video stream")? as usize;
    let src_h = info["streams"][0]["height"].as_u64().context("no video stream")? as usize;
    let duration: f64 = info["format"]["duration"].as_str().and_then(|d| d.parse().ok()).context("unknown duration")?;
    let width = MAX_ANALYSIS_WIDTH.min(src_w) & !1;
    let height = ((src_h as f64 * width as f64 / src_w as f64 / 2.0).round() as usize * 2).max(2);

    let mut frames = Vec::with_capacity(count);
    for i in 0..count {
        let t = (i as f64 + 0.5) * duration / count as f64;
        let out = Command::new("ffmpeg")
            .args(["-v", "error", "-ss", &format!("{t:.3}"), "-i"])
            .arg(video)
            .args(["-frames:v", "1", "-vf", &format!("scale={width}:{height}"), "-f", "rawvideo", "-pix_fmt", "rgba", "-"])
            .output()
            .context("running ffmpeg")?;
        ensure!(out.status.success(), "ffmpeg failed at t={t:.2}s: {}", String::from_utf8_lossy(&out.stderr));
        if out.stdout.len() == width * height * 4 {
            frames.push(out.stdout);
        }
    }
    ensure!(!frames.is_empty(), "no frames decoded from {}", video.display());
    Ok((frames, width, height))
}

fn embed_video(model: &SystemOne, video: &Path, count: usize) -> anyhow::Result<Vec<Vec<f32>>> {
    let (frames, w, h) = sample_frames(video, count)?;
    let start = Instant::now();
    let embeddings = frames.iter().map(|rgba| model.embed_frame(&Frame::new(rgba, w, h)?)).collect::<anyhow::Result<Vec<_>>>()?;
    eprintln!(
        "{}: {} frames at {w}x{h}, {:.0} ms/frame",
        video.display(),
        embeddings.len(),
        start.elapsed().as_secs_f64() * 1e3 / embeddings.len() as f64
    );
    Ok(embeddings)
}

/// Minimal .npy (v1.0) writer for a 2-D little-endian f32 array.
fn write_npy(path: &Path, rows: &[Vec<f32>]) -> anyhow::Result<()> {
    let cols = rows.first().map_or(0, Vec::len);
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}", rows.len(), cols);
    let unpadded = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - unpadded % 64) % 64));
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for row in rows {
        bytes.extend(row.iter().flat_map(|v| v.to_le_bytes()));
    }
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}

fn read_request(path: &Path) -> anyhow::Result<Request> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Request::parse(&text).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
}

fn main() -> anyhow::Result<()> {
    let args = parse_args()?;
    match args.command.as_str() {
        "evaluate" => {
            let [video] = args.positional.as_slice() else { bail!("evaluate takes one video\n{USAGE}") };
            let request = read_request(args.questions.as_deref().context("--questions is required")?)?;
            let mut model = load_model(&args.models, args.head.as_deref())?;
            let compiled = model.compile(&request)?;
            let frames = embed_video(&model, video, args.frames)?;
            println!("{}", serde_json::to_string_pretty(&model.answer(&frames, &compiled)?)?);
        }
        "embed-video" => {
            ensure!(!args.positional.is_empty(), "embed-video needs at least one video\n{USAGE}");
            let out_dir = args.out_dir.context("--out-dir is required")?;
            std::fs::create_dir_all(&out_dir)?;
            let model = load_model(&args.models, None)?;
            for video in &args.positional {
                let stem = video.file_stem().context("video path has no file name")?;
                let dest = out_dir.join(stem).with_extension("npy");
                if dest.exists() {
                    continue;
                }
                match embed_video(&model, video, args.frames) {
                    Ok(frames) => write_npy(&dest, &frames)?,
                    Err(e) => eprintln!("skipping {}: {e:#}", video.display()),
                }
            }
        }
        "embed-questions" => {
            // One Jev question per line. Writes texts.npy (unique text
            // embeddings) and index.jsonl (per question: question row,
            // option rows), so shared options are embedded once.
            let [path] = args.positional.as_slice() else { bail!("embed-questions takes one file\n{USAGE}") };
            let out_dir = args.out_dir.context("--out-dir is required")?;
            std::fs::create_dir_all(&out_dir)?;
            let mut model = load_model(&args.models, None)?;
            let mut rows: Vec<Vec<f32>> = Vec::new();
            let mut seen: std::collections::HashMap<(String, String), usize> = std::collections::HashMap::new();
            let mut index = String::new();
            let start = Instant::now();
            let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
            for (line_no, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
                let q: Question = serde_json::from_str(line).with_context(|| format!("line {}", line_no + 1))?;
                let prompts = vs1_core::prompts::build(&q);
                let mut row_of = |context: &str, focus: &str| -> anyhow::Result<usize> {
                    let key = (context.to_string(), focus.to_string());
                    if let Some(&i) = seen.get(&key) {
                        return Ok(i);
                    }
                    rows.push(model.embed_text(context, focus)?.0);
                    seen.insert(key, rows.len() - 1);
                    Ok(rows.len() - 1)
                };
                let question = row_of("", &prompts.question)?;
                let options = prompts.options.iter().map(|o| row_of(&o.context, &o.focus)).collect::<anyhow::Result<Vec<_>>>()?;
                index.push_str(&serde_json::json!({ "question": question, "options": options }).to_string());
                index.push('\n');
            }
            write_npy(&out_dir.join("texts.npy"), &rows)?;
            std::fs::write(out_dir.join("index.jsonl"), index)?;
            eprintln!("{} unique texts in {:.1} s", rows.len(), start.elapsed().as_secs_f64());
        }
        "check-head" => {
            let [head_path, parity_path] = args.positional.as_slice() else { bail!("check-head takes two files\n{USAGE}") };
            let head = Head::Trained(vs1_core::head::TrainedHead::from_safetensors(&std::fs::read(head_path)?)?);
            let cases: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(parity_path)?)?;
            let floats = |v: &serde_json::Value| -> Vec<f32> { v.as_array().into_iter().flatten().filter_map(|x| x.as_f64()).map(|x| x as f32).collect() };
            let rows = |v: &serde_json::Value| -> Vec<Vec<f32>> { v.as_array().into_iter().flatten().map(floats).collect() };
            let mut worst = 0.0f32;
            for case in &cases {
                let probs = head.probabilities(&rows(&case["frames"]), &floats(&case["question"]), &rows(&case["options"]));
                for (a, b) in probs.iter().zip(floats(&case["probabilities"])) {
                    worst = worst.max((a - b).abs());
                }
            }
            println!("{} cases, max |rust - python| probability difference = {worst:.2e}", cases.len());
            ensure!(worst < 1e-3, "head parity check failed");
        }
        other => bail!("unknown command `{other}`\n{USAGE}"),
    }
    Ok(())
}
