//! Native runner for vs1-core. Frames are decoded with ffmpeg at the same
//! timestamps and size the web page uses, so results match the browser build.
//!
//!   vs1 evaluate <video> --questions q.json [--frames 16] [--head h.safetensors]
//!   vs1 embed-video <video>... --out-dir DIR [--frames 16]     # [T, 512] f32 .npy per video
//!   vs1 embed-video <video>... --out-dir DIR --crops-only       # crops of the crop frames: <stem>.crops.npy
//!   vs1 embed-state <video> --out rows.f32 [--frames 16]        # a video state for the web page (full rows, then crops)
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
  vs1 evaluate <video> --questions q.json [--frames 16] [--crop-frames 4] [--head h.safetensors] [--choice-head c.safetensors] [--models DIR]
  vs1 embed-video <video>... --out-dir DIR [--frames 16] [--crops-only [--crop-frames 4]] [--models DIR]
  vs1 embed-state <video> --out rows.f32 [--frames 16] [--crop-frames 4] [--models DIR]
  vs1 embed-questions <questions.jsonl> --out-dir DIR [--models DIR]
  vs1 embed-texts <questions.json> --out texts.f32 [--models DIR]
  vs1 check-head <head.safetensors> <parity.json>";

struct Args {
    command: String,
    positional: Vec<PathBuf>,
    questions: Option<PathBuf>,
    frames: usize,
    /// Frames that also get crop embeddings (`vs1_core::crop_frame_indices`).
    crop_frames: usize,
    crops_only: bool,
    head: Option<PathBuf>,
    choice_head: Option<PathBuf>,
    models: PathBuf,
    out_dir: Option<PathBuf>,
    out: Option<PathBuf>,
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
        crop_frames: vs1_core::CROP_FRAMES,
        crops_only: false,
        head: None,
        choice_head: None,
        models: PathBuf::from(DEFAULT_MODELS),
        out_dir: None,
        out: None,
    };
    while let Some(a) = it.next() {
        let mut value = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--questions" => args.questions = Some(value()?.into()),
            "--frames" => args.frames = value()?.parse()?,
            "--crop-frames" => args.crop_frames = value()?.parse()?,
            "--crops-only" => args.crops_only = true,
            "--head" => args.head = Some(value()?.into()),
            "--choice-head" => args.choice_head = Some(value()?.into()),
            "--models" => args.models = value()?.into(),
            "--out-dir" => args.out_dir = Some(value()?.into()),
            "--out" => args.out = Some(value()?.into()),
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
        vision_onnx: Some(&read("vision.onnx")?),
        text: Some((&read("text.onnx")?, &read("tokenizer.json")?)),
        head: head.as_deref(),
    })?;
    eprintln!("loaded {} in {:.0} ms", model.model_name(), start.elapsed().as_secs_f64() * 1e3);
    Ok(model)
}

/// Uniformly sampled RGBA frames: T frames at the midpoints of T equal spans
/// (only those at `only` indices, if given).
fn sample_frames(video: &Path, count: usize, only: Option<&[usize]>) -> anyhow::Result<(Vec<Vec<u8>>, usize, usize)> {
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
    for i in (0..count).filter(|i| only.is_none_or(|o| o.contains(i))) {
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

/// The video state's embedding rows: every sampled frame's full view, plus
/// crops of `crop_frames` of them (as the browser computes it). Also returns
/// the number of frames decoded.
fn embed_video(model: &SystemOne, video: &Path, count: usize, crop_frames: usize) -> anyhow::Result<(Vec<Vec<f32>>, usize)> {
    let (frames, w, h) = sample_frames(video, count, None)?;
    let start = Instant::now();
    let with_crops = vs1_core::crop_frame_indices(frames.len(), crop_frames);
    // Full views first, then crops (the order `SystemOne::answer` expects).
    let mut rows = Vec::new();
    let mut crops = Vec::new();
    for (i, rgba) in frames.iter().enumerate() {
        let frame = Frame::new(rgba, w, h)?;
        rows.push(model.vision()?.embed(&frame)?);
        if with_crops.contains(&i) {
            crops.extend(model.vision()?.embed_crops(&frame)?);
        }
    }
    rows.extend(crops);
    eprintln!(
        "{}: {} frames at {w}x{h} → {} rows, {:.0} ms/row",
        video.display(),
        frames.len(),
        rows.len(),
        start.elapsed().as_secs_f64() * 1e3 / rows.len() as f64
    );
    Ok((rows, frames.len()))
}

/// Only the crop rows (training stores them next to the full-frame rows).
fn embed_crops(model: &SystemOne, video: &Path, count: usize, crop_frames: usize) -> anyhow::Result<Vec<Vec<f32>>> {
    let idx = vs1_core::crop_frame_indices(count, crop_frames);
    let (frames, w, h) = sample_frames(video, count, Some(&idx))?;
    let mut rows = Vec::new();
    for rgba in &frames {
        rows.extend(model.vision()?.embed_crops(&Frame::new(rgba, w, h)?)?);
    }
    Ok(rows)
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
            if let Some(p) = &args.choice_head {
                model.load_choice_head(&std::fs::read(p).with_context(|| format!("reading {}", p.display()))?)?;
            }
            let compiled = model.compile(&request)?;
            let (rows, frames) = embed_video(&model, video, args.frames, args.crop_frames)?;
            println!("{}", serde_json::to_string_pretty(&model.answer(&rows, frames, &compiled)?)?);
        }
        "embed-video" => {
            ensure!(!args.positional.is_empty(), "embed-video needs at least one video\n{USAGE}");
            let out_dir = args.out_dir.context("--out-dir is required")?;
            std::fs::create_dir_all(&out_dir)?;
            let model = load_model(&args.models, None)?;
            for video in &args.positional {
                let stem = video.file_stem().context("video path has no file name")?;
                let name = format!("{}{}", stem.to_string_lossy(), if args.crops_only { ".crops.npy" } else { ".npy" });
                let dest = out_dir.join(name);
                if dest.exists() {
                    continue;
                }
                let rows = if args.crops_only {
                    embed_crops(&model, video, args.frames, args.crop_frames)
                } else {
                    embed_video(&model, video, args.frames, 0).map(|(rows, _)| rows)
                };
                match rows {
                    Ok(rows) => write_npy(&dest, &rows)?,
                    Err(e) => eprintln!("skipping {}: {e:#}", video.display()),
                }
            }
        }
        "embed-state" => {
            // Raw little-endian f32 rows, exactly what the page computes for
            // the video: one full view per frame, then the crop rows.
            let [video] = args.positional.as_slice() else { bail!("embed-state takes one video\n{USAGE}") };
            let out = args.out.context("--out is required")?;
            let model = load_model(&args.models, None)?;
            let (rows, frames) = embed_video(&model, video, args.frames, args.crop_frames)?;
            let bytes: Vec<u8> = rows.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(&out, bytes).with_context(|| format!("writing {}", out.display()))?;
            println!("{}", serde_json::json!({ "frames": frames, "rows": rows.len() }));
        }
        "embed-texts" => {
            // Text embeddings for a JSON list of questions (a string is a
            // yes/no statement, an object a Jev question), for the web page to
            // answer them before its text encoder downloads: 513 floats per
            // text (embedding, then token count) in texts.f32, and the
            // (context, focus) keys in texts.json next to it.
            let [path] = args.positional.as_slice() else { bail!("embed-texts takes one file\n{USAGE}") };
            let out = args.out.context("--out is required")?;
            let items: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(path)?)?;
            let questions: serde_json::Map<String, serde_json::Value> = items
                .into_iter()
                .enumerate()
                .map(|(i, q)| match q {
                    serde_json::Value::String(s) => (format!("s{i}"), serde_json::json!({ "type": "noul", "instructions": s })),
                    q => (format!("s{i}"), q),
                })
                .collect();
            let request = Request::parse(&serde_json::json!({ "questions": questions }).to_string())
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut model = load_model(&args.models, None)?;
            let keys = model.missing_texts(&request)?;
            let mut floats = Vec::with_capacity(keys.len() * 513);
            for (context, focus) in &keys {
                let (emb, tokens) = model.embed_text(context, focus)?;
                floats.extend(emb);
                floats.push(tokens as f32);
            }
            let bytes: Vec<u8> = floats.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(&out, bytes)?;
            std::fs::write(out.with_extension("json"), serde_json::to_string(&keys)?)?;
            println!("{}", serde_json::json!({ "texts": keys.len() }));
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
                index.push_str(
                    &serde_json::json!({ "question": question, "options": options, "negated": prompts.negated }).to_string(),
                );
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
                let frames = rows(&case["frames"]);
                let n_full = case["n_full"].as_u64().map_or(frames.len(), |n| n as usize);
                let probs = head.probabilities(&frames, n_full, &floats(&case["question"]), &rows(&case["options"]));
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
