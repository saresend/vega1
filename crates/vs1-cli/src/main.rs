//! Native runner: `vs1 <video> [--fps 4] [--width 640] [--model PATH] [--frames out.jsonl]`
//!
//! Decodes with ffmpeg at the same sampling the web page uses, so results
//! here match what the browser build produces.

use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{bail, Context};

const DEFAULT_MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/yolox_nano.onnx");

struct Args {
    video: PathBuf,
    fps: f64,
    width: usize,
    model: PathBuf,
    frames_out: Option<PathBuf>,
}

fn parse_args() -> anyhow::Result<Args> {
    let mut args = std::env::args().skip(1);
    let mut video = None;
    let mut parsed = Args {
        video: PathBuf::new(),
        fps: 4.0,
        width: 640,
        model: PathBuf::from(DEFAULT_MODEL),
        frames_out: None,
    };
    while let Some(a) = args.next() {
        let mut value = || args.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--fps" => parsed.fps = value()?.parse()?,
            "--width" => parsed.width = value()?.parse()?,
            "--model" => parsed.model = value()?.into(),
            "--frames" => parsed.frames_out = Some(value()?.into()),
            "-h" | "--help" => {
                println!("usage: vs1 <video> [--fps 4] [--width 640] [--model PATH] [--frames out.jsonl]");
                std::process::exit(0);
            }
            _ if a.starts_with("--") => bail!("unknown flag {a}"),
            _ => video = Some(PathBuf::from(a)),
        }
    }
    parsed.video = video.context("missing video path (see --help)")?;
    Ok(parsed)
}

struct VideoInfo {
    width: usize,
    height: usize,
    duration: f64,
}

fn probe_video(path: &PathBuf) -> anyhow::Result<VideoInfo> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration"])
        .args(["-of", "json"])
        .arg(path)
        .output()
        .context("running ffprobe (is ffmpeg installed?)")?;
    if !out.status.success() {
        bail!("ffprobe failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout)?;
    let stream = &v["streams"][0];
    Ok(VideoInfo {
        width: stream["width"].as_u64().context("no video stream")? as usize,
        height: stream["height"].as_u64().context("no video stream")? as usize,
        duration: v["format"]["duration"].as_str().and_then(|d| d.parse().ok()).unwrap_or(0.0),
    })
}

fn main() -> anyhow::Result<()> {
    let args = parse_args()?;
    let info = probe_video(&args.video)?;
    let width = args.width.min(info.width) & !1;
    let height = ((info.height as f64 * width as f64 / info.width as f64 / 2.0).round() as usize * 2).max(2);

    let model = std::fs::read(&args.model).with_context(|| format!("reading model {}", args.model.display()))?;
    let load = Instant::now();
    let mut analyzer = vs1_core::Analyzer::new(&model)?;
    eprintln!("model loaded in {:.0} ms", load.elapsed().as_secs_f64() * 1e3);

    let file_bytes = std::fs::read(&args.video)?;
    let probe = analyzer.probe_container(&file_bytes);
    eprintln!("container: {} ({} text tracks)", probe.format, probe.text_tracks.len());

    let mut ffmpeg = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&args.video)
        .args(["-vf", &format!("fps={},scale={width}:{height}", args.fps)])
        .args(["-f", "rawvideo", "-pix_fmt", "rgba", "-"])
        .stdout(Stdio::piped())
        .spawn()
        .context("running ffmpeg")?;
    let mut stdout = ffmpeg.stdout.take().expect("piped");

    let mut frames_out = match &args.frames_out {
        Some(p) => Some(BufWriter::new(std::fs::File::create(p)?)),
        None => None,
    };
    let mut buf = vec![0u8; width * height * 4];
    let mut index = 0usize;
    let mut busy = 0.0f64;
    while stdout.read_exact(&mut buf).is_ok() {
        let t = index as f64 / args.fps;
        let frame = vs1_core::Frame::new(&buf, width, height)?;
        let start = Instant::now();
        let result = analyzer.push_frame(&frame, t)?;
        busy += start.elapsed().as_secs_f64();
        if let Some(w) = frames_out.as_mut() {
            serde_json::to_writer(&mut *w, &result)?;
            w.write_all(b"\n")?;
        }
        index += 1;
    }
    ffmpeg.wait()?;
    if index == 0 {
        bail!("no frames decoded");
    }
    eprintln!(
        "{index} frames at {width}x{height}, {:.1} ms/frame average",
        busy * 1e3 / index as f64
    );

    let duration = if info.duration > 0.0 { info.duration } else { index as f64 / args.fps };
    println!("{}", serde_json::to_string_pretty(&analyzer.report(duration))?);
    Ok(())
}
