//! Person detection with YOLOX-nano (Apache-2.0, COCO, ~0.9M params) via tract.
//!
//! The exported ONNX graph (release 0.1.1rc0) expects a 416x416 BGR image in
//! the 0..255 range, letterboxed into the top-left corner with padding value
//! 114, and emits raw per-anchor predictions `[1, 3549, 85]` with sigmoid
//! already applied to objectness and class scores. Box decoding (grid offsets
//! and `exp` on width/height) happens here.

use anyhow::{ensure, Context};
use serde::Serialize;
use tract_onnx::prelude::*;

use crate::image::Frame;

pub const INPUT_SIZE: usize = 416;
const STRIDES: [usize; 3] = [8, 16, 32];
const OUTPUTS_PER_ANCHOR: usize = 85; // 4 box + 1 objectness + 80 COCO classes
const PERSON_CLASS: usize = 0;
const PAD_VALUE: f32 = 114.0;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)
    }

    pub fn diagonal(&self) -> f32 {
        self.width().hypot(self.height())
    }

    pub fn shifted(&self, dx: f32, dy: f32) -> BBox {
        BBox { x1: self.x1 + dx, y1: self.y1 + dy, x2: self.x2 + dx, y2: self.y2 + dy }
    }

    pub fn iou(&self, other: &BBox) -> f32 {
        let ix = (self.x2.min(other.x2) - self.x1.max(other.x1)).max(0.0);
        let iy = (self.y2.min(other.y2) - self.y1.max(other.y1)).max(0.0);
        let inter = ix * iy;
        let union = self.area() + other.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Detection {
    pub bbox: BBox,
    pub score: f32,
}

pub struct PersonDetector {
    model: TypedRunnableModel<TypedModel>,
    pub score_threshold: f32,
    pub nms_iou: f32,
    /// Boxes shorter than this fraction of the frame height are ignored.
    pub min_height_fraction: f32,
}

impl PersonDetector {
    pub fn from_onnx_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(bytes))
            .context("parsing detector ONNX")?
            .with_input_fact(0, f32::fact([1, 3, INPUT_SIZE, INPUT_SIZE]).into())?
            .into_optimized()
            .context("optimizing detector graph")?
            .into_runnable()?;
        Ok(Self { model, score_threshold: 0.35, nms_iou: 0.45, min_height_fraction: 0.04 })
    }

    pub fn detect(&self, frame: &Frame) -> anyhow::Result<Vec<Detection>> {
        let (input, scale) = letterbox(frame);
        let outputs = self.model.run(tvec!(input.into()))?;
        let raw = outputs[0].to_array_view::<f32>()?;
        let raw = raw.as_slice().context("detector output is not contiguous")?;
        let candidates = decode(raw, scale, self.score_threshold, frame.width as f32, frame.height as f32)?;
        let min_height = self.min_height_fraction * frame.height as f32;
        let candidates = candidates.into_iter().filter(|d| d.bbox.height() >= min_height).collect();
        Ok(nms(candidates, self.nms_iou))
    }
}

/// Resize preserving aspect ratio into the top-left of an `INPUT_SIZE` square,
/// returning the NCHW BGR tensor and the resize ratio.
fn letterbox(frame: &Frame) -> (Tensor, f32) {
    let s = INPUT_SIZE;
    let ratio = (s as f32 / frame.width as f32).min(s as f32 / frame.height as f32);
    let new_w = ((frame.width as f32 * ratio) as usize).min(s);
    let new_h = ((frame.height as f32 * ratio) as usize).min(s);
    let plane = s * s;
    let mut data = vec![PAD_VALUE; 3 * plane];
    for y in 0..new_h {
        let fy = (y as f32 + 0.5) / ratio - 0.5;
        for x in 0..new_w {
            let fx = (x as f32 + 0.5) / ratio - 0.5;
            let [r, g, b] = frame.sample_bilinear(fx, fy);
            let i = y * s + x;
            data[i] = b;
            data[plane + i] = g;
            data[2 * plane + i] = r;
        }
    }
    let tensor = Tensor::from_shape(&[1, 3, s, s], &data).expect("shape matches buffer");
    (tensor, ratio)
}

fn decode(raw: &[f32], ratio: f32, threshold: f32, frame_w: f32, frame_h: f32) -> anyhow::Result<Vec<Detection>> {
    let anchors: usize = STRIDES.iter().map(|st| (INPUT_SIZE / st).pow(2)).sum();
    ensure!(
        raw.len() == anchors * OUTPUTS_PER_ANCHOR,
        "unexpected detector output size {} (want {anchors}x{OUTPUTS_PER_ANCHOR})",
        raw.len()
    );
    let mut out = Vec::new();
    let mut rows = raw.chunks_exact(OUTPUTS_PER_ANCHOR);
    for stride in STRIDES {
        let grid = INPUT_SIZE / stride;
        let stride = stride as f32;
        for gy in 0..grid {
            for gx in 0..grid {
                let row = rows.next().expect("anchor count checked above");
                let score = row[4] * row[5 + PERSON_CLASS];
                if score < threshold {
                    continue;
                }
                let cx = (row[0] + gx as f32) * stride / ratio;
                let cy = (row[1] + gy as f32) * stride / ratio;
                let w = row[2].exp() * stride / ratio;
                let h = row[3].exp() * stride / ratio;
                let bbox = BBox {
                    x1: (cx - w * 0.5).clamp(0.0, frame_w),
                    y1: (cy - h * 0.5).clamp(0.0, frame_h),
                    x2: (cx + w * 0.5).clamp(0.0, frame_w),
                    y2: (cy + h * 0.5).clamp(0.0, frame_h),
                };
                out.push(Detection { bbox, score });
            }
        }
    }
    Ok(out)
}

pub fn nms(mut dets: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<Detection> = Vec::new();
    for d in dets {
        if keep.iter().all(|k| k.bbox.iou(&d.bbox) < iou_threshold) {
            keep.push(d);
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bx(x1: f32, y1: f32, x2: f32, y2: f32) -> BBox {
        BBox { x1, y1, x2, y2 }
    }

    #[test]
    fn iou_basics() {
        let a = bx(0.0, 0.0, 10.0, 10.0);
        assert_eq!(a.iou(&a), 1.0);
        assert_eq!(a.iou(&bx(20.0, 20.0, 30.0, 30.0)), 0.0);
        assert!((a.iou(&bx(5.0, 0.0, 15.0, 10.0)) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn nms_drops_overlaps_keeps_best() {
        let dets = vec![
            Detection { bbox: bx(0.0, 0.0, 10.0, 10.0), score: 0.6 },
            Detection { bbox: bx(1.0, 0.0, 11.0, 10.0), score: 0.9 },
            Detection { bbox: bx(50.0, 50.0, 60.0, 60.0), score: 0.5 },
        ];
        let kept = nms(dets, 0.45);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].score, 0.9);
    }

    #[test]
    fn decode_places_box_on_grid() {
        let anchors: usize = STRIDES.iter().map(|s| (INPUT_SIZE / s).pow(2)).sum();
        let mut raw = vec![0.0f32; anchors * OUTPUTS_PER_ANCHOR];
        // Anchor at stride 8, gx=3, gy=10 → index 10*52+3.
        let row = &mut raw[(10 * 52 + 3) * OUTPUTS_PER_ANCHOR..][..OUTPUTS_PER_ANCHOR];
        row[0] = 0.5;
        row[1] = 0.5;
        row[2] = (4.0f32).ln(); // w = 4 * 8 = 32
        row[3] = (8.0f32).ln(); // h = 64
        row[4] = 0.9;
        row[5] = 0.9;
        let dets = decode(&raw, 1.0, 0.5, 416.0, 416.0).unwrap();
        assert_eq!(dets.len(), 1);
        let (cx, cy) = dets[0].bbox.center();
        assert!((cx - 28.0).abs() < 1e-3 && (cy - 84.0).abs() < 1e-3);
        assert!((dets[0].bbox.height() - 64.0).abs() < 1e-3);
    }
}
