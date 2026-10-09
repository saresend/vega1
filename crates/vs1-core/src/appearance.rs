//! Appearance embedding for re-identifying subjects across gaps and cuts.
//!
//! Prototype: hue/saturation colour histograms of the upper and lower body,
//! Hellinger-normalised so a dot product gives a Bhattacharyya similarity in
//! [0, 1]. This is cheap and needs no model, but it confuses people wearing
//! similar colours — swap for a learned re-ID embedding when one is trained.

use crate::detect::BBox;
use crate::image::Frame;

const HUE_BINS: usize = 12;
const SAT_BINS: usize = 2;
const GRAY_BINS: usize = 4;
const BINS_PER_PART: usize = HUE_BINS * SAT_BINS + GRAY_BINS;
pub const EMBED_DIM: usize = BINS_PER_PART * 2;

/// Sampling grid per box; keeps cost independent of box size.
const GRID_W: usize = 16;
const GRID_H: usize = 32;

pub fn embed(frame: &Frame, bbox: &BBox) -> Vec<f32> {
    let mut hist = vec![0.0f32; EMBED_DIM];
    // Central 60% of the width to reduce background; skip the top 10% (often
    // background above the head) and the bottom 5% (floor).
    let x0 = bbox.x1 + 0.2 * bbox.width();
    let w = 0.6 * bbox.width();
    let y0 = bbox.y1 + 0.10 * bbox.height();
    let h = 0.85 * bbox.height();
    for gy in 0..GRID_H {
        let v = (gy as f32 + 0.5) / GRID_H as f32;
        let part = if v < 0.45 { 0 } else { 1 };
        let py = ((y0 + v * h) as usize).min(frame.height - 1);
        for gx in 0..GRID_W {
            let px = ((x0 + (gx as f32 + 0.5) / GRID_W as f32 * w) as usize).min(frame.width - 1);
            let bin = color_bin(frame.rgb(px, py));
            hist[part * BINS_PER_PART + bin] += 1.0;
        }
    }
    for part in hist.chunks_mut(BINS_PER_PART) {
        let total: f32 = part.iter().sum();
        if total > 0.0 {
            part.iter_mut().for_each(|b| *b = (*b / total).sqrt());
        }
    }
    normalize(&mut hist);
    hist
}

/// Similarity in [0, 1]; 1 means identical colour distributions.
pub fn similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>().clamp(0.0, 1.0)
}

/// Exponential moving average of embeddings, renormalised.
pub fn blend(into: &mut [f32], new: &[f32], keep: f32) {
    for (a, b) in into.iter_mut().zip(new) {
        *a = keep * *a + (1.0 - keep) * b;
    }
    normalize(into);
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

fn color_bin([r, g, b]: [u8; 3]) -> usize {
    let (r, g, b) = (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let sat = if max > 0.0 { delta / max } else { 0.0 };
    if sat < 0.2 || max < 0.15 {
        // Achromatic: bucket by brightness.
        return HUE_BINS * SAT_BINS + ((max * GRAY_BINS as f32) as usize).min(GRAY_BINS - 1);
    }
    let hue = if max == r {
        ((g - b) / delta).rem_euclid(6.0)
    } else if max == g {
        (b - r) / delta + 2.0
    } else {
        (r - g) / delta + 4.0
    } / 6.0;
    let h = ((hue * HUE_BINS as f32) as usize).min(HUE_BINS - 1);
    let s = if sat < 0.55 { 0 } else { 1 };
    h * SAT_BINS + s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_tone(top: [u8; 3], bottom: [u8; 3]) -> Vec<u8> {
        let (w, h) = (20, 40);
        let mut px = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            let c = if y < h / 2 { top } else { bottom };
            for _ in 0..w {
                px.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        px
    }

    #[test]
    fn same_outfit_matches_different_outfit_does_not() {
        let bbox = BBox { x1: 0.0, y1: 0.0, x2: 20.0, y2: 40.0 };
        let red_blue = two_tone([200, 30, 30], [30, 30, 200]);
        let green_gray = two_tone([30, 200, 30], [128, 128, 128]);
        let a = embed(&Frame::new(&red_blue, 20, 40).unwrap(), &bbox);
        let b = embed(&Frame::new(&red_blue, 20, 40).unwrap(), &bbox);
        let c = embed(&Frame::new(&green_gray, 20, 40).unwrap(), &bbox);
        assert!(similarity(&a, &b) > 0.99);
        assert!(similarity(&a, &c) < 0.2);
    }
}
