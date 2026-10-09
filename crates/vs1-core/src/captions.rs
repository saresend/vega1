//! Burned-in caption detection.
//!
//! Per frame we look for a *caption-like text band*: a short horizontal strip
//! (one or two lines tall) in the lower ~2/3 of the frame whose density of
//! strong vertical edges is far above its surroundings, roughly centred, and
//! filled like a line of words. Across frames, captions differ from other
//! on-screen text by *behaviour*: they sit at a stable height and their
//! content keeps changing, whereas logos and watermarks never change.
//!
//! Prototype heuristic, no model. The intended replacement is a small learned
//! text-region head on the shared backbone, trained on the same signals.

use serde::Serialize;

use crate::image::Gray;

const REGION_TOP: f32 = 0.35;
const REGION_BOTTOM: f32 = 0.98;
const EDGE_THRESHOLD: i16 = 48;
const ROW_DENSITY_THRESHOLD: f32 = 0.06;
const SIG_ROWS: usize = 4;
const SIG_COLS: usize = 32;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct CaptionObservation {
    /// 0..1 confidence that this frame shows a caption-like text band.
    pub score: f32,
    /// Band position as fractions of frame size.
    pub y_top: f32,
    pub y_bottom: f32,
    pub x_left: f32,
    pub x_right: f32,
    #[serde(skip)]
    signature: u128,
}

pub fn observe(gray: &Gray) -> Option<CaptionObservation> {
    let (w, h) = (gray.width, gray.height);
    let y0 = (h as f32 * REGION_TOP) as usize;
    let y1 = ((h as f32 * REGION_BOTTOM) as usize).min(h);
    let step = (w / 640).max(1);
    let cols = (w - step) / step;
    if y1 <= y0 + 4 || cols < 32 {
        return None;
    }
    let rows = y1 - y0;

    // Strong horizontal-gradient edges (vertical strokes) per row.
    let mut edges = vec![false; rows * cols];
    let mut density = vec![0.0f32; rows];
    for r in 0..rows {
        let y = y0 + r;
        let mut count = 0;
        for c in 0..cols {
            let x = c * step;
            let d = gray.at(x + step, y) as i16 - gray.at(x, y) as i16;
            if d.abs() >= EDGE_THRESHOLD {
                edges[r * cols + c] = true;
                count += 1;
            }
        }
        density[r] = count as f32 / cols as f32;
    }
    let smooth: Vec<f32> = (0..rows)
        .map(|r| {
            let lo = r.saturating_sub(1);
            let hi = (r + 2).min(rows);
            density[lo..hi].iter().sum::<f32>() / (hi - lo) as f32
        })
        .collect();

    // Densest run of text-like rows, bridging small gaps between lines.
    let max_gap = (h / 100).max(2);
    let mut best: Option<(usize, usize, f32)> = None;
    let mut r = 0;
    while r < rows {
        if smooth[r] < ROW_DENSITY_THRESHOLD {
            r += 1;
            continue;
        }
        let start = r;
        let mut end = r;
        let mut gap = 0;
        while r < rows {
            if smooth[r] >= ROW_DENSITY_THRESHOLD {
                end = r;
                gap = 0;
            } else {
                gap += 1;
                if gap > max_gap {
                    break;
                }
            }
            r += 1;
        }
        let mass: f32 = smooth[start..=end].iter().sum();
        if best.map_or(true, |b| mass > b.2) {
            best = Some((start, end, mass));
        }
    }
    let (b0, b1, _) = best?;
    let band_h = b1 - b0 + 1;
    if (band_h as f32) < 0.02 * h as f32 || (band_h as f32) > 0.2 * h as f32 {
        return None;
    }

    // Contrast against the rest of the search region.
    let band_mean = density[b0..=b1].iter().sum::<f32>() / band_h as f32;
    let (mut out_sum, mut out_n) = (0.0, 0);
    for (r, d) in density.iter().enumerate() {
        if r + 3 < b0 || r > b1 + 3 {
            out_sum += d;
            out_n += 1;
        }
    }
    let outside_mean = if out_n > 0 { out_sum / out_n as f32 } else { 0.0 };
    let contrast = band_mean / (outside_mean + 0.01);

    // Horizontal extent: columns carrying edges within the band.
    let active: Vec<bool> = (0..cols)
        .map(|c| (b0..=b1).filter(|&r| edges[r * cols + c]).count() >= 2)
        .collect();
    let window = 16.min(cols);
    let dense_at = |c: usize| active[c..(c + window).min(cols)].iter().filter(|&&a| a).count() >= 4;
    let x_left = (0..cols).find(|&c| dense_at(c))?;
    let x_right = (0..cols).rev().find(|&c| dense_at(c.saturating_sub(window - 1)))?;
    if x_right <= x_left {
        return None;
    }
    // Edges sit on stroke boundaries, so treat columns within 2 of an edge as inked.
    let covered = (x_left..=x_right).filter(|&c| active[c.saturating_sub(2)..(c + 3).min(cols)].iter().any(|&a| a));
    let fill = covered.count() as f32 / (x_right - x_left + 1) as f32;
    let width_frac = (x_right - x_left) as f32 / cols as f32;
    let center = (x_left + x_right) as f32 * 0.5 / cols as f32;

    let s_center = (1.0 - ((center - 0.5).abs() / 0.25)).clamp(0.0, 1.0).sqrt();
    let s_width = if (0.08..=0.95).contains(&width_frac) { 1.0 } else { 0.0 };
    let s_contrast = ((contrast - 1.5) / 3.0).clamp(0.0, 1.0);
    let s_fill = ((fill - 0.2) / 0.4).clamp(0.0, 1.0);
    let score = s_center * s_width * s_contrast * s_fill;
    if score < 0.3 {
        return None;
    }

    // Coarse binary layout of the band; changes when the words change.
    let mut signature = 0u128;
    for sr in 0..SIG_ROWS {
        for sc in 0..SIG_COLS {
            let (ra, rb) = (b0 + sr * band_h / SIG_ROWS, b0 + (sr + 1) * band_h / SIG_ROWS);
            let span = x_right - x_left + 1;
            let (ca, cb) = (x_left + sc * span / SIG_COLS, x_left + (sc + 1) * span / SIG_COLS);
            let cells = (rb - ra).max(1) * (cb - ca).max(1);
            let hits = (ra..rb).flat_map(|r| (ca..cb).map(move |c| (r, c))).filter(|&(r, c)| edges[r * cols + c]).count();
            if hits as f32 / cells as f32 > 0.15 {
                signature |= 1u128 << (sr * SIG_COLS + sc);
            }
        }
    }

    Some(CaptionObservation {
        score,
        y_top: (y0 + b0) as f32 / h as f32,
        y_bottom: (y0 + b1 + 1) as f32 / h as f32,
        x_left: (x_left * step) as f32 / w as f32,
        x_right: (x_right * step) as f32 / w as f32,
        signature,
    })
}

fn is_change(a: &CaptionObservation, b: &CaptionObservation) -> bool {
    let hamming = (a.signature ^ b.signature).count_ones();
    let extent_shift = (a.x_left - b.x_left).abs() + (a.x_right - b.x_right).abs();
    hamming > 20 || extent_shift > 0.06
}

#[derive(Debug, Clone, Serialize)]
pub struct BurnedInCaptions {
    pub detected: bool,
    pub probability: f32,
    /// Fraction of sampled frames showing a caption-like text band.
    pub presence_ratio: f32,
    pub text_changes_per_minute: f32,
    /// Typical vertical position of the band (fraction of frame height).
    pub typical_y: Option<f32>,
    /// Text that is present but never changes (logo, watermark, static title).
    pub static_text_overlay: bool,
}

#[derive(Default)]
pub struct CaptionAggregator {
    frames: usize,
    present: usize,
    centers: Vec<f32>,
    changes: usize,
    previous: Option<CaptionObservation>,
}

impl CaptionAggregator {
    pub fn push(&mut self, obs: Option<&CaptionObservation>) {
        self.frames += 1;
        if let Some(o) = obs {
            self.present += 1;
            self.centers.push((o.y_top + o.y_bottom) * 0.5);
            // New content, or text reappearing after a gap (but not the very
            // first appearance, which a static logo also has).
            let changed = match &self.previous {
                Some(p) => is_change(p, o),
                None => self.present > 1,
            };
            if changed {
                self.changes += 1;
            }
        }
        self.previous = obs.copied();
    }

    pub fn summary(&self, duration_secs: f64) -> BurnedInCaptions {
        let presence_ratio = if self.frames > 0 { self.present as f32 / self.frames as f32 } else { 0.0 };
        let minutes = (duration_secs / 60.0).max(1.0 / 60.0) as f32;
        let text_changes_per_minute = self.changes as f32 / minutes;
        let (typical_y, spread) = if self.centers.is_empty() {
            (None, 1.0)
        } else {
            let mut sorted = self.centers.clone();
            sorted.sort_by(f32::total_cmp);
            let median = sorted[sorted.len() / 2];
            let mad = {
                let mut dev: Vec<f32> = sorted.iter().map(|c| (c - median).abs()).collect();
                dev.sort_by(f32::total_cmp);
                dev[dev.len() / 2]
            };
            (Some(median), mad)
        };
        let p_presence = ((presence_ratio - 0.1) / 0.3).clamp(0.0, 1.0);
        let p_change = ((text_changes_per_minute - 1.0) / 5.0).clamp(0.0, 1.0);
        let p_stable = (1.0 - (spread - 0.03) / 0.07).clamp(0.0, 1.0);
        let probability = (p_presence * p_change * p_stable).cbrt();
        BurnedInCaptions {
            detected: probability >= 0.5,
            probability,
            presence_ratio,
            text_changes_per_minute,
            typical_y,
            static_text_overlay: presence_ratio > 0.3 && text_changes_per_minute < 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mid-gray frame with an optional block of fake "glyphs" (alternating
    /// black/white 3px strokes with word gaps) near the bottom.
    fn frame(text: Option<u32>) -> Gray {
        let (w, h) = (640, 360);
        let mut data = vec![110u8; w * h];
        if let Some(seed) = text {
            for y in 300..318 {
                for x in 160..480 {
                    let word_gap = ((x as u32 + seed * 7) / 40) % 5 == 0;
                    if !word_gap {
                        data[y * w + x] = if ((x as u32 + seed) / 3) % 2 == 0 { 250 } else { 10 };
                    }
                }
            }
        }
        Gray { data, width: w, height: h }
    }

    #[test]
    fn finds_text_band_and_ignores_blank() {
        assert!(observe(&frame(None)).is_none());
        let o = observe(&frame(Some(0))).expect("band found");
        assert!(o.y_top > 0.8 && o.y_bottom < 0.9, "{o:?}");
        assert!(o.x_left < 0.3 && o.x_right > 0.7, "{o:?}");
    }

    #[test]
    fn changing_text_reads_as_captions_static_text_does_not() {
        let mut captions = CaptionAggregator::default();
        let mut logo = CaptionAggregator::default();
        for i in 0..40 {
            captions.push(observe(&frame(Some(i / 4 * 11))).as_ref());
            logo.push(observe(&frame(Some(0))).as_ref());
        }
        let c = captions.summary(10.0);
        let l = logo.summary(10.0);
        assert!(c.detected, "{c:?}");
        assert!(!l.detected && l.static_text_overlay, "{l:?}");
    }
}
