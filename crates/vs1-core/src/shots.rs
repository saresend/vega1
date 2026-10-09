//! Hard-cut detection between consecutive sampled frames, using a luma
//! histogram (robust to motion) gated by a thumbnail difference (robust to
//! global lighting changes that keep the layout).

use crate::image::Gray;

const HIST_BINS: usize = 16;
const THUMB_W: usize = 32;
const THUMB_H: usize = 18;

pub struct ShotDetector {
    prev: Option<([f32; HIST_BINS], Vec<f32>)>,
    pub hist_threshold: f32,
    pub thumb_threshold: f32,
}

impl Default for ShotDetector {
    fn default() -> Self {
        Self { prev: None, hist_threshold: 0.5, thumb_threshold: 0.12 }
    }
}

impl ShotDetector {
    /// Returns true if `gray` starts a new shot relative to the previous call.
    pub fn is_cut(&mut self, gray: &Gray) -> bool {
        let sig = signature(gray);
        let cut = match &self.prev {
            Some((hist, thumb)) => {
                let hist_dist: f32 = hist.iter().zip(&sig.0).map(|(a, b)| (a - b).abs()).sum();
                let thumb_dist =
                    thumb.iter().zip(&sig.1).map(|(a, b)| (a - b).abs()).sum::<f32>() / thumb.len() as f32;
                hist_dist > self.hist_threshold && thumb_dist > self.thumb_threshold
            }
            None => false,
        };
        self.prev = Some(sig);
        cut
    }
}

fn signature(gray: &Gray) -> ([f32; HIST_BINS], Vec<f32>) {
    let mut hist = [0.0f32; HIST_BINS];
    let mut thumb = vec![0.0f32; THUMB_W * THUMB_H];
    let mut counts = vec![0u32; THUMB_W * THUMB_H];
    let step = (gray.width.max(gray.height) / 160).max(1);
    let mut n = 0.0;
    for y in (0..gray.height).step_by(step) {
        let ty = y * THUMB_H / gray.height;
        for x in (0..gray.width).step_by(step) {
            let v = gray.at(x, y);
            hist[v as usize * HIST_BINS / 256] += 1.0;
            let ti = ty * THUMB_W + x * THUMB_W / gray.width;
            thumb[ti] += v as f32 / 255.0;
            counts[ti] += 1;
            n += 1.0;
        }
    }
    hist.iter_mut().for_each(|h| *h /= n);
    for (t, c) in thumb.iter_mut().zip(&counts) {
        if *c > 0 {
            *t /= *c as f32;
        }
    }
    (hist, thumb)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(v: u8) -> Gray {
        Gray { data: vec![v; 64 * 36], width: 64, height: 36 }
    }

    fn gradient(offset: u8) -> Gray {
        let data = (0..36).flat_map(|_| (0..64).map(move |x| (x as u8 * 3).wrapping_add(offset))).collect();
        Gray { data, width: 64, height: 36 }
    }

    #[test]
    fn detects_cut_but_not_small_change() {
        let mut d = ShotDetector::default();
        assert!(!d.is_cut(&gradient(0)));
        assert!(!d.is_cut(&gradient(2)));
        assert!(d.is_cut(&flat(230)));
        assert!(!d.is_cut(&flat(228)));
    }
}
