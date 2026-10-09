//! Minimal image types. Frames arrive as RGBA8 (what browsers hand out via
//! `getImageData` and what ffmpeg emits with `-pix_fmt rgba`).

use anyhow::ensure;

/// An RGBA8 frame borrowed from the caller.
#[derive(Clone, Copy)]
pub struct Frame<'a> {
    pub rgba: &'a [u8],
    pub width: usize,
    pub height: usize,
}

impl<'a> Frame<'a> {
    pub fn new(rgba: &'a [u8], width: usize, height: usize) -> anyhow::Result<Self> {
        ensure!(width > 1 && height > 1, "frame must be at least 2x2, got {width}x{height}");
        ensure!(
            rgba.len() == width * height * 4,
            "expected {} bytes for a {width}x{height} RGBA frame, got {}",
            width * height * 4,
            rgba.len()
        );
        Ok(Self { rgba, width, height })
    }

    #[inline]
    pub fn rgb(&self, x: usize, y: usize) -> [u8; 3] {
        let i = (y * self.width + x) * 4;
        [self.rgba[i], self.rgba[i + 1], self.rgba[i + 2]]
    }

    /// Bilinear RGB sample at continuous pixel coordinates (clamped to the frame).
    pub fn sample_bilinear(&self, fx: f32, fy: f32) -> [f32; 3] {
        let fx = fx.clamp(0.0, (self.width - 1) as f32);
        let fy = fy.clamp(0.0, (self.height - 1) as f32);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let (ax, ay) = (fx - x0 as f32, fy - y0 as f32);
        let (p00, p10, p01, p11) = (self.rgb(x0, y0), self.rgb(x1, y0), self.rgb(x0, y1), self.rgb(x1, y1));
        let mut out = [0.0; 3];
        for c in 0..3 {
            let top = p00[c] as f32 * (1.0 - ax) + p10[c] as f32 * ax;
            let bottom = p01[c] as f32 * (1.0 - ax) + p11[c] as f32 * ax;
            out[c] = top * (1.0 - ay) + bottom * ay;
        }
        out
    }

    pub fn to_gray(&self) -> Gray {
        let data = self
            .rgba
            .chunks_exact(4)
            .map(|p| (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) as u8)
            .collect();
        Gray { data, width: self.width, height: self.height }
    }
}

/// 8-bit luma image.
pub struct Gray {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl Gray {
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.width + x]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrong_buffer_size() {
        assert!(Frame::new(&[0; 10], 2, 2).is_err());
        assert!(Frame::new(&[0; 16], 2, 2).is_ok());
    }

    #[test]
    fn bilinear_interpolates_midpoint() {
        // 2x1 frame: black then white.
        let px = [0, 0, 0, 255, 255, 255, 255, 255];
        let f = Frame::new(&px, 2, 1).err();
        assert!(f.is_some(), "1-row frames are rejected");
        let px = [0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255];
        let f = Frame::new(&px, 2, 2).unwrap();
        let s = f.sample_bilinear(0.5, 0.0);
        assert!((s[0] - 127.5).abs() < 0.01);
    }
}
