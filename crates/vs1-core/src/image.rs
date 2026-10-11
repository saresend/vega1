//! Frames arrive as RGBA8 (what browsers hand out via `getImageData` and what
//! ffmpeg emits with `-pix_fmt rgba`). Training features are extracted through
//! this same code (via the CLI), so train and serve preprocessing can't drift.

use anyhow::ensure;

/// A rectangle within a frame, in pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Region {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

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
    fn rgb(&self, x: usize, y: usize) -> [f32; 3] {
        let i = (y * self.width + x) * 4;
        [self.rgba[i] as f32, self.rgba[i + 1] as f32, self.rgba[i + 2] as f32]
    }

    /// The whole frame as a region.
    pub fn full(&self) -> Region {
        Region { x: 0, y: 0, width: self.width, height: self.height }
    }

    /// Six square crops on a 3x2 grid (2x3 for portrait) that together cover
    /// the whole frame, including the sides the full view's centre crop drops.
    /// Each is half the short side (a third of the long side if larger, so
    /// three span it): small objects appear at up to twice the full view's scale.
    pub fn crop_regions(&self) -> Vec<Region> {
        let (short, long) = (self.width.min(self.height), self.width.max(self.height));
        let side = (short / 2).max(long.div_ceil(3)).min(short).max(1);
        let (cols, rows) = if self.width >= self.height { (3, 2) } else { (2, 3) };
        let spread = |extent: usize, n: usize, i: usize| if n == 1 { 0 } else { (extent - side) * i / (n - 1) };
        let mut out = Vec::with_capacity(cols * rows);
        for r in 0..rows {
            for c in 0..cols {
                out.push(Region { x: spread(self.width, cols, c), y: spread(self.height, rows, r), width: side, height: side });
            }
        }
        out
    }

    /// CLIP-style input for the whole frame (see [`Frame::region_to_clip_input`]).
    pub fn to_clip_input(&self, size: usize) -> Vec<f32> {
        self.region_to_clip_input(self.full(), size)
    }

    /// CLIP-style input for a region: resize its shortest side to `size`,
    /// centre-crop to `size`x`size`, scale to 0..1, planar CHW. Downscaling
    /// averages each output pixel's source footprint (area resampling) to
    /// avoid aliasing; upscaling falls back to bilinear.
    pub fn region_to_clip_input(&self, region: Region, size: usize) -> Vec<f32> {
        let side = region.width.min(region.height) as f32;
        let x0 = region.x as f32 + (region.width as f32 - side) / 2.0;
        let y0 = region.y as f32 + (region.height as f32 - side) / 2.0;
        let step = side / size as f32;
        let plane = size * size;
        let mut out = vec![0.0f32; 3 * plane];
        for oy in 0..size {
            for ox in 0..size {
                let rgb = if step > 1.0 {
                    self.area_average(x0 + ox as f32 * step, y0 + oy as f32 * step, step)
                } else {
                    self.bilinear(x0 + (ox as f32 + 0.5) * step - 0.5, y0 + (oy as f32 + 0.5) * step - 0.5)
                };
                let i = oy * size + ox;
                for c in 0..3 {
                    out[c * plane + i] = rgb[c] / 255.0;
                }
            }
        }
        out
    }

    fn area_average(&self, fx: f32, fy: f32, step: f32) -> [f32; 3] {
        let xa = (fx.round() as usize).min(self.width - 1);
        let ya = (fy.round() as usize).min(self.height - 1);
        let xb = ((fx + step).round() as usize).clamp(xa + 1, self.width);
        let yb = ((fy + step).round() as usize).clamp(ya + 1, self.height);
        let mut acc = [0.0f32; 3];
        for y in ya..yb {
            for x in xa..xb {
                let p = self.rgb(x, y);
                acc[0] += p[0];
                acc[1] += p[1];
                acc[2] += p[2];
            }
        }
        let n = ((xb - xa) * (yb - ya)) as f32;
        acc.map(|v| v / n)
    }

    fn bilinear(&self, fx: f32, fy: f32) -> [f32; 3] {
        let fx = fx.clamp(0.0, (self.width - 1) as f32);
        let fy = fy.clamp(0.0, (self.height - 1) as f32);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let (ax, ay) = (fx - x0 as f32, fy - y0 as f32);
        let (p00, p10, p01, p11) = (self.rgb(x0, y0), self.rgb(x1, y0), self.rgb(x0, y1), self.rgb(x1, y1));
        let mut out = [0.0; 3];
        for c in 0..3 {
            let top = p00[c] * (1.0 - ax) + p10[c] * ax;
            let bottom = p01[c] * (1.0 - ax) + p11[c] * ax;
            out[c] = top * (1.0 - ay) + bottom * ay;
        }
        out
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
    fn clip_input_center_crops_and_scales() {
        // 8x4 frame: left quarter red, middle half green, right quarter blue.
        let (w, h) = (8, 4);
        let mut px = Vec::new();
        for _ in 0..h {
            for x in 0..w {
                let c = if x < 2 { [255, 0, 0] } else if x < 6 { [0, 255, 0] } else { [0, 0, 255] };
                px.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        let f = Frame::new(&px, w, h).unwrap();
        let out = f.to_clip_input(2);
        // Centre 4x4 crop is all green; downscaled 2x2 stays green, in 0..1.
        assert_eq!(&out[0..4], &[0.0; 4], "red plane");
        assert_eq!(&out[4..8], &[1.0; 4], "green plane");
        assert_eq!(&out[8..12], &[0.0; 4], "blue plane");
    }

    #[test]
    fn crops_tile_the_frame_at_half_scale() {
        let px = vec![0u8; 640 * 360 * 4];
        let crops = Frame::new(&px, 640, 360).unwrap().crop_regions();
        assert_eq!(crops.len(), 6);
        assert!(crops.iter().all(|r| r.width == 214 && r.height == 214 && r.x + 214 <= 640 && r.y + 214 <= 360));
        assert_eq!((crops[0].x, crops[1].x, crops[2].x, crops[5].y), (0, 213, 426, 146));
        // Every pixel is covered.
        let covered = |x: usize, y: usize| crops.iter().any(|r| (r.x..r.x + r.width).contains(&x) && (r.y..r.y + r.height).contains(&y));
        assert!((0..640).all(|x| (0..360).all(|y| covered(x, y))));
        // 4:3 portrait: half the short side.
        let px = vec![0u8; 240 * 320 * 4];
        let crops = Frame::new(&px, 240, 320).unwrap().crop_regions();
        assert_eq!((crops.len(), crops[5].width, crops[5].x, crops[5].y), (6, 120, 120, 200));
    }

    #[test]
    fn region_input_reads_only_the_region() {
        // Left half red, right half blue; the right-half region is all blue.
        let (w, h) = (8, 4);
        let px: Vec<u8> = (0..w * h).flat_map(|i| if i % w < 4 { [255, 0, 0, 255] } else { [0, 0, 255, 255] }).collect();
        let f = Frame::new(&px, w, h).unwrap();
        let out = f.region_to_clip_input(Region { x: 4, y: 0, width: 4, height: 4 }, 2);
        assert_eq!((&out[0..4], &out[8..12]), (&[0.0; 4][..], &[1.0; 4][..]));
    }

    #[test]
    fn area_average_blends_footprint() {
        // 4x4 checkerboard of 0/255 downscaled to 1x1 averages to mid-gray.
        let px: Vec<u8> = (0..16).flat_map(|i| { let v = if (i / 4 + i % 4) % 2 == 0 { 0 } else { 255 }; [v, v, v, 255] }).collect();
        let out = Frame::new(&px, 4, 4).unwrap().to_clip_input(1);
        assert!((out[0] - 0.5).abs() < 1e-6);
    }
}
