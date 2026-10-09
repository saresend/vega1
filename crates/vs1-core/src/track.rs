//! Multi-subject tracking and unique-subject counting.
//!
//! Online: greedy association of detections to live tracks using predicted-box
//! IoU, centre distance and appearance (appearance only right after a shot
//! cut, since motion is meaningless across cuts). Tentative tracks that never
//! reach `min_hits` are discarded, which suppresses one-frame false positives.
//!
//! Offline (`subjects`): confirmed tracks that never coexist in time and look
//! alike are merged into one subject, so someone leaving and re-entering the
//! frame is counted once.

use serde::Serialize;

use crate::appearance::{blend, similarity};
use crate::detect::{BBox, Detection};

#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// A confirmed track stays matchable this long after its last detection.
    pub max_lost_secs: f64,
    /// A tentative track is dropped if unmatched for this long.
    pub tentative_max_lost_secs: f64,
    /// Detections needed before a track counts as a real subject.
    pub min_hits: u32,
    /// Minimum appearance similarity for a motion-consistent match.
    pub min_appearance: f32,
    /// Minimum appearance similarity to match across a shot cut.
    pub cut_min_appearance: f32,
    /// Minimum appearance similarity to merge non-overlapping tracks.
    pub reid_min_appearance: f32,
    /// Subjects seen in fewer detections than this are ignored in the count.
    pub min_subject_hits: u32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            max_lost_secs: 2.5,
            tentative_max_lost_secs: 0.6,
            min_hits: 2,
            min_appearance: 0.5,
            cut_min_appearance: 0.8,
            reid_min_appearance: 0.88,
            min_subject_hits: 3,
        }
    }
}

#[derive(Debug, Clone)]
struct Track {
    id: u32,
    bbox: BBox,
    /// Centre velocity in pixels per second.
    velocity: (f32, f32),
    appearance: Vec<f32>,
    hits: u32,
    first_seen: f64,
    last_seen: f64,
    active: bool,
}

/// Per-detection tracking result for one frame.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Assignment {
    pub track_id: u32,
    pub confirmed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Subject {
    pub id: u32,
    pub track_ids: Vec<u32>,
    pub first_seen: f64,
    pub last_seen: f64,
    pub visible_seconds: f64,
    pub detections: u32,
}

pub struct Tracker {
    cfg: TrackerConfig,
    tracks: Vec<Track>,
    next_id: u32,
}

impl Tracker {
    pub fn new(cfg: TrackerConfig) -> Self {
        Self { cfg, tracks: Vec::new(), next_id: 1 }
    }

    pub fn update(&mut self, t: f64, dets: &[Detection], embeds: &[Vec<f32>], after_cut: bool) -> Vec<Assignment> {
        assert_eq!(dets.len(), embeds.len());
        let mut candidates = Vec::new();
        for (ti, tr) in self.tracks.iter().enumerate().filter(|(_, tr)| tr.active) {
            let dt = (t - tr.last_seen).clamp(0.0, 1.0) as f32;
            let predicted = tr.bbox.shifted(tr.velocity.0 * dt, tr.velocity.1 * dt);
            for (di, (det, emb)) in dets.iter().zip(embeds).enumerate() {
                let app = similarity(&tr.appearance, emb);
                let cost = if after_cut {
                    if app < self.cfg.cut_min_appearance {
                        continue;
                    }
                    1.0 - app
                } else {
                    let iou = predicted.iou(&det.bbox);
                    let (pc, dc) = (predicted.center(), det.bbox.center());
                    let dist = (pc.0 - dc.0).hypot(pc.1 - dc.1) / tr.bbox.diagonal().max(1.0);
                    if (iou < 0.05 && dist > 1.0) || app < self.cfg.min_appearance {
                        continue;
                    }
                    0.5 * (1.0 - iou) + 0.3 * (1.0 - app) + 0.2 * dist.min(1.0)
                };
                candidates.push((cost, ti, di));
            }
        }
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut det_track: Vec<Option<usize>> = vec![None; dets.len()];
        let mut track_taken = vec![false; self.tracks.len()];
        for (_, ti, di) in candidates {
            if track_taken[ti] || det_track[di].is_some() {
                continue;
            }
            track_taken[ti] = true;
            det_track[di] = Some(ti);
            let tr = &mut self.tracks[ti];
            let dt = (t - tr.last_seen) as f32;
            if !after_cut && dt > 0.0 {
                let (old, new) = (tr.bbox.center(), dets[di].bbox.center());
                let v = ((new.0 - old.0) / dt, (new.1 - old.1) / dt);
                tr.velocity = (0.5 * tr.velocity.0 + 0.5 * v.0, 0.5 * tr.velocity.1 + 0.5 * v.1);
            } else {
                tr.velocity = (0.0, 0.0);
            }
            tr.bbox = dets[di].bbox;
            blend(&mut tr.appearance, &embeds[di], 0.8);
            tr.hits += 1;
            tr.last_seen = t;
        }

        for (di, slot) in det_track.iter_mut().enumerate() {
            if slot.is_none() {
                self.tracks.push(Track {
                    id: self.next_id,
                    bbox: dets[di].bbox,
                    velocity: (0.0, 0.0),
                    appearance: embeds[di].clone(),
                    hits: 1,
                    first_seen: t,
                    last_seen: t,
                    active: true,
                });
                self.next_id += 1;
                *slot = Some(self.tracks.len() - 1);
            }
        }

        let assignments = det_track
            .iter()
            .map(|ti| {
                let tr = &self.tracks[ti.expect("every detection assigned")];
                Assignment { track_id: tr.id, confirmed: tr.hits >= self.cfg.min_hits }
            })
            .collect();

        // Expire stale tracks; forget tentative ones entirely.
        let cfg = &self.cfg;
        for tr in &mut self.tracks {
            let lost = t - tr.last_seen;
            let limit = if tr.hits >= cfg.min_hits { cfg.max_lost_secs } else { cfg.tentative_max_lost_secs };
            if lost > limit {
                tr.active = false;
            }
        }
        self.tracks.retain(|tr| tr.active || tr.hits >= cfg.min_hits);
        assignments
    }

    /// Unique subjects so far, after merging re-identified tracks.
    pub fn subjects(&self) -> (Vec<Subject>, usize) {
        let mut confirmed: Vec<&Track> = self.tracks.iter().filter(|t| t.hits >= self.cfg.min_hits).collect();
        confirmed.sort_by(|a, b| a.first_seen.total_cmp(&b.first_seen));

        struct Group<'a> {
            members: Vec<&'a Track>,
            appearance: Vec<f32>,
        }
        let mut groups: Vec<Group> = Vec::new();
        let mut merges = 0;
        for tr in confirmed {
            let best = groups
                .iter()
                .enumerate()
                .filter(|(_, g)| {
                    g.members.iter().all(|m| m.last_seen < tr.first_seen || tr.last_seen < m.first_seen)
                })
                .map(|(gi, g)| (gi, similarity(&g.appearance, &tr.appearance)))
                .filter(|(_, sim)| *sim >= self.cfg.reid_min_appearance)
                .max_by(|a, b| a.1.total_cmp(&b.1));
            match best {
                Some((gi, _)) => {
                    let g = &mut groups[gi];
                    blend(&mut g.appearance, &tr.appearance, 0.5);
                    g.members.push(tr);
                    merges += 1;
                }
                None => groups.push(Group { members: vec![tr], appearance: tr.appearance.clone() }),
            }
        }

        let subjects = groups
            .into_iter()
            .map(|g| Subject {
                id: 0,
                track_ids: g.members.iter().map(|m| m.id).collect(),
                first_seen: g.members.iter().map(|m| m.first_seen).fold(f64::INFINITY, f64::min),
                last_seen: g.members.iter().map(|m| m.last_seen).fold(f64::NEG_INFINITY, f64::max),
                visible_seconds: g.members.iter().map(|m| m.last_seen - m.first_seen).sum(),
                detections: g.members.iter().map(|m| m.hits).sum(),
            })
            .filter(|s| s.detections >= self.cfg.min_subject_hits)
            .enumerate()
            .map(|(i, s)| Subject { id: i as u32 + 1, ..s })
            .collect();
        (subjects, merges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32) -> Detection {
        Detection { bbox: BBox { x1: x, y1: 100.0, x2: x + 50.0, y2: 250.0 }, score: 0.9 }
    }

    fn emb(hot: usize) -> Vec<f32> {
        let mut v = vec![0.0; 8];
        v[hot] = 1.0;
        v
    }

    #[test]
    fn walking_subject_keeps_one_id() {
        let mut tr = Tracker::new(TrackerConfig::default());
        let mut ids = Vec::new();
        for i in 0..10 {
            let a = tr.update(i as f64 * 0.25, &[det(10.0 + i as f32 * 8.0)], &[emb(0)], false);
            ids.push(a[0].track_id);
        }
        assert!(ids.iter().all(|&id| id == ids[0]));
        assert_eq!(tr.subjects().0.len(), 1);
    }

    #[test]
    fn two_people_side_by_side_count_two() {
        let mut tr = Tracker::new(TrackerConfig::default());
        for i in 0..8 {
            tr.update(i as f64 * 0.25, &[det(10.0), det(300.0)], &[emb(0), emb(1)], false);
        }
        assert_eq!(tr.subjects().0.len(), 2);
    }

    #[test]
    fn subject_leaving_and_returning_is_counted_once() {
        let mut tr = Tracker::new(TrackerConfig::default());
        for i in 0..6 {
            tr.update(i as f64 * 0.25, &[det(10.0)], &[emb(0)], false);
        }
        for i in 6..30 {
            tr.update(i as f64 * 0.25, &[], &[], false);
        }
        for i in 30..36 {
            tr.update(i as f64 * 0.25, &[det(400.0)], &[emb(0)], false);
        }
        let (subjects, merges) = tr.subjects();
        assert_eq!(merges, 1);
        assert_eq!(subjects.len(), 1);
        assert_eq!(subjects[0].track_ids.len(), 2);
    }

    #[test]
    fn single_frame_false_positive_is_ignored() {
        let mut tr = Tracker::new(TrackerConfig::default());
        tr.update(0.0, &[det(10.0)], &[emb(3)], false);
        for i in 1..10 {
            tr.update(i as f64 * 0.25, &[], &[], false);
        }
        assert_eq!(tr.subjects().0.len(), 0);
    }

    #[test]
    fn cut_matches_by_appearance_only() {
        let mut tr = Tracker::new(TrackerConfig::default());
        for i in 0..4 {
            tr.update(i as f64 * 0.25, &[det(10.0)], &[emb(0)], false);
        }
        // After a cut the same person appears elsewhere; a stranger appears where they were.
        let a = tr.update(1.0, &[det(10.0), det(500.0)], &[emb(1), emb(0)], true);
        assert_eq!(a[1].track_id, 1);
        assert_ne!(a[0].track_id, 1);
    }
}
