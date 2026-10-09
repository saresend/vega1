//! vs1-core: a compact, fully on-device "System 1" video analyzer.
//!
//! Feed sampled RGBA frames with timestamps to [`Analyzer::push_frame`], then
//! call [`Analyzer::report`] for answers to the built-in questions:
//!
//! * `subject_count` — how many distinct people appear across the video
//!   (detector → tracker → re-identification merge).
//! * `captioned` — whether the video carries captions, either as a container
//!   text track or burned into the picture.
//!
//! Everything here compiles to `wasm32-unknown-unknown`; see `vs1-wasm`.

pub mod appearance;
pub mod captions;
pub mod container;
pub mod detect;
pub mod image;
pub mod shots;
pub mod track;

use serde::Serialize;

use captions::{BurnedInCaptions, CaptionAggregator, CaptionObservation};
use container::{ContainerProbe, TextTrack};
use detect::{BBox, PersonDetector};
pub use image::Frame;
use shots::ShotDetector;
use track::{Subject, Tracker, TrackerConfig};

/// What one sampled frame contributed, for live overlays.
#[derive(Debug, Clone, Serialize)]
pub struct FrameResult {
    pub t: f64,
    pub shot_cut: bool,
    pub subjects: Vec<FrameSubject>,
    pub caption: Option<CaptionObservation>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FrameSubject {
    pub bbox: BBox,
    pub score: f32,
    pub track_id: u32,
    pub confirmed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub duration_seconds: f64,
    pub frames_analyzed: usize,
    pub shot_cuts: Vec<f64>,
    pub answers: Answers,
}

#[derive(Debug, Clone, Serialize)]
pub struct Answers {
    pub subject_count: SubjectCount,
    pub captioned: Captioned,
}

#[derive(Debug, Clone, Serialize)]
pub struct SubjectCount {
    pub value: usize,
    pub confidence: f32,
    pub max_in_single_frame: usize,
    pub reidentified_merges: usize,
    pub subjects: Vec<Subject>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Captioned {
    pub value: bool,
    pub confidence: f32,
    /// "caption_track", "burned_in" or "none".
    pub source: &'static str,
    pub caption_tracks: Vec<TextTrack>,
    pub container_format: Option<&'static str>,
    pub burned_in: BurnedInCaptions,
}

pub struct Analyzer {
    detector: PersonDetector,
    tracker_config: TrackerConfig,
    tracker: Tracker,
    shots: ShotDetector,
    captions: CaptionAggregator,
    container: Option<ContainerProbe>,
    frames: usize,
    max_in_frame: usize,
    cut_times: Vec<f64>,
}

impl Analyzer {
    pub fn new(detector_onnx: &[u8]) -> anyhow::Result<Self> {
        let detector = PersonDetector::from_onnx_bytes(detector_onnx)?;
        let tracker_config = TrackerConfig::default();
        Ok(Self {
            detector,
            tracker: Tracker::new(tracker_config.clone()),
            tracker_config,
            shots: ShotDetector::default(),
            captions: CaptionAggregator::default(),
            container: None,
            frames: 0,
            max_in_frame: 0,
            cut_times: Vec::new(),
        })
    }

    /// Clears per-video state, keeping the loaded model.
    pub fn reset(&mut self) {
        self.tracker = Tracker::new(self.tracker_config.clone());
        self.shots = ShotDetector::default();
        self.captions = CaptionAggregator::default();
        self.container = None;
        self.frames = 0;
        self.max_in_frame = 0;
        self.cut_times.clear();
    }

    /// Inspects the raw file bytes for caption/subtitle tracks.
    pub fn probe_container(&mut self, file_bytes: &[u8]) -> &ContainerProbe {
        self.container.insert(container::probe(file_bytes))
    }

    /// Processes one sampled frame. Timestamps must be non-decreasing.
    pub fn push_frame(&mut self, frame: &Frame, t: f64) -> anyhow::Result<FrameResult> {
        let gray = frame.to_gray();
        let shot_cut = self.shots.is_cut(&gray);
        if shot_cut {
            self.cut_times.push(t);
        }

        let dets = self.detector.detect(frame)?;
        let embeds: Vec<Vec<f32>> = dets.iter().map(|d| appearance::embed(frame, &d.bbox)).collect();
        let assignments = self.tracker.update(t, &dets, &embeds, shot_cut);
        self.max_in_frame = self.max_in_frame.max(dets.len());

        let caption = captions::observe(&gray);
        self.captions.push(caption.as_ref());
        self.frames += 1;

        let subjects = dets
            .iter()
            .zip(assignments)
            .map(|(d, a)| FrameSubject { bbox: d.bbox, score: d.score, track_id: a.track_id, confirmed: a.confirmed })
            .collect();
        Ok(FrameResult { t, shot_cut, subjects, caption })
    }

    pub fn report(&self, duration_seconds: f64) -> Report {
        let (subjects, merges) = self.tracker.subjects();
        let subject_confidence = if self.frames == 0 {
            0.0
        } else {
            let mut c = 0.9f32;
            c -= (0.05 * merges as f32).min(0.3);
            c -= (0.05 * self.cut_times.len() as f32).min(0.2);
            if self.max_in_frame >= 6 {
                c -= 0.15; // crowds: occlusion and ID switches
            }
            c.clamp(0.2, 0.95)
        };

        let burned_in = self.captions.summary(duration_seconds);
        let tracks = self.container.as_ref().map(|c| c.text_tracks.clone()).unwrap_or_default();
        let (value, confidence, source) = if !tracks.is_empty() {
            (true, 0.99, "caption_track")
        } else if burned_in.detected {
            (true, burned_in.probability, "burned_in")
        } else {
            (false, 1.0 - burned_in.probability, "none")
        };

        Report {
            duration_seconds,
            frames_analyzed: self.frames,
            shot_cuts: self.cut_times.clone(),
            answers: Answers {
                subject_count: SubjectCount {
                    value: subjects.len(),
                    confidence: subject_confidence,
                    max_in_single_frame: self.max_in_frame,
                    reidentified_merges: merges,
                    subjects,
                },
                captioned: Captioned {
                    value,
                    confidence,
                    source,
                    caption_tracks: tracks,
                    container_format: self.container.as_ref().map(|c| c.format),
                    burned_in,
                },
            },
        }
    }
}
