//! wasm-bindgen wrapper around `vs1_core`. JSON crosses the boundary for
//! requests/responses; embeddings cross as flat Float32Arrays.
//!
//! Frames can be embedded by any number of workers: one `SystemOne` (vision,
//! text and head) plus any `FrameEmbedder`s (vision only). The caller gathers
//! the rows and hands them to `SystemOne::set_state`.

use vs1_core::api::Request;
use vs1_core::encoders::EMBED_DIM;
use vs1_core::{Frame, FrameEmbedder as CoreEmbedder, ModelFiles, SystemOne as Core, TextEmbedder as CoreText};
use wasm_bindgen::prelude::*;

fn js_err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

/// Embeds one RGBA frame: its full view, then (if `crops`) its six crops,
/// flattened row after row.
fn embed(vision: &CoreEmbedder, rgba: &[u8], width: u32, height: u32, crops: bool) -> Result<Vec<f32>, JsError> {
    let frame = Frame::new(rgba, width as usize, height as usize).map_err(js_err)?;
    let rows = vision.embed_rows(&frame, crops).map_err(|e| js_err(format!("{e:#}")))?;
    Ok(rows.concat())
}

fn unflatten(flat: &[f32]) -> Result<Vec<Vec<f32>>, JsError> {
    if flat.len() % EMBED_DIM != 0 {
        return Err(js_err(format!("embedding rows must be {EMBED_DIM} floats each, got {} floats", flat.len())));
    }
    Ok(flat.chunks_exact(EMBED_DIM).map(<[f32]>::to_vec).collect())
}

/// Vision-only embedder, for extra workers that embed frames in parallel.
#[wasm_bindgen]
pub struct FrameEmbedder {
    inner: CoreEmbedder,
}

#[wasm_bindgen]
impl FrameEmbedder {
    #[wasm_bindgen(constructor)]
    pub fn new(vision_onnx: &[u8]) -> Result<FrameEmbedder, JsError> {
        Ok(FrameEmbedder { inner: CoreEmbedder::new(vision_onnx).map_err(|e| js_err(format!("{e:#}")))? })
    }

    #[wasm_bindgen(js_name = embedFrame)]
    pub fn embed_frame(&self, rgba: &[u8], width: u32, height: u32, crops: bool) -> Result<Vec<f32>, JsError> {
        embed(&self.inner, rgba, width, height, crops)
    }
}

/// Text-only embedder, for extra workers that encode question text in parallel.
#[wasm_bindgen]
pub struct TextEmbedder {
    inner: CoreText,
}

#[wasm_bindgen]
impl TextEmbedder {
    #[wasm_bindgen(constructor)]
    pub fn new(text_onnx: &[u8], tokenizer_json: &[u8]) -> Result<TextEmbedder, JsError> {
        Ok(TextEmbedder { inner: CoreText::new(text_onnx, tokenizer_json).map_err(|e| js_err(format!("{e:#}")))? })
    }

    /// The embedding of a context/focus pair, then the number of tokens used
    /// (513 floats).
    #[wasm_bindgen(js_name = embedText)]
    pub fn embed_text(&self, context: &str, focus: &str) -> Result<Vec<f32>, JsError> {
        let (mut emb, tokens) = self.inner.embed(context, focus).map_err(|e| js_err(format!("{e:#}")))?;
        emb.push(tokens as f32);
        Ok(emb)
    }
}

#[wasm_bindgen]
pub struct SystemOne {
    inner: Core,
    /// Video state: full-view rows first, then crop rows.
    rows: Vec<Vec<f32>>,
    frames: usize,
}

#[wasm_bindgen]
impl SystemOne {
    /// Loads the answer head (`head` may be empty for zero-shot). The
    /// encoders load later (`loadText`, `loadVision`): precomputed frame and
    /// text embeddings need neither.
    #[wasm_bindgen(constructor)]
    pub fn new(head: &[u8]) -> Result<SystemOne, JsError> {
        let inner = Core::new(ModelFiles { vision_onnx: None, text: None, head: (!head.is_empty()).then_some(head) })
            .map_err(|e| js_err(format!("{e:#}")))?;
        Ok(SystemOne { inner, rows: Vec::new(), frames: 0 })
    }

    #[wasm_bindgen(js_name = loadVision)]
    pub fn load_vision(&mut self, vision_onnx: &[u8]) -> Result<(), JsError> {
        self.inner.load_vision(vision_onnx).map_err(|e| js_err(format!("{e:#}")))
    }

    #[wasm_bindgen(js_name = loadText)]
    pub fn load_text(&mut self, text_onnx: &[u8], tokenizer_json: &[u8]) -> Result<(), JsError> {
        self.inner.load_text(text_onnx, tokenizer_json).map_err(|e| js_err(format!("{e:#}")))
    }

    #[wasm_bindgen(getter, js_name = modelName)]
    pub fn model_name(&self) -> String {
        self.inner.model_name().to_string()
    }

    /// Embeds one RGBA frame (see `FrameEmbedder::embed_frame`).
    #[wasm_bindgen(js_name = embedFrame)]
    pub fn embed_frame(&self, rgba: &[u8], width: u32, height: u32, crops: bool) -> Result<Vec<f32>, JsError> {
        embed(self.inner.vision().map_err(js_err)?, rgba, width, height, crops)
    }

    /// Sets the video state: one full-view row per sampled frame, and the crop
    /// rows, each flattened.
    #[wasm_bindgen(js_name = setState)]
    pub fn set_state(&mut self, full: &[f32], crops: &[f32]) -> Result<(), JsError> {
        let mut rows = unflatten(full)?;
        self.frames = rows.len();
        rows.extend(unflatten(crops)?);
        self.rows = rows;
        Ok(())
    }

    /// The texts a request needs that aren't embedded yet, as JSON
    /// `[[context, focus], ...]` (embed them with `TextEmbedder`s, then
    /// `insertText` each).
    #[wasm_bindgen(js_name = missingTexts)]
    pub fn missing_texts(&self, request_json: &str) -> Result<String, JsError> {
        let request = Request::parse(request_json).map_err(js_err)?;
        let missing = self.inner.missing_texts(&request).map_err(|e| js_err(format!("{e:#}")))?;
        serde_json::to_string(&missing).map_err(js_err)
    }

    /// Embeds a text into this instance's cache (its share of `missingTexts`).
    #[wasm_bindgen(js_name = encodeText)]
    pub fn encode_text(&mut self, context: &str, focus: &str) -> Result<(), JsError> {
        self.inner.embed_text(context, focus).map(|_| ()).map_err(|e| js_err(format!("{e:#}")))
    }

    /// Adds a text embedded by a `TextEmbedder` (its 513-float output).
    #[wasm_bindgen(js_name = insertText)]
    pub fn insert_text(&mut self, context: &str, focus: &str, embedded: &[f32]) -> Result<(), JsError> {
        let (emb, tokens) = embedded.split_at(embedded.len().saturating_sub(1));
        let tokens = tokens.first().copied().unwrap_or(0.0) as usize;
        self.inner.insert_text(context, focus, emb.to_vec(), tokens).map_err(|e| js_err(format!("{e:#}")))
    }

    /// Validates a request and embeds its questions (cached for `evaluate`),
    /// so bad requests fail before any video is decoded.
    pub fn validate(&mut self, request_json: &str) -> Result<(), JsError> {
        let request = Request::parse(request_json).map_err(js_err)?;
        self.inner.compile(&request).map_err(|e| js_err(format!("{e:#}")))?;
        Ok(())
    }

    /// Answers a Jev-style request (`{model?, questions}`, JSON) against the
    /// current video state. Returns the Jev-style response as JSON.
    pub fn evaluate(&mut self, request_json: &str) -> Result<String, JsError> {
        let request = Request::parse(request_json).map_err(js_err)?;
        let compiled = self.inner.compile(&request).map_err(|e| js_err(format!("{e:#}")))?;
        let response = self.inner.answer(&self.rows, self.frames, &compiled).map_err(|e| js_err(format!("{e:#}")))?;
        serde_json::to_string(&response).map_err(js_err)
    }
}
