//! wasm-bindgen wrapper around `vs1_core::SystemOne`. JSON crosses the
//! boundary for requests/responses; embeddings stay inside WASM memory.

use vs1_core::api::Request;
use vs1_core::{Frame, ModelFiles, SystemOne as Core};
use wasm_bindgen::prelude::*;

fn js_err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

#[wasm_bindgen]
pub struct SystemOne {
    inner: Core,
    frames: Vec<Vec<f32>>,
}

#[wasm_bindgen]
impl SystemOne {
    /// Loads encoders from raw bytes. `head` may be empty for zero-shot.
    #[wasm_bindgen(constructor)]
    pub fn new(vision_onnx: &[u8], text_onnx: &[u8], tokenizer_json: &[u8], head: &[u8]) -> Result<SystemOne, JsError> {
        let inner = Core::new(ModelFiles {
            vision_onnx,
            text_onnx,
            tokenizer_json,
            head: (!head.is_empty()).then_some(head),
        })
        .map_err(|e| js_err(format!("{e:#}")))?;
        Ok(SystemOne { inner, frames: Vec::new() })
    }

    #[wasm_bindgen(getter, js_name = modelName)]
    pub fn model_name(&self) -> String {
        self.inner.model_name().to_string()
    }

    /// Starts a new video state.
    #[wasm_bindgen(js_name = clearFrames)]
    pub fn clear_frames(&mut self) {
        self.frames.clear();
    }

    /// Embeds one RGBA frame into the current video state.
    #[wasm_bindgen(js_name = pushFrame)]
    pub fn push_frame(&mut self, rgba: &[u8], width: u32, height: u32) -> Result<(), JsError> {
        let frame = Frame::new(rgba, width as usize, height as usize).map_err(js_err)?;
        let emb = self.inner.embed_frame(&frame).map_err(|e| js_err(format!("{e:#}")))?;
        self.frames.push(emb);
        Ok(())
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
        let response = self.inner.answer(&self.frames, &compiled).map_err(|e| js_err(format!("{e:#}")))?;
        serde_json::to_string(&response).map_err(js_err)
    }
}
