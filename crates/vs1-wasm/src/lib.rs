//! wasm-bindgen wrapper around `vs1_core::Analyzer`. Results cross the
//! boundary as JSON strings to keep the binding surface small.

use wasm_bindgen::prelude::*;

fn js_err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

#[wasm_bindgen]
pub struct Analyzer {
    inner: vs1_core::Analyzer,
}

#[wasm_bindgen]
impl Analyzer {
    /// Loads the YOLOX-nano ONNX detector from its raw bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(detector_onnx: &[u8]) -> Result<Analyzer, JsError> {
        let inner = vs1_core::Analyzer::new(detector_onnx).map_err(|e| js_err(format!("{e:#}")))?;
        Ok(Analyzer { inner })
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Scans the whole video file for caption/subtitle tracks. Returns JSON.
    #[wasm_bindgen(js_name = probeContainer)]
    pub fn probe_container(&mut self, file_bytes: &[u8]) -> Result<String, JsError> {
        serde_json::to_string(self.inner.probe_container(file_bytes)).map_err(js_err)
    }

    /// Analyzes one RGBA frame sampled at `t` seconds. Returns JSON.
    #[wasm_bindgen(js_name = pushFrame)]
    pub fn push_frame(&mut self, rgba: &[u8], width: u32, height: u32, t: f64) -> Result<String, JsError> {
        let frame = vs1_core::Frame::new(rgba, width as usize, height as usize).map_err(js_err)?;
        let result = self.inner.push_frame(&frame, t).map_err(|e| js_err(format!("{e:#}")))?;
        serde_json::to_string(&result).map_err(js_err)
    }

    /// Final answers for everything pushed since the last reset. Returns JSON.
    pub fn report(&self, duration_seconds: f64) -> Result<String, JsError> {
        serde_json::to_string(&self.inner.report(duration_seconds)).map_err(js_err)
    }
}
