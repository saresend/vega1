// Runs the WASM analyzer off the main thread. Request/response protocol:
// main → { id, type, ...payload }, worker → { id, ok, result | error }.
import init, { Analyzer } from './pkg/vs1_wasm.js';

let analyzer = null;

const handlers = {
  async init() {
    await init();
    const resp = await fetch('./models/yolox_nano.onnx');
    if (!resp.ok) throw new Error(`model fetch failed: ${resp.status}`);
    const t0 = performance.now();
    analyzer = new Analyzer(new Uint8Array(await resp.arrayBuffer()));
    return { loadMs: performance.now() - t0 };
  },
  reset() {
    analyzer.reset();
    return {};
  },
  probe({ bytes }) {
    return JSON.parse(analyzer.probeContainer(new Uint8Array(bytes)));
  },
  frame({ rgba, width, height, t }) {
    const t0 = performance.now();
    const result = JSON.parse(analyzer.pushFrame(new Uint8Array(rgba), width, height, t));
    result.ms = performance.now() - t0;
    return result;
  },
  report({ duration }) {
    return JSON.parse(analyzer.report(duration));
  },
};

self.onmessage = async ({ data }) => {
  const { id, type, ...payload } = data;
  try {
    const result = await handlers[type](payload);
    self.postMessage({ id, ok: true, result });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err?.message ?? err) });
  }
};
