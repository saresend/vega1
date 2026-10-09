// Runs the WASM model off the main thread. Protocol:
// main → { id, type, ...payload }, worker → { id, ok, result | error, status }.
import init, { SystemOne } from './pkg/vs1_wasm.js';

const CACHE = 'vs1-models-v1';
const FILES = { vision: 'models/vision_fp16w.onnx', text: 'models/text_fp16w.onnx', tokenizer: 'models/tokenizer.json' };

let model = null;

// Model files are large; keep them in the Cache API across visits when allowed.
async function fetchBytes(url) {
  let cache = null;
  try {
    cache = await caches.open(CACHE);
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch {
    cache = null; // private mode, blocked storage, etc.
  }
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetching ${url}: HTTP ${resp.status}`);
  if (cache) {
    try {
      await cache.put(url, resp.clone());
    } catch {
      // Quota exceeded: still usable, just not cached.
    }
  }
  return new Uint8Array(await resp.arrayBuffer());
}

const handlers = {
  async init({ baseUrl, head }) {
    const t0 = performance.now();
    await init();
    const url = (path) => new URL(path, baseUrl).href;
    const [vision, text, tokenizer, headBytes] = await Promise.all([
      fetchBytes(url(FILES.vision)),
      fetchBytes(url(FILES.text)),
      fetchBytes(url(FILES.tokenizer)),
      head ? fetchBytes(new URL(head, baseUrl).href) : Promise.resolve(new Uint8Array()),
    ]);
    const t1 = performance.now();
    model = new SystemOne(vision, text, tokenizer, headBytes);
    return { model: model.modelName, downloadMs: t1 - t0, loadMs: performance.now() - t1 };
  },
  validate({ request }) {
    try {
      model.validate(request);
    } catch (err) {
      throw Object.assign(err, { status: 422 });
    }
    return {};
  },
  clear() {
    model.clearFrames();
    return {};
  },
  frame({ rgba, width, height }) {
    const t0 = performance.now();
    model.pushFrame(new Uint8Array(rgba), width, height);
    return { ms: performance.now() - t0 };
  },
  evaluate({ request }) {
    return JSON.parse(model.evaluate(request));
  },
};

self.onmessage = async ({ data }) => {
  const { id, type, ...payload } = data;
  try {
    self.postMessage({ id, ok: true, result: await handlers[type](payload) });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err?.message ?? err), status: err?.status });
  }
};
