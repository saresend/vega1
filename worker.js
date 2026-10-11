// Runs the WASM model off the main thread. Protocol:
// main → { id, type, ...payload }, worker → { id, ok, result | error, status }.
import init, { FrameEmbedder, SystemOne } from './pkg/vs1_wasm.js';

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
  // role "main": the answer head now, the encoders when asked (loadText,
  // loadVision); role "embed": the vision encoder only, for parallel frames.
  async init({ baseUrl, head, choiceHead, role = 'main' }) {
    const t0 = performance.now();
    await init();
    self.baseUrl = baseUrl;
    if (role === 'embed') {
      model = new FrameEmbedder(await fetchBytes(new URL(FILES.vision, baseUrl).href));
      return { ms: performance.now() - t0 };
    }
    const headBytes = head ? await fetchBytes(new URL(head, baseUrl).href) : new Uint8Array();
    model = new SystemOne(headBytes);
    if (choiceHead) model.loadChoiceHead(await fetchBytes(new URL(choiceHead, baseUrl).href));
    return { model: model.modelName, ms: performance.now() - t0 };
  },
  async loadText() {
    const t0 = performance.now();
    const [text, tokenizer] = await Promise.all([
      fetchBytes(new URL(FILES.text, self.baseUrl).href),
      fetchBytes(new URL(FILES.tokenizer, self.baseUrl).href),
    ]);
    model.loadText(text, tokenizer);
    return { ms: performance.now() - t0 };
  },
  async loadVision() {
    const t0 = performance.now();
    model.loadVision(await fetchBytes(new URL(FILES.vision, self.baseUrl).href));
    return { ms: performance.now() - t0 };
  },
  validate({ request }) {
    try {
      model.validate(request);
    } catch (err) {
      throw Object.assign(err, { status: 422 });
    }
    return {};
  },
  // One frame's rows (full view, then crops if asked), flattened.
  embed({ rgba, width, height, crops }) {
    return model.embedFrame(new Uint8Array(rgba), width, height, crops);
  },
  // Parallel text encoding: the main worker lists what a request still needs,
  // every worker encodes a share, the main worker caches the results.
  missingTexts({ request }) {
    try {
      return JSON.parse(model.missingTexts(request));
    } catch (err) {
      throw Object.assign(err, { status: 422 });
    }
  },
  insertTexts({ texts, embedded }) {
    texts.forEach(([context, focus], i) => model.insertText(context, focus, embedded[i]));
    return {};
  },
  setState({ full, crops }) {
    model.setState(full, crops);
    return {};
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
