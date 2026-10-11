// Public browser API — the Jev System One contract with a video as `state`.
//
//   import { SystemOne } from './vs1.js';
//   const s1 = await SystemOne.load();
//   const res = await s1.evaluate({
//     state: fileOrBlobOrUrl,
//     questions: { captioned: { type: 'noul', instructions: 'The video is captioned' } },
//   });
//   res.answers.captioned.noul  // 0..1
//
// Everything runs locally: the video never leaves the page.

const MAX_ANALYSIS_WIDTH = 640;
const EMBED_DIM = 512;
// How many sampled frames also get crop embeddings (vs1_core::CROP_FRAMES).
const CROP_FRAMES = 4;
const MAX_WORKERS = 4;

/** Indices of the frames whose crops are embedded too (vs1_core::crop_frame_indices). */
export function cropFrameIndices(frames, cropFrames = CROP_FRAMES) {
  const n = Math.min(cropFrames, frames);
  return Array.from({ length: n }, (_, j) => Math.floor(((2 * j + 1) * frames) / (2 * n)));
}

/** What identifies a video state: the object itself, or a <video>'s source. */
const keyOf = (state) => (state instanceof HTMLVideoElement ? state.currentSrc : state);

/** A promise-based wrapper around one worker. */
class WorkerClient {
  #worker;
  #pending = new Map();
  #nextId = 0;

  constructor() {
    this.#worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    this.#worker.onmessage = ({ data }) => {
      const p = this.#pending.get(data.id);
      this.#pending.delete(data.id);
      if (data.ok) p.resolve(data.result);
      else p.reject(Object.assign(new Error(data.error), { status: data.status }));
    };
  }

  call(type, payload = {}, transfer = []) {
    return new Promise((resolve, reject) => {
      const id = ++this.#nextId;
      this.#pending.set(id, { resolve, reject });
      this.#worker.postMessage({ id, type, ...payload }, transfer);
    });
  }

  terminate() {
    this.#worker.terminate();
  }
}

export class SystemOne {
  #main; // the answer head, the encoders once loaded, and the video state
  #embedders = []; // vision-only workers that embed frames in parallel
  #baseUrl;
  #workers;
  #text = null; // promise: text encoder loaded
  #textDone = false;
  #vision = null; // promise: vision encoder loaded in every worker
  #queue = Promise.resolve();
  #embedded = null; // { key, frames } of the state whose rows the main worker holds
  #prepared = null; // the last request passed to prepare()

  constructor(main, baseUrl, workers) {
    this.#main = main;
    this.#baseUrl = baseUrl;
    this.#workers = workers;
  }

  /**
   * Loads the answer head in a worker; that is enough to answer from
   * precomputed embeddings (`useEmbeddings`, `addTexts`). The text encoder
   * (~90 MB) loads with `loadText()` or when a question needs it; the vision
   * encoder (~23 MB, in `workers` workers: up to 4, 2 on phones) loads with
   * `loadVision()` or when a video must be embedded. `head` is an optional
   * trained-head URL; `choiceHead` an optional category head (vega1-cat) that
   * answers `choice` and `score` questions.
   */
  static async load({ baseUrl = new URL('.', import.meta.url).href, head = null, choiceHead = null, workers } = {}) {
    const mobile = navigator.userAgentData?.mobile ?? (/Android|iPhone|iPad|iPod|Mobi/i.test(navigator.userAgent) ||
      (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1));
    const cores = (navigator.hardwareConcurrency || 2) - 1;
    const n = Math.max(1, Math.min(mobile ? 2 : MAX_WORKERS, workers ?? cores));
    const main = new WorkerClient();
    const info = await main.call('init', { baseUrl, head, choiceHead, role: 'main' });
    const s1 = new SystemOne(main, baseUrl, n);
    s1.info = { ...info, workers: n };
    return s1;
  }

  /** Loads the text encoder (once); resolves when questions can be encoded. */
  loadText() {
    this.#text ??= this.#main.call('loadText').catch((err) => {
      this.#text = null;
      throw err;
    });
    return this.#text;
  }

  get textLoaded() {
    return this.#textDone;
  }

  /** Loads the vision encoder in all workers (once). */
  loadVision() {
    this.#vision ??= (async () => {
      this.#embedders = Array.from({ length: this.#workers - 1 }, () => new WorkerClient());
      await Promise.all([
        this.#main.call('loadVision'),
        ...this.#embedders.map((w) => w.call('init', { baseUrl: this.#baseUrl, role: 'embed' })),
      ]);
    })().catch((err) => {
      this.#vision = null;
      throw err;
    });
    return this.#vision;
  }

  /**
   * Adds precomputed text embeddings (from `vs1 embed-texts`): `keys` is its
   * JSON list of [context, focus], `floats` its f32 data, 513 floats per text.
   */
  addTexts(keys, floats) {
    const embedded = keys.map((_, i) => floats.slice(i * 513, (i + 1) * 513));
    return this.#main.call('insertTexts', { texts: keys, embedded });
  }

  /** Makes sure a request's texts are encodable: loads the text encoder if any is new. */
  async #encodeTexts(request) {
    if (this.#textDone) return;
    const missing = await this.#main.call('missingTexts', { request });
    if (missing.length) {
      await this.loadText();
      this.#textDone = true;
    }
  }

  /**
   * Uses precomputed embedding rows as `state`'s video state (e.g. shipped
   * with a sample clip, from `vs1 embed-state`), so asking about it needs no
   * frame embedding. `full`: one 512-float row per frame (`frames` of them),
   * `crops`: the crop rows, each flattened.
   */
  useEmbeddings(state, { frames, full, crops }) {
    const run = async () => {
      this.#embedded = null;
      await this.#main.call('setState', { full, crops });
      this.#embedded = { key: keyOf(state), frames };
    };
    const result = this.#queue.then(run, run);
    this.#queue = result.catch(() => {});
    return result;
  }

  /**
   * Encodes questions ahead of time (e.g. while they are typed), so a later
   * `evaluate` with them only runs the answer head. Resolves when done;
   * rejects (status 422) on an invalid request.
   */
  prepare({ model, questions }) {
    const request = JSON.stringify({ model, questions });
    if (request === this.#prepared) return Promise.resolve();
    this.#prepared = request;
    return this.#encodeTexts(request).then(() => this.#main.call('validate', { request })).catch((err) => {
      this.#prepared = null;
      throw err;
    });
  }

  /**
   * Jev-style evaluation. `state` is a File, Blob, URL string or <video>.
   * Options: `frames` (uniform samples, default 16), `onProgress(done, total, index)`,
   * `onFrame(index, canvas, cropped)` (each sampled frame as it is decoded,
   * e.g. to draw thumbnails; `cropped` if its crops are embedded too).
   * Calls are serialized, since the main worker holds one video state at a time.
   * Asking again about the same state (same object, same `frames`) reuses its
   * frame embeddings, so only new question text is encoded.
   */
  evaluate({ state, model, questions }, { frames = 16, onProgress, onFrame } = {}) {
    const run = () => this.#evaluate({ state, model, questions }, { frames, onProgress, onFrame });
    const result = this.#queue.then(run, run);
    this.#queue = result.catch(() => {});
    return result;
  }

  async #evaluate({ state, model, questions }, { frames, onProgress, onFrame }) {
    // Validate questions before spending time on video decoding.
    const request = JSON.stringify({ model, questions });
    await this.#encodeTexts(request);
    await this.#main.call('validate', { request });
    const key = keyOf(state);
    if (this.#embedded?.key === key && this.#embedded.frames === frames) {
      return await this.#main.call('evaluate', { request });
    }

    this.#embedded = null;
    const { video, release } = await toVideo(state);
    try {
      const rows = await this.#embedFrames(video, frames, onProgress, onFrame);
      // Full views first (one per frame, in order), then the crop rows.
      const full = new Float32Array(frames * EMBED_DIM);
      const crops = [];
      rows.forEach((r, i) => {
        full.set(r.subarray(0, EMBED_DIM), i * EMBED_DIM);
        if (r.length > EMBED_DIM) crops.push(r.subarray(EMBED_DIM));
      });
      const cropRows = new Float32Array(crops.reduce((n, c) => n + c.length, 0));
      crops.reduce((at, c) => (cropRows.set(c, at), at + c.length), 0);
      await this.#main.call('setState', { full, crops: cropRows });
      this.#embedded = { key, frames };
      return await this.#main.call('evaluate', { request });
    } finally {
      release();
    }
  }

  /** Seeks frame by frame (one video element) while the workers embed in parallel. */
  async #embedFrames(video, frames, onProgress, onFrame) {
    await this.loadVision();
    const width = Math.min(MAX_ANALYSIS_WIDTH, video.videoWidth) & ~1;
    const height = Math.max(2, Math.round((video.videoHeight * width) / video.videoWidth / 2) * 2);
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const duration = video.duration;
    if (!Number.isFinite(duration) || duration <= 0) throw new Error('video has no usable duration');
    const withCrops = new Set(cropFrameIndices(frames));
    const idle = [this.#main, ...this.#embedders];
    const waiting = []; // resolvers for a free worker
    const acquire = () => (idle.length ? Promise.resolve(idle.pop()) : new Promise((r) => waiting.push(r)));
    const releaseWorker = (w) => (waiting.length ? waiting.shift()(w) : idle.push(w));
    const rows = new Array(frames);
    const jobs = [];
    let done = 0;
    for (let i = 0; i < frames; i++) {
      const worker = await acquire();
      await seek(video, ((i + 0.5) * duration) / frames);
      ctx.drawImage(video, 0, 0, width, height);
      onFrame?.(i, canvas, withCrops.has(i));
      const { data } = ctx.getImageData(0, 0, width, height);
      jobs.push(
        worker
          .call('embed', { rgba: data.buffer, width, height, crops: withCrops.has(i) }, [data.buffer])
          .then((r) => {
            rows[i] = r;
            onProgress?.(++done, frames, i);
          })
          .finally(() => releaseWorker(worker)),
      );
    }
    await Promise.all(jobs);
    return rows;
  }
}

async function toVideo(state) {
  if (state instanceof HTMLVideoElement) {
    if (state.readyState < 1) await once(state, 'loadedmetadata');
    return { video: state, release: () => {} };
  }
  const url = typeof state === 'string' ? state : URL.createObjectURL(state);
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.src = url;
  await once(video, 'loadedmetadata');
  return {
    video,
    release: () => {
      video.removeAttribute('src');
      video.load();
      if (typeof state !== 'string') URL.revokeObjectURL(url);
    },
  };
}

function once(el, event) {
  return new Promise((resolve, reject) => {
    el.addEventListener(event, resolve, { once: true });
    el.addEventListener('error', () => reject(new Error(`could not load video (${event})`)), { once: true });
  });
}

async function seek(video, t) {
  video.currentTime = Math.min(t, Math.max(0, video.duration - 0.001));
  await once(video, 'seeked');
}
