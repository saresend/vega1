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

export class SystemOne {
  #worker;
  #pending = new Map();
  #nextId = 0;
  #queue = Promise.resolve();

  constructor(worker) {
    this.#worker = worker;
    worker.onmessage = ({ data }) => {
      const p = this.#pending.get(data.id);
      this.#pending.delete(data.id);
      if (data.ok) p.resolve(data.result);
      else p.reject(Object.assign(new Error(data.error), { status: data.status }));
    };
  }

  /** Loads the model in a worker. `head` is an optional trained-head URL. */
  static async load({ baseUrl = new URL('.', import.meta.url).href, head = null } = {}) {
    const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    const s1 = new SystemOne(worker);
    s1.info = await s1.#call('init', { baseUrl, head });
    return s1;
  }

  #call(type, payload = {}, transfer = []) {
    return new Promise((resolve, reject) => {
      const id = ++this.#nextId;
      this.#pending.set(id, { resolve, reject });
      this.#worker.postMessage({ id, type, ...payload }, transfer);
    });
  }

  /**
   * Jev-style evaluation. `state` is a File, Blob, URL string or <video>.
   * Options: `frames` (uniform samples, default 16), `onProgress(done, total)`.
   * Calls are serialized, since the worker holds one video state at a time.
   */
  evaluate({ state, model, questions }, { frames = 16, onProgress } = {}) {
    const run = () => this.#evaluate({ state, model, questions }, { frames, onProgress });
    const result = this.#queue.then(run, run);
    this.#queue = result.catch(() => {});
    return result;
  }

  async #evaluate({ state, model, questions }, { frames, onProgress }) {
    // Validate questions before spending time on video decoding.
    const request = JSON.stringify({ model, questions });
    await this.#call('validate', { request });

    const { video, release } = await toVideo(state);
    try {
      const width = Math.min(MAX_ANALYSIS_WIDTH, video.videoWidth) & ~1;
      const height = Math.max(2, Math.round((video.videoHeight * width) / video.videoWidth / 2) * 2);
      const canvas = new OffscreenCanvas(width, height);
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      await this.#call('clear');
      const duration = video.duration;
      if (!Number.isFinite(duration) || duration <= 0) throw new Error('video has no usable duration');
      for (let i = 0; i < frames; i++) {
        await seek(video, ((i + 0.5) * duration) / frames);
        ctx.drawImage(video, 0, 0, width, height);
        const { data } = ctx.getImageData(0, 0, width, height);
        await this.#call('frame', { rgba: data.buffer, width, height }, [data.buffer]);
        onProgress?.(i + 1, frames);
      }
      return await this.#call('evaluate', { request });
    } finally {
      release();
    }
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
