// What both demo pages share: the model, the video panel (samples, your own
// video, frame strip), asking with "latest call wins", and the status lines.
// A page supplies its question list (`page` below); index.html asks yes/no
// statements (main.js), vega1-cat.html category questions (cat.js).

import { SystemOne, cropFrameIndices } from './vs1.js';

export const $ = (id) => document.getElementById(id);
export const escape = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
export const ms = (t) => (t < 1000 ? `${Math.round(t)} ms` : `${(t / 1000).toFixed(1)} s`);
export const pct = (p) => (p < 0.01 ? '<1%' : p > 0.99 ? '>99%' : `${Math.round(p * 100)}%`);
export const store = {
  get(key) { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ } },
};
export const setFoot = (html) => { $('foot-text').innerHTML = html; };

/**
 * Starts the page. `page` supplies:
 *   name           model name for status lines ("Vega1")
 *   noun           "statement" / "question"
 *   heads          { head, choiceHead } model files
 *   texts          precomputed text embeddings, e.g. "samples/texts" (.json + .f32)
 *   count()        number of items in the list
 *   request(extra) the Jev request for the list (+ draft items)
 *   apply(res)     store a response's answers
 *   onSample(entry) a sample was opened (replace the list unless edited by hand)
 *   clear()        forget all answers
 *   render(pending) redraw; `pending` while answering
 * Returns { changed, prepareSoon, wantText, sample() }.
 */
export function startDemo(page) {
  let s1 = null;
  let file = null; // the current video (File)
  let sample = null; // its manifest entry, if it is a sample
  let preloaded = null; // { frames, full, crops } embeddings shipped with that sample
  let frames = 16;
  let analyzed = false; // whether the worker holds this video's frames
  const askTimes = []; // recent post-embedding ask latencies (ms)
  const plural = (n) => `${n} ${page.noun}${n === 1 ? '' : 's'}`;
  const idleFoot = () => setFoot(file ? `Add a ${page.noun}. Then the model answers it.` : `Select a video. Then the model answers the ${page.noun}s.`);

  function setModelStatus(state, text) {
    $('model-pill').dataset.state = state;
    $('model-status').textContent = text;
  }

  function recordAsk(t) {
    askTimes.push(t);
    if (askTimes.length > 50) askTimes.shift();
    const mean = askTimes.reduce((a, b) => a + b, 0) / askTimes.length;
    $('stat-latency').textContent = ms(mean);
    $('stat-latency-label').textContent = `average time for each question on this device (${askTimes.length} question${askTimes.length > 1 ? 's' : ''})`;
  }

  // ---------------------------------------------------------------- asking

  let askTimer = null;
  let asking = false;
  let askAgain = false;
  const askSoon = () => {
    clearTimeout(askTimer);
    askTimer = setTimeout(ask, 0);
  };

  let prepareTimer = null;
  function prepareSoon(extra = []) {
    clearTimeout(prepareTimer);
    prepareTimer = setTimeout(() => {
      if (s1 && (page.count() || extra.length)) s1.prepare(page.request(extra)).catch(() => {});
    }, 200);
  }

  /** The list changed: answer right away if the video is ready, else just encode it. */
  function changed() {
    page.render(false);
    if (analyzed) askSoon();
    else prepareSoon();
  }

  /** Answers the list for the current video; the latest call wins. */
  async function ask() {
    if (!s1 || !file || !page.count()) {
      if (!page.count()) idleFoot();
      return;
    }
    if (asking) {
      askAgain = true;
      return;
    }
    asking = true;
    const req = page.request();
    const n = page.count();
    const forFile = file;
    page.render(true);
    const first = !analyzed;
    const started = performance.now();
    try {
      if (first) startFrames();
      const slow = setTimeout(() => {
        if (!s1.textLoaded) {
          wantText();
          setFoot('The text model loads (90 MB, one time only). Please wait.');
        }
      }, 300);
      const res = await s1.evaluate({ state: file, ...req }, { frames, onFrame: drawFrame, onProgress: frameDone });
      clearTimeout(slow);
      $('viewer').classList.remove('analyzing');
      const took = performance.now() - started;
      if (forFile !== file) return; // the video changed meanwhile
      analyzed = true;
      page.apply(res);
      if (first) {
        setFoot(`Time to analyze ${res.usage.input_frames} frames and answer ${plural(n)}: ${ms(took)}`);
        $('frames-status').textContent = `${frames} frames · analysis time ${ms(took)}`;
      } else {
        recordAsk(took);
        setFoot(`<span class="speed">${ms(took)}</span> to answer ${plural(n)}`);
      }
    } catch (err) {
      setFoot(`<span class="error">${escape(err.status === 422 ? `The ${page.noun} is not correct: ${err.message}` : err.message)}</span>`);
      if (first) $('frames-status').textContent = 'the model cannot read this video';
    } finally {
      asking = false;
      page.render(false);
      if (askAgain) {
        askAgain = false;
        ask();
      }
    }
  }

  // ---------------------------------------------------------------- video

  function setFrames(n) {
    frames = n;
    for (const o of document.querySelectorAll('[data-frames]')) o.setAttribute('aria-pressed', String(Number(o.dataset.frames) === n));
  }

  async function setFile(next, entry = null, embeddings = null) {
    file = next;
    sample = entry;
    preloaded = embeddings;
    analyzed = false;
    page.clear();
    $('picker').hidden = !!file;
    $('viewer').hidden = !file;
    for (const c of document.querySelectorAll('.sample')) c.classList.toggle('active', c === entry?.card);
    if (!file) {
      idleFoot();
      return page.render(false);
    }
    if (entry) page.onSample(entry);
    $('preview').src = URL.createObjectURL(file);
    $('preview').play().catch(() => {});
    $('file-name').textContent = entry?.title ?? file.name;
    $('file-meta').innerHTML = entry?.credit
      ? entry.source ? `<a href="${escape(entry.source)}" target="_blank" rel="noopener">${escape(entry.credit)}</a>` : escape(entry.credit)
      : '';
    $('preview').onloadedmetadata = () => {
      const d = $('preview').duration;
      if (Number.isFinite(d) && !entry) $('file-meta').textContent = `${d.toFixed(1)} s`;
    };
    if (embeddings) setFrames(embeddings.frames);
    page.render(false);
    if (!s1) return setFoot('The model loads. Please wait.');
    await usePreloaded();
    ask();
  }

  /** A sample's precomputed embeddings become the video state: no frame embedding. */
  async function usePreloaded() {
    if (!preloaded || !s1 || preloaded.frames !== frames) return;
    await s1.useEmbeddings(file, preloaded);
    analyzed = true;
    showSprite();
    $('frames-status').textContent = `${frames} frames · embeddings ready`;
    setFoot(`The embeddings are ready. The model answers the ${page.noun}s.`);
  }

  function frameCells() {
    const strip = $('strip');
    strip.style.gridTemplateColumns = `repeat(${Math.min(frames, 16) / 2}, 1fr)`;
    const crops = new Set(cropFrameIndices(frames));
    strip.replaceChildren(...Array.from({ length: frames }, (_, i) => {
      const cell = document.createElement('div');
      cell.className = crops.has(i) ? 'f crops' : 'f';
      return cell;
    }));
    return [...strip.children];
  }

  /** The sample's frame strip (one image, `frames` tiles wide). */
  function showSprite() {
    frameCells().forEach((cell, i) => {
      cell.style.backgroundImage = `url("samples/${encodeURI(sample.strip)}")`;
      cell.style.backgroundSize = `${frames * 100}% 100%`;
      cell.style.backgroundPosition = `${(i / (frames - 1)) * 100}% 0`;
    });
    $('progress').style.width = '100%';
  }

  function startFrames() {
    $('viewer').classList.add('analyzing');
    for (const cell of frameCells()) cell.innerHTML = '<canvas width="96" height="54"></canvas>';
    $('progress').style.width = '0%';
    $('frames-status').textContent = `analysis of ${frames} frames…`;
    setFoot('The model analyzes the video in your browser. Please wait.');
  }

  function drawFrame(i, canvas) {
    const cell = $('strip').children[i];
    const c = cell?.querySelector('canvas');
    if (!c) return;
    const scale = Math.max(c.width / canvas.width, c.height / canvas.height);
    const w = canvas.width * scale;
    const h = canvas.height * scale;
    c.getContext('2d').drawImage(canvas, (c.width - w) / 2, (c.height - h) / 2, w, h);
    cell.classList.add('decoded');
  }

  function frameDone(done, total, i) {
    $('strip').children[i]?.classList.add('done');
    $('progress').style.width = `${(done / total) * 100}%`;
    $('frames-status').textContent = `analysis: ${done} of ${total} frames`;
  }

  for (const b of document.querySelectorAll('[data-frames]')) {
    b.addEventListener('click', async () => {
      setFrames(Number(b.dataset.frames));
      if (!file) return;
      analyzed = false;
      page.clear();
      await usePreloaded();
      ask();
    });
  }

  $('browse').addEventListener('click', () => $('file').click());
  $('file').addEventListener('change', (e) => e.target.files[0] && setFile(e.target.files[0]));
  $('change').addEventListener('click', () => {
    $('file').value = '';
    $('file').click();
  });
  const drop = $('drop');
  drop.addEventListener('dragover', (e) => {
    e.preventDefault();
    drop.classList.add('over');
  });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('over');
    const f = [...e.dataTransfer.files].find((x) => x.type.startsWith('video/')) ?? e.dataTransfer.files[0];
    if (f) setFile(f);
  });

  // Sample clips with precomputed embeddings: samples/samples.json, written by scripts/fetch-samples.py.
  async function loadSamples() {
    const manifest = await fetch('samples/samples.json').then((r) => (r.ok ? r.json() : [])).catch(() => []);
    const cards = manifest.map((entry) => {
      const b = document.createElement('button');
      b.className = 'sample';
      b.title = entry.credit ?? '';
      // Thumbnail: one tile of the clip's frame strip (lighter than a <video> per card).
      const tile = Math.min(entry.frames - 1, Math.max(0, Math.round(entry.frames / 2) - 1));
      b.innerHTML = `<span class="thumb" style="background-image:url('samples/${encodeURI(entry.strip)}');` +
        `background-size:${entry.frames * 100}% 100%;background-position:${(tile / (entry.frames - 1)) * 100}% 0"></span>` +
        `<span class="sname">${escape(entry.title)}</span><span class="smeta">${escape(entry.subtitle ?? '')}</span>`;
      b.addEventListener('click', () => openSample(entry));
      entry.card = b;
      return b;
    });
    $('samples').replaceChildren(...cards);
    $('samples-wrap').hidden = !cards.length;
    return manifest;
  }

  async function openSample(entry) {
    const [blob, rows] = await Promise.all([
      fetch(`samples/${encodeURI(entry.file)}`).then((r) => r.blob()),
      entry.embeddings ? fetch(`samples/${encodeURI(entry.embeddings)}`).then((r) => (r.ok ? r.arrayBuffer() : null)).catch(() => null) : null,
    ]);
    let embeddings = null;
    if (rows) {
      const all = new Float32Array(rows);
      const split = entry.frames * 512;
      embeddings = { frames: entry.frames, full: all.slice(0, split), crops: all.slice(split) };
    }
    setFile(new File([blob], entry.file, { type: blob.type || 'video/mp4' }), entry, embeddings);
  }

  /** Precomputed text embeddings for every offered question (scripts/fetch-samples.py). */
  async function addPrecomputedTexts() {
    const [keys, data] = await Promise.all([
      fetch(`${page.texts}.json`).then((r) => (r.ok ? r.json() : null)),
      fetch(`${page.texts}.f32`).then((r) => (r.ok ? r.arrayBuffer() : null)),
    ]).catch(() => [null, null]);
    if (keys && data) await s1.addTexts(keys, new Float32Array(data));
  }

  /** Starts the text encoder download (~90 MB) once a custom question is likely. */
  let textLoading = false;
  function wantText() {
    if (!s1 || s1.textLoaded || textLoading) return;
    textLoading = true;
    $('text-status').textContent = 'the text model loads (90 MB)…';
    s1.loadText().then(() => { $('text-status').textContent = 'the text model is ready'; })
      .catch(() => { textLoading = false; $('text-status').textContent = 'the text model did not load'; });
  }

  // ---------------------------------------------------------------- start

  page.render(false);
  const manifest = loadSamples();

  // Phones: samples go below the questions, so the answers are in the first screen.
  const phone = matchMedia('(max-width: 560px)');
  const placeSamples = () => (phone.matches ? $('samples-mobile') : $('samples-home')).append($('samples-wrap'));
  phone.addEventListener('change', placeSamples);
  placeSamples();

  // The trained heads ship in models/ when built; `?head=` (empty) forces zero-shot,
  // `?head=path` loads another head (for both question types).
  async function loadModel() {
    const params = new URLSearchParams(location.search);
    if (params.has('head')) return SystemOne.load({ head: params.get('head') || null });
    const shipped = async (url) => url && (await fetch(url, { method: 'HEAD' }).then((r) => r.ok).catch(() => false)) ? url : null;
    const [head, choiceHead] = await Promise.all([shipped(page.heads.head), shipped(page.heads.choiceHead)]);
    return SystemOne.load({ head, choiceHead });
  }

  (async () => {
    try {
      const t0 = performance.now();
      s1 = await loadModel();
      await addPrecomputedTexts();
      setModelStatus('ready', `ready (${ms(performance.now() - t0)})`);
      if (file) {
        await usePreloaded();
        ask();
      } else {
        const samples = await manifest;
        if (samples.length) openSample(samples[0]); // open straight into an answered sample
      }
    } catch (err) {
      setModelStatus('error', `${page.name} did not load`);
      setFoot(`<span class="error">${escape(err.message)}</span>`);
    }
  })();

  return { changed, prepareSoon, wantText, sample: () => sample };
}
