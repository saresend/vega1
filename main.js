import { SystemOne, cropFrameIndices } from './vs1.js';

const $ = (id) => document.getElementById(id);
const escape = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

// Example and suggested statements (their text embeddings ship in samples/texts.*).
const { defaults: DEFAULT_STATEMENTS, suggestions: SUGGESTIONS } = await fetch('statements.json')
  .then((r) => r.json()).catch(() => ({ defaults: ['There are people in the video', 'There is a dog'], suggestions: [] }));
const BAR = 14; // characters in a probability bar

const store = {
  get(key) { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ } },
};

// ---------------------------------------------------------------- state

let s1 = null;
let file = null; // the current video (File)
let sample = null; // its manifest entry, if it is a sample
let preloaded = null; // { frames, full, crops } embeddings shipped with that sample
let frames = 16;
let analyzed = false; // whether the worker holds this video's frames
let nextId = 1;
let statements = (store.get('vs1-statements-v3') ?? DEFAULT_STATEMENTS).map((text) => ({ id: nextId++, text }));
let custom = store.get('vs1-custom') ?? false; // edited by hand: samples don't replace the list
const results = new Map(); // statement id → probability, for the current video
let pending = new Set(); // statement ids being answered
const askTimes = []; // recent post-embedding ask latencies (ms)

const saveStatements = () => {
  store.set('vs1-statements-v3', statements.map((s) => s.text));
  store.set('vs1-custom', custom);
};
const request = (extra = []) => ({
  questions: Object.fromEntries([
    ...statements.map((s) => [`s${s.id}`, { type: 'noul', instructions: s.text }]),
    ...extra.map((t, i) => [`draft${i}`, { type: 'noul', instructions: t }]),
  ]),
});

// ---------------------------------------------------------------- status

function setModelStatus(state, text) {
  $('model-pill').dataset.state = state;
  $('model-status').textContent = text;
}
const setFoot = (html) => { $('foot-text').innerHTML = html; };
const ms = (t) => (t < 1000 ? `${Math.round(t)} ms` : `${(t / 1000).toFixed(1)} s`);

function recordAsk(t) {
  askTimes.push(t);
  if (askTimes.length > 50) askTimes.shift();
  const mean = askTimes.reduce((a, b) => a + b, 0) / askTimes.length;
  $('stat-latency').textContent = ms(mean);
  $('stat-latency-label').textContent = `average time for each question on this device (${askTimes.length} question${askTimes.length > 1 ? 's' : ''})`;
}

// ---------------------------------------------------------------- statements

const pct = (p) => (p < 0.01 ? '<1%' : p > 0.99 ? '>99%' : `${Math.round(p * 100)}%`);
const verdict = (p) => (p >= 0.7 ? 'yes' : p <= 0.3 ? 'no' : 'unsure');

function rowFor(s) {
  let li = document.querySelector(`.stmt[data-id="${s.id}"]`);
  if (!li) {
    li = document.createElement('li');
    li.dataset.id = s.id;
    li.innerHTML = '<span class="text"></span><span class="bar" aria-hidden="true"><span class="on"></span><span class="off"></span></span>' +
      '<span class="pct"></span><span class="tag"></span>' +
      '<button class="remove" aria-label="Remove statement"><svg class="i" viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg></button>';
    li.querySelector('.remove').addEventListener('click', () => removeStatement(s.id));
  }
  li.querySelector('.text').textContent = s.text;
  return li;
}

function renderStatements() {
  const rows = statements.map(rowFor);
  $('stmts').replaceChildren(...rows);
  for (const [i, s] of statements.entries()) {
    const li = rows[i];
    const p = results.get(s.id);
    const state = pending.has(s.id) ? 'pending' : p === undefined ? 'idle' : verdict(p);
    const on = p === undefined ? 0 : Math.round(p * BAR);
    li.className = `stmt ${state}`;
    li.querySelector('.on').textContent = '█'.repeat(on);
    li.querySelector('.off').textContent = '░'.repeat(BAR - on);
    li.querySelector('.pct').textContent = p === undefined ? '–' : pct(p);
    li.querySelector('.tag').textContent = state === 'unsure' ? 'not sure' : state;
    li.title = p === undefined ? '' : `P(true) = ${(p * 100).toFixed(1)}%`;
  }
  $('empty').hidden = statements.length > 0;
  $('count').textContent = statements.length ? String(statements.length) : '';
  renderSuggestions();
}

function renderSuggestions() {
  const have = new Set(statements.map((s) => s.text.toLowerCase()));
  const pool = [...(sample?.statements ?? []), ...SUGGESTIONS];
  const picks = [...new Set(pool)].filter((t) => !have.has(t.toLowerCase())).slice(0, 6);
  $('suggest').replaceChildren(...picks.map((t) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = t;
    b.addEventListener('click', () => addStatements([t]));
    return b;
  }));
}

function addStatements(texts, byHand = true) {
  const have = new Set(statements.map((s) => s.text.toLowerCase()));
  for (const t of texts.map((x) => x.trim()).filter(Boolean)) {
    if (have.has(t.toLowerCase())) continue;
    have.add(t.toLowerCase());
    statements.push({ id: nextId++, text: t });
  }
  if (byHand) custom = true;
  changed();
}

function removeStatement(id) {
  statements = statements.filter((s) => s.id !== id);
  results.delete(id);
  custom = true;
  changed();
}

function setStatements(texts) {
  statements = [];
  results.clear();
  custom = false;
  addStatements(texts, false);
}

/** Statements changed: answer right away if the video is ready, else just encode them. */
function changed() {
  saveStatements();
  renderStatements();
  if (analyzed) askSoon();
  else prepareSoon();
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
    if (s1 && (statements.length || extra.length)) s1.prepare(request(extra)).catch(() => {});
  }, 200);
}

/** Answers the current statements for the current video; the latest call wins. */
async function ask() {
  if (!s1 || !file || !statements.length) {
    if (!statements.length) setFoot(file ? 'Add a statement. Then the model answers it.' : 'Select a video. Then the model answers the statements.');
    return;
  }
  if (asking) {
    askAgain = true;
    return;
  }
  asking = true;
  const asked = statements.slice();
  const forFile = file;
  pending = new Set(asked.filter((s) => !results.has(s.id)).map((s) => s.id));
  renderStatements();
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
    const res = await s1.evaluate({ state: file, ...request() }, { frames, onFrame: drawFrame, onProgress: frameDone });
    clearTimeout(slow);
    $('viewer').classList.remove('analyzing');
    const took = performance.now() - started;
    if (forFile !== file) return; // the video changed meanwhile
    analyzed = true;
    for (const s of asked) {
      const a = res.answers[`s${s.id}`];
      if (a) results.set(s.id, a.noul);
    }
    const n = asked.length;
    const what = `${n} statement${n === 1 ? '' : 's'}`;
    if (first) {
      setFoot(`Time to analyze ${res.usage.input_frames} frames and answer ${what}: ${ms(took)}`);
      $('frames-status').textContent = `${frames} frames · analysis time ${ms(took)}`;
    } else {
      recordAsk(took);
      setFoot(`<span class="speed">${ms(took)}</span> to answer ${what}`);
    }
  } catch (err) {
    setFoot(`<span class="error">${escape(err.status === 422 ? `The statement is not correct: ${err.message}` : err.message)}</span>`);
    if (first) $('frames-status').textContent = 'the model cannot read this video';
  } finally {
    asking = false;
    pending = new Set();
    renderStatements();
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
  results.clear();
  $('picker').hidden = !!file;
  $('viewer').hidden = !file;
  for (const c of document.querySelectorAll('.sample')) c.classList.toggle('active', c === entry?.card);
  if (!file) {
    setFoot('Select a video. Then the model answers the statements.');
    return renderStatements();
  }
  if (entry && !custom) setStatements(entry.statements);
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
  renderStatements();
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
  setFoot('The embeddings are ready. The model answers the statements.');
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
    results.clear();
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

/** Precomputed text embeddings for every offered statement (scripts/fetch-samples.py). */
async function addPrecomputedTexts() {
  const [keys, data] = await Promise.all([
    fetch('samples/texts.json').then((r) => (r.ok ? r.json() : null)),
    fetch('samples/texts.f32').then((r) => (r.ok ? r.arrayBuffer() : null)),
  ]).catch(() => [null, null]);
  if (keys && data) await s1.addTexts(keys, new Float32Array(data));
}

/** Starts the text encoder download (~90 MB) once a custom statement is likely. */
function wantText() {
  if (!s1 || s1.textLoaded || textLoading) return;
  textLoading = true;
  $('text-status').textContent = 'the text model loads (90 MB)…';
  s1.loadText().then(() => { $('text-status').textContent = 'the text model is ready'; })
    .catch(() => { textLoading = false; $('text-status').textContent = 'the text model did not load'; });
}
let textLoading = false;

// ---------------------------------------------------------------- statement editing

// Encode the draft while it's typed, so asking only runs the answer head.
$('add-input').addEventListener('focus', wantText);
$('add-input').addEventListener('input', () => {
  const draft = $('add-input').value.trim();
  $('add-btn').disabled = !draft;
  if (draft.length > 2) prepareSoon([draft]);
});
$('add-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const text = $('add-input').value.trim();
  if (!text) return;
  $('add-input').value = '';
  $('add-btn').disabled = true;
  addStatements([text]);
});

let bulk = false;
$('bulk-toggle').addEventListener('click', () => {
  bulk = !bulk;
  if (bulk) {
    $('bulk').value = statements.map((s) => s.text).join('\n');
  } else {
    // Keep answers for statements that didn't change.
    const before = new Map(statements.map((s) => [s.text, s]));
    const texts = [...new Set($('bulk').value.split('\n').map((l) => l.trim()).filter(Boolean))];
    statements = texts.map((t) => before.get(t) ?? { id: nextId++, text: t });
    for (const id of [...results.keys()]) if (!statements.some((s) => s.id === id)) results.delete(id);
    custom = true;
    changed();
  }
  $('bulk-view').hidden = !bulk;
  $('list-view').hidden = bulk;
  $('bulk-toggle').textContent = bulk ? 'list view' : 'text view';
});
$('reset').addEventListener('click', () => setStatements(sample?.statements?.length ? sample.statements : DEFAULT_STATEMENTS));

// ---------------------------------------------------------------- start

renderStatements();
const manifest = loadSamples();

// Phones: samples go below the statements, so the answers are in the first screen.
const phone = matchMedia('(max-width: 560px)');
const placeSamples = () => (phone.matches ? $('samples-mobile') : $('samples-home')).append($('samples-wrap'));
phone.addEventListener('change', placeSamples);
placeSamples();

// The trained head ships in models/ when built; `?head=` (empty) forces zero-shot,
// `?head=path` loads another head.
async function loadModel() {
  const params = new URLSearchParams(location.search);
  if (params.has('head')) return SystemOne.load({ head: params.get('head') || null });
  const head = 'models/head.safetensors';
  const shipped = await fetch(head, { method: 'HEAD' }).then((r) => r.ok).catch(() => false);
  return SystemOne.load({ head: shipped ? head : null });
}

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
  setModelStatus('error', 'Vega1 did not load');
  setFoot(`<span class="error">${escape(err.message)}</span>`);
}
