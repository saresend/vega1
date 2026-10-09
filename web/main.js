// Main thread: picks frames out of a <video> by seeking, hands RGBA pixels to
// the worker, draws boxes/caption bands, and renders the final answers.

const $ = (id) => document.getElementById(id);
const video = $('video');
const overlay = $('overlay');
const octx = overlay.getContext('2d');
const grab = document.createElement('canvas');
const gctx = grab.getContext('2d', { willReadFrequently: true });

const MAX_ANALYSIS_WIDTH = 640;
const PROBE_MAX_BYTES = 1 << 30;
const TRACK_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#9a6324'];

const worker = new Worker('./worker.js', { type: 'module' });
const pending = new Map();
let nextId = 0;
worker.onmessage = ({ data }) => {
  const p = pending.get(data.id);
  pending.delete(data.id);
  data.ok ? p.resolve(data.result) : p.reject(new Error(data.error));
};
function call(type, payload = {}, transfer = []) {
  return new Promise((resolve, reject) => {
    const id = ++nextId;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, type, ...payload }, transfer);
  });
}

const once = (el, ev) => new Promise((r) => el.addEventListener(ev, r, { once: true }));
let cancelled = false;

function setStatus(text) {
  $('status').textContent = text;
}

async function analyze(file) {
  cancelled = false;
  $('file').disabled = true;
  $('cancel').hidden = false;
  $('results').hidden = true;
  setStatus('Loading video…');

  video.src = URL.createObjectURL(file);
  await once(video, 'loadedmetadata');
  const duration = video.duration;
  const fps = Number($('fps').value);
  const width = Math.min(MAX_ANALYSIS_WIDTH, video.videoWidth) & ~1;
  const height = Math.max(2, Math.round((video.videoHeight * width) / video.videoWidth / 2) * 2);
  grab.width = overlay.width = width;
  grab.height = overlay.height = height;

  await call('reset');
  if (file.size <= PROBE_MAX_BYTES) {
    setStatus('Reading container…');
    const bytes = await file.arrayBuffer();
    await call('probe', { bytes }, [bytes]);
  }

  const total = Math.max(1, Math.floor(duration * fps));
  let busyMs = 0;
  const started = performance.now();
  for (let i = 0; i < total && !cancelled; i++) {
    const t = i / fps;
    video.currentTime = t;
    await once(video, 'seeked');
    gctx.drawImage(video, 0, 0, width, height);
    const { data } = gctx.getImageData(0, 0, width, height);
    const fr = await call('frame', { rgba: data.buffer, width, height, t }, [data.buffer]);
    busyMs += fr.ms;
    drawOverlay(fr);
    $('progress').value = (i + 1) / total;
    setStatus(`Frame ${i + 1}/${total} · ${fr.ms.toFixed(0)} ms inference · ${fr.subjects.length} people in view`);
  }

  const report = await call('report', { duration });
  const wall = (performance.now() - started) / 1000;
  renderReport(report, { busyMs, wall, cancelled });
  setStatus(cancelled ? 'Cancelled — partial results below.' : `Done in ${wall.toFixed(1)} s.`);
  $('file').disabled = false;
  $('cancel').hidden = true;
}

function drawOverlay(fr) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  octx.lineWidth = 2;
  octx.font = '12px system-ui, sans-serif';
  for (const s of fr.subjects) {
    const { x1, y1, x2, y2 } = s.bbox;
    const color = TRACK_COLORS[s.track_id % TRACK_COLORS.length];
    octx.strokeStyle = color;
    octx.setLineDash(s.confirmed ? [] : [4, 3]);
    octx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    octx.fillStyle = color;
    octx.fillText(`#${s.track_id}`, x1 + 2, Math.max(12, y1 - 3));
  }
  octx.setLineDash([]);
  if (fr.caption) {
    const c = fr.caption;
    octx.strokeStyle = '#ffd400';
    octx.strokeRect(c.x_left * overlay.width, c.y_top * overlay.height,
      (c.x_right - c.x_left) * overlay.width, (c.y_bottom - c.y_top) * overlay.height);
  }
  if (fr.shot_cut) {
    octx.fillStyle = '#ffd400';
    octx.fillText('CUT', 6, 14);
  }
}

const pct = (x) => `${Math.round(x * 100)}%`;
const secs = (x) => `${x.toFixed(1)} s`;

function renderReport(r, { busyMs, wall }) {
  const sc = r.answers.subject_count;
  const cap = r.answers.captioned;
  $('subjects-value').textContent = sc.value;
  $('subjects-detail').textContent =
    `confidence ${pct(sc.confidence)} · at most ${sc.max_in_single_frame} in one frame · ` +
    `${sc.reidentified_merges} re-identified return${sc.reidentified_merges === 1 ? '' : 's'}`;
  $('subjects-list').innerHTML = sc.subjects
    .map((s) => `<li>Subject ${s.id}: ${secs(s.first_seen)} – ${secs(s.last_seen)} ` +
      `<span class="muted">(tracks ${s.track_ids.join(', ')})</span></li>`)
    .join('');

  $('captions-value').textContent = cap.value ? 'Yes' : 'No';
  const b = cap.burned_in;
  const via = {
    caption_track: `subtitle track in ${cap.container_format}: ` +
      cap.caption_tracks.map((t) => `${t.codec}${t.language ? ` (${t.language})` : ''}`).join(', '),
    burned_in: 'burned into the picture',
    none: b.static_text_overlay ? 'only static on-screen text found (logo or watermark)' : 'no captions found',
  }[cap.source];
  $('captions-detail').textContent = `confidence ${pct(cap.confidence)} · ${via}`;
  $('captions-list').innerHTML =
    `<li>Burned-in text present in ${pct(b.presence_ratio)} of frames</li>` +
    `<li>${b.text_changes_per_minute.toFixed(1)} text changes per minute</li>`;

  $('perf').textContent =
    `${r.frames_analyzed} frames · ${(busyMs / Math.max(1, r.frames_analyzed)).toFixed(0)} ms/frame inference · ` +
    `${wall.toFixed(1)} s total · ${r.shot_cuts.length} shot cut${r.shot_cuts.length === 1 ? '' : 's'}`;
  $('raw').textContent = JSON.stringify(r, null, 2);
  $('results').hidden = false;
}

$('file').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) analyze(file).catch((err) => {
    setStatus(`Error: ${err.message}`);
    $('file').disabled = false;
    $('cancel').hidden = true;
  });
});
$('cancel').addEventListener('click', () => { cancelled = true; });

setStatus('Loading model…');
call('init')
  .then(({ loadMs }) => {
    setStatus(`Model ready (${loadMs.toFixed(0)} ms). Choose a video.`);
    $('file').disabled = false;
  })
  .catch((err) => setStatus(`Failed to load model: ${err.message}`));
