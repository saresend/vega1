import { SystemOne } from './vs1.js';

const $ = (id) => document.getElementById(id);
let s1 = null;
let file = null;

const setStatus = (text, isError = false) => {
  $('status').textContent = text;
  $('status').className = isError ? 'error' : 'muted';
};
const fmt = (p) => `${(p * 100).toFixed(1)}%`;
const escape = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

function bars(entries) {
  return `<div class="bars">${entries
    .map(([label, p]) => `<div class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i><span>${escape(label)}</span><b>${fmt(p)}</b></div>`)
    .join('')}</div>`;
}

function renderAnswers(res, questions) {
  $('answers').innerHTML = Object.entries(res.answers)
    .map(([key, a]) => {
      const q = questions[key];
      const title = typeof q.instructions === 'string' ? q.instructions : JSON.stringify(q.instructions);
      let body;
      if (a.type === 'noul') {
        body = `<div class="value">${fmt(a.noul)}</div><div class="muted">probability "yes"</div>` +
          bars([['yes', a.noul], ['no', 1 - a.noul]]);
      } else if (a.type === 'choice') {
        body = `<div class="value">${escape(a.choice)}</div><div class="muted">confidence ${a.confidence.toFixed(2)}</div>` +
          bars(Object.entries(a.probabilities));
      } else {
        body = `<div class="value">${a.score.toFixed(2)}</div><div class="muted">of ${Object.keys(a.legend).length - 1} · confidence ${a.confidence.toFixed(2)}</div>` +
          bars(Object.entries(a.probabilities).map(([i, p]) => [`${i}: ${a.legend[i]}`, p]));
      }
      return `<div class="panel answer"><div class="key">${escape(key)} · ${a.type}</div><div>${escape(title)}</div>${body}</div>`;
    })
    .join('');
  $('raw').textContent = JSON.stringify(res, null, 2);
  $('raw-wrap').hidden = false;
}

async function run() {
  let request;
  try {
    request = JSON.parse($('questions').value);
  } catch (err) {
    setStatus(`Questions are not valid JSON: ${err.message}`, true);
    return;
  }
  $('run').disabled = true;
  const started = performance.now();
  try {
    const res = await s1.evaluate(
      { state: file, model: request.model, questions: request.questions },
      {
        frames: Number($('frames').value),
        onProgress: (done, total) => {
          $('progress').value = done / total;
          setStatus(`Embedding frame ${done}/${total}…`);
        },
      },
    );
    renderAnswers(res, request.questions);
    setStatus(`Answered ${Object.keys(res.answers).length} questions from ${res.usage.input_frames} frames in ` +
      `${((performance.now() - started) / 1000).toFixed(1)} s · model ${res.model}`);
  } catch (err) {
    setStatus(err.status === 422 ? `Invalid request: ${err.message}` : `Error: ${err.message}`, true);
  } finally {
    $('run').disabled = !file;
  }
}

$('file').addEventListener('change', (e) => {
  file = e.target.files[0] ?? null;
  if (file) $('preview').src = URL.createObjectURL(file);
  $('run').disabled = !file || !s1;
});
$('run').addEventListener('click', run);

const examples = await fetch('./questions.json').then((r) => r.text()).catch(() => '{"questions": {}}');
$('questions').value = examples;

try {
  const params = new URLSearchParams(location.search);
  s1 = await SystemOne.load({ head: params.get('head') });
  const { model, downloadMs, loadMs } = s1.info;
  setStatus(`Model ${model} ready (download ${(downloadMs / 1000).toFixed(1)} s, init ${(loadMs / 1000).toFixed(1)} s). Choose a video.`);
  $('run').disabled = !file;
} catch (err) {
  setStatus(`Failed to load model: ${err.message}`, true);
}
