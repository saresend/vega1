// Vega1 page (index.html): yes/no statements about the video.
import { $, pct, store, startDemo } from './demo.js';

// Example and suggested statements (their text embeddings ship in samples/texts.*).
const { defaults: DEFAULT_STATEMENTS, suggestions: SUGGESTIONS } = await fetch('statements.json')
  .then((r) => r.json()).catch(() => ({ defaults: ['There are people in the video', 'There is a dog'], suggestions: [] }));
const BAR = 14; // characters in a probability bar

let nextId = 1;
let statements = (store.get('vs1-statements-v3') ?? DEFAULT_STATEMENTS).map((text) => ({ id: nextId++, text }));
let custom = store.get('vs1-custom') ?? false; // edited by hand: samples don't replace the list
const results = new Map(); // statement id → probability, for the current video
let pending = new Set(); // statement ids being answered

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

// ---------------------------------------------------------------- statements

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
  const pool = [...(demo?.sample()?.statements ?? []), ...SUGGESTIONS];
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

function changed() {
  saveStatements();
  demo.changed();
}

let demo = null;
demo = startDemo({
  name: 'Vega1',
  noun: 'statement',
  heads: { head: 'models/head.safetensors' },
  texts: 'samples/texts',
  count: () => statements.length,
  request,
  apply(res) {
    for (const s of statements) {
      const a = res.answers[`s${s.id}`];
      if (a) results.set(s.id, a.noul);
    }
  },
  onSample(entry) {
    if (!custom) setStatements(entry.statements);
  },
  clear: () => results.clear(),
  render(busy) {
    pending = busy ? new Set(statements.filter((s) => !results.has(s.id)).map((s) => s.id)) : new Set();
    renderStatements();
  },
});

// ---------------------------------------------------------------- statement editing

// Encode the draft while it's typed, so asking only runs the answer head.
$('add-input').addEventListener('focus', () => demo.wantText());
$('add-input').addEventListener('input', () => {
  const draft = $('add-input').value.trim();
  $('add-btn').disabled = !draft;
  if (draft.length > 2) demo.prepareSoon([draft]);
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
$('reset').addEventListener('click', () => setStatements(demo.sample()?.statements?.length ? demo.sample().statements : DEFAULT_STATEMENTS));
