// Vega1-cat page (vega1-cat.html): category questions about the video, each a
// question and a set of options; the model selects one option.
import { $, escape, pct, store, startDemo } from './demo.js';

// Example and suggested questions (their text embeddings ship in samples/texts-cat.*).
const CATEGORIES = await fetch('categories.json').then((r) => r.json())
  .catch(() => ({ defaults: [{ q: 'How many people are in the video?', options: ['no people', 'one person', 'a group of people'] }], suggestions: [], samples: {} }));
const BAR = 14; // characters in a probability bar
const SURE = 0.5; // below this, the top option is "not sure"

let nextId = 1;
const withId = (q) => ({ id: nextId++, q: q.q, options: q.options });
let questions = (store.get('vega1-cat-questions-v1') ?? CATEGORIES.defaults).map(withId);
let custom = store.get('vega1-cat-custom') ?? false; // edited by hand: samples don't replace the list
const results = new Map(); // question id → { label: probability }, for the current video
let pending = new Set(); // question ids being answered

const save = () => {
  store.set('vega1-cat-questions-v1', questions.map(({ q, options }) => ({ q, options })));
  store.set('vega1-cat-custom', custom);
};
const jev = (q) => ({ type: 'choice', instructions: q.q, criteria: Object.fromEntries(q.options.map((o) => [o, null])) });
const request = (extra = []) => ({
  questions: Object.fromEntries([
    ...questions.map((q) => [`q${q.id}`, jev(q)]),
    ...extra.map((q, i) => [`draft${i}`, jev(q)]),
  ]),
});
const sameQuestion = (a, b) => a.q.toLowerCase() === b.q.toLowerCase() && a.options.join('|') === b.options.join('|');

// ---------------------------------------------------------------- questions

function render() {
  $('cqs').replaceChildren(...questions.map((q) => {
    const probs = results.get(q.id);
    const best = probs ? q.options.reduce((a, b) => (probs[b] > probs[a] ? b : a)) : null;
    const state = pending.has(q.id) ? 'pending' : !probs ? 'idle' : probs[best] >= SURE ? 'sure' : 'unsure';
    const li = document.createElement('li');
    li.className = `cq ${state}`;
    li.innerHTML = `<div class="cq-head"><span class="text">${escape(q.q)}</span>` +
      `<span class="tag">${state === 'unsure' ? 'not sure' : escape(best ?? '')}</span>` +
      '<button class="remove" aria-label="Remove question"><svg class="i" viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg></button></div>' +
      `<ul class="opts">${q.options.map((o) => {
        const p = probs?.[o];
        const on = p === undefined ? 0 : Math.round(p * BAR);
        return `<li class="opt${o === best ? ' best' : ''}"><span class="label">${escape(o)}</span>` +
          `<span class="bar" aria-hidden="true"><span class="on">${'█'.repeat(on)}</span><span class="off">${'░'.repeat(BAR - on)}</span></span>` +
          `<span class="pct">${p === undefined ? '–' : pct(p)}</span></li>`;
      }).join('')}</ul>`;
    li.querySelector('.remove').addEventListener('click', () => remove(q.id));
    return li;
  }));
  $('empty').hidden = questions.length > 0;
  $('count').textContent = questions.length ? String(questions.length) : '';
  renderSuggestions();
}

function renderSuggestions() {
  const pool = [...(CATEGORIES.samples[demo?.sample()?.file] ?? []), ...CATEGORIES.defaults, ...CATEGORIES.suggestions];
  const picks = pool.filter((s, i) => pool.findIndex((t) => sameQuestion(s, t)) === i)
    .filter((s) => !questions.some((q) => sameQuestion(q, s))).slice(0, 6);
  $('suggest').replaceChildren(...picks.map((s) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = s.q;
    b.title = s.options.join(' · ');
    b.addEventListener('click', () => add([s]));
    return b;
  }));
}

function add(items, byHand = true) {
  for (const item of items) if (!questions.some((q) => sameQuestion(q, item))) questions.push(withId(item));
  if (byHand) custom = true;
  changed();
}

function remove(id) {
  questions = questions.filter((q) => q.id !== id);
  results.delete(id);
  custom = true;
  changed();
}

function setQuestions(items) {
  questions = [];
  results.clear();
  custom = false;
  add(items, false);
}

function changed() {
  save();
  demo.changed();
}

let demo = null;
demo = startDemo({
  name: 'Vega1-cat',
  noun: 'question',
  heads: { head: null, choiceHead: 'models/head-cat.safetensors' }, // only choice questions here
  texts: 'samples/texts-cat',
  count: () => questions.length,
  request,
  apply(res) {
    for (const q of questions) {
      const a = res.answers[`q${q.id}`];
      if (a) results.set(q.id, a.probabilities);
    }
  },
  onSample(entry) {
    const items = CATEGORIES.samples[entry.file];
    if (!custom && items) setQuestions(items);
  },
  clear: () => results.clear(),
  render(busy) {
    pending = busy ? new Set(questions.filter((q) => !results.has(q.id)).map((q) => q.id)) : new Set();
    render();
  },
});

// ---------------------------------------------------------------- question editing

/** The question being typed, if it has a question and at least two options. */
function draft() {
  const q = $('q-input').value.trim();
  const options = [...new Set($('opt-input').value.split(/,|\n|\s+or\s+/i).map((o) => o.trim()).filter(Boolean))];
  return q && options.length >= 2 ? { q, options } : null;
}

for (const input of [$('q-input'), $('opt-input')]) {
  input.addEventListener('focus', () => demo.wantText());
  // Encode the draft while it's typed, so asking only runs the answer head.
  input.addEventListener('input', () => {
    const d = draft();
    $('add-btn').disabled = !d;
    if (d) demo.prepareSoon([d]);
  });
}
$('q-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !draft()) {
    e.preventDefault();
    $('opt-input').focus();
  }
});
$('add-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const d = draft();
  if (!d) return;
  $('q-input').value = '';
  $('opt-input').value = '';
  $('add-btn').disabled = true;
  add([d]);
});
$('reset').addEventListener('click', () => setQuestions(CATEGORIES.samples[demo.sample()?.file] ?? CATEGORIES.defaults));
