# video-system1

A general **System One model for video** that runs entirely in the browser.

It uses the same contract as TypeSafe's [Jev](https://docs.typesafe.ai/api): a state plus a map of typed questions in,
calibrated typed answers out, in one pass with no text generation. The difference is that the state is a video,
and inference runs locally in Rust/WASM. The video never leaves the page.

```js
import { SystemOne } from './vs1.js';

const s1 = await SystemOne.load();
const res = await s1.evaluate({
  state: videoFile,
  questions: {
    subjects:  { type: 'choice', instructions: 'How many people appear throughout the video?',
                 criteria: { '0': null, '1': null, '2': null, '3-5': null, '6 or more': null } },
    captioned: { type: 'noul', instructions: 'The video is captioned' },
    energy:    { type: 'score', instructions: 'How much movement is there?',
                 criteria: ['still', 'calm movement', 'fast and energetic'] },
  },
});
res.answers.captioned  // { type: 'noul', noul: 0.85 }
res.answers.subjects   // { type: 'choice', choice: '2', probabilities: {...}, confidence: 0.41 }
res.answers.energy     // { type: 'score', score: 1.2, legend: {...}, probabilities: {...}, confidence: 0.3 }
```

Questions are plain data. Adding one takes no new code.

## How it works

```
frames ──image encoder──► rows: 16 full views + 6 crops × 4 frames ─┐
                                                                      ├──► answer head ──► option probabilities ──► typed answers
question ──prompts──text encoder──► option embeddings ───────────────┘
```

- **Every question type is "pick among described options."** `noul` has two options (the `true` and `false`
  descriptions, defaulting to the statement vs. "a video"). A negated statement ("there is no dog", "nobody is
  swimming") is answered as the complement of its positive form, because CLIP-style text encoders largely ignore
  "no"/"not" and would otherwise give a statement and its negation the same answer. `choice` has one option per criteria entry. `score` has
  one per level, and `score` is the probability-weighted level number.
- **Encoders:** MobileCLIP-S0 (image 11M params, text 42M params), run with [tract](https://github.com/sonos/tract).
  The same Rust code runs natively and in WASM.
- **Video state:** one embedding per sampled frame (the full view, centre-cropped to a square) plus six square crops
  that tile the whole frame for 4 of the frames, so small objects and the edges of wide frames are seen at up to twice
  the scale.
- **Answer head:**
  - *Zero-shot*: option logit = soft-max-over-frames similarity (log-mean-exp) × scale, on the full views.
  - *Trained*: a small question-conditioned attention head (about 1M params, plain Rust) over all rows, adding a
    learned correction and two crop terms (soft max and mean over crops, which reward evidence localised in some
    crops) to the zero-shot logits. Initialised to exactly zero-shot, which is where training starts.
- **Browser:** up to 4 workers embed frames, and with 8 GB or more of memory also question text, in parallel;
  statements are encoded while they are typed, so asking about an embedded video only runs the answer head.
- **`score` and `confidence`** follow Jev's documented formulas
  ([confidence](https://docs.typesafe.ai/confidence), [score](https://docs.typesafe.ai/primitives/score)).
  The unit tests check them against the docs' worked examples.

## Layout

| Path | What |
|---|---|
| `crates/vs1-core` | API types and validation, prompts, encoders, heads, `SystemOne` |
| `crates/vs1-cli` | `vs1 evaluate`, `embed-video`, `embed-questions`, `check-head` |
| `crates/vs1-wasm`, `web/` | Browser build: `vs1.js` (public API), worker, demo page |
| `train/` | Model preparation, dataset building, head training (Python, uv) |
| `examples/questions.json` | Example request |

## Running it

```sh
# 1. Models: fetch MobileCLIP-S0 ONNX (Xenova/mobileclip_s0) into models/mobileclip_s0, then prepare them
cd train && uv run python prepare_models.py && cd ..

# 2. Native
cargo build --release -p vs1-cli
./target/release/vs1 evaluate some.mp4 --questions examples/questions.json [--head models/vs1/head.safetensors]

# 3. Browser
./scripts/build-web.sh
python3 scripts/fetch-samples.py       # optional: sample clips with precomputed embeddings (web/samples)
python3 scripts/serve.py               # open http://localhost:8765 (or this machine's LAN address on a phone)
```

The demo page takes a list of yes/no statements ("There is a dog", "The scene is outdoors") and answers each with the
probability that it is true. Dropping in a video embeds its frames in the browser (4–6 s for 16 frames); the sample
clips ship with their embeddings (`vs1 embed-state`), so picking one answers immediately. Once a video is embedded,
adding or removing a statement re-answers the whole list live. The page uses the trained head in `web/models/` when
present (`?head=` forces zero-shot, `?head=path` loads another). `SystemOne.evaluate` in `web/vs1.js` takes any Jev
request, including `choice` and `score`.

Sample clips come from OpenCV's test video, UCF101's test split (research use, kept local) and Wikimedia Commons
(credited per clip in `web/samples/samples.json`); `scripts/fetch-samples.py` lists them.

Training (CPU is fine):

```sh
cd train
uv run python build_dataset.py --ucf ~/Development/Projects/vega/data/ucf101/UCF-101 \
                               --coco ~/Development/Projects/vega/data/coco   # val2017 + annotations; train2017 images are fetched
uv run python train_head.py            # prints zero-shot vs trained metrics, exports + parity-checks the head
```

`build_dataset.py` writes examples in a generic format:
`{video, question: <Jev question>, target: [p per option], split, tags}`. Labels come for free from data we control:

| Source | `noul` statements |
|---|---|
| UCF-101 clips | activity ("someone is doing X"), hand-labelled class attributes (water, ball, animal, outdoors, sport, ...) |
| COCO images as short pan/zoom clips | object present/absent ("there is a dog"), negations ("there is no dog"), people present / exactly one / a group, captions true/false, supercategories ("there is an animal"), and scene attributes decided by caption consensus (night, snow, beach, street, kitchen, indoors, sitting, riding, eating, ...) |
| Synthetic edits of any clip | burned-in subtitles, watermarks/logos, grayscale; each edit is a negative for the others |

Test and model-selection sets hold out videos (UCF groups, COCO images), whole statement subjects (20 UCF classes,
16 COCO categories, some attributes) and phrasings, so the report measures statements the head has never seen.
`statements.py` holds the phrasings and label tables. A video-language teacher model that writes the same format can
label any question, including counting.

## vega1-cat: category questions

A second head with the same architecture, trained only on `choice`
questions ("which animal is in the video?", "how many people?", "where was
this filmed?"). Load it next to the noul head (`--choice-head` in the CLI,
`loadChoiceHead` in WASM, `choiceHead` in `vs1.js`): `noul` questions go to
vega1, `choice`/`score` questions to vega1-cat, in the same request.

```
cd train
uv run python build_dataset.py --ucf ... --coco ... --task cat --out data/cat   # reuses data/'s videos
uv run python train_head.py --data data/cat --name vega1-cat --out data/runs/cat1/head.safetensors
uv run python finalize_cat.py data/runs/cat1/head.safetensors ../models/vs1/head-cat.safetensors
```

`finalize_cat.py` halves the learned correction (better on labels never
trained on) and fits a temperature, both on the selection split.

| test (6,075 questions, 13 kinds) | zero-shot | vega1-cat |
|---|---|---|
| accuracy | 81.5% | 93.8% |
| calibration error (top choice) | 0.107 | 0.027 |
| unseen subjects (681) | 91.3% | 90.5% |
| held-out phrasings (3,196) | 80.9% | 92.7% |
| `time_of_day` (kind never trained on) | 89.9% | 89.3% |

There is no "none of these" option: when no option fits, the probabilities
spread out and the confidence is low.

## Status and known limits

Current head (`models/vs1/head.safetensors`), on the held-out test set (`train_head.py` report):

| `noul` slice | zero-shot | trained |
|---|---|---|
| all (12.4k statements) | 74.5%, ECE 0.08 | **91.8%, ECE 0.023** |
| statements about subjects never trained on (objects, stuff, attributes, activities) | 84.6% | **86.1%** |
| held-out phrasings | 74.6% | **91.4%** |
| objects / negated objects | 75% / 76–83% | **88% / 88–90%** |
| objects covering < 1% / 1–5% of the image (`eval_by_size.py`) | — | **72% / 79%** (66% / 74% without crops) |
| people present / exactly one / a group | 42–57% / 58–62% / 65–73% | **95–96% / 89% / 97–98%** |
| scene (night, snow, street, indoors, ...) / background stuff (grass, trees, sky, ...) | 78% / 67–68% | **94–96% / 88–90%** |
| captioned / watermark / black and white / computer graphics | 64–69% / 47–64% / 78–84% / 66–89% | **95–97% / 90–96% / 100% / 49–100%** |

`eval_heads.py` compares heads side by side on the same test set; crops raised overall accuracy by a point and
watermark/caption/small-object accuracy by 5–13 points, at a cost of 1–2 points on statements about unseen subjects.

- **Very small objects are still missed.** Crops help (above), but a dog a few dozen pixels tall at a person's feet in
  320×240 footage stays at "no" ("someone is walking a dog" is answered correctly).
- **Counting is weak** beyond "one person" vs "a group". There are no finer count labels yet; a video-language
  teacher writing the example format can add them.
- **On-screen text leaks into answers:** CLIP reads it, so subtitles like "The final round" nudge "someone is playing a sport".
- **Co-occurrence guesses:** a snowy street with trees can get "there is grass". The head learns what usually appears
  together; unusual scenes expose that.
- **Latency** (Chrome, idle i5-12400F, 4 workers): first answer for a dropped-in video 4–6 s at 16 frames (40 image
  encodes in parallel); picking a sample with preloaded embeddings about 85 ms. After that, per question asked: about
  16 ms of compute and 70 ms until it is on screen when typed (the text is encoded while typing), about 260 ms when a
  new statement is submitted instantly, and about 25 ms to re-answer 20+ known statements. Pasting 10 new statements at
  once takes about 420 ms. Budget: 500 ms per ask on a typical laptop. The model download is about 108 MB, cached after
  the first visit; the text encoder is most of that.
- MobileCLIP weights are under Apple's sample-code license (`models/vs1/LICENSE-mobileclip`).
