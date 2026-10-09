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
frames ──image encoder──► frame embeddings ───────┐
                                                  ├──► answer head ──► option probabilities ──► typed answers
question ──prompts──text encoder──► option embeddings ┘
```

- **Every question type is "pick among described options."** `noul` has two options (the `true` and `false`
  descriptions, defaulting to the statement vs. "a video"). `choice` has one option per criteria entry. `score` has
  one per level, and `score` is the probability-weighted level number.
- **Encoders:** MobileCLIP-S0 (image 11M params, text 42M params), run with [tract](https://github.com/sonos/tract).
  The same Rust code runs natively and in WASM.
- **Answer head:**
  - *Zero-shot*: option logit = soft-max-over-frames similarity (log-mean-exp) × scale.
  - *Trained*: a small question-conditioned attention head (about 1M params, plain Rust) that adds a learned
    correction to the zero-shot logits. With `gamma = 0` it is exactly zero-shot, which is where training starts.
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
python3 -m http.server -d web 8000     # open http://localhost:8000 (add ?head=models/head.safetensors for the trained head)
```

Training (CPU is fine):

```sh
cd train
uv run python build_dataset.py --ucf ~/Development/Projects/vega/data/ucf101/UCF-101
uv run python train_head.py            # prints zero-shot vs trained metrics, exports + parity-checks the head
```

`build_dataset.py` writes examples in a generic format:
`{video, question: <Jev question>, target: [p per option], split, tags}`. Labels currently come from UCF-101
activity classes and synthetically burned-in captions. A video-language teacher model that writes the same format
can label any question, including counting.

## Status and known limits

- **Counting is weak.** CLIP-style frame embeddings carry little count information, and there are no count labels
  yet. Next step: teacher labels for counting, and per-patch tokens instead of one vector per frame.
- **Zero-shot calibration is rough.** The trained head fixes this for question types it has seen; check the
  per-slice ECE in `train_head.py`'s report.
- **Browser speed:** about 0.8 s per frame in Chrome (single-threaded WASM SIMD), so roughly 13 s for 16 frames.
  The model download is about 108 MB, cached after the first visit. The text encoder is most of that; questions
  known in advance could be pre-embedded on a server and the text encoder dropped from the page.
- MobileCLIP weights are under Apple's sample-code license (`models/vs1/LICENSE-mobileclip`).
