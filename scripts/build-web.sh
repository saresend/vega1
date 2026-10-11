#!/usr/bin/env bash
# Builds the WASM package into web/pkg and copies the browser model files into
# web/models. Prerequisite: train/prepare_models.py has written models/vs1.
# Then serve:  python3 -m http.server -d web 8000
set -euo pipefail

cd "$(dirname "$0")/.."
cargo build --release --target wasm32-unknown-unknown -p vs1-wasm
wasm-bindgen --target web --no-typescript --out-dir web/pkg \
  target/wasm32-unknown-unknown/release/vs1_wasm.wasm
if command -v wasm-opt >/dev/null; then
  wasm-opt -O3 --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int \
    web/pkg/vs1_wasm_bg.wasm -o web/pkg/vs1_wasm_bg.wasm
fi

mkdir -p web/models
for f in vision_fp16w.onnx text_fp16w.onnx tokenizer.json LICENSE-mobileclip; do
  cp models/vs1/$f web/models/
done
if [[ -f models/vs1/head.safetensors ]]; then
  # Half-size copy for the page (the head still computes in f32).
  uv run --project train python scripts/half-head.py models/vs1/head.safetensors web/models/head.safetensors
fi
if [[ -f models/vs1/head-cat.safetensors ]]; then
  # vega1-cat: the category head (choice/score questions), from train/finalize_cat.py.
  uv run --project train python scripts/half-head.py models/vs1/head-cat.safetensors web/models/head-cat.safetensors
fi
# Sample videos for the demo (not committed): python3 scripts/fetch-samples.py
[[ -f web/samples/samples.json ]] || echo "tip: run scripts/fetch-samples.py to add the demo's sample videos"
ls -la web/pkg web/models
