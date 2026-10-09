#!/usr/bin/env bash
# Builds the WASM package into web/pkg and copies the model next to it.
# Then: python3 -m http.server -d web 8000
set -euo pipefail

cd "$(dirname "$0")/.."
cargo build --release --target wasm32-unknown-unknown -p vs1-wasm
wasm-bindgen --target web --no-typescript --out-dir web/pkg \
  target/wasm32-unknown-unknown/release/vs1_wasm.wasm
mkdir -p web/models
cp models/yolox_nano.onnx web/models/
ls -la web/pkg web/models
