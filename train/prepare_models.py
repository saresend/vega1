"""Prepare MobileCLIP-S0 ONNX encoders for tract (native and WASM).

For each encoder:
  1. pin the input shape (batch 1; text length 77),
  2. constant-fold with ONNX Runtime's *basic* optimizer (standard ONNX ops
     only), which removes the dynamic Range/Shape subgraphs tract can't analyse,
  3. write compressed-weight variants that tract expands back to fp32 at load
     (the dequantising ops are constant-folded, so inference runs in fp32):
       *_fp16w: large initializers stored as fp16 + Cast,
       *_int8w: large initializers stored as int8 with a per-row scale (Cast + Mul),
  4. check every variant against the original model.

Usage: uv run python prepare_models.py [--src ../models/mobileclip_s0] [--out ../models/vs1]
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ENCODERS = {
    "vision": ("vision_model.onnx", [1, 3, 256, 256]),
    "text": ("text_model.onnx", [1, 77]),
}
MIN_FP16_ELEMENTS = 1024


def pin_input_shape(model: onnx.ModelProto, shape: list[int]) -> None:
    dims = model.graph.input[0].type.tensor_type.shape.dim
    assert len(dims) == len(shape), (len(dims), shape)
    for dim, value in zip(dims, shape):
        dim.Clear()
        dim.dim_value = value


def constant_fold(src: Path, dst: Path) -> None:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.optimized_model_filepath = str(dst)
    ort.InferenceSession(str(src), opts, providers=["CPUExecutionProvider"])


def fp16_weights(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    casts = []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        arr = numpy_helper.to_array(init)
        if arr.size < MIN_FP16_ELEMENTS:
            continue
        name = init.name
        stored = name + "__fp16"
        init.CopyFrom(numpy_helper.from_array(arr.astype(np.float16), stored))
        casts.append(helper.make_node("Cast", [stored], [name], to=TensorProto.FLOAT, name=name + "__cast"))
    # Casts must precede their consumers; prepend keeps topological order valid.
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    return model


def int8_weights(model: onnx.ModelProto) -> onnx.ModelProto:
    """Symmetric int8 per first-axis row; dequantised by Cast(float) * scale."""
    graph = model.graph
    nodes = []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        arr = numpy_helper.to_array(init)
        if arr.size < MIN_FP16_ELEMENTS or arr.ndim < 2:
            continue
        name = init.name
        rows = arr.reshape(arr.shape[0], -1)
        scale = np.abs(rows).max(axis=1) / 127.0
        scale[scale == 0] = 1.0
        q = np.clip(np.round(rows / scale[:, None]), -127, 127).astype(np.int8).reshape(arr.shape)
        scale = scale.astype(np.float32).reshape((arr.shape[0],) + (1,) * (arr.ndim - 1))
        init.CopyFrom(numpy_helper.from_array(q, name + "__q"))
        graph.initializer.append(numpy_helper.from_array(scale, name + "__scale"))
        nodes.append(helper.make_node("Cast", [name + "__q"], [name + "__f"], to=TensorProto.FLOAT, name=name + "__cast"))
        nodes.append(helper.make_node("Mul", [name + "__f", name + "__scale"], [name], name=name + "__dequant"))
    all_nodes = nodes + list(graph.node)
    del graph.node[:]
    graph.node.extend(all_nodes)
    return model


def sample_input(kind: str, shape: list[int]) -> np.ndarray:
    rng = np.random.default_rng(0)
    if kind == "vision":
        return rng.random(shape, dtype=np.float32)
    ids = np.zeros(shape, dtype=np.int64)
    ids[0, :6] = [49406, 1047, 3941, 530, 320, 49407]
    return ids


def run(path: Path, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, {sess.get_inputs()[0].name: x})[0]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("../models/mobileclip_s0"))
    ap.add_argument("--out", type=Path, default=Path("../models/vs1"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for kind, (filename, shape) in ENCODERS.items():
        original = args.src / "onnx" / filename
        model = onnx.load(original)
        pin_input_shape(model, shape)
        with tempfile.TemporaryDirectory() as tmp:
            pinned = Path(tmp) / "pinned.onnx"
            onnx.save(model, pinned)
            fp32_path = args.out / f"{kind}.onnx"
            constant_fold(pinned, fp32_path)

        fp16_path = args.out / f"{kind}_fp16w.onnx"
        onnx.save(fp16_weights(onnx.load(fp32_path)), fp16_path)
        int8_path = args.out / f"{kind}_int8w.onnx"
        onnx.save(int8_weights(onnx.load(fp32_path)), int8_path)

        x = sample_input(kind, shape)
        ref = run(original, x)
        ops = sorted({n.op_type for n in onnx.load(fp32_path).graph.node})
        for path in (fp32_path, fp16_path, int8_path):
            print(f"{path.name:22s} {path.stat().st_size / 1e6:6.1f} MB  cos vs original {cosine(ref, run(path, x)):.6f}")
        print(f"  ops: {', '.join(ops)}")

    shutil.copy(args.src / "tokenizer.json", args.out / "tokenizer.json")
    shutil.copy(args.src / "LICENSE", args.out / "LICENSE-mobileclip")


if __name__ == "__main__":
    main()
