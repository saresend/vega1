"""Accuracy of object statements on the test split, by object size: does a
head see small objects? Compares heads given as safetensors paths.

Size = the largest instance of the statement's category in the image, as a
fraction of the image (absent objects are their own bucket).

Usage: uv run python eval_by_size.py ../models/vs1/head.safetensors data/runs/v5/head.safetensors
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

import statements as st
import train_head as t
from head import AnswerHead

COCO = Path.home() / "Development/Projects/vega/data/coco/annotations/instances_val2017.json"
BUCKETS = ["absent", "< 1%", "1-5%", "> 5%"]


def bucket(frac: float) -> str:
    return "absent" if frac == 0 else "< 1%" if frac < 0.01 else "1-5%" if frac < 0.05 else "> 5%"


def load_head(path: str) -> AnswerHead:
    tensors = load_file(path)
    names = {"wq": "wq.weight", "bq": "wq.bias", "wk": "wk.weight", "bk": "wk.bias", "wv": "wv.weight", "bv": "wv.bias",
             "ln_g": "ln.weight", "ln_b": "ln.bias", "w1": "w1.weight", "b1": "w1.bias", "w2": "w2.weight",
             "b2": "w2.bias", "wz": "wz.weight", "bz": "wz.bias"}
    head = AnswerHead(hidden=tensors["bq"].shape[0], mlp=tensors["b1"].shape[0], crops="a_crop" in tensors,
                      crop_attention="type_crop" in tensors)
    state = {names.get(k, k): v for k, v in tensors.items() if k not in ("ws", "bs", "wsq", "bsq")}
    head.load_state_dict(state, strict=False)
    return head.eval()


def main() -> None:
    inst = json.loads(COCO.read_text())
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    sizes = {img["id"]: img["width"] * img["height"] for img in inst["images"]}
    largest = defaultdict(float)  # (image, category) -> largest area fraction
    for a in inst["annotations"]:
        key = (a["image_id"], cats[a["category_id"]])
        largest[key] = max(largest[key], a["area"] / sizes[a["image_id"]])
    nouns = sorted(((st.COCO_NAMES.get(c, c), c) for c in cats.values()), key=lambda x: -len(x[0]))

    crops = all(t.crops_path(v).exists() for v in {e["video"] for e in map(json.loads, open(t.DATA / "examples.jsonl"))})
    examples, texts, videos = t.load(crops)
    test = [e for e in examples if e["split"] == "test" and e["tags"]["kind"] in ("object", "object_negated")
            and e["tags"]["source"] == "coco"]
    for e in test:  # the category is the longest COCO noun in the statement
        text = e["question"]["instructions"].lower()
        cat = next((c for n, c in nouns if n in text), None)
        e["bucket"] = bucket(largest.get((e["tags"]["image"], cat), 0.0))

    heads = {Path(p).parent.name or p: load_head(p) for p in sys.argv[1:]}
    results = {name: defaultdict(lambda: [0, 0]) for name in heads}
    with torch.no_grad():
        for start in range(0, len(test), 512):
            chunk = test[start:start + 512]
            frames, question, options, mask, target = t.batch(chunk, texts, videos)
            for name, head in heads.items():
                probs = head(frames, question, options, mask, n_full=t.FRAMES).softmax(-1)
                for e, p, y in zip(chunk, probs[:, 0].tolist(), target[:, 0].tolist()):
                    r = results[name][e["bucket"]]
                    r[0] += 1
                    r[1] += (p > 0.5) == (y > 0.5)
    print(f"{'object size':12s} {'n':>5s}  " + "  ".join(f"{n:>10s}" for n in heads))
    for b in BUCKETS:
        n = next(iter(results.values()))[b][0]
        print(f"{b:12s} {n:5d}  " + "  ".join(f"{results[h][b][1] / max(n, 1):10.1%}" for h in heads))


if __name__ == "__main__":
    main()
