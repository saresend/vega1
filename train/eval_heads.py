"""Per-slice test accuracy / NLL / ECE of several exported heads on the current
dataset, side by side (fair comparison: same examples for every head).

Usage: uv run python eval_heads.py data/runs/v4/head.safetensors data/runs/v5/head.safetensors ...
"""

import json
import sys
from pathlib import Path

import train_head as t
from eval_by_size import load_head


def main() -> None:
    crops = all(t.crops_path(v).exists() for v in {e["video"] for e in map(json.loads, open(t.DATA / "examples.jsonl"))})
    examples, texts, videos = t.load(crops)
    test = [e for e in examples if e["split"] == "test"]
    names = [Path(p).parent.name for p in sys.argv[1:]]
    results = [t.evaluate(load_head(p), test, texts, videos) for p in sys.argv[1:]]
    slices = sorted(results[0], key=lambda n: (n.startswith("ALL"), n))
    print(f"{'slice':38s} {'n':>5s}  " + "  ".join(f"{n:>13s}" for n in names) + "   (acc / ECE)")
    for s in slices:
        if s.startswith("activity_choice"):
            continue
        cells = [f"{r[s]['acc']:6.1%} {r[s]['ece'] or 0:.3f}" for r in results]
        print(f"{s:38s} {results[0][s]['n']:5d}  " + "  ".join(f"{c:>13s}" for c in cells))


if __name__ == "__main__":
    main()
