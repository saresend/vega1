"""Exports the vega1-cat head: a trained category head with its learned
correction (gamma) scaled down, which trades a little fit on trained labels
for much better answers about labels never trained on (picked on the
selection split: unseen subjects 84% → 88%, all questions unchanged), then
a temperature fitted on the selection split (scaling every logit term by the
same factor) to restore calibration.

Usage: uv run python finalize_cat.py data/runs/cat1/head.safetensors ../models/vs1/head-cat.safetensors [--gamma 0.5]
"""

import argparse
import json
from datetime import date
from pathlib import Path

import torch

import train_head as t
from eval_by_size import load_head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("head", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--gamma", type=float, default=0.5, help="factor on the learned correction")
    ap.add_argument("--data", type=Path, default=t.DATA / "cat")
    args = ap.parse_args()
    t.DATA = args.data
    model = load_head(str(args.head))
    with torch.no_grad():
        model.gamma.mul_(args.gamma)
    examples, texts, videos = t.load(model.crops)
    test = [e for e in examples if e["split"] == "test"]
    selection = [e for e in examples if e["split"] == "selection"]
    base = {k: v.clone() for k, v in model.state_dict().items()}

    def with_temperature(scale: float) -> None:
        model.load_state_dict(base)
        with torch.no_grad():
            for p in (model.a_zs, model.gamma, *([model.a_crop] if model.crops else [])):
                p.mul_(scale)

    scores = {}
    for scale in [0.8 + 0.05 * i for i in range(17)]:
        with_temperature(scale)
        scores[scale] = t.evaluate(model, selection, texts, videos)["ALL"]["nll"]
    best = min(scores, key=scores.get)
    print(f"logit scale {best:.2f} (selection NLL {scores[1.0]:.4f} → {scores[best]:.4f})")
    with_temperature(best)
    zero = load_head(str(args.head))
    with torch.no_grad():
        zero.gamma.zero_()
        zero.a_zs.fill_(50.0)
        zero.a_crop.zero_()
        zero.pool_beta.fill_(10.0)
    t.report(t.evaluate(zero, test, texts, videos), metrics := t.evaluate(model, test, texts, videos))
    model.export(args.out, f"vega1-cat-{date.today().isoformat()}")
    args.out.with_suffix(".metrics.json").write_text(json.dumps({"source": str(args.head), "gamma": args.gamma, "logit_scale": best,
                                                                 "trained": metrics}, indent=2, default=float))
    print(f"exported → {args.out}")


if __name__ == "__main__":
    main()
