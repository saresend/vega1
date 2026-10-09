"""Train the answer head on examples from build_dataset.py (or any teacher that
writes the same format), compare against zero-shot, export for Rust/WASM.

Reports, per validation slice (question kind × seen/unseen class × seen/
held-out phrasing): accuracy, negative log-likelihood and — for `noul` —
expected calibration error, before (zero-shot) and after training.

Usage: uv run python train_head.py [--epochs 40] [--out ../models/vs1/head.safetensors]
"""

import argparse
import json
import random
import subprocess
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import torch

from head import AnswerHead

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
VS1 = ROOT.parent / "target" / "release" / "vs1"


def load():
    examples = [json.loads(l) for l in open(DATA / "examples.jsonl")]
    index = [json.loads(l) for l in open(DATA / "text" / "index.jsonl")]
    assert len(examples) == len(index), "examples and question index are out of sync; rebuild the dataset"
    texts = torch.from_numpy(np.load(DATA / "text" / "texts.npy"))
    videos = {p: torch.from_numpy(np.load(ROOT / p)) for p in {e["video"] for e in examples}}
    for e, ix in zip(examples, index):
        e["q_row"], e["opt_rows"] = ix["question"], ix["options"]
    return examples, texts, videos


def batch(examples, texts, videos):
    k = max(len(e["opt_rows"]) for e in examples)
    b = len(examples)
    frames = torch.stack([videos[e["video"]] for e in examples])
    question = texts[[e["q_row"] for e in examples]]
    options = torch.zeros(b, k, texts.shape[1])
    mask = torch.zeros(b, k, dtype=torch.bool)
    target = torch.zeros(b, k)
    for i, e in enumerate(examples):
        n = len(e["opt_rows"])
        options[i, :n] = texts[e["opt_rows"]]
        mask[i, :n] = True
        target[i, :n] = torch.tensor(e["target"])
    return frames, question, options, mask, target


def soft_ce(logits, target, mask):
    logp = torch.log_softmax(logits, dim=-1).masked_fill(~mask, 0.0)
    return -(target * logp).sum(-1)


def slice_name(e) -> str:
    t = e["tags"]
    cls = "unseen-class" if t["unseen_class"] else "seen-class"
    phrase = "heldout-phrase" if t["heldout_phrase"] else "train-phrase"
    return f"{t['kind']:15s} {cls:12s} {phrase}" if t["kind"] != "caption_noul" else f"{t['kind']:15s} {'':12s} {phrase}"


@torch.no_grad()
def evaluate(model, examples, texts, videos):
    model.eval()
    groups = defaultdict(lambda: {"n": 0, "correct": 0, "nll": 0.0, "noul": []})
    for start in range(0, len(examples), 512):
        chunk = examples[start:start + 512]
        frames, question, options, mask, target = batch(chunk, texts, videos)
        logits = model(frames, question, options, mask)
        probs = logits.softmax(-1)
        nll = soft_ce(logits, target, mask)
        for i, e in enumerate(chunk):
            for name in (slice_name(e), "ALL"):
                g = groups[name]
                g["n"] += 1
                g["correct"] += int(probs[i].argmax() == target[i].argmax())
                g["nll"] += float(nll[i])
                if e["question"]["type"] == "noul":
                    g["noul"].append((float(probs[i, 0]), float(target[i, 0])))
    model.train()
    out = {}
    for name, g in groups.items():
        ece = None
        if g["noul"]:
            p = np.array(g["noul"])
            bins = np.clip((p[:, 0] * 10).astype(int), 0, 9)
            ece = sum(abs(p[bins == b, 0].mean() - p[bins == b, 1].mean()) * (bins == b).sum()
                      for b in range(10) if (bins == b).any()) / len(p)
        out[name] = {"n": g["n"], "acc": g["correct"] / g["n"], "nll": g["nll"] / g["n"], "ece": ece}
    return out


def report(before, after):
    print(f"\n{'slice':45s} {'n':>5s}  {'acc zero-shot→trained':>22s}  {'NLL':>15s}  {'ECE (noul)':>15s}")
    for name in sorted(after, key=lambda n: (n == "ALL", n)):
        b, a = before[name], after[name]
        ece = f"{b['ece']:.3f}→{a['ece']:.3f}" if a["ece"] is not None else ""
        print(f"{name:45s} {a['n']:5d}  {b['acc']:9.1%} → {a['acc']:8.1%}  {b['nll']:6.3f}→{a['nll']:6.3f}  {ece:>15s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--label-smoothing", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT.parent / "models" / "vs1" / "head.safetensors")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    examples, texts, videos = load()
    train = [e for e in examples if e["split"] == "train"]
    val = [e for e in examples if e["split"] == "val"]
    print(f"{len(train)} train / {len(val)} val examples, {len(videos)} videos, {texts.shape[0]} unique texts")

    model = AnswerHead()
    before = evaluate(model, val, texts, videos)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best, best_state = float("inf"), None
    for epoch in range(args.epochs):
        random.shuffle(train)
        total = 0.0
        for start in range(0, len(train), args.batch):
            frames, question, options, mask, target = batch(train[start:start + args.batch], texts, videos)
            k = mask.sum(-1, keepdim=True)
            smoothed = (target * (1 - args.label_smoothing) + args.label_smoothing / k) * mask
            loss = soft_ce(model(frames, question, options, mask), smoothed, mask).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                model.pool_beta.clamp_(min=1.0)
            total += float(loss) * len(frames)
        sched.step()
        metrics = evaluate(model, val, texts, videos)["ALL"]
        if metrics["nll"] < best:
            best, best_state = metrics["nll"], {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  train loss {total / len(train):.4f}  val nll {metrics['nll']:.4f}  val acc {metrics['acc']:.1%}")

    model.load_state_dict(best_state)
    after = evaluate(model, val, texts, videos)
    report(before, after)
    print(f"\nlearned: a_zs={model.a_zs.item():.2f} gamma={model.gamma.item():.3f} pool_beta={model.pool_beta.item():.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    name = f"vs1-head-ucf101-{date.today().isoformat()}"
    model.export(args.out, name)
    (args.out.with_suffix(".metrics.json")).write_text(json.dumps({"zero_shot": before, "trained": after}, indent=2))
    print(f"exported {name} → {args.out}")

    # Parity: the Rust head must reproduce these probabilities.
    cases = []
    with torch.no_grad():
        model.eval()
        for e in random.sample(val, 8):
            frames, question, options, mask, _ = batch([e], texts, videos)
            probs = model(frames, question, options, mask).softmax(-1)[0]
            cases.append({"frames": frames[0].tolist(), "question": question[0].tolist(),
                          "options": options[0].tolist(), "probabilities": probs.tolist()})
    parity = args.out.with_suffix(".parity.json")
    parity.write_text(json.dumps(cases))
    subprocess.run([str(VS1), "check-head", str(args.out), str(parity)], check=True)


if __name__ == "__main__":
    main()
