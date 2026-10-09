"""Build a labelled training set of Jev-style questions over UCF-101 clips.

Labels come for free from data we control:
  * activity  — UCF-101 class labels → `choice` ("which activity?") and
                `noul` ("someone is doing X") questions,
  * captions  — half the clips get randomly styled subtitles burned in →
                `noul` ("the video is captioned") questions.

Every example is written in the generic format a VLM teacher would also emit:
    {"video": <embedding .npy>, "question": <Jev question>, "target": [p_option...],
     "split": "train"|"val", "tags": {...}}
so swapping in teacher labels for arbitrary questions (counting, etc.) needs
no change to training.

Generalisation is measured, not assumed. Validation holds out:
  * people/scenes:   UCF groups g01–g07 (the official split-1 test groups),
  * activity classes: HELDOUT_CLASSES are never seen in training,
  * phrasings:        HELDOUT_* templates are never seen in training.

Frame embeddings are computed by the Rust CLI (`vs1 embed-video`) — the same
code path the browser runs — so train and serve features match exactly.

Usage: uv run python build_dataset.py --ucf ~/Development/Projects/vega/data/ucf101/UCF-101
"""

import argparse
import json
import random
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VS1 = ROOT.parent / "target" / "release" / "vs1"
DATA = ROOT / "data"

VAL_GROUPS = {f"g{i:02d}" for i in range(1, 8)}
NUM_HELDOUT_CLASSES = 20

ACTIVITY_CHOICE = ["What activity is shown in the video?", "Which action is being performed?",
                   "What is happening in this video?", "What sport or activity does this video show?"]
HELDOUT_ACTIVITY_CHOICE = ["Identify the activity in this clip."]
ACTIVITY_NOUL = ["The video shows someone doing {a}", "Someone in the video is doing {a}", "This is a video of {a}"]
HELDOUT_ACTIVITY_NOUL = ["The clip contains {a}"]
CAPTION_NOUL = ["The video is captioned", "There are subtitles in the video", "Text captions are shown on screen",
                "The video has subtitles"]
HELDOUT_CAPTION_NOUL = ["The clip has burned-in subtitles"]

SUBTITLE_LINES = [
    "Welcome back to the channel", "Let's see how this goes", "This is the best part", "Watch closely now",
    "I can't believe it", "Here we go again", "That was amazing", "Try this at home", "Almost there",
    "Keep going, you've got this", "What do you think?", "Look at that form", "And that's how it's done",
    "Don't forget to subscribe", "One more time", "Perfect timing", "Nobody expected that", "Slow and steady",
    "This took a lot of practice", "The final round", "Thanks for watching", "Here's a quick tip",
]


def humanize(cls: str) -> str:
    """'ApplyEyeMakeup' → 'apply eye makeup'."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", cls).lower()


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def burn_captions(src: Path, dst: Path, rng: random.Random) -> None:
    """Burns randomly styled, regularly changing subtitles into `src`."""
    total = duration(src)
    lines, t = [], 0.0
    while t < total:
        span = rng.uniform(1.2, 3.0)
        lines.append((t, min(total, t + span), rng.choice(SUBTITLE_LINES)))
        t += span
    style = ",".join([
        f"FontSize={rng.randint(14, 24)}",
        f"PrimaryColour={rng.choice(['&H00FFFFFF', '&H0000FFFF', '&H00F0F0F0'])}",
        f"Outline={rng.randint(1, 3)}",
        f"BorderStyle={rng.choice([1, 1, 3])}",
        f"Alignment={rng.choice([2, 2, 2, 8])}",
        f"MarginV={rng.randint(6, 24)}",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".srt", delete=False) as f:
        for i, (a, b, text) in enumerate(lines, 1):
            f.write(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n\n")
        srt = f.name
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", f"subtitles={srt}:force_style='{style}'",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-an", str(dst)],
                       check=True)
    finally:
        Path(srt).unlink()


def embed_videos(videos: list[Path], out_dir: Path, workers: int, frames: int) -> None:
    todo = [v for v in videos if not (out_dir / f"{v.stem}.npy").exists()]
    chunks = [todo[i::workers] for i in range(workers) if todo[i::workers]]
    print(f"embedding {len(todo)} videos with {len(chunks)} processes")

    def run(chunk: list[Path]) -> None:
        subprocess.run([str(VS1), "embed-video", *map(str, chunk), "--out-dir", str(out_dir), "--frames", str(frames)],
                       check=True, stderr=subprocess.DEVNULL)

    with ThreadPoolExecutor(len(chunks)) as pool:
        list(pool.map(run, chunks))


def choice_question(template: str, options: list[str]) -> dict:
    return {"type": "choice", "instructions": template, "criteria": {humanize(c): None for c in options}}


def noul_question(text: str) -> dict:
    return {"type": "noul", "instructions": text}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucf", type=Path, required=True, help="UCF-101 directory with one folder per class")
    ap.add_argument("--train-per-class", type=int, default=6)
    ap.add_argument("--val-per-class", type=int, default=3)
    ap.add_argument("--caption-fraction", type=float, default=0.5)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    classes = sorted(p.name for p in args.ucf.iterdir() if p.is_dir())
    heldout = set(rng.sample(classes, NUM_HELDOUT_CLASSES))
    seen = [c for c in classes if c not in heldout]

    clips = []  # (path, class, split)
    for cls in classes:
        files = sorted((args.ucf / cls).glob("*.avi"))
        val = [f for f in files if f.stem.split("_")[2] in VAL_GROUPS]
        train = [f for f in files if f.stem.split("_")[2] not in VAL_GROUPS]
        clips += [(f, cls, "val") for f in rng.sample(val, min(args.val_per_class, len(val)))]
        if cls not in heldout:
            clips += [(f, cls, "train") for f in rng.sample(train, min(args.train_per_class, len(train)))]

    captioned_dir = DATA / "captioned"
    captioned_dir.mkdir(parents=True, exist_ok=True)
    to_caption = [(f, cls, split) for f, cls, split in clips if rng.random() < args.caption_fraction]
    jobs = []
    for f, _, _ in to_caption:
        dst = captioned_dir / f"{f.stem}_cap.mp4"
        if not dst.exists():
            jobs.append((f, dst, random.Random(f"{args.seed}:{f.stem}")))
    print(f"{len(clips)} clips ({sum(s == 'train' for *_, s in clips)} train), burning captions into {len(jobs)}")
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda j: burn_captions(*j), jobs))

    videos = [(f, cls, split, False) for f, cls, split in clips]
    videos += [(captioned_dir / f"{f.stem}_cap.mp4", cls, split, True) for f, cls, split in to_caption]
    emb_dir = DATA / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    embed_videos([v for v, *_ in videos], emb_dir, args.workers, args.frames)

    examples = []
    for path, cls, split, captioned in videos:
        npy = emb_dir / f"{path.stem}.npy"
        if not npy.exists():
            continue
        is_val = split == "val"
        phrase_sets = [(False, ACTIVITY_CHOICE, ACTIVITY_NOUL, CAPTION_NOUL)]
        if is_val:
            phrase_sets.append((True, HELDOUT_ACTIVITY_CHOICE, HELDOUT_ACTIVITY_NOUL, HELDOUT_CAPTION_NOUL))
        pool = classes if is_val else seen
        base_tags = {"class": cls, "captioned": captioned, "unseen_class": cls in heldout}

        def add(question: dict, target: list[float], kind: str, heldout_phrase: bool) -> None:
            examples.append({"video": str(npy.relative_to(ROOT)), "split": split, "question": question,
                             "target": target, "tags": {**base_tags, "kind": kind, "heldout_phrase": heldout_phrase}})

        for heldout_phrase, choice_t, noul_t, caption_t in phrase_sets:
            for _ in range(2):
                k = rng.randint(4, 8)
                options = [cls] + rng.sample([c for c in pool if c != cls], k - 1)
                rng.shuffle(options)
                add(choice_question(rng.choice(choice_t), options), [float(o == cls) for o in options],
                    "activity_choice", heldout_phrase)
            other = rng.choice([c for c in pool if c != cls])
            for target_cls, truth in ((cls, 1.0), (other, 0.0)):
                text = rng.choice(noul_t).format(a=humanize(target_cls))
                add(noul_question(text), [truth, 1.0 - truth], "activity_noul", heldout_phrase)
            add(noul_question(rng.choice(caption_t)), [float(captioned), 1.0 - float(captioned)],
                "caption_noul", heldout_phrase)

    with open(DATA / "examples.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    with open(DATA / "questions.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex["question"]) + "\n")
    subprocess.run([str(VS1), "embed-questions", str(DATA / "questions.jsonl"), "--out-dir", str(DATA / "text")],
                   check=True)
    (DATA / "split.json").write_text(json.dumps({"heldout_classes": sorted(heldout), "val_groups": sorted(VAL_GROUPS)},
                                                indent=2))
    print(f"wrote {len(examples)} examples "
          f"({sum(e['split'] == 'train' for e in examples)} train / {sum(e['split'] == 'val' for e in examples)} val)")


if __name__ == "__main__":
    main()
