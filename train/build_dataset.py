"""Build a labelled training set of Jev-style questions over video.

Labels come for free from data we control:
  * UCF-101 clips — class labels → `choice` ("which activity?") and `noul`
    ("someone is doing X") questions; hand-labelled class attributes → `noul`
    ("there is water", "the scene is outdoors", ...).
  * COCO images, turned into short pan/zoom clips — instance annotations →
    object presence/absence, negated statements and people counts; captions →
    "the video shows <caption>" true/false.
  * Synthetic edits on any clip — burned-in subtitles, watermarks/logos and
    grayscale → `noul` questions about each, every edit a negative for the others.

Every example is written in the generic format a VLM teacher would also emit:
    {"video": <embedding .npy>, "question": <Jev question>, "target": [p_option...],
     "split": "train"|"selection"|"test", "tags": {...}}
so swapping in teacher labels for arbitrary questions needs no change to training.

Generalisation is measured, not assumed. Selection (model picking) and test
hold out:
  * people/scenes: UCF groups g01–g07 (official split-1 test groups) and 30% of COCO images,
  * statements:    20 UCF classes, 16 COCO categories and 2 UCF attributes are never trained on,
  * phrasings:     held-out templates (statements.py) are never trained on.
Test-only held-out subjects are never used for model selection either.

Frame embeddings are computed by the Rust CLI (`vs1 embed-video`) — the same
code path the browser runs — so train and serve features match exactly.

Usage: uv run python build_dataset.py --ucf ~/Development/Projects/vega/data/ucf101/UCF-101 \
                                      --coco ~/Development/Projects/vega/data/coco
"""

import argparse
import json
import random
import re
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import statements as st

ROOT = Path(__file__).resolve().parent
VS1 = ROOT.parent / "target" / "release" / "vs1"
DATA = ROOT / "data"

VAL_GROUPS = {f"g{i:02d}" for i in range(1, 8)}
SELECTION_GROUPS = {"g01", "g02", "g03"}
NUM_HELDOUT_CLASSES = 20
# Objects smaller than this fraction of the image are too ambiguous to call
# present or absent, so statements about them are skipped. People count from a
# smaller size: distant pedestrians are clearly "people in the video".
MIN_OBJECT_AREA = 0.004
MIN_PERSON_AREA = 0.002
MIN_STUFF_AREA = 0.02
# Frames per training video whose crops are embedded. Inference crops 4
# (vs1_core::CROP_FRAMES); training uses 2 to halve the cost, and the loader
# cycles them to the same row count (pooling is invariant to repeating rows).
CROP_FRAMES = 2
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"]

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


def subtitle_filter(src: Path, rng: random.Random, tmp: Path) -> str:
    """Randomly styled, regularly changing subtitles."""
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
    srt = tmp / "subs.srt"
    srt.write_text("".join(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n\n" for i, (a, b, text) in enumerate(lines, 1)))
    return f"subtitles={srt}:force_style='{style}'"


def watermark_filter(rng: random.Random, tmp: Path) -> str:
    """A static, semi-transparent channel name or logo text."""
    text = tmp / "mark.txt"
    text.write_text(rng.choice(st.WATERMARK_TEXTS))
    margin = "h/30"
    x, y = rng.choice([("w-tw-" + margin, margin), (margin, margin), ("w-tw-" + margin, "h-th-" + margin),
                       (margin, "h-th-" + margin), ("(w-tw)/2", "h-th-" + margin)])
    box = f":box=1:boxcolor=black@{rng.uniform(0.2, 0.6):.2f}:boxborderw=6" if rng.random() < 0.3 else ""
    return (f"drawtext=fontfile={rng.choice(FONTS)}:textfile={text}:fontsize=h/{rng.randint(10, 20)}"
            f":fontcolor={rng.choice(['white', 'white', 'yellow', 'gray'])}@{rng.uniform(0.45, 0.95):.2f}"
            f":borderw={rng.randint(0, 2)}:x={x}:y={y}{box}")


def make_variant(src: Path, dst: Path, edits: frozenset[str], rng: random.Random) -> None:
    """Re-encodes `src` with the given edits: cap (subtitles), wm (watermark), bw (grayscale)."""
    with tempfile.TemporaryDirectory() as tmp:
        filters = []
        if "cap" in edits:
            filters.append(subtitle_filter(src, rng, Path(tmp)))
        if "wm" in edits:
            filters.append(watermark_filter(rng, Path(tmp)))
        if "bw" in edits:
            filters.append("hue=s=0")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", ",".join(filters), "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)


def ken_burns(image: Path, dst: Path, width: int, height: int, rng: random.Random, frames: int = 48) -> None:
    """A clip (12 fps) slowly zooming into or out of a still image, so COCO
    stills go through the same video pipeline (and have frame-to-frame change)."""
    zoom = rng.uniform(0.12, 0.3)
    z = f"1+{zoom / frames:.5f}*on" if rng.random() < 0.5 else f"{1 + zoom:.4f}-{zoom / frames:.5f}*on"
    fx, fy = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)  # zoom focus
    w, h = width // 2 * 2, height // 2 * 2
    vf = f"scale={w}:{h},zoompan=z='{z}':x='(iw-iw/zoom)*{fx:.3f}':y='(ih-ih/zoom)*{fy:.3f}':d=1:s={w}x{h}:fps=12"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(image), "-vf", vf, "-frames:v", str(frames),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(dst)], check=True)


def synth_clip(dst: Path, rng: random.Random) -> str:
    """A 4 s computer-generated clip (fractal, cellular automaton, test pattern,
    gradient, plasma or noise). Returns its kind."""
    w, h = rng.choice([(640, 360), (480, 360), (640, 480)])
    color = lambda: f"0x{rng.randrange(1 << 24):06x}"
    kind = rng.choice(["mandelbrot", "life", "cellauto", "testsrc2", "bars", "gradients", "sierpinski", "plasma",
                       "noise", "zoneplate", "mptestsrc", "colorspectrum"])
    size = f"size={w}x{h}:rate=12"
    src = {
        "mandelbrot": f"mandelbrot={size}:start_scale={rng.uniform(0.5, 3):.2f}:end_scale={rng.uniform(0.001, 0.3):.3f}"
                      f":start_x={rng.uniform(-1.5, 0.3):.3f}:start_y={rng.uniform(-0.6, 0.6):.3f}"
                      f":inner={rng.choice(['black', 'period', 'convergence', 'mincol'])}"
                      f":outer={rng.choice(['iteration_count', 'normalized_iteration_count', 'outz'])}",
        "life": f"life=size={w // 4}x{h // 4}:rate=12:ratio={rng.uniform(0.05, 0.5):.2f}:mold={rng.randint(0, 50)}"
                f":life_color={color()}:death_color={color()}:mold_color={color()},scale={w}:{h}:flags=neighbor",
        "cellauto": f"cellauto={size}:rule={rng.randint(1, 255)}:random_fill_ratio={rng.uniform(0.2, 0.8):.2f}:scroll=1",
        "testsrc2": f"testsrc2={size}",
        "bars": f"{rng.choice(['smptehdbars', 'pal75bars', 'rgbtestsrc'])}={size}",
        "gradients": f"gradients={size}:speed={rng.uniform(0.005, 0.05):.3f}:nb_colors={rng.randint(2, 8)}"
                     f":type={rng.choice(['linear', 'radial', 'circular', 'spiral'])}",
        "sierpinski": f"sierpinski={size}:type={rng.choice(['carpet', 'triangle'])}",
        "plasma": f"nullsrc={size},format=yuv444p,geq=lum='128+100*sin(X/{rng.uniform(8, 40):.1f}+T*{rng.uniform(0.5, 3):.1f})'"
                  f":cb='128+100*sin(Y/{rng.uniform(8, 40):.1f}+T)':cr='128+100*sin((X+Y)/{rng.uniform(8, 40):.1f}+T*2)'",
        "noise": f"nullsrc={size},format=yuv444p,geq=lum='random(1)*255':cb='{rng.choice(['128', 'random(2)*255'])}'"
                 f":cr='{rng.choice(['128', 'random(3)*255'])}'",
        "zoneplate": f"zoneplate={size}:kt={rng.randint(1, 4)}:kx={rng.randint(1, 8)}:ky={rng.randint(1, 8)}",
        "mptestsrc": f"mptestsrc=rate=12:test={rng.choice(['dc_luma', 'dc_chroma', 'freq_luma', 'freq_chroma', 'amp_luma', 'cbp', 'mv', 'ring1', 'ring2'])},scale={w}:{h}",
        "colorspectrum": f"colorspectrum={size}:type={rng.choice(['black', 'white', 'all'])}",
    }[kind]
    vf = f"hue=h={rng.uniform(0, 360):.0f}" if rng.random() < 0.4 else "null"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", src, "-t", "4", "-vf", vf, "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(dst)], check=True)
    return kind


def embed_videos(videos: list[Path], out_dir: Path, workers: int, frames: int, crops: bool = False) -> None:
    """Full-frame rows to <stem>.npy, or with `crops` the crop rows of the
    crop frames (vs1_core::crop_frame_indices) to <stem>.crops.npy."""
    suffix = ".crops.npy" if crops else ".npy"
    todo = [v for v in videos if not (out_dir / f"{v.stem}{suffix}").exists()]
    chunks = [todo[i::workers] for i in range(workers) if todo[i::workers]]
    print(f"embedding {'crops of ' if crops else ''}{len(todo)} videos with {len(chunks)} processes")
    if not chunks:
        return

    def run(chunk: list[Path]) -> None:
        subprocess.run([str(VS1), "embed-video", *map(str, chunk), "--out-dir", str(out_dir), "--frames", str(frames),
                        *(["--crops-only", "--crop-frames", str(CROP_FRAMES)] if crops else [])],
                       check=True, stderr=subprocess.DEVNULL)

    with ThreadPoolExecutor(len(chunks)) as pool:
        list(pool.map(run, chunks))


def embed_questions(questions: list[dict], out_dir: Path, workers: int) -> None:
    """`vs1 embed-questions` in parallel chunks, merged into one texts.npy +
    index.jsonl (a text shared by two chunks is simply stored twice)."""
    import numpy as np

    chunks = [questions[i::workers] for i in range(workers)]
    with tempfile.TemporaryDirectory(dir=DATA) as tmp:
        def run(i: int) -> None:
            path = Path(tmp) / f"q{i}.jsonl"
            path.write_text("".join(json.dumps(q) + "\n" for q in chunks[i]))
            subprocess.run([str(VS1), "embed-questions", str(path), "--out-dir", str(Path(tmp) / f"out{i}")], check=True,
                           stderr=subprocess.DEVNULL)

        parallel([lambda i=i: run(i) for i in range(workers) if chunks[i]], workers)
        texts, index, offset = [], [None] * len(questions), 0
        for i in range(workers):
            if not chunks[i]:
                continue
            rows = np.load(Path(tmp) / f"out{i}" / "texts.npy")
            lines = (Path(tmp) / f"out{i}" / "index.jsonl").read_text().splitlines()
            for j, line in enumerate(lines):
                ix = json.loads(line)
                index[i + j * workers] = {**ix, "question": ix["question"] + offset,
                                          "options": [o + offset for o in ix["options"]]}
            texts.append(rows)
            offset += len(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "texts.npy", np.concatenate(texts))
    (out_dir / "index.jsonl").write_text("".join(json.dumps(ix) + "\n" for ix in index))
    print(f"{offset} text embeddings")


def parallel(jobs, workers: int) -> None:
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda j: j(), jobs))


# ---------------------------------------------------------------- sources


def ucf_clips(ucf: Path, rng: random.Random, train_per_class: int, val_per_class: int):
    """Returns (clips, heldout classes); clip = dict(path, cls, split)."""
    classes = sorted(p.name for p in ucf.iterdir() if p.is_dir())
    heldout = set(rng.sample(classes, NUM_HELDOUT_CLASSES))
    selection_unseen = set(sorted(heldout)[: len(heldout) // 2])
    clips = []
    for cls in classes:
        files = sorted((ucf / cls).glob("*.avi"))
        val = [f for f in files if f.stem.split("_")[2] in VAL_GROUPS]
        train = [f for f in files if f.stem.split("_")[2] not in VAL_GROUPS]
        for f in rng.sample(val, min(val_per_class, len(val))):
            if cls in heldout:
                split = "selection" if cls in selection_unseen else "test"
            else:
                split = "selection" if f.stem.split("_")[2] in SELECTION_GROUPS else "test"
            clips.append({"path": f, "cls": cls, "split": split})
        if cls not in heldout:
            clips += [{"path": f, "cls": cls, "split": "train"} for f in rng.sample(train, min(train_per_class, len(train)))]
    return clips, classes, heldout


def coco_records(coco: Path, subset: str):
    """Per-image records for a COCO subset ("val2017"/"train2017") with
    object/person labels, captions and caption-consensus attributes."""
    inst = json.loads((coco / "annotations" / f"instances_{subset}.json").read_text())
    caps = json.loads((coco / "annotations" / f"captions_{subset}.json").read_text())
    cats = {c["id"]: c for c in inst["categories"]}
    supercat = {c["name"]: c["supercategory"] for c in inst["categories"]}
    by_image = defaultdict(list)
    for a in inst["annotations"]:
        by_image[a["image_id"]].append(a)
    captions = defaultdict(list)
    for c in caps["annotations"]:
        captions[c["image_id"]].append(c["caption"].strip())
    stuff = {}  # image id → [(panoptic stuff category, area)]
    panoptic = coco / "annotations" / f"panoptic_{subset}.json"
    if panoptic.exists():
        pan = json.loads(panoptic.read_text())
        names = {c["id"]: c["name"] for c in pan["categories"] if not c["isthing"]}
        for a in pan["annotations"]:
            stuff[a["image_id"]] = [(names[g["category_id"]], g["area"]) for g in a["segments_info"]
                                    if g["category_id"] in names]
    out = []
    for img in sorted(inst["images"], key=lambda i: i["id"]):
        area = img["width"] * img["height"]
        present, seen = set(), set()
        persons, crowd, big_persons = 0, False, 0
        for a in by_image[img["id"]]:
            name = cats[a["category_id"]]["name"]
            seen.add(name)
            big = a["area"] / area >= (MIN_PERSON_AREA if name == "person" else MIN_OBJECT_AREA)
            if a["iscrowd"] or big:
                present.add(name)
            if name == "person":
                persons += 1
                crowd |= bool(a["iscrowd"])
                big_persons += int(big and not a["iscrowd"])
        rec = {"path": coco / subset / img["file_name"], "id": img["id"], "width": img["width"],
               "height": img["height"], "present": present, "seen": seen, "persons": persons, "crowd": crowd,
               "big_persons": big_persons, "captions": captions[img["id"]]}
        rec["attributes"] = caption_attributes(rec, supercat)
        rec["stuff"] = stuff_labels(stuff[img["id"]], area) if img["id"] in stuff else {}
        out.append(rec)
    return out, supercat


def stuff_labels(segments: list[tuple[str, int]], area: int) -> dict[str, bool]:
    """Stuff name → present (covers MIN_STUFF_AREA of the image) / absent (no
    segment at all); smaller patches are left undecided."""
    covered = defaultdict(int)
    for name, a in segments:
        covered[name] += a
    out = {}
    for name, (_, _, cats) in st.STUFF.items():
        total = sum(covered.get(c, 0) for c in cats)
        if total / area >= MIN_STUFF_AREA:
            out[name] = True
        elif total == 0:
            out[name] = False
    return out


_ATTRIBUTE_RES = {name: (re.compile(pos), re.compile(rule["avoid"]), rule.get("negative") and re.compile(rule["negative"]))
                  for name, (_, _, pos, rule) in st.COCO_ATTRIBUTES.items()}


def caption_attributes(rec: dict, supercat: dict) -> dict[str, bool]:
    """Attribute → truth, for attributes the captions (and annotations) decide."""
    caps = [c.lower() for c in rec["captions"]]
    out = {}
    for name, (pos, avoid, negative) in _ATTRIBUTE_RES.items():
        if sum(bool(pos.search(c)) for c in caps) >= 2:
            out[name] = True
        elif not any(avoid.search(c) for c in caps) and (
                negative is None or sum(bool(negative.search(c)) for c in caps) >= 2):
            out[name] = False
    # Annotations overrule captions where they speak to the same thing.
    food = {c for c in rec["seen"] if supercat[c] == "food"}
    if any(supercat[c] == "food" for c in rec["present"]):
        out["food"] = True
    elif food and out.get("food") is False:
        del out["food"]
    if "cell phone" in rec["present"]:
        out["phone"] = True
    elif "cell phone" in rec["seen"] and out.get("phone") is False:
        del out["phone"]
    if rec["persons"] == 0:  # "someone is sitting" is false with nobody there
        for name in ("sitting", "standing", "eating", "riding"):
            out[name] = False
    for group in st.SUPERCATEGORIES:
        if any(supercat[c] == group for c in rec["present"]):
            out[f"super_{group}"] = True
        elif not any(supercat[c] == group for c in rec["seen"]):
            out[f"super_{group}"] = False
    return out


def select_coco_train(records: list[dict], count: int, rng: random.Random) -> list[dict]:
    """A training subset of train2017: every positive for rare attributes (up
    to a cap each), topped up with random images."""
    chosen = {}
    by_attr = defaultdict(list)
    for r in records:
        for name, truth in r["attributes"].items():
            if truth and not name.startswith("super_"):
                by_attr[name].append(r)
    for name, recs in sorted(by_attr.items()):
        for r in rng.sample(recs, min(len(recs), count // 20)):
            chosen[r["id"]] = r
    rest = [r for r in records if r["id"] not in chosen]
    for r in rng.sample(rest, max(0, count - len(chosen))):
        chosen[r["id"]] = r
    return sorted(chosen.values(), key=lambda r: r["id"])


def download(records: list[dict], subset: str, workers: int) -> None:
    import urllib.request

    todo = [r for r in records if not r["path"].exists()]
    print(f"downloading {len(todo)} {subset} images")
    todo and todo[0]["path"].parent.mkdir(parents=True, exist_ok=True)

    def fetch(r):
        tmp = r["path"].with_suffix(".part")
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(f"http://images.cocodataset.org/{subset}/{r['path'].name}", tmp)
                tmp.rename(r["path"])
                return
            except OSError:
                if attempt == 3:
                    print(f"failed to download {r['path'].name}")

    with ThreadPoolExecutor(workers * 2) as pool:
        list(pool.map(fetch, todo))


def is_grayscale_image(path: Path) -> bool:
    """True for single-channel images and RGB images with (almost) no colour."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", "scale=32:32", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True)
    px = out.stdout
    if len(px) < 3:
        return False
    diff = sum(abs(px[i] - px[i + 1]) + abs(px[i + 1] - px[i + 2]) for i in range(0, len(px) - 2, 3))
    return diff / (len(px) // 3) < 3.0


def pick_edits(rng: random.Random, rate: float) -> list[frozenset[str]]:
    """Edited variants to make of one clip (each a set of edits)."""
    out = []
    if rng.random() < 0.5 * rate:
        out.append(frozenset({"cap"} | ({"wm"} if rng.random() < 0.25 else set()) | ({"bw"} if rng.random() < 0.2 else set())))
    if rng.random() < 0.3 * rate:
        out.append(frozenset({"wm"} | ({"bw"} if rng.random() < 0.25 else set())))
    if rng.random() < 0.2 * rate:
        out.append(frozenset({"bw"}))
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------- questions


class Writer:
    """Collects examples for one video, choosing phrasings by split."""

    def __init__(self, examples: list, rng: random.Random, npy: Path, split: str, tags: dict):
        self.examples, self.rng, self.npy, self.split, self.tags = examples, rng, npy, split, tags

    def phrasing(self, train: list[str], heldout: list[str]) -> tuple[str, bool]:
        if self.split != "train" and heldout and (not train or self.rng.random() < 0.5):
            return self.rng.choice(heldout), True
        return self.rng.choice(train), False

    def add(self, question: dict, target: list[float], kind: str, heldout_phrase: bool, unseen: bool = False) -> None:
        self.examples.append({"video": str(self.npy.relative_to(ROOT)), "split": self.split, "question": question,
                              "target": target, "tags": {**self.tags, "kind": kind, "unseen": unseen,
                                                         "heldout_phrase": heldout_phrase}})

    def noul(self, kind: str, truth: bool, unseen: bool = False, templates=None, **fmt) -> None:
        train, held = templates or st.NOUL[kind]
        text, heldout_phrase = self.phrasing(train, held)
        text = text.format(**fmt)
        self.add({"type": "noul", "instructions": text[0].upper() + text[1:]}, [float(truth), 1.0 - float(truth)],
                 kind, heldout_phrase, unseen)


def edit_questions(w: Writer, edits: frozenset[str], gray_native: bool, rng: random.Random) -> None:
    for kind, positive in (("caption", "cap" in edits), ("watermark", "wm" in edits),
                           ("grayscale", "bw" in edits or gray_native)):
        if positive or rng.random() < 0.3:
            if kind == "grayscale" and rng.random() < 0.25:
                w.noul("color", not positive)
            else:
                w.noul(kind, positive)


def ucf_questions(w: Writer, clip: dict, classes: list[str], heldout: set[str], rng: random.Random) -> None:
    cls, split = clip["cls"], clip["split"]
    unseen = cls in heldout
    pool = classes if split != "train" else [c for c in classes if c not in heldout]
    for _ in range(2):
        k = rng.randint(4, 8)
        options = [cls] + rng.sample([c for c in pool if c != cls], k - 1)
        rng.shuffle(options)
        text, heldout_phrase = w.phrasing(st.ACTIVITY_CHOICE, st.HELDOUT_ACTIVITY_CHOICE)
        w.add({"type": "choice", "instructions": text, "criteria": {humanize(c): None for c in options}},
              [float(o == cls) for o in options], "activity_choice", heldout_phrase, unseen)
    other = rng.choice([c for c in pool if c != cls])
    for target_cls, truth in ((cls, True), (other, False)):
        w.noul("activity", truth, unseen, a=humanize(target_cls))

    use_heldout = split == "test" and rng.random() < 0.5
    names = [a for a in st.UCF_ATTRIBUTES if (a in st.HELDOUT_ATTRIBUTES) == use_heldout]
    labelled = [(a, cls in st.UCF_ATTRIBUTES[a][2]) for a in names
                if cls in st.UCF_ATTRIBUTES[a][2] or cls in st.UCF_ATTRIBUTES[a][3]]
    pos = [x for x in labelled if x[1]]
    neg = [x for x in labelled if not x[1]]
    for group in (pos, neg):
        if group:
            name, truth = rng.choice(group)
            train, held, *_ = st.UCF_ATTRIBUTES[name]
            w.noul("attribute", truth, name in st.HELDOUT_ATTRIBUTES, templates=(train, held))
    if rng.random() < 0.3:
        w.noul("people_any", True)


UNSEEN_STUFF = set(st.SELECTION_STUFF) | set(st.TEST_STUFF)


def stuff_question(w: Writer, name: str, truth: bool, rng: random.Random) -> None:
    phrase, plural, _ = st.STUFF[name]
    bare = re.sub(r"^(a|an|the) ", "", phrase)
    fmt = {"o": phrase, "O": phrase[0].upper() + phrase[1:], "o_bare": bare}
    negated = rng.random() < 0.15
    templates = (st.NOUL["stuff_plural_negated" if negated else "stuff_plural"] if plural
                 else st.NOUL["object_negated" if negated else "object"])
    w.noul("stuff_negated" if negated else "stuff", truth != negated, name in UNSEEN_STUFF, templates=templates, **fmt)


def allowed_stuff_for(split: str) -> set[str]:
    hidden = {"train": UNSEEN_STUFF, "selection": set(st.TEST_STUFF), "test": set(st.SELECTION_STUFF)}[split]
    return set(st.STUFF) - hidden


def coco_questions(w: Writer, img: dict, allowed: set[str], unseen_cats: set[str], supercat: dict,
                   all_images: list[dict], rng: random.Random) -> None:
    allowed_stuff = allowed_stuff_for(w.split)
    def obj(name: str, truth: bool) -> None:
        noun = st.COCO_NAMES.get(name, name)
        fmt = {"o": st.article(noun), "O": st.article(noun).capitalize(), "o_bare": noun}
        if rng.random() < 0.15:
            w.noul("object_negated", not truth, name in unseen_cats, **fmt)
        else:
            w.noul("object", truth, name in unseen_cats, **fmt)

    present = [c for c in img["present"] if c in allowed and c != "person"]
    absent = [c for c in allowed if c not in img["seen"] and c != "person"]
    for c in rng.sample(present, min(2, len(present))):
        obj(c, True)
    hard = [c for c in absent if any(supercat[c] == supercat[p] for p in img["present"])]
    negatives = rng.sample(hard, min(1, len(hard)))
    rest = [c for c in absent if c not in negatives]
    for c in negatives + rng.sample(rest, min(2 - len(negatives), len(rest))):
        obj(c, False)

    stuff = [(n, t) for n, t in img["stuff"].items() if n in allowed_stuff]
    for group in ([x for x in stuff if x[1]], [x for x in stuff if not x[1]]):
        if group:
            stuff_question(w, *rng.choice(group), rng)

    heldout = st.HELDOUT_COCO_ATTRIBUTES | {f"super_{g}" for g in st.HELDOUT_SUPERCATEGORIES}
    attrs = [(a, t) for a, t in img["attributes"].items() if a != "black_and_white"
             and (a not in heldout or w.split == "test")]
    for group in ([x for x in attrs if x[1]], [x for x in attrs if not x[1]]):
        if group:
            name, truth = rng.choice(group)
            if name.startswith("super_"):
                w.noul("supercategory", truth, name in heldout, templates=st.SUPERCATEGORIES[name[6:]])
            else:
                w.noul("scene", truth, name in heldout, templates=st.COCO_ATTRIBUTES[name][:2])

    n, big, crowd = img["persons"], img["big_persons"], img["crowd"]
    people = []
    if big >= 1 or crowd or n == 0:
        people.append(("people_any", big >= 1 or crowd))
    if (n == 1 and not crowd) or big >= 2 or n == 0:
        people.append(("people_one", n == 1 and not crowd))
    if big >= 3 or crowd or n <= 1:
        people.append(("people_group", big >= 3 or crowd))
    if people:
        w.noul(*rng.choice(people))

    if img["captions"]:
        truth = rng.random() < 0.5
        if truth:
            caption = rng.choice(img["captions"])
        else:
            others = [o for o in rng.sample(all_images, 30) if o["captions"] and not (o["seen"] & img["seen"])]
            caption = rng.choice(rng.choice(others)["captions"]) if others else None
        if caption and not (w.split == "train" and any(c in caption.lower() for c in unseen_cats)):
            caption = caption.rstrip(".").strip()
            w.noul("caption_match", truth, c=caption, c_lower=caption[0].lower() + caption[1:])


def synth_questions(w: Writer, allowed: set[str], classes: list[str], heldout_classes: set[str], supercat: dict,
                    all_images: list[dict], rng: random.Random) -> None:
    """Computer-generated clips: graphics yes, the real world no."""
    w.noul("graphics", True)
    if rng.random() < 0.5:
        w.noul("camera", False)
    w.noul(rng.choice(["people_any", "people_any", "people_one", "people_group"]), False)
    for c in rng.sample(sorted(c for c in allowed if c != "person"), 2):
        noun = st.COCO_NAMES.get(c, c)
        w.noul("object", False, False, o=st.article(noun), O=st.article(noun).capitalize(), o_bare=noun)
    stuff_question(w, rng.choice(sorted(allowed_stuff_for(w.split))), False, rng)
    for name in rng.sample(["outdoors", "indoors", "street", "sport", "water", "kitchen", "beach", "snow", "bedroom"], 2):
        w.noul("scene", False, templates=st.COCO_ATTRIBUTES[name][:2])
    group = rng.choice(["animal", "vehicle", "furniture"])
    w.noul("supercategory", False, templates=st.SUPERCATEGORIES[group])
    pool = classes if w.split != "train" else [c for c in classes if c not in heldout_classes]
    w.noul("activity", False, a=humanize(rng.choice(pool)))
    caption = rng.choice(rng.choice([i for i in rng.sample(all_images, 5) if i["captions"]] or all_images)["captions"])
    caption = caption.rstrip(".").strip()
    w.noul("caption_match", False, c=caption, c_lower=caption[0].lower() + caption[1:])


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucf", type=Path, required=True, help="UCF-101 directory with one folder per class")
    ap.add_argument("--coco", type=Path, help="COCO directory with val2017/ and annotations/")
    ap.add_argument("--coco-images", type=int, default=5000, help="val2017 images (train/selection/test)")
    ap.add_argument("--coco-train-images", type=int, default=6000, help="extra train2017 images (training only)")
    ap.add_argument("--synth-clips", type=int, default=300, help="computer-generated clips (negatives for the real world)")
    ap.add_argument("--train-per-class", type=int, default=6)
    ap.add_argument("--val-per-class", type=int, default=3)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    clips, classes, heldout = ucf_clips(args.ucf, rng, args.train_per_class, args.val_per_class)
    st.finish_attributes(classes)
    for c in clips:
        c.update(source="ucf", gray_native=False, frames=args.frames)

    coco, supercat = [], {}
    if args.coco:
        coco_rng = random.Random(f"{args.seed}:coco")
        coco, supercat = coco_records(args.coco, "val2017")
        coco_rng.shuffle(coco)
        coco = coco[:args.coco_images]
        for i, img in enumerate(coco):
            frac = i / len(coco)
            img.update(split="train" if frac < 0.7 else "selection" if frac < 0.85 else "test", frames=args.frames)
        if args.coco_train_images:
            extra, _ = coco_records(args.coco, "train2017")
            extra = select_coco_train(extra, args.coco_train_images, coco_rng)
            download(extra, "train2017", args.workers)
            extra = [r for r in extra if r["path"].exists()]
            # Training only; 8 frames suffice for a still (the head is invariant
            # to repeating every frame, so they're tiled to 16 in training).
            for img in extra:
                img.update(split="train", frames=8)
            coco += extra
        stills = DATA / "stills"
        stills.mkdir(parents=True, exist_ok=True)
        jobs = []
        for img in coco:
            dst = stills / f"coco_{img['id']}.mp4"
            if not dst.exists():
                jobs.append(lambda img=img, dst=dst: ken_burns(img["path"], dst, img["width"], img["height"],
                                                                random.Random(f"{args.seed}:{img['id']}"),
                                                                48 if img["frames"] >= 16 else 24))
        print(f"{len(coco)} COCO images, rendering {len(jobs)} clips")
        parallel(jobs, args.workers)
        with ThreadPoolExecutor(args.workers) as pool:
            gray = list(pool.map(lambda i: is_grayscale_image(i["path"]), coco))
        for img, g in zip(coco, gray):
            img.update(source="coco", src=img["path"], path=stills / f"coco_{img['id']}.mp4",
                       gray_native=g or img["attributes"].get("black_and_white") is True)
        print(f"{sum(i['gray_native'] for i in coco)} COCO images are black and white")

    synth = []
    if args.synth_clips:
        synth_dir = DATA / "synth"
        synth_dir.mkdir(parents=True, exist_ok=True)
        synth = [{"path": synth_dir / f"synth_{i:04d}.mp4", "id": i, "source": "synth", "frames": args.frames,
                  "split": "train" if i % 20 < 14 else "selection" if i % 20 < 17 else "test"}
                 for i in range(args.synth_clips)]
        parallel([lambda c=c: synth_clip(c["path"], random.Random(f"{args.seed}:synth:{c['id']}"))
                  for c in synth if not c["path"].exists()], args.workers)
        with ThreadPoolExecutor(args.workers) as pool:
            gray = list(pool.map(lambda c: is_grayscale_image(c["path"]), synth))
        for c, g in zip(synth, gray):
            c["gray_native"] = g
        print(f"{len(synth)} synthetic clips ({sum(gray)} grayscale)")

    # Edited variants of every clip; each inherits its base clip's labels.
    variants_dir = DATA / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    videos, jobs = [], []
    for base in clips + coco + synth:
        videos.append({**base, "edits": frozenset()})
        for edits in pick_edits(rng, 1.0 if base["source"] == "ucf" else 0.3):
            dst = variants_dir / f"{base['path'].stem}_{'-'.join(sorted(edits))}.mp4"
            videos.append({**base, "path": dst, "edits": edits})
            if not dst.exists():
                jobs.append(lambda base=base, dst=dst, edits=edits: make_variant(
                    base["path"], dst, edits, random.Random(f"{args.seed}:{dst.stem}")))
    print(f"{len(clips)} UCF clips + {len(coco)} COCO clips + {len(synth)} synthetic clips → {len(videos)} videos, "
          f"rendering {len(jobs)} variants")
    parallel(jobs, args.workers)

    emb_dir = DATA / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    for frames in sorted({v["frames"] for v in videos}):
        embed_videos([v["path"] for v in videos if v["frames"] == frames], emb_dir, args.workers, frames)
    # Crops come from the same timestamps as the 16-frame grid for every video
    # (stills are near-static, so their 8-frame grid doesn't matter).
    embed_videos([v["path"] for v in videos], emb_dir, args.workers, args.frames, crops=True)

    unseen_sel, unseen_test = set(st.SELECTION_COCO_CATEGORIES), set(st.TEST_COCO_CATEGORIES)
    all_cats = set(supercat)
    coco_val = [c for c in coco if c["frames"] >= 16]
    allowed = {"train": all_cats - unseen_sel - unseen_test, "selection": all_cats - unseen_test,
               "test": all_cats - unseen_sel}
    examples = []
    for v in videos:
        npy = emb_dir / f"{v['path'].stem}.npy"
        if not npy.exists():
            continue
        tags = {"source": v["source"], "edits": sorted(v["edits"]),
                **({"class": v["cls"]} if v["source"] == "ucf" else {"image": v["id"]})}
        w = Writer(examples, rng, npy, v["split"], tags)
        if v["source"] == "ucf":
            ucf_questions(w, v, classes, heldout, rng)
        elif v["source"] == "coco":
            coco_questions(w, v, allowed[v["split"]], unseen_sel | unseen_test, supercat, coco_val, rng)
        else:
            synth_questions(w, allowed[v["split"]], classes, heldout, supercat, coco_val, rng)
        if v["source"] != "synth" and rng.random() < 0.08:
            kind = rng.choice(["graphics", "camera"])
            w.noul(kind, kind == "camera")
        edit_questions(w, v["edits"], v["gray_native"], rng)

    with open(DATA / "examples.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    with open(DATA / "questions.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex["question"]) + "\n")
    embed_questions([ex["question"] for ex in examples], DATA / "text", args.workers)
    (DATA / "split.json").write_text(json.dumps({
        "heldout_classes": sorted(heldout), "val_groups": sorted(VAL_GROUPS),
        "heldout_coco_categories": {"selection": sorted(unseen_sel), "test": sorted(unseen_test)},
        "heldout_attributes": sorted(st.HELDOUT_ATTRIBUTES | st.HELDOUT_COCO_ATTRIBUTES),
        "heldout_supercategories": sorted(st.HELDOUT_SUPERCATEGORIES),
        "heldout_stuff": {"selection": st.SELECTION_STUFF, "test": st.TEST_STUFF}}, indent=2))
    counts = defaultdict(int)
    for e in examples:
        counts[e["split"]] += 1
    print(f"wrote {len(examples)} examples ({', '.join(f'{n} {s}' for s, n in sorted(counts.items()))})")


if __name__ == "__main__":
    main()
