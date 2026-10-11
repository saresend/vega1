#!/usr/bin/env python3
"""Builds the demo's sample videos: web/samples/<slug>.mp4 plus samples.json.

Clips are openly licensed videos from Wikimedia Commons (credited from each
file's own metadata) and two local test clips. Each is trimmed to a few
seconds, scaled to 640 px wide and encoded as H.264 with faststart.
Downloads are sequential and cached (default: ~/Development/Projects/vega/data/commons_samples).

Usage: python3 scripts/fetch-samples.py [--cache DIR] [--local-only] [--texts-only]
Standard library + ffmpeg only.
"""

import argparse
import html
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "samples"
USER_AGENT = "video-system1-demo/0.1 (local research demo)"
VS1 = ROOT / "target" / "release" / "vs1"  # cargo build --release -p vs1-cli
FRAMES = 16

# slug, title, subtitle, source (local path or Wikimedia Commons file), start (s),
# duration (s), poster (s into the clip), suggested statements. UCF-101 clips are
# from its test groups (never used for training) and stay local (research use).
UCF = "~/Development/Projects/vega/data/ucf101/UCF-101"
UCF_CREDIT = "UCF101 data set (Soomro et al. 2012) · for research only"
SAMPLES = [
    dict(slug="street", title="street", subtitle="people, cars, subtitles", local="testdata/people_burned.mp4",
         start=10, duration=12, poster=4, credit="OpenCV sample video (vtest.avi) with subtitles that we added",
         statements=["There are people in the video", "People are walking", "There is a car", "The video is captioned",
                     "There is grass", "Nobody is in the video"]),
    dict(slug="dog", title="dog on a leash", subtitle="person and dog, grass", local=f"{UCF}/WalkingWithDog/v_WalkingWithDog_g05_c01.avi",
         start=0, duration=10, poster=3, credit=UCF_CREDIT,
         statements=["There is a dog", "Someone is walking a dog", "There is grass", "There is a cat", "The scene is indoors",
                     "There is a horse"]),
    dict(slug="cat-play", title="cat", subtitle="close-up, watermark", commons="Cat Play 03.webm",
         start=0, duration=8, poster=2,
         statements=["There is a cat", "There is a watermark or logo", "There is a dog", "There is an animal",
                     "There are people in the video", "The video is captioned"]),
    dict(slug="kitchen", title="kitchen", subtitle="knife and cucumber", local=f"{UCF}/CuttingInKitchen/v_CuttingInKitchen_g04_c01.avi",
         start=0, duration=10, poster=3, credit=UCF_CREDIT,
         statements=["Someone is preparing food", "There is a knife", "There is food", "The scene is outdoors",
                     "There is a dog", "Someone is playing a sport"]),
    dict(slug="surf", title="surfer", subtitle="sea and waves", local=f"{UCF}/Surfing/v_Surfing_g06_c01.avi",
         start=0, duration=10, poster=4, credit=UCF_CREDIT,
         statements=["Someone is surfing", "There is water", "There is snow", "There is a boat",
                     "Someone is playing a sport", "There is exactly one person"]),
    dict(slug="guitar", title="acoustic guitar", subtitle="indoors, one person", local=f"{UCF}/PlayingGuitar/v_PlayingGuitar_g06_c01.avi",
         start=0, duration=10, poster=3, credit=UCF_CREDIT,
         statements=["Someone is playing a guitar", "Someone is playing a musical instrument", "The scene is indoors",
                     "There is exactly one person", "There is a crowd", "There is a piano"]),
    dict(slug="horse", title="horse and rider", subtitle="field, fence", local=f"{UCF}/HorseRiding/v_HorseRiding_g04_c01.avi",
         start=0, duration=10, poster=4, credit=UCF_CREDIT,
         statements=["There is a horse", "Someone is riding a horse", "There is a fence", "There are trees",
                     "There is a dog", "It is nighttime"]),
    dict(slug="baby", title="baby", subtitle="on the floor, indoors", local=f"{UCF}/BabyCrawling/v_BabyCrawling_g05_c01.avi",
         start=0, duration=10, poster=3, credit=UCF_CREDIT,
         statements=["There is a baby", "The scene is indoors", "There are toys", "There is a dog",
                     "There is exactly one person", "Someone is playing a sport"]),
    dict(slug="fractal", title="fractal zoom", subtitle="computer graphics", local="testdata/no_people.mp4",
         start=2, duration=10, poster=3, credit="Computer graphics from the ffmpeg mandelbrot source",
         statements=["This is computer-generated graphics", "There are people in the video", "The scene is outdoors",
                     "There is a tree", "This is real camera footage"]),
]
# More Commons clips that suit the demo (beach, city at night, skier, traffic, ...)
# were rate-limited on download; add entries here to include them.


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 4:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 15 * 2 ** attempt
            print(f"  rate limited, waiting {wait} s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def commons_info(title: str) -> dict:
    query = urllib.parse.urlencode({"action": "query", "titles": f"File:{title}", "prop": "imageinfo", "format": "json",
                                    "iiprop": "url|extmetadata", "iiextmetadatafilter": "Artist|LicenseShortName"})
    page = next(iter(json.loads(get(f"https://commons.wikimedia.org/w/api.php?{query}"))["query"]["pages"].values()))
    info = page["imageinfo"][0]
    meta = info.get("extmetadata", {})
    artist = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")))).strip()
    license_ = meta.get("LicenseShortName", {}).get("value", "")
    return {"url": info["url"].split("?")[0], "page": info.get("descriptionurl", ""),
            "credit": " · ".join(x for x in (artist or "Unknown author", license_, "Wikimedia Commons") if x)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path.home() / "Development/Projects/vega/data/commons_samples")
    ap.add_argument("--local-only", action="store_true", help="skip the Wikimedia Commons clips (offline)")
    ap.add_argument("--texts-only", action="store_true", help="only re-embed the pages' texts (samples.json exists)")
    args = ap.parse_args()
    if args.texts_only:
        return embed_texts(json.loads((OUT / "samples.json").read_text()))
    args.cache.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for s in SAMPLES:
        entry = {k: s[k] for k in ("title", "subtitle", "poster", "statements")}
        entry["file"] = f"{s['slug']}.mp4"
        if "local" in s:
            src = Path(s["local"]).expanduser() if s["local"].startswith("~") else ROOT / s["local"]
            if not src.exists():
                print(f"skip {s['slug']}: {src} not found")
                continue
            entry["credit"] = s["credit"]
        elif args.local_only:
            continue
        else:
            try:
                info = commons_info(s["commons"])
                src = args.cache / s["commons"].replace(" ", "_")
                if not src.exists():
                    print(f"downloading {s['commons']}")
                    data = get(info["url"])
                    src.with_suffix(".part").write_bytes(data)
                    src.with_suffix(".part").rename(src)
                    time.sleep(2)
            except Exception as e:  # network or API trouble: leave this sample out
                print(f"skip {s['slug']}: {e}")
                continue
            entry["credit"] = info["credit"]
            entry["source"] = info["page"]
        dst = OUT / entry["file"]
        if not dst.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(s["start"]), "-t", str(s["duration"]), "-i", str(src),
                            "-vf", "scale='min(640,iw)':-2", "-c:v", "libx264", "-preset", "slow", "-crf", "24",
                            "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(dst)], check=True)
        # Precomputed video state (the page skips embedding) and a strip of the
        # sampled frames for display.
        state = OUT / f"{s['slug']}.f32"
        if not state.exists():
            out = subprocess.run([str(VS1), "embed-state", str(dst), "--out", str(state), "--frames", str(FRAMES)],
                                 check=True, capture_output=True, text=True).stdout
            json.loads(out)
        entry.update(embeddings=state.name, frames=FRAMES, rows=state.stat().st_size // (4 * 512))
        strip = OUT / f"{s['slug']}.frames.jpg"
        if not strip.exists():
            seconds = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                            str(dst)], capture_output=True, text=True, check=True).stdout)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(dst), "-vf",
                            f"fps={FRAMES / seconds:.4f},scale=160:90:force_original_aspect_ratio=increase,crop=160:90,"
                            f"tile={FRAMES}x1", "-frames:v", "1", "-q:v", "4", str(strip)], check=True)
        entry["strip"] = strip.name
        print(f"{entry['file']:18s} {dst.stat().st_size / 1e6:5.1f} MB  {entry['rows']} rows  {entry['credit']}")
        manifest.append(entry)
    (OUT / "samples.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    embed_texts(manifest)
    print(f"wrote {len(manifest)} samples to {OUT}")


def run_embed_texts(items: list, out: Path) -> int:
    listing = OUT / "texts.tmp.json"
    listing.write_text(json.dumps(items))
    res = subprocess.run([str(VS1), "embed-texts", str(listing), "--out", str(out)],
                         check=True, capture_output=True, text=True).stdout
    listing.unlink()
    return json.loads(res)["texts"]


def embed_texts(manifest: list) -> None:
    # Text embeddings for every statement the page offers (samples, defaults,
    # suggestions): the page answers these before its text encoder downloads.
    page = json.loads((ROOT / "web" / "statements.json").read_text())
    texts = list(dict.fromkeys([t for e in manifest for t in e["statements"]] + page["defaults"] + page["suggestions"]))
    print(f"{len(texts)} statements → {run_embed_texts(texts, OUT / 'texts.f32')} text embeddings")
    # The same for the category page (vega1-cat.html): web/categories.json.
    cats = json.loads((ROOT / "web" / "categories.json").read_text())
    items = cats["defaults"] + cats["suggestions"] + [q for qs in cats["samples"].values() for q in qs]
    questions = list({json.dumps(q, sort_keys=True): {"type": "choice", "instructions": q["q"],
                                                      "criteria": {o: None for o in q["options"]}} for q in items}.values())
    print(f"{len(questions)} category questions → {run_embed_texts(questions, OUT / 'texts-cat.f32')} text embeddings")


if __name__ == "__main__":
    main()
