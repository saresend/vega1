#!/usr/bin/env python3
"""Copies the built site (web/) to a GitHub Pages checkout and commits it.

Leaves out sample clips that may not be published (UCF-101: research use
only) and their credit. Push the result yourself:  git -C DIR push origin gh-pages

Usage: python3 scripts/build-pages.py [DIR]   (default: ../vega1-pages, a gh-pages checkout)
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "vega1-pages"
    assert (out / ".git").exists(), f"{out} is not a git checkout of the gh-pages branch"
    for p in out.iterdir():
        if p.name != ".git":
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    shutil.copytree(WEB, out, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.part", "*.tmp.json"))
    (out / ".nojekyll").touch()
    samples = out / "samples"
    manifest = json.loads((samples / "samples.json").read_text())
    keep = [e for e in manifest if "UCF101" not in e.get("credit", "")]
    for e in manifest:
        if e not in keep:
            for f in samples.glob(e["file"].rsplit(".", 1)[0] + ".*"):
                f.unlink()
    (samples / "samples.json").write_text(json.dumps(keep, indent=2, ensure_ascii=False))
    for page in out.glob("*.html"):
        page.write_text(page.read_text().replace(" · UCF101 (for research only) · ", " · ffmpeg · "))
    subprocess.run(["git", "-C", str(out), "add", "-A"], check=True)
    if subprocess.run(["git", "-C", str(out), "diff", "--cached", "--quiet"]).returncode:
        subprocess.run(["git", "-C", str(out), "commit", "-q", "-m",
                        "Deploy Vega1 demo\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"], check=True)
    print(f"{out}: {len(keep)} samples ({', '.join(e['title'] for e in keep)})")


if __name__ == "__main__":
    main()
