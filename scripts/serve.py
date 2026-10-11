#!/usr/bin/env python3
"""Serves web/ for the demo, with HTTP range requests (Safari needs them for
<video>) and no caching of the page's own code. Binds all interfaces so a
phone on the same network can open it.

Usage: python3 scripts/serve.py [--port 8765] [--host 0.0.0.0]
"""

import argparse
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".mjs": "text/javascript"}

    def end_headers(self):
        if self.path.split("?")[0].endswith((".html", ".js", "/")):
            self.send_header("Cache-Control", "no-cache")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        path = self.translate_path(self.path)
        if not match or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        start, end = match.groups()
        start, end = (int(start), int(end or size - 1)) if start else (max(0, size - int(end)), size - 1)
        if start >= size:
            self.send_error(416, "Range not satisfiable")
            return None
        end = min(end, size - 1)
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self.range_left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "range_left", None)
        if left is None:
            return super().copyfile(source, outputfile)
        while left > 0:
            chunk = source.read(min(64 * 1024, left))
            if not chunk:
                break
            outputfile.write(chunk)
            left -= len(chunk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=str(WEB)))
    print(f"serving {WEB} on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
