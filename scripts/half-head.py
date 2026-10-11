#!/usr/bin/env python3
"""Writes a half-size (f16) copy of an answer head for the web page; the Rust
head computes in f32 either way. Usage: half-head.py in.safetensors out.safetensors"""

import sys

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

src, dst = sys.argv[1:3]
with safe_open(src, "np") as f:
    meta = f.metadata()
    tensors = {k: f.get_tensor(k).astype(np.float16) for k in f.keys()}
save_file(tensors, dst, metadata=meta)
