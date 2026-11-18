"""Benchmark the CLIP embedder on the laptop: one device/precision per run.

    python /bench/bench.py <image_dir> <out.npz>

Uses the app's own embedder (DEVICE / PRECISION env vars), so it measures
exactly what the indexer would do. Decode+resize is timed separately from
the model, since only the model part changes between CPU and GPU.
Writes the vectors so compare.py can check agreement between runs.
"""
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from app import config, embedder

img_dir, out = Path(sys.argv[1]), sys.argv[2]
paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
bs = config.BATCH_SIZE

t0 = time.time()
embedder._load()
load_s = time.time() - t0
dev = embedder._device

t0 = time.time()
imgs = [embedder.load_and_downsample(p) for p in paths]
decode_s = time.time() - t0

def sync():
    if dev == "xpu":
        torch.xpu.synchronize()

# warm-up: first batches compile kernels on XPU -- not representative
embedder.embed_batch(imgs[:bs]); sync()
if dev == "xpu":
    torch.xpu.reset_peak_memory_stats()

t0 = time.time()
vecs = []
for i in range(0, len(imgs), bs):
    vecs.append(embedder.embed_batch(imgs[i:i + bs]))
sync()
embed_s = time.time() - t0
vecs = np.concatenate(vecs)

t0 = time.time()
for _ in range(5):
    embedder.embed_text("a dog playing on the beach")
sync()
text_ms = (time.time() - t0) / 5 * 1000

rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
gpu_gb = torch.xpu.max_memory_allocated() / 1024**3 if dev == "xpu" else 0.0

print(f"device={dev} precision={config.PRECISION if dev != 'cpu' else 'fp32'} "
      f"batch={bs} threads={torch.get_num_threads()} torch={torch.__version__}")
print(f"  model load      {load_s:6.1f} s")
print(f"  decode+resize   {len(paths) / decode_s:6.1f} img/s  ({len(paths)} images, CPU, same for every run)")
print(f"  model only      {len(paths) / embed_s:6.1f} img/s")
print(f"  text query      {text_ms:6.0f} ms")
print(f"  peak RAM (proc) {rss_gb:6.2f} GB   peak GPU alloc {gpu_gb:5.2f} GB")
np.savez(out, vecs=vecs, names=np.array([p.name for p in paths]))
