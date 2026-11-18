"""Compare vectors from bench.py runs against a reference run.

    python compare.py ref.npz other.npz [other2.npz ...]

Per-image cosine similarity (1.0 = identical vector), and how many of each
image's top-10 nearest neighbours match the reference -- i.e. would a search
return the same results.
"""
import sys

import numpy as np

ref = np.load(sys.argv[1])
R = ref["vecs"]
k = 10
ref_top = np.argsort(-(R @ R.T), axis=1)[:, 1:k + 1]

for f in sys.argv[2:]:
    o = np.load(f)
    assert (o["names"] == ref["names"]).all(), f"{f}: different image set"
    V = o["vecs"]
    cos = (R * V).sum(1)
    top = np.argsort(-(V @ V.T), axis=1)[:, 1:k + 1]
    overlap = np.mean([len(set(a) & set(b)) / k for a, b in zip(ref_top, top)])
    print(f"{f}: cosine vs ref min {cos.min():.5f} mean {cos.mean():.5f} | "
          f"top-{k} search overlap {overlap * 100:.1f}%")
