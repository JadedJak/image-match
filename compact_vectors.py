"""Rebuild the FAISS index with exactly one vector per catalog row: drops
orphan vectors (no row) and, where an id was stored twice, keeps the most
recently added vector. HNSW can't delete in place, so this writes a fresh
index. The previous one is kept as vectors.faiss.bak.

Run with the indexer NOT running; restart the UI afterwards so it loads the
new index. Not baked into the image -- pipe it in from the host:
    Get-Content compact_vectors.py | docker compose run -T --rm imgmatch-general python -
"""
import shutil
import sqlite3

import faiss
import numpy as np

from app import config

old = faiss.read_index(str(config.FAISS_PATH))
base = faiss.downcast_index(old.index)
ids = faiss.vector_to_array(old.id_map)
vecs = base.reconstruct_n(0, old.ntotal)

c = sqlite3.connect(str(config.DB_PATH))
row_ids = {fid for (fid,) in c.execute("SELECT faiss_id FROM files WHERE faiss_id IS NOT NULL")}

# later position wins -- the newest vector for a re-processed file
latest = {}
for pos, fid in enumerate(ids.tolist()):
    if fid in row_ids:
        latest[fid] = pos
keep = np.array(sorted(latest.values()), dtype="int64")
missing = len(row_ids - latest.keys())

print(f"{old.ntotal} vectors -> {len(keep)} ({old.ntotal - len(keep)} dropped); "
      f"{missing} rows have no vector")

new_base = faiss.IndexHNSWFlat(old.d, base.hnsw.nb_neighbors(1), faiss.METRIC_INNER_PRODUCT)
new_base.hnsw.efConstruction = base.hnsw.efConstruction
new_base.hnsw.efSearch = base.hnsw.efSearch
new = faiss.IndexIDMap2(new_base)
print("building new index (a few minutes on CPU)...", flush=True)
new.add_with_ids(vecs[keep], ids[keep])

shutil.copy2(config.FAISS_PATH, str(config.FAISS_PATH) + ".bak")
tmp = str(config.FAISS_PATH) + ".tmp"
faiss.write_index(new, tmp)
shutil.move(tmp, config.FAISS_PATH)
print(f"done: {new.ntotal} vectors written, old index saved as {config.FAISS_PATH.name}.bak")
