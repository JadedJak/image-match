"""Compare FAISS vector ids against catalog rows: how many vectors are
harmless orphans (no row -- never shown) vs. duplicated ids (one row, several
vectors -- can show the same photo twice in results).

Not baked into the image -- pipe it in from the host so it runs without a rebuild:
    Get-Content check_vectors.py | docker compose run -T --rm imgmatch-general python -
"""
import sqlite3
from collections import Counter

import faiss

from app import config

index = faiss.read_index(str(config.FAISS_PATH))
ids = Counter(faiss.vector_to_array(index.id_map).tolist())
c = sqlite3.connect(str(config.DB_PATH))
row_ids = {fid for (fid,) in c.execute("SELECT faiss_id FROM files WHERE faiss_id IS NOT NULL")}

orphans = sum(n for fid, n in ids.items() if fid not in row_ids)
dup_ids = {fid: n for fid, n in ids.items() if n > 1 and fid in row_ids}
no_vector = len(row_ids - ids.keys())

print(f"{index.ntotal} vectors, {len(row_ids)} catalog rows")
print(f"orphan vectors (no row, never shown):        {orphans}")
print(f"rows with duplicate vectors (can show twice): {len(dup_ids)}  ({sum(dup_ids.values()) - len(dup_ids)} extra vectors)")
print(f"rows with no vector (never found by search):  {no_vector}")
