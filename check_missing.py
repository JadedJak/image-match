"""List catalog rows whose source file no longer exists at the recorded path,
grouped by likely cause.

Not baked into the image -- pipe it in from the host so it runs without a rebuild:
    Get-Content check_missing.py | docker compose run -T --rm imgmatch-general python -
"""
import errno
import os
import sqlite3
from collections import Counter

from app import config

c = sqlite3.connect(str(config.DB_PATH))
rows = c.execute("SELECT path FROM files").fetchall()
source_root = str(config.SOURCE_DIR)

groups = {"outside SOURCE_DIR": [], "name/path too long": [], "folder gone": [], "file gone, folder exists": []}
for (p,) in rows:
    try:
        os.stat(p)
        continue
    except OSError as e:
        too_long = e.errno == errno.ENAMETOOLONG
    if not p.startswith(source_root + "/"):
        groups["outside SOURCE_DIR"].append(p)
    elif too_long:
        groups["name/path too long"].append(p)
    elif not os.path.isdir(os.path.dirname(p)):
        groups["folder gone"].append(p)
    else:
        groups["file gone, folder exists"].append(p)

total = sum(len(v) for v in groups.values())
print(f"{total} missing of {len(rows)} indexed (SOURCE_DIR={source_root})")
for name, paths in groups.items():
    if not paths:
        continue
    longest_part = max(len(part.encode()) for p in paths for part in p.split("/"))
    print(f"\n== {name}: {len(paths)}  (longest path {max(len(p.encode()) for p in paths)} bytes, "
          f"longest single name {longest_part} bytes)")
    if name == "folder gone":
        top = Counter(os.path.dirname(p) for p in paths).most_common(10)
        print("  most affected folders:")
        for d, n in top:
            print(f"    {n:6d}  {d}")
    for p in paths[:5]:
        print("  ", p)
