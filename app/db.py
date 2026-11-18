import sqlite3
from contextlib import contextmanager

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    phash TEXT NOT NULL,
    faiss_id INTEGER UNIQUE,
    proxy_path TEXT,
    indexed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
"""


@contextmanager
def connect():
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with connect() as conn:
        conn.executescript(SCHEMA)


def get_by_path(conn, path):
    return conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()


def upsert_file(conn, path, filename, size, mtime, phash, faiss_id, proxy_path, indexed_at):
    conn.execute(
        """INSERT INTO files (path, filename, size, mtime, phash, faiss_id, proxy_path, indexed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(path) DO UPDATE SET
             filename=excluded.filename, size=excluded.size, mtime=excluded.mtime,
             phash=excluded.phash, faiss_id=excluded.faiss_id, proxy_path=excluded.proxy_path,
             indexed_at=excluded.indexed_at""",
        (path, filename, size, mtime, phash, faiss_id, proxy_path, indexed_at),
    )


def get_all_meta(conn):
    """path -> {size, mtime, faiss_id, phash, proxy_path} for every indexed
    file, loaded once. Used by the indexer's skip-check (size+mtime) and
    move-detection (size+phash) so a large catalog doesn't mean per-file
    SQLite queries during the scan phase."""
    rows = conn.execute("SELECT path, size, mtime, faiss_id, phash, proxy_path FROM files").fetchall()
    return {
        r["path"]: {
            "size": r["size"], "mtime": r["mtime"], "faiss_id": r["faiss_id"],
            "phash": r["phash"], "proxy_path": r["proxy_path"],
        }
        for r in rows
    }


def update_moved(conn, old_path, new_path, new_filename, new_mtime, new_proxy_path):
    """Repoint an existing row to a new path -- same faiss_id/phash/vector,
    just the location changed. Used when a file is detected as moved
    (matched by size+phash) rather than genuinely new."""
    conn.execute(
        "UPDATE files SET path=?, filename=?, mtime=?, proxy_path=? WHERE path=?",
        (new_path, new_filename, new_mtime, new_proxy_path, old_path),
    )


def delete_by_path(conn, path):
    """Remove a row whose file is gone and wasn't matched to a new location
    (see indexer's move-detection/prune pass). Note: this does not remove
    the corresponding vector from the FAISS HNSW index -- that index type
    doesn't support true deletion. The vector becomes an inert orphan: it
    can never surface in search results, since results are always joined
    against this table and a missing row means no match. A full from-scratch
    reindex is the only way to reclaim that space, and is rarely needed."""
    conn.execute("DELETE FROM files WHERE path = ?", (path,))


def search_by_text(conn, term, limit=200):
    """Substring match against path/filename -- for finding a file you
    already know the name (or part of it) of, as opposed to reverse image
    search. Plain SQL LIKE, not fuzzy -- fine at this scale (indexed
    columns, low hundreds of thousands of rows) without needing FTS5."""
    like = f"%{term}%"
    return conn.execute(
        "SELECT * FROM files WHERE path LIKE ? OR filename LIKE ? ORDER BY filename LIMIT ?",
        (like, like, limit),
    ).fetchall()


def get_by_faiss_ids(conn, faiss_ids):
    if not faiss_ids:
        return {}
    q = f"SELECT * FROM files WHERE faiss_id IN ({','.join('?' * len(faiss_ids))})"
    rows = conn.execute(q, faiss_ids).fetchall()
    return {r["faiss_id"]: r for r in rows}


def count(conn):
    return conn.execute("SELECT COUNT(*) c FROM files").fetchone()["c"]


def max_faiss_id(conn):
    r = conn.execute("SELECT MAX(faiss_id) m FROM files").fetchone()
    return r["m"] if r["m"] is not None else -1
