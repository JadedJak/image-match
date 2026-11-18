"""Walks SOURCE_DIR, embeds + hashes any new or changed image, and writes
into the SQLite metadata table + FAISS index for this category.

Resumable: a file already indexed at its current size/mtime is skipped, so
re-running after an interruption -- or after dropping new files onto the
source volume -- only does incremental work. Run with:

    docker compose run --rm imgmatch-general python -m app.indexer
    docker compose run --rm imgmatch-personal python -m app.indexer

I/O-bound-workload notes (large sources on a mechanical encrypted HDD):
- File reads/decodes for a batch run on a thread pool (IO_WORKERS) so
  multiple reads are in flight rather than strictly one-at-a-time.
- GPU embedding happens once per batch (BATCH_SIZE), not once per file.
- No cross-batch prefetching: GPU time per batch is milliseconds next to
  disk time per batch (seconds, seek-bound), so the GPU sits idle either
  way -- overlapping wouldn't hide anything worth the added complexity.

Move detection: files that get reorganized/reclassified into new folders
are recognized by content signature (exact file size + perceptual hash),
not by path. A "new" file whose signature matches a row whose old path has
vanished this run is treated as a move -- the existing row is repointed to
the new path (same faiss_id, same vector, no re-embedding) instead of
being indexed as a duplicate. Anything left over at the end (missing, not
matched to any new location) is pruned from the DB -- see db.delete_by_path
for why that's safe even though the FAISS index itself can't delete.
"""
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import config, db, embedder, phash_util
from .faiss_store import FaissStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("indexer")


# Windows system dirs that can appear anywhere a whole drive/volume gets
# mounted in as SOURCE_DIR. $RECYCLE.BIN in particular holds obfuscated
# fragments of deleted files -- real image extensions, not real image data --
# which just show up as permanent decode failures (retried every run, since
# a failed file is never marked processed) if they aren't skipped outright.
SKIP_DIRNAMES = {"$recycle.bin", "system volume information"}


def _is_proxy_dir(p: Path) -> bool:
    """True if p is the same directory as config.PROXY_DIR -- checked by
    device+inode (os.path.samefile), not path string equality, because
    SOURCE_DIR and PROXY_DIR are two independent bind mounts inside the
    container (/data/source, /data/proxy). If your .env happens to nest
    PROXY_DIR under SOURCE_DIR on the host (e.g. SOURCE_DIR_PERSONAL=I:/,
    PROXY_DIR_PERSONAL=I:/proxy-cache), the *path strings* /data/source/
    proxy-cache and /data/proxy never match even though they're the same
    underlying files -- samefile() sees through that, comparing what's
    actually on disk rather than which mount you walked in through."""
    if config.PROXY_DIR is None:
        return False
    try:
        return p.is_dir() and os.path.samefile(p, config.PROXY_DIR)
    except OSError:
        return False


def iter_images(root: Path):
    # os.walk (not root.rglob) so TRASH_DIR/SKIP_DIRNAMES/PROXY_DIR can be
    # pruned from the walk entirely, not just filtered after the fact.
    # TRASH_DIR is a direct child of SOURCE_DIR (see config.TRASH_DIR), so
    # that check only needs to run at the top level -- trashed files are
    # pending permanent deletion via the UI's "Empty trash" button, not part
    # of the live catalog, so if they got scanned here, a soft-deleted file
    # whose DB row we already removed would just get silently re-indexed on
    # the next run. PROXY_DIR has no such guarantee on where it sits, so
    # that check (_is_proxy_dir) runs at every level instead -- if it ever
    # got scanned, proxy JPEGs would get indexed as source photos and get
    # *their own* proxies generated in turn (proxy-of-a-proxy), corrupting
    # the catalog with mismatched paths.
    for dirpath, dirnames, filenames in os.walk(root):
        if Path(dirpath) == root:
            dirnames[:] = [d for d in dirnames if d != config.TRASH_DIR.name]
        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRNAMES]
        dirnames[:] = [d for d in dirnames if not _is_proxy_dir(Path(dirpath) / d)]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in config.VALID_EXTENSIONS:
                yield p


def batched(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def proxy_path_for(path: Path):
    if not config.PROXY_DIR:
        return None
    rel = path.relative_to(config.SOURCE_DIR)
    # append (not replace) the extension so e.g. photo.png and photo.jpg
    # in the same folder can't collide once both become proxy JPEGs
    return config.PROXY_DIR / (str(rel) + ".jpg")


def load_one(path):
    """Runs on a worker thread: stat, decode+downsample, hash, and (if
    PROXY_DIR is configured) write the small proxy copy -- all from the one
    decode, no second read/decode pass. Returns an error alongside the path
    instead of raising, so one bad/corrupt file doesn't kill the whole batch."""
    try:
        stat = path.stat()
        img = embedder.load_and_downsample(path)
        phash = phash_util.compute_phash(img)
        proxy_path = proxy_path_for(path)
        if proxy_path:
            embedder.save_proxy(img, proxy_path)
        return path, stat, img, phash, proxy_path, None
    except Exception as e:
        return path, None, None, None, None, e


def run():
    db.init_db()
    store = FaissStore(embedder.dim())

    added = 0
    skipped = 0
    failed = 0
    moved = 0
    pruned = 0
    t_start = time.time()
    bytes_read = 0

    with db.connect() as conn:
        existing_meta = db.get_all_meta(conn)
        next_id = (max((m["faiss_id"] for m in existing_meta.values() if m["faiss_id"] is not None), default=-1)) + 1

        log.info("scanning %s ...", config.SOURCE_DIR)
        to_process = []
        current_paths = set()
        scanned = 0
        for path in iter_images(config.SOURCE_DIR):
            str_path = str(path)
            current_paths.add(str_path)
            stat = path.stat()
            scanned += 1
            # This walk stat()s every file (even ones that turn out to be
            # unchanged and get skipped below), which is real per-file disk
            # I/O -- on a large tree, especially a mechanical/encrypted
            # drive, that can take a long time with nothing else printed in
            # between. A checkpoint every 5000 files here is just
            # visibility into "still working" vs. "actually stuck"; it
            # doesn't affect what gets indexed.
            if scanned % 5000 == 0:
                log.info("scanning: %d files checked so far...", scanned)
            prior = existing_meta.get(str_path)
            proxy_intact = (
                not (config.PROXY_DIR and prior and prior["proxy_path"])
                or Path(prior["proxy_path"]).exists()
            )
            if prior and prior["size"] == stat.st_size and prior["mtime"] == stat.st_mtime and proxy_intact:
                skipped += 1
                continue
            # size/mtime unchanged but the recorded proxy file is gone --
            # falls through to reprocessing below instead of being skipped,
            # so a plain re-run can self-heal proxies that got clobbered
            # (e.g. by the PROXY_DIR-nested-under-SOURCE_DIR bug _is_proxy_dir
            # now prevents) without needing a full wipe-and-reindex.
            to_process.append(path)

        # DB rows whose recorded path no longer exists on disk this run --
        # candidates for "moved" (if a to_process file's content matches) or
        # "deleted" (if nothing claims them by the end of this run).
        # An empty scan against a non-empty catalog is almost always an
        # unmounted volume (VeraCrypt not mounted yet, Docker started before
        # it, wrong .env path) -- not the user deleting every photo. Without
        # this, the prune pass below would wipe every row and proxy.
        if scanned == 0 and existing_meta:
            log.error(
                "found 0 images in %s but the catalog has %d -- source looks unmounted or empty. "
                "Nothing changed. Mount it (and restart Docker/Podman if needed) and re-run.",
                config.SOURCE_DIR, len(existing_meta),
            )
            raise SystemExit(1)

        missing_paths = set(existing_meta.keys()) - current_paths
        # signature -> list of missing paths, not a single path: identical
        # copies of the same photo (same size+phash) in several folders are
        # common, and a plain dict kept only one of them per signature. The
        # rest were silently dropped from both move-matching and the prune
        # pass below, so their rows survived every re-run pointing at a path
        # that no longer exists -- thumbnail still showed (proxy intact) but
        # high-res preview / full image 404'd.
        by_signature = defaultdict(list)
        for p in missing_paths:
            by_signature[(existing_meta[p]["size"], existing_meta[p]["phash"])].append(p)
        log.info(
            "%d files to (re)index, %d already up to date, %d missing from last known location",
            len(to_process), skipped, len(missing_paths),
        )

        with ThreadPoolExecutor(max_workers=config.IO_WORKERS) as pool:
            for batch_paths in batched(to_process, config.BATCH_SIZE):
                loaded = list(pool.map(load_one, batch_paths))

                new_imgs, new_metas = [], []
                for path, stat, img, phash, proxy_path, err in loaded:
                    if err is not None:
                        log.warning("failed to process %s: %s", path, err)
                        failed += 1
                        continue
                    bytes_read += stat.st_size

                    sig = (stat.st_size, phash)
                    # Only a path with no row of its own can be a move
                    # target. A path already in the DB is just being
                    # reprocessed (changed mtime, missing proxy) -- repointing
                    # a missing row onto it would collide with its existing
                    # row (UNIQUE files.path). Its matching missing copy is
                    # left for the prune pass instead.
                    candidates = by_signature.get(sig) if str(path) not in existing_meta else None
                    old_path = candidates.pop() if candidates else None
                    if old_path is not None:
                        old_meta = existing_meta[old_path]
                        new_proxy = str(proxy_path) if proxy_path else None
                        db.update_moved(conn, old_path, str(path), path.name, stat.st_mtime, new_proxy)
                        if old_meta["proxy_path"] and old_meta["proxy_path"] != new_proxy:
                            Path(old_meta["proxy_path"]).unlink(missing_ok=True)
                        moved += 1
                        continue

                    new_imgs.append(img)
                    new_metas.append((path, stat, phash, proxy_path))

                if new_imgs:
                    vecs = embedder.embed_batch(new_imgs)
                    ids = []
                    for path, stat, phash, proxy_path in new_metas:
                        str_path = str(path)
                        # Always a fresh id, even when re-processing a path
                        # that already has a row. HNSW can't remove vectors,
                        # so reusing the old id left two vectors under one
                        # id and the photo showed up twice in results. A
                        # fresh id instead orphans the old vector (no row
                        # points at it -- never shown; compact_vectors.py
                        # reclaims the space).
                        faiss_id = next_id
                        next_id += 1
                        ids.append(faiss_id)
                        db.upsert_file(
                            conn, str_path, path.name, stat.st_size, stat.st_mtime,
                            phash, faiss_id, str(proxy_path) if proxy_path else None, time.time(),
                        )
                    store.add(vecs, np.array(ids, dtype="int64"))
                    added += len(new_imgs)

                if (added + moved) % config.SAVE_EVERY < config.BATCH_SIZE:
                    store.save()
                    conn.commit()
                    elapsed = time.time() - t_start
                    rate = added / elapsed if elapsed > 0 else 0
                    mb_rate = (bytes_read / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                    log.info(
                        "checkpoint: %d added, %d moved, %d skipped, %d failed | %.1f img/s, %.1f MB/s",
                        added, moved, skipped, failed, rate, mb_rate,
                    )

        # anything left in by_signature never showed up anywhere in this
        # scan -- genuinely gone, not just relocated. Prune it.
        for old_path in (p for paths in by_signature.values() for p in paths):
            old_meta = existing_meta[old_path]
            if old_meta["proxy_path"]:
                Path(old_meta["proxy_path"]).unlink(missing_ok=True)
            db.delete_by_path(conn, old_path)
            pruned += 1
        if pruned:
            conn.commit()

    store.save()
    elapsed = time.time() - t_start
    rate = added / elapsed if elapsed > 0 else 0
    log.info(
        "done: %d added, %d moved, %d pruned, %d skipped, %d failed, %d total in index | %.1f img/s avg over %.0fs",
        added, moved, pruned, skipped, failed, store.ntotal(), rate, elapsed,
    )


if __name__ == "__main__":
    run()
