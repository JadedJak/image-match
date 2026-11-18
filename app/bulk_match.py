"""Bulk-matches a folder of low-quality images against this category's FAISS
index and copies the best higher-resolution match for each into an output
folder. Built for "I have a pile of low-res copies, want to find their
high-res source in the catalog" -- read-search-copy only, never touches the
index/catalog or the low-quality inputs themselves. Run with:

    docker compose run --rm imgmatch-general python -m app.bulk_match
    docker compose run --rm imgmatch-personal python -m app.bulk_match

(or the `matchfind <category>` alias in powershell-aliases.ps1)

Threshold is MATCH_SIMILARITY_THRESHOLD (config.py, default 0.90) --
deliberately below the near-duplicate-grade 0.95+ range: testing found some
genuine matches (same photo, cropped differently) score a bit lower on raw
CLIP cosine similarity than an uncropped near-dup would. 0.90 accepts some
extra false positives in exchange for not missing those -- see match-log.csv
in the output folder to spot-check them; nothing here deletes/moves the
low-quality inputs, so a bad match costs you nothing but a stray extra file.

A candidate only counts if it's also higher resolution (by total pixel
count) than the query -- this both enforces "the point is finding the
higher-quality original" and, as a side effect, keeps a low-quality image
from matching itself if it happens to already be indexed in this catalog
(same file, same resolution, so it can never be "higher").
"""
import csv
import logging
import shutil
import time
from pathlib import Path

from PIL import Image

from . import config, db, embedder
from .faiss_store import FaissStore
from .search_engine import search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bulk_match")

# How many candidates to pull per query before filtering by threshold/
# resolution -- search()'s result order interleaves near-duplicate-first
# grouping with similarity-descending, so it's not one clean sorted list to
# just take the top of; pulling a generous window and filtering explicitly
# below is simpler and more correct than assuming a particular ordering.
CANDIDATES_PER_QUERY = 20


def _resolution(path):
    with Image.open(path) as img:
        return img.size  # (width, height)


def _dest_name(matched_path, used_names):
    """Plain basename normally; only disambiguated if two different matched
    source files would otherwise collide on the same output filename (e.g.
    IMG_0001.jpg from two different subfolders)."""
    name = matched_path.name
    prior_source = used_names.get(name)
    if prior_source is None or prior_source == str(matched_path):
        used_names[name] = str(matched_path)
        return name
    stem, suffix = matched_path.stem, matched_path.suffix
    i = 2
    while f"{stem}_{i}{suffix}" in used_names:
        i += 1
    new_name = f"{stem}_{i}{suffix}"
    used_names[new_name] = str(matched_path)
    return new_name


def run():
    if not config.MATCH_INPUT_DIR or not config.MATCH_OUTPUT_DIR:
        log.error("MATCH_INPUT_DIR/MATCH_OUTPUT_DIR not configured -- see .env.example")
        return

    config.MATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(
        p for p in config.MATCH_INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in config.VALID_EXTENSIONS
    )
    if not inputs:
        log.info("nothing to process in %s", config.MATCH_INPUT_DIR)
        return

    db.init_db()
    store = FaissStore(embedder.dim())

    log_path = config.MATCH_OUTPUT_DIR / "match-log.csv"
    is_new_log = not log_path.exists()
    matched = unmatched = failed = 0
    used_names = {}
    t_start = time.time()

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new_log:
            writer.writerow(["low_quality_path", "matched_path", "similarity", "lowres_dims", "match_dims"])

        log.info("processing %d file(s) from %s (threshold=%.2f)",
                  len(inputs), config.MATCH_INPUT_DIR, config.MATCH_SIMILARITY_THRESHOLD)

        for query_path in inputs:
            try:
                query_w, query_h = _resolution(query_path)
            except Exception as e:
                log.warning("failed to read %s: %s", query_path, e)
                failed += 1
                continue

            try:
                results = search(store, query_path, top_k=CANDIDATES_PER_QUERY)
            except Exception as e:
                log.warning("search failed for %s: %s", query_path, e)
                failed += 1
                continue

            best = None
            for r in results:
                if r["similarity"] < config.MATCH_SIMILARITY_THRESHOLD:
                    continue
                try:
                    cand_w, cand_h = _resolution(r["path"])
                except Exception:
                    continue  # source unreadable/moved since indexing -- skip it, don't crash the run
                if (cand_w * cand_h) <= (query_w * query_h):
                    continue  # not actually higher-res -- also filters out self-matches
                if best is None or r["similarity"] > best["similarity"]:
                    best = {**r, "width": cand_w, "height": cand_h}

            if best is None:
                writer.writerow([str(query_path), "", "", f"{query_w}x{query_h}", ""])
                unmatched += 1
                continue

            matched_path = Path(best["path"])
            dest_name = _dest_name(matched_path, used_names)
            dest_path = config.MATCH_OUTPUT_DIR / dest_name
            try:
                shutil.copy2(matched_path, dest_path)
            except OSError as e:
                log.warning("copy failed for match of %s: %s", query_path, e)
                failed += 1
                continue

            writer.writerow([
                str(query_path), str(matched_path), f"{best['similarity']:.4f}",
                f"{query_w}x{query_h}", f"{best['width']}x{best['height']}",
            ])
            matched += 1
            f.flush()

    elapsed = time.time() - t_start
    log.info(
        "done: %d matched, %d unmatched, %d failed, %d total | %.0fs | log: %s",
        matched, unmatched, failed, len(inputs), elapsed, log_path,
    )


if __name__ == "__main__":
    run()
