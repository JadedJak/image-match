"""Given a query image, returns ranked matches: near-duplicates (by
perceptual hash, within NEAR_DUP_HAMMING_THRESHOLD bits) surfaced first,
followed by semantically similar images (by CLIP cosine similarity),
each with its full path and filename."""
import numpy as np

from . import config, db, embedder, phash_util


def _vector_search(store, vec, top_k):
    """Shared plumbing for both image-query and text-query search: run the
    FAISS lookup, drop -1 padding, and join the surviving ids against
    SQLite. Returns (score, row) pairs, unsorted."""
    scores, ids = store.search(vec, top_k)
    # FAISS pads results with -1 when fewer than top_k candidates exist --
    # filter scores and ids together so a dropped -1 can't shift the
    # pairing between the two (zip(scores, ids) on mismatched-length lists
    # would silently attach the wrong score to the wrong id).
    pairs = [(float(s), int(i)) for s, i in zip(scores, ids) if i != -1]
    # An index built before the indexer stopped reusing ids on re-process
    # can still hold two vectors under one id -- keep only the best-scoring
    # one so the photo isn't listed twice (FAISS returns best-first).
    seen = set()
    pairs = [(s, i) for s, i in pairs if not (i in seen or seen.add(i))]
    with db.connect() as conn:
        rows = db.get_by_faiss_ids(conn, [i for _, i in pairs])
    return [(score, rows[fid]) for score, fid in pairs if fid in rows]


def _query_image(query_path, crop):
    """The image a search embeds: the whole query file, or only the part
    inside crop (x0, y0, x1, y1 fractions -- the UI's box search)."""
    if crop:
        return embedder.load_region(query_path, crop)
    return embedder.load_and_downsample(query_path)


def _hamming_threshold(hamming_threshold, crop):
    # A crop's perceptual hash isn't comparable to the whole-photo hashes in
    # the catalog, so a box search never tags anything DUPLICATE.
    if crop:
        return -1
    return config.NEAR_DUP_HAMMING_THRESHOLD if hamming_threshold is None else hamming_threshold


def text_search(store, term, top_k=None):
    """Semantic search from a plain-language prompt instead of a query
    image -- CLIP's text and image towers share one embedding space, so a
    term like "dog on a beach" can be searched against the same FAISS
    index of image vectors directly. No near-duplicate/hamming concept
    here since there's no query image to hash."""
    top_k = top_k or config.TOP_K
    vec = embedder.embed_text(term)
    results = []
    for score, row in _vector_search(store, vec, top_k):
        results.append({
            "path": row["path"],
            "filename": row["filename"],
            "proxy_path": row["proxy_path"],
            "similarity": score,
            "mtime": row["mtime"],
        })
    results.sort(key=lambda r: -r["similarity"])
    return results


def search(store, query_path, top_k=None, hamming_threshold=None, crop=None):
    top_k = top_k or config.TOP_K
    hamming_threshold = _hamming_threshold(hamming_threshold, crop)
    img = _query_image(query_path, crop)
    vec = embedder.embed(img)
    query_phash = phash_util.compute_phash(img)

    results = []
    for score, row in _vector_search(store, vec, top_k):
        hd = phash_util.hamming(query_phash, row["phash"])
        is_near_dup = hd <= hamming_threshold
        results.append({
            "path": row["path"],
            "filename": row["filename"],
            "proxy_path": row["proxy_path"],
            "similarity": score,
            "hamming": hd,
            "near_duplicate": is_near_dup,
            "mtime": row["mtime"],
        })

    # near-duplicates first (closest hash first), then descending semantic similarity
    results.sort(key=lambda r: (not r["near_duplicate"], r["hamming"] if r["near_duplicate"] else 0, -r["similarity"]))
    return results


def search_combined(store, query_path, refine_text, image_weight=0.6, top_k=None, hamming_threshold=None, crop=None):
    """Like search(), but nudges ranking toward a plain-language detail as
    well as the query image -- e.g. a house photo + "blue sky" biases
    results toward similar houses that also look sunny. image_weight in
    [0, 1] controls the balance (1.0 = ignore the text entirely, 0.0 =
    ignore the image entirely); the UI's default is 0.6, favoring the
    image somewhat.

    This is a soft nudge, not a hard filter: image and text vectors are
    both already L2-normalized (see embedder.embed/embed_text), so the
    weighted sum is renormalized and handed to the exact same FAISS
    lookup every other search here uses (_vector_search). Cosine/inner-
    product similarity is linear, so the blended vector genuinely scores
    "close to both inputs at once" -- but there's no guarantee every
    result satisfies the text description, only that it's ranked toward
    images that do.

    Near-duplicate detection still runs off the query image's own phash,
    not the blended vector -- "is this the same photo" isn't something a
    text nudge should be able to influence.
    """
    top_k = top_k or config.TOP_K
    hamming_threshold = _hamming_threshold(hamming_threshold, crop)
    img = _query_image(query_path, crop)
    image_vec = embedder.embed(img)
    text_vec = embedder.embed_text(refine_text)
    combined = image_weight * image_vec + (1 - image_weight) * text_vec
    combined = (combined / np.linalg.norm(combined)).astype("float32")
    query_phash = phash_util.compute_phash(img)

    results = []
    for score, row in _vector_search(store, combined, top_k):
        hd = phash_util.hamming(query_phash, row["phash"])
        is_near_dup = hd <= hamming_threshold
        results.append({
            "path": row["path"],
            "filename": row["filename"],
            "proxy_path": row["proxy_path"],
            "similarity": score,
            "hamming": hd,
            "near_duplicate": is_near_dup,
            "mtime": row["mtime"],
        })

    results.sort(key=lambda r: (not r["near_duplicate"], r["hamming"] if r["near_duplicate"] else 0, -r["similarity"]))
    return results
