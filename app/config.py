import os
from pathlib import Path

CATEGORY = os.environ.get("CATEGORY", "general")  # "general" or "personal"

SOURCE_DIR = Path(os.environ.get("SOURCE_DIR", "/data/source"))
INDEX_DIR = Path(os.environ.get("INDEX_DIR", "/data/index"))
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Where the UI's "delete" action moves files instead of unlinking them
# outright -- a same-volume rename (fast, atomic), not a copy, since it
# lives inside SOURCE_DIR itself. This is a soft delete: the file is gone
# from the catalog (its DB row is removed, so it can never surface in
# search again) but still recoverable by hand until someone clicks "Empty
# trash" in the UI, which is the only thing that actually unlinks it.
# Requires SOURCE_DIR to be mounted read-write (see docker-compose.yml) --
# it used to be :ro; that changed specifically to support this feature.
# Excluded from the indexer's scan (see indexer.iter_images) so trashed
# files can't get silently re-indexed.
TRASH_DIR = SOURCE_DIR / "_deleted"
TRASH_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = INDEX_DIR / "catalog.sqlite3"
FAISS_PATH = INDEX_DIR / "vectors.faiss"

# ViT-H-14 (LAION-2B weights) over the smaller ViT-B-32 default this project
# started with: this workload is disk-seek-bound, not GPU-bound (see README),
# so the GPU sits idle most of the time regardless of model size -- there was
# no real throughput cost to the smaller model, only a match-quality one.
# LAION-trained checkpoints specifically (vs OpenAI's own CLIP releases) also
# matter here: LAION's scrape-based datasets are far less filtered than
# OpenAI's proprietary training set, so LAION checkpoints have materially
# better recall for NSFW-adjacent text/image concepts if that's part of your
# catalog -- relevant for a private personal-photo search tool.
CLIP_MODEL = os.environ.get("CLIP_MODEL", "ViT-H-14")
CLIP_PRETRAINED = os.environ.get("CLIP_PRETRAINED", "laion2b_s32b_b79k")
DEVICE = os.environ.get("DEVICE", "cuda")  # cuda | xpu (Intel GPU) | cpu -- falls back to cpu automatically if unavailable
# Model weight/compute precision: fp32 (default) | fp16 | bf16. 16-bit halves
# the model's memory and uses Intel XMX / tensor cores, at a tiny cost in
# vector precision (output vectors are always stored as float32). GPU-only:
# ignored (fp32) on cpu, where 16-bit is usually slower. Pick one per catalog
# before its first full index run.
PRECISION = os.environ.get("PRECISION", "fp32")

# In-memory downsample cap applied on read, before hashing/embedding.
# Never written back to the source file. CLIP resizes to 224x224 internally
# and pHash works off an even smaller representation, so this just cuts
# decode/IO time on large source images without changing match quality.
# The same downsampled image also becomes the proxy/thumbnail below, so
# raise this if you want larger UI thumbnails -- no extra decode cost either way.
MAX_SIDE = int(os.environ.get("MAX_SIDE", "512"))

# Optional: persistent downsized copy of each indexed image, written once
# during indexing as a side effect of the decode that's already happening
# (no second read/decode pass). Used only as embedding input + UI thumbnail
# source -- never treated as the authoritative file location, which is
# always the original `path`. Leave PROXY_DIR unset to disable entirely.
_proxy_dir = os.environ.get("PROXY_DIR")
PROXY_DIR = Path(_proxy_dir) if _proxy_dir else None
if PROXY_DIR:
    PROXY_DIR.mkdir(parents=True, exist_ok=True)
PROXY_QUALITY = int(os.environ.get("PROXY_QUALITY", "85"))

# Hover-preview target size/quality -- deliberately separate from MAX_SIDE/
# PROXY_QUALITY above. The small search-thumbnail proxy looks blurry once
# CSS stretches it to ~90vw for the preview overlay, but sending the raw
# multi-thousand-pixel original on every hover is wasteful and slow over a
# tailnet link. This is the middle ground: resized on the fly from the true
# source (see the /img-preview route), never written to disk.
PREVIEW_MAX_SIDE = int(os.environ.get("PREVIEW_MAX_SIDE", "1600"))
PREVIEW_QUALITY = int(os.environ.get("PREVIEW_QUALITY", "90"))

# "Download medium"/"Download small" context-menu options -- resized on the
# fly from the true source via the same embedder.load_and_downsample() path
# as the preview route above, written to a temp file and handed to
# ui.download(), never persisted. Kept as separate knobs from PREVIEW_MAX_SIDE
# even though the medium default matches it, since these are a distinct
# feature (an actual saved file, not a hover overlay) that may want tuning
# independently later.
DOWNLOAD_MEDIUM_MAX_SIDE = int(os.environ.get("DOWNLOAD_MEDIUM_MAX_SIDE", "1600"))
DOWNLOAD_SMALL_MAX_SIDE = int(os.environ.get("DOWNLOAD_SMALL_MAX_SIDE", "800"))
DOWNLOAD_QUALITY = int(os.environ.get("DOWNLOAD_QUALITY", "90"))

NEAR_DUP_HAMMING_THRESHOLD = int(os.environ.get("NEAR_DUP_HAMMING_THRESHOLD", "8"))
TOP_K = int(os.environ.get("TOP_K", "100"))

# Bulk match/upgrade-hunt tool (app/bulk_match.py) -- drop low-quality
# images in MATCH_INPUT_DIR, run it, and it copies the best higher-res match
# for each (found via the same FAISS index/search path as the UI) into
# MATCH_OUTPUT_DIR. Both optional -- unset means the tool refuses to run
# rather than silently doing nothing (see bulk_match.run()).
_match_input_dir = os.environ.get("MATCH_INPUT_DIR")
MATCH_INPUT_DIR = Path(_match_input_dir) if _match_input_dir else None
_match_output_dir = os.environ.get("MATCH_OUTPUT_DIR")
MATCH_OUTPUT_DIR = Path(_match_output_dir) if _match_output_dir else None
# Below the near-duplicate-grade 0.95+ range on purpose -- cropped-vs-
# uncropped copies of the same photo can score a bit lower on raw CLIP
# cosine similarity, and missing those defeats the point of this tool. See
# bulk_match.py's docstring for the false-positive tradeoff this implies.
MATCH_SIMILARITY_THRESHOLD = float(os.environ.get("MATCH_SIMILARITY_THRESHOLD", "0.90"))

# Indexer throughput tuning. IO_WORKERS lets multiple file reads/decodes be
# in flight concurrently -- on a single mechanical HDD this won't beat the
# physical seek limit, but it keeps decode from idling dead-serial on each
# disk wait and lets the OS/drive schedule pending reads more efficiently.
IO_WORKERS = int(os.environ.get("IO_WORKERS", "8"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "500"))  # files, not batches

UI_HOST = os.environ.get("UI_HOST", "0.0.0.0")
UI_PORT = int(os.environ.get("UI_PORT", "7860"))
# "nicegui" (default) or "gradio" -- see app/main.py. Switchable via .env
# without a rebuild: NiceGUI is the primary UI now (real per-element click/
# dblclick events, native scroll control -- Gradio's js= escape hatch kept
# fighting us on exactly this kind of fine-grained interactivity). Gradio
# is kept as a working fallback, not deleted, in case NiceGUI hits its own
# unforeseen wall.
UI_ENGINE = os.environ.get("UI_ENGINE", "nicegui")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
