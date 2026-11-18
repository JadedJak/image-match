# image-match

Private, local reverse-image-search over a large local catalog. Given a
query image, returns ranked matches (near-duplicates first, then
semantically similar images) with full path and filename, via a small
local web UI.

## Design

- **Near-duplicate detection:** perceptual hashing (`imagehash.phash`) —
  catches resizes, recompressions, minor crops/edits.
- **Semantic similarity:** CLIP embeddings (`open_clip`, `ViT-H-14`,
  LAION-2B weights) run on GPU, indexed with FAISS (HNSW, cosine/inner-
  product) — catches "looks like this" matches across different source
  photos. The same model's text encoder also powers plain-language search
  (a description like "dog on a beach" embeds into the same vector space
  as the images) — see the UI's search box.
- **Two fully separate instances/indexes:** `imgmatch-general` (not
  personal) and `imgmatch-personal`, each with its own SQLite metadata DB
  and FAISS index, so the personal catalog never shares an index or a
  network-reachable port with the general one.
- **Nothing leaves the box.** No cloud APIs. Source volumes are mounted
  read-write into the containers (they used to be read-only, until the
  UI's delete-to-trash feature needed to move files within them -- see
  "Deleting files from the UI" below).
- **Encrypted at rest:** source and index directories are expected to
  live on VeraCrypt volumes, mounted as drive letters on the Windows host
  (VeraCrypt has a "TrueCrypt Mode" for your existing old TC volume — GUI:
  *Mount Options → TrueCrypt Mode*, CLI: `veracrypt --truecrypt -t <path>`).
  Use a *new native* VeraCrypt volume for the index (no TC-mode needed
  there, since it's new data).

## Why images are downsampled in memory, not on disk

CLIP resizes every input to 224×224 internally regardless of source
resolution, and pHash works off an even coarser representation — so a
4000×3000 original and a 512px version produce essentially identical
embeddings/hashes. `MAX_SIDE` (default 512, see `.env`) shrinks each
image **in memory only** right before hashing/embedding; it never
rewrites anything on the source volume. This is mainly a decode/IO win
when reading large originals off an encrypted volume.

## Setup

Runs natively on Windows: invoke `docker compose` from PowerShell (not a
WSL2 shell), with drive-letter paths in `.env` (e.g. `F:/personal`) —
Docker Desktop's WSL2 backend resolves these transparently.

**CLIP weight cache (metered connection, read this before your first build):**
the CLIP weights (~3.9GB for the current `ViT-H-14` default) live in a
plain host folder (`HF_CACHE_DIR` in `.env`, default `./hf-cache`),
bind-mounted **read-only** into both containers at `/opt/hf-cache`, with
`HF_HUB_OFFLINE=1` set unconditionally so a container can never silently
fall back to a network fetch -- it fails loudly instead if the folder is
somehow empty. This is deliberately *not* baked into the image and *not*
a Docker build-cache mount -- two earlier approaches here, tried in that
order, each looked like it solved the redundant-download problem and then
didn't: an image layer gets rebuilt when anything upstream of it in the
Dockerfile changes, and a BuildKit cache mount turned out to be a
separate cache namespace from the image layer that had already downloaded
the weights once, so switching to it caused a *third* download instead of
reusing the first two. A plain host folder can't be invalidated by either
failure mode -- it's just a directory that sits there until you delete it.

One-time cache warming (only step that's ever allowed to touch the
network for this):

```bash
mkdir hf-cache   # or wherever HF_CACHE_DIR points
docker run --rm -v ${HF_CACHE_DIR:-./hf-cache}:/opt/hf-cache -e HF_HOME=/opt/hf-cache -e HF_HUB_OFFLINE=0 \
  image-match:latest python -c "import open_clip; open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')"
```

(`-e HF_HUB_OFFLINE=0` overrides the image's baked-in offline default for
this one command only -- everything else, indexer runs and the UI alike,
stays permanently offline.) Do this once per model; changing
`CLIP_MODEL`/`CLIP_PRETRAINED` in `app/config.py` means re-running it for
the new model, since the old weights obviously won't satisfy a different
model name/checkpoint.

1. Mount your VeraCrypt volumes on Windows (old catalog in TrueCrypt
   Mode for source, fresh volumes for each index and proxy cache),
   confirm each drive letter is actually mounted and browsable in
   Explorer before proceeding.
2. `copy .env.example .env` and fill in the real drive-letter paths.
3. Confirm GPU passthrough works: `docker run --rm --gpus all
   nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi` should show your 3090.
4. Build: `docker compose build`

Every time you restart the machine or dismount/remount a VeraCrypt
volume, re-mount it and confirm it's live **before** running any
`docker compose` command — see the warning in `.env.example` about
Docker silently creating an empty directory for a missing host path.

## Managing containers: `powershell-aliases.ps1`

`powershell-aliases.ps1` at the project root wraps the raw `docker
compose` commands used throughout this README into two PowerShell
functions, `image-match` (container lifecycle) and `indexer` (run the
indexer). Both are category-scoped — a bare `image-match up`/`down`/
`stop`/`restart`/`logs` acts on both categories, or pass `general`/
`personal` to target just one.

**Install (one-time):** add this line to your PowerShell `$PROFILE`
(dot-sourcing, not just running the script, so the functions stay
defined in every new session):

```powershell
. D:\image-match\powershell-aliases.ps1
```

**Reload after editing the script itself:** open a new PowerShell
window. `. $PROFILE` in the same window is supposed to re-run the
dot-source line above, but in practice it didn't pick up an updated
script (2026-11-11), while a new window always does.

```powershell
image-match up                  # start both UIs (docker compose up -d)
image-match up personal         # start just imgmatch-personal
image-match down                # full teardown (no per-category scoping — see below)
image-match stop personal       # stop just imgmatch-personal, container stays for a fast `up` later
image-match restart general     # rebuild-aware restart: up -d --force-recreate
image-match logs personal       # follow imgmatch-personal logs
image-match build               # docker compose build (builds the one shared image tag)
indexer general                 # run the indexer for one category
indexer all                     # run both, sequentially (see I/O bottleneck notes below)
```

`stop` vs `down`: `stop` leaves the container in place (fast `up` to
resume, and it's the one that actually frees GPU VRAM for just one
category, since CUDA context dies with the process either way). `down`
tears down the whole Compose project — it has no per-service form, so it
always affects both categories regardless of the argument you'd pass to
other actions.

## Indexing (run once per category, then incrementally as needed)

```bash
docker compose run --rm imgmatch-general python -m app.indexer
docker compose run --rm imgmatch-personal python -m app.indexer
```

Resumable — safe to interrupt and re-run; only new/changed files (by
size+mtime) get re-embedded. Start with a source dir containing just your
test set of a couple thousand images before pointing it at the full 8TB.

If `SOURCE_DIR` is a whole mounted drive/volume, the indexer skips
`$RECYCLE.BIN` and `System Volume Information` automatically — these
Windows system dirs can hold obfuscated fragments of deleted files (real
image extensions, not real image data) that would otherwise show up as
permanent decode failures, retried every single run since a failed file
is never marked processed. It also skips whatever directory `PROXY_DIR_*`
points at, if that happens to sit nested under `SOURCE_DIR` — otherwise
proxy JPEGs would get indexed as source photos and get proxies generated
*of* their proxies.

## Running the UI

```bash
docker compose up -d
```

- General: `http://<bind-host>:7860`. `GENERAL_BIND_HOST` in `.env`
  defaults to `0.0.0.0` (LAN-reachable) if unset.
- Personal: `http://<bind-host>:7861`. `PERSONAL_BIND_HOST` defaults to
  `127.0.0.1` (this machine only) if unset — the more locked-down default
  reflects what's normally in that catalog, but it's just as configurable
  as `GENERAL_BIND_HOST`, not hardcoded.

Recommended for both: set `GENERAL_BIND_HOST`/`PERSONAL_BIND_HOST` in
`.env` to this box's Tailscale IP (`tailscale ip -4`) rather than leaving
either default — `0.0.0.0` is also reachable on the raw LAN (anyone on
the same network, not just your tailnet), and `127.0.0.1` can't be
reached remotely at all, including from your own other devices over
Tailscale. Same IP for both entries — it's this one host's tailnet
address, not a per-category value.

**No authentication in front of either UI.** Binding to a Tailscale IP
narrows *who can reach the port* to your tailnet, but anyone who can
reach it can use every action in the UI, including delete-to-trash — this
is a private single-user tool, not a multi-user one. If that's ever not
acceptable, the intended fix is a reverse proxy (Caddy/nginx) doing HTTP
Basic Auth in front of Compose, covering the NiceGUI page routes *and*
the `/img-source`, `/img-proxy`, `/img-preview` routes uniformly — not
yet implemented.

After running the indexer while the UI is already up, `docker compose
restart imgmatch-general` (or `-personal`) to pick up new vectors — the
running UI process holds the FAISS index in memory for query speed and
doesn't watch for changes yet.

**Any change to a file under `app/` needs a rebuild, not just a
restart/recreate.** The Dockerfile bakes the whole `app/` folder into the
image at build time (`COPY app ./app`) and, unlike `SOURCE_DIR`/
`INDEX_DIR`/`HF_CACHE_DIR`, it is **not** bind-mounted — so the running
container has no way to see an edited `.py` file until a new image is
built from it. `image-match restart` (even with `--force-recreate`)
recreates the container from whatever image was last built; it can't
pick up source changes on its own. Always `image-match build` first,
*then* `image-match restart <category>`, for any `app/` edit — including
UI-only/JS changes, not just Python logic changes.

## Two UI implementations: NiceGUI (default) and Gradio (fallback)

`UI_ENGINE` in `.env` picks which frontend runs (`app/main.py` dispatches
based on it) — `nicegui` (default) or `gradio`. Both talk to the exact
same backend (`search_engine.py`, `db.py`, `embedder.py`, the FaissStore/
CLIP model) — switching frontends never touches the CLIP weight cache or
the index, and switching is just an env var + container recreate:

```bash
# in .env: UI_ENGINE=gradio (or nicegui)
image-match restart general   # or: docker compose up -d --force-recreate imgmatch-general
```

NiceGUI became the default after Gradio's `js=` mechanism repeatedly
failed to reliably run client-side code for things like double-click
detection and scroll-position control — real per-element DOM event
binding and a native `scroll_area.scroll_to()` in NiceGUI fixed both
outright instead of fighting a bolt-on escape hatch. Gradio (`app/ui.py`)
is kept working, not deleted, as a fallback in case NiceGUI hits its own
unforeseen wall.

## UI features (NiceGUI)

- **Hover preview:** pause over any result thumbnail to see it at its
  real (non-square) aspect ratio in a full-screen overlay — the grid
  crops everything to a square via CSS for scanning, which hides the
  actual shape. Delay and on/off are both adjustable in the sidebar.
  Purely client-side (no extra request) *unless* the quality toggle below
  is set to High-res.
- **Preview quality toggle** (defaults to **Proxy**): Proxy reuses the
  already-loaded small search-thumbnail (`MAX_SIDE`/`PROXY_QUALITY` in
  `.env`) — free, but looks soft once stretched to fill the overlay.
  High-res instead hits a dedicated `/img-preview` route that re-decodes
  the *true source file* and resizes it on the fly to `PREVIEW_MAX_SIDE`
  px (default 1600, quality `PREVIEW_QUALITY` default 90) — sharper, but
  a real network round trip per hover, so it's the one to leave off over
  a slow tailnet link. Nothing from this route is ever cached to disk.
- **Max results slider:** 300–1000, default 500, step 50. Every card in a result
  set is built as real DOM (image + caption + right-click menu, ~9
  elements each) synchronously at query time — thumbnail *bytes* lazy-
  load as you scroll (native `<img loading="lazy">` behavior), but the
  DOM construction itself doesn't, so a query that actually returns
  several thousand results can make the page visibly hitch while it
  renders. `reconnect_timeout=30.0` (`app/ui_nicegui.py`) gives that
  headroom before NiceGUI would otherwise treat a slow render as a dead
  connection and reset the whole page/state — raise it further if you
  regularly push toward the high end of the slider and still see resets.
- **Right-click any result:** search similar, copy path, open full-size
  original in a new tab, download original, or move to trash (see
  "Deleting files from the UI" below).
- **Refine with text (optional):** add a plain-language detail to an
  image search — a house photo + "blue sky" biases ranking toward similar
  houses that also look sunny. Blends the image and text CLIP vectors
  into one query vector (`search_engine.search_combined`) rather than
  filtering results after the fact, using the same shared FAISS lookup
  every other search here uses. This is a soft nudge, not a hard filter —
  there's no guarantee every result matches the text, only that results
  are ranked toward images that do. The weight slider controls the
  balance (default 0.6, favoring the image); leaving the text box empty
  runs a plain image search exactly as before.
- **Selected Image:** a fixed 320×320px preview (no longer resizable; the
  old adjustable size slider was removed once 320px settled as the size
  that actually gets used) plus a plain "Select Image" button that opens
  the OS's native file-picker dialog — Quasar's fancier drag-and-drop
  QUploader widget was tried first but its internal styling (text color,
  then box height) never fully cooperated, so it was dropped in favor of
  a plain button triggering a hidden `<input type="file">`. You can still
  drag-and-drop a file anywhere on the page instead of clicking the
  button — both paths go through the same upload code. "Search similar to
  selected" sits directly below.
- **Box search:** drag on the query image to draw a box; letting go
  searches only what's inside it. Drag inside the box to move it, drag a
  corner to resize it, and each change re-runs the search. "Clear box"
  goes back to the whole image. The query image itself never changes, so
  you can search several parts of one photo in a row. The crop is cut
  from the full-resolution original (not the 512px proxy) so small boxes
  keep their detail; it re-runs whichever search was last used (Find
  matches with refine text, or plain). Box changes replace the results in
  place rather than adding Back entries, but Back/Forward does restore the
  box that was active when you left a view. Limitation: the index holds
  one vector per *whole* photo, so a box finds photos whose overall
  content resembles the boxed part — great when that thing is the main
  subject elsewhere, weak for finding it as a small detail inside other
  photos. A box search never tags results DUPLICATE (a crop's perceptual
  hash isn't comparable to whole-photo hashes).
- **Responsive layout:** the sidebar stacks above the results grid below
  Tailwind's `md` breakpoint (768px) instead of squeezing both into one
  row, and the results grid runs 2–5 columns depending on viewport width
  instead of a fixed 5. Hover preview is mouse-only, though — it doesn't
  have a touch equivalent yet, so that one feature is desktop-only for now.
- **Back / Forward:** browser-history-style navigation across query hops
  — search an image, double-click a result to search similar to *it*,
  then Back returns to the first image's results without re-querying
  (everything's restored from an in-memory snapshot: query image, result
  list, sort mode, and any active text search/refine-text). Works across
  both image-similarity chains and text searches. Forward stack clears on
  a genuinely new search, same as a browser.

## I/O is the real bottleneck, not GPU or CPU

At ~10MB/file across millions of files on a mechanical, encrypted HDD, this
will be **disk-seek-bound**, not compute-bound — there's a hard physical
floor here that no amount of software tuning removes: every byte has to be
read off the platter at least once. What the indexer does try to avoid is
*stacking* extra cost on top of that floor:

- `Image.draft()` decodes JPEGs at a reduced DCT scale directly, instead of
  decoding full resolution and shrinking after — a large win specifically
  because your originals (~10MB, presumably high-megapixel) are so much
  bigger than the 512px `MAX_SIDE` target.
- Reads/decodes for a batch run on `IO_WORKERS` threads concurrently
  (default 8) instead of strictly one file at a time, so the OS/drive can
  schedule pending reads more efficiently than dead-serial access.
- GPU embedding runs once per `BATCH_SIZE` (default 32) instead of once
  per file — removes per-call overhead, though on this workload the GPU
  will be idle most of the time regardless; I/O dominates.
- Already resumable/checkpointed (`SAVE_EVERY`), so the right way to run
  the full 8TB pass is: kick it off, let it run unattended over however
  many days it takes, safe to interrupt and resume at any point.

**Get real numbers before extrapolating to the full catalog:** run the
indexer against your few-thousand-image test set first and watch the
`img/s, MB/s` figures in the checkpoint log lines — that's your actual
throughput on this HDD/volume, not a guess, and `(total source size) /
(observed MB/s)` gives a real floor estimate for the full run.

Actual catalog size: ~270k personal (~10MB each, ~2.7TB) + ~60k
not-personal (mixed, up to 4MB, ~0.1-0.24TB) ≈ **~3TB total**, ~330k
files. At a conservative 30-40MB/s that's ~20-28 hours; at a more typical
60-80MB/s for reads this size (files this large amortize seek cost well
compared to a catalog of tiny thumbnails), more like ~10-14 hours --
realistically an overnight job rather than multi-day.

**Don't run `imgmatch-general` and `imgmatch-personal` indexers at the
same time if their source volumes sit on the same physical disk** —
two processes both issuing reads against one HDD head will contend and
likely be slower combined than running them one after another. If they
end up on genuinely separate physical drives, running both concurrently
is fine and roughly halves total wall-clock.

If you find `IO_WORKERS=8` isn't actually helping (some HDD/controller
combos see no benefit past 2-4 concurrent readers, or even regress), drop
it in `.env` and re-test — there's no universal right number here, it
depends on the specific drive/enclosure/controller.

## Proxy cache on fast storage

If you have spare NVMe, a downsized "proxy" copy of every indexed image
is generated automatically (see `PROXY_DIR_*` in `.env`) as a side effect
of the decode the indexer already does — not a separate preprocessing
pass. This is worth being precise about what it does and doesn't buy you:

- **Does not speed up the first full index build.** Every byte of the
  source still has to be read off the encrypted HDD once, no matter what
  tool does the decoding — that cost is fixed by disk physics, not
  software. Running a separate tool (e.g. XnView) as a pre-pass would
  mean *two* full decode cycles instead of one; folding proxy generation
  into the indexer's existing decode avoids that duplication.
- **Does speed up everything after that.** Any future re-index (new CLIP
  model, different threshold, whatever) reads from NVMe instead of the
  HDD — GPU-bound instead of disk-bound, likely minutes instead of hours.
- **Does fix result-browsing latency.** The results gallery renders each
  match's thumbnail by reading its image file — without a proxy, that
  means an HDD seek per result on every search. With proxies, thumbnails
  load from NVMe.
- **Storage is not tight.** Proxies are capped at `MAX_SIDE` (default
  512px) JPEGs — tens of KB each. At ~330k files total that's tens of GB,
  a small fraction of a 1TB drive, even before considering you could
  raise `MAX_SIDE` for nicer thumbnails and still have huge headroom.
- **The proxy is never the answer to "where is this file."** The DB
  tracks the true original `path` separately from `proxy_path`; captions
  in the UI always show the original location. Proxies are a disposable
  cache — safe to delete and regenerate any time by re-running the
  indexer (it'll treat existing DB rows as already-indexed and only
  regenerate proxies that are actually missing... note: if you delete
  proxy files without touching the DB/source mtimes, the indexer's
  skip-check won't notice they're gone, since it only compares source
  size/mtime. Wipe the DB too if you need a clean regenerate).
- **Privacy note:** if `PROXY_DIR_PERSONAL` points at the spare NVMe,
  that drive now holds downsized copies of personal photos. It's a
  performance cache, not a security boundary — worth encrypting that
  drive too (gocryptfs/VeraCrypt) if it's meant to satisfy the same
  privacy bar as everything else here.
- **EXIF orientation is now respected** (`ImageOps.exif_transpose`)
  when generating both proxies and embeddings — phone/camera photos with
  rotation metadata would otherwise get embedded and thumbnailed sideways.

Proxy generation is always on (it's cheap and fixes result-browsing
latency regardless of where it lives) — leaving `PROXY_DIR_GENERAL` /
`PROXY_DIR_PERSONAL` unset in `.env` just means proxies land in a
project-local `./proxy-cache/` folder instead of the NVMe, still
functional, just not on fast storage. Point those two vars at your NVMe
mount to get the actual speed benefit.

## Deleting files from the UI

Right-click any result (NiceGUI UI only) → **Move to trash...**. This is a
two-step, soft-delete-then-empty design, not an immediate hard delete:

1. **Move to trash** — the original file is renamed into `_deleted/` at
   the root of that category's `SOURCE_DIR` (a same-volume rename, so it's
   instant regardless of file size). Its proxy copy (if any) is deleted
   outright — proxies are cheap to regenerate and aren't the thing you're
   trying to protect. Its catalog row is removed immediately, which is
   what actually makes it stop appearing in search results and, on the
   next indexer run, `_deleted/` is skipped entirely so it can't get
   silently re-indexed. The FAISS vector itself is left behind as an
   inert orphan (HNSW doesn't support true deletion) — harmless, since
   results are always joined against the catalog table and a missing row
   means no match, ever. This step is fully reversible by hand: the file
   is still sitting on disk under `_deleted/`, just prefixed with a random
   ID to avoid name collisions between files that shared a filename in
   different folders.
2. **Empty trash** (button next to the index stats, top of the page) —
   the only step that's actually irreversible. Permanently unlinks
   everything currently in `_deleted/`. Confirms before running, and
   nothing is ever auto-emptied — trash only grows until you click it.

This is also why `SOURCE_DIR` is mounted read-write now instead of
read-only (see "Design" above) — moving/unlinking files requires write
access to the volume they live on.

## Files moving/getting reclassified into new folders

**Steps required: re-run the indexer. `docker compose up` alone does
nothing here** -- `up` only starts/serves the UI container, it never
scans the source tree by itself. Moves are picked up by running the
indexer again, same command as any other incremental pass:

```bash
docker compose run --rm imgmatch-general python -m app.indexer
docker compose run --rm imgmatch-personal python -m app.indexer
```

If the UI is already running (`docker compose up -d`), it holds the
FAISS index in memory and won't see the updated vectors/paths until you
restart it afterward:

```bash
docker compose restart imgmatch-general   # or -personal
```

A plain `restart` is genuinely enough *here specifically* -- the indexer
only wrote to files on an already-mounted volume (`INDEX_DIR_*`), no image
involved, so restarting the existing container's process to make it
re-read those files is all that's needed. **This is not true after a code
change.** `docker compose restart` never swaps in a newly built image --
it restarts the *existing* container on whatever image it already had.
After `docker compose build` (app code, Dockerfile, requirements.txt --
anything that changes the image), you need the container recreated from
the new image instead:

```bash
docker compose up -d --force-recreate imgmatch-general   # or -personal
```

(`image-match build` + `image-match restart <category>` via
`powershell-aliases.ps1` already does the right one of these for you --
its `restart` action runs `up -d --force-recreate` specifically because
"restart after a rebuild" is what it's almost always used for in
practice, not the narrower reindex-only case above.)

That's the whole procedure — no manual deleting, no full reindex, no
`.env` changes, nothing else needed for routine file moves/reclassifying.

Handled automatically -- no full reindex needed. Every run re-walks the
whole source tree, and a moved file is recognized by **content signature**
(exact file size + perceptual hash), not by its path:

- If a "new" file's signature matches a DB row whose old path just
  disappeared this run, that row is **repointed to the new path** (same
  `faiss_id`, same vector, no re-embedding) instead of being indexed as a
  fresh duplicate.
- Anything still missing at the end of the run -- not matched to any new
  location -- is genuinely gone (actually deleted, not moved) and gets
  **pruned** from the database automatically.
- Checkpoint/summary log lines report `moved` and `pruned` counts
  alongside `added`, so you can see this happening.

One limitation worth knowing: FAISS's HNSW index type doesn't support true
vector deletion, so a pruned file's vector isn't reclaimed from the
`.faiss` file -- it just becomes inert. This is harmless for correctness:
search results are always joined against the SQLite table, so a pruned
row's vector can never be returned (no row to attach a path/filename to).
It's purely a small amount of unreclaimed disk space, not a behavior bug.
If that ever accumulates enough to matter (heavy, sustained reclassification
churn over a long time), a from-scratch reindex compacts it away -- not a
routine need.

**If you do wipe an index to force a from-scratch rebuild** (e.g. after
changing `CLIP_MODEL`/`CLIP_PRETRAINED` -- a different model usually means a
different vector dimension, which makes the existing `.faiss` file outright
incompatible, not just stale), delete `catalog.sqlite3` **and**
`vectors.faiss` together, never just one. The two have to stay in lockstep:
the indexer computes the next `faiss_id` from whatever rows exist in the
SQLite DB, while the actual vector IDs live in the `.faiss` file. Clear only
one and the id sequence restarts from 0 while the other file still has
those same IDs occupied -- silently corrupting the id↔vector↔row mapping
(wrong vector attached to wrong row) with no error raised to warn you.
Always delete/recreate both files as one atomic step.

Edge case worth knowing about: this matches on `(exact size, perceptual
hash)`, which is reliable in practice but not cryptographically exact --
if you have two genuinely different files that happen to share both the
same byte size and the same phash (extremely unlikely for actually
different content), a stale prune candidate could match against the wrong
new file. Not a practical concern at real-world catalog sizes.

## Bulk matching low-quality images against the catalog

For the "I have a pile of low-res/thumbnail copies and want to find their
high-res originals already sitting in the catalog" workflow -- `app/bulk_match.py`.
Drop the low-quality images into that category's `MATCH_INPUT_DIR`, then:

```bash
docker compose run --rm imgmatch-general python -m app.bulk_match
docker compose run --rm imgmatch-personal python -m app.bulk_match
```

(or `matchfind general` / `matchfind personal` via `powershell-aliases.ps1`)

For each input image, this runs the exact same CLIP search the UI's "Find
matches" uses against that category's index, keeps candidates scoring at
least `MATCH_SIMILARITY_THRESHOLD` (`.env`, default `0.90`) that are also
higher resolution (by total pixel count) than the query, and copies the
single best one into `MATCH_OUTPUT_DIR`. A `match-log.csv` in the output
folder records every input processed -- matched path, similarity, both
resolutions -- with a blank `matched_path` for anything that found nothing
above the threshold, so you know at a glance which inputs still need manual
hunting. Runs are additive: the log is appended to, not overwritten, so
repeat runs across multiple batches build one running audit trail.

**Deliberately manual, not automatic.** This does not run on `docker compose
up`/`restart`, and there's no folder-watcher polling `MATCH_INPUT_DIR` in
the background -- same reasoning as indexing (see above): drop files in, run
the command when you're ready, get a predictable one-shot result. Nothing
about the main catalog/index is touched -- this is pure read-search-copy, so
there's no risk to the existing data even with a bad match.

**0.90, not 0.95+.** This sits below the near-duplicate-grade threshold the
UI's own "similar/duplicate" tagging uses on purpose: a genuine match that's
been cropped differently from the query can score a bit lower on raw CLIP
cosine similarity than an uncropped near-duplicate would, and missing those
defeats the point of this tool. The tradeoff is more false positives to spot
in `match-log.csv` -- since nothing here deletes or moves the low-quality
inputs, a bad match only costs you one stray extra file in the output
folder, not lost data.

Left unset in `.env`, `MATCH_INPUT_DIR_*`/`MATCH_OUTPUT_DIR_*` default to
project-local `./match-input/<category>` and `./match-output/<category>`
folders (same fallback pattern as `PROXY_DIR_*`) -- fine for occasional use;
point them at real drive paths if you want the folders somewhere specific.

## Scaling notes (current: HNSW, CPU search, GPU embedding)

- Fine as-is through the low millions of vectors per category. HNSW needs
  no training step (unlike IVF), so indexing stays simple as it grows.
- If per-query latency becomes a problem at the full ~3M/~2M scale,
  options are: move FAISS search to GPU, shard the index, or switch to
  IVF+PQ for a smaller memory footprint at some recall cost — not needed
  to start.
- The near-duplicate pass currently only compares hashes among the
  top-K semantic candidates returned per query (cheap, no global hash
  index needed). If you also want "find every near-duplicate of X
  anywhere in the catalog" independent of semantic neighborhood, that's
  a separate global hash index (e.g. a BK-tree) — can add later if needed.


## Laptop (Fedora + Podman, Intel Arc GPU)

The laptop runs the same system against its own files -- separate,
independent catalogs, same `general`/`personal` categories and ports.
`laptop/` holds what differs: a `Containerfile` built for the Arc B390
(Fedora 44 base, PyTorch XPU build), a `compose.yml` for `podman compose`
(`/dev/dri` passed in, SELinux `:z` mounts), its own `.env.example`,
`image-match.sh` -- bash versions of `image-match`, `indexer` and `matchfind`
with the same arguments -- and `install.sh`, which installs or updates
everything into `~/image-match` for the user running it.

Runs the model on the Intel GPU in fp16 (`DEVICE=xpu`, `PRECISION=fp16`).
Benchmarked 2026-11-05 on 300 photos, model-only: CPU 1.7 img/s, GPU fp32
11.8, GPU fp16 45.6, with 99.7% identical top-10 search results vs fp32;
the full indexer did 38.7 img/s end to end. The image falls back to CPU by
itself if the GPU isn't visible -- check the `embedder: ... on xpu` line at
the start of indexer output / container logs. `DEVICE=cpu` forces CPU.
`laptop/bench.py` and `laptop/compare.py` re-run the benchmark.

Memory: the GPU shares system RAM. Each container peaks at ~8 GB while
loading the model, then settles to ~2 GB of model plus ~650 MB. If a large
Ollama model is loaded, start one container at a time.

To move photos between machines, copy the original files into the other
machine's source folder and run `indexer` there. Never copy index folders
between machines -- FAISS ids and paths belong to the catalog that built
them.

**Indexer safety (both machines):** if a scan finds 0 images but the
catalog isn't empty, the indexer stops without changing anything -- that's
almost always an unmounted VeraCrypt volume, and pruning would otherwise
wipe the whole catalog and its thumbnails.
