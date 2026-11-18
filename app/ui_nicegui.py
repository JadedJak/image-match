"""NiceGUI version of the local web UI -- same backend (search_engine, db,
embedder, FaissStore) as app/ui.py (Gradio), different frontend. This is the
primary UI now: NiceGUI gives real per-element click/dblclick events and a
native scroll_area.scroll_to() instead of Gradio's js= escape hatch, which
kept fighting us on exactly this kind of fine-grained interactivity (see
app/ui.py's git history / README for the specifics). app/ui.py is kept
working as a fallback -- switch back via UI_ENGINE=gradio in .env, no
rebuild needed, see app/main.py.

Per-browser-tab state (current query image, current result list, current
selection) lives in a plain dict captured in the @ui.page('/') closure --
NiceGUI runs that function fresh per client connection, so tabs don't
interfere with each other. The FaissStore/CLIP model are still one shared
singleton at module level, same as the Gradio version -- read-only search
against them is fine from multiple concurrent clients.
"""
import io
import json
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import UploadFile
from fastapi.responses import FileResponse, Response
from nicegui import app, ui

from . import config, db, embedder
from .faiss_store import FaissStore
from .search_engine import search, search_combined, text_search

db.init_db()
_store = FaissStore(embedder.dim())

# Serve the source volume and proxy cache as static routes so <img> tags can
# reference them by URL -- NiceGUI (like Gradio) won't serve an arbitrary
# local filesystem path directly, it needs an explicit registered route.
app.add_static_files("/img-source", str(config.SOURCE_DIR))
if config.PROXY_DIR:
    app.add_static_files("/img-proxy", str(config.PROXY_DIR))

def _sweep_temp_dir(d: Path):
    """Delete anything left over in a scratch temp dir from a previous run
    of this container. Called once at import time (i.e. container startup)
    for both _UPLOAD_DIR and _DOWNLOAD_TMP_DIR below -- nothing else cleans
    either of them during a session, so without this they'd grow for the
    lifetime of the container's writable layer (which survives
    stop/restart, only wiped on rebuild/recreate) instead of just one run.
    Note this is startup-only, not continuous: files uploaded/downloaded
    since the last restart still sit here until the next one. Best-effort --
    a stray permissions error shouldn't block the app from starting."""
    if not d.exists():
        return
    for p in d.iterdir():
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
        except OSError:
            pass


# Scratch space for freshly-uploaded query images (not yet part of the
# indexed catalog, so they don't live under SOURCE_DIR/PROXY_DIR). Swept on
# every startup (see _sweep_temp_dir) -- still accumulates *within* a run,
# but no longer forever.
_UPLOAD_DIR = Path(tempfile.gettempdir()) / "image-match-uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_sweep_temp_dir(_UPLOAD_DIR)
app.add_static_files("/img-upload", str(_UPLOAD_DIR))

# Scratch space for the "Download medium/small" context-menu options below --
# same lifecycle/sweep treatment as _UPLOAD_DIR above.
_DOWNLOAD_TMP_DIR = Path(tempfile.gettempdir()) / "image-match-downloads"
_DOWNLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)
_sweep_temp_dir(_DOWNLOAD_TMP_DIR)


def _public_url(local_path: str) -> str:
    """Map an absolute local path to whatever static route can serve it --
    PROXY_DIR/SOURCE_DIR get their registered routes above; anything else
    (a freshly uploaded query image sitting in _UPLOAD_DIR) is served via
    its own dedicated route (see drop_upload/handle_page_drop)."""
    p = Path(local_path)
    if config.PROXY_DIR:
        try:
            return f"/img-proxy/{p.relative_to(config.PROXY_DIR).as_posix()}"
        except ValueError:
            pass
    try:
        return f"/img-source/{p.relative_to(config.SOURCE_DIR).as_posix()}"
    except ValueError:
        pass
    try:
        return f"/img-upload/{p.relative_to(_UPLOAD_DIR).as_posix()}"
    except ValueError:
        pass
    return str(local_path)  # last resort, unlikely to actually resolve


def _preview_url(local_path: str) -> str:
    """Like _public_url, but for the hover-preview's dedicated on-the-fly
    resize route (below) instead of a static file -- only meaningful for
    files under SOURCE_DIR. Falls back to _public_url for anything else
    (e.g. an uploaded query image), which just means the preview shows the
    same asset the thumbnail already used, not a resized-from-source copy."""
    p = Path(local_path)
    try:
        return f"/img-preview/{p.relative_to(config.SOURCE_DIR).as_posix()}"
    except ValueError:
        return _public_url(local_path)


@app.get("/img-preview/{rel_path:path}")
def img_preview(rel_path: str):
    """On-the-fly resize of the true source file for the hover-preview --
    the small search-thumbnail proxy (PROXY_QUALITY/MAX_SIDE) looks blurry
    once CSS stretches it to ~90vw, but sending the raw original on every
    hover is wasteful/slow over a tailnet link. Nothing is cached or
    written to disk; this re-decodes+resizes per request (see
    embedder.load_and_downsample's JPEG-draft-mode note for why that's
    still cheap), so cost is only paid while a preview is actually open.
    """
    full_path = (config.SOURCE_DIR / rel_path).resolve()
    try:
        full_path.relative_to(config.SOURCE_DIR.resolve())  # guard against ../.. escaping SOURCE_DIR
    except ValueError:
        return Response(status_code=404)
    if not full_path.is_file():
        # Original gone (moved/deleted since the last index run) -- show
        # the proxy rather than a broken image. Same naming rule as indexer.proxy_path_for.
        proxy = config.PROXY_DIR / (rel_path + ".jpg") if config.PROXY_DIR else None
        if proxy and proxy.is_file():
            return FileResponse(proxy, media_type="image/jpeg")
        return Response(status_code=404)
    img = embedder.load_and_downsample(full_path, max_side=config.PREVIEW_MAX_SIDE)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=config.PREVIEW_QUALITY)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


# Receives a file from either the whole-page drop listener or the "Select
# Image" button's hidden file input (see uploadFile() in main_page() below)
# -- both funnel through this one endpoint. NiceGUI's `app` object is a
# real FastAPI app, so a plain route works normally here.
@app.post("/api/drop-upload")
async def drop_upload(file: UploadFile):
    dest = _UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"path": str(dest), "url": _public_url(str(dest))}


def index_stats() -> str:
    with db.connect() as conn:
        n = db.count(conn)
    return (
        f"**{n}** images indexed in `{config.CATEGORY}` -- {_store.ntotal()} vectors loaded "
        f"(restart this container after reindexing to pick up newly added ones)"
    )


def trash_stats() -> str:
    n, total_bytes = 0, 0
    if config.TRASH_DIR.exists():
        for p in config.TRASH_DIR.iterdir():  # flat -- see request_delete's dest naming
            if p.is_file():
                n += 1
                total_bytes += p.stat().st_size
    mb = total_bytes / (1024 * 1024)
    return f"\U0001F5D1️ **{n}** file(s) in trash ({mb:.1f} MB)"


@ui.page("/")
def main_page():
    ui.dark_mode(True)  # NiceGUI defaults to light; match the old Gradio theme
    ui.colors(primary="#3B4B6B")  # muted slate-blue instead of Quasar's bright default primary

    # Per-tab state -- fresh dict every time a browser connects, see module
    # docstring. result_items holds (true_path, proxy_path_or_None) tuples
    # in the exact order rendered, so a click's index maps back correctly.
    state = {
        "query_path": None, "result_items": [], "result_items_full": [], "selected": None,
        "hover_preview_enabled": True, "preview_highres": False,
        # last_items: the full result set from the most recent query, always
        # in original match-rank order -- never overwritten by a sort
        # toggle (only replaced by a fresh search, or filtered by a delete),
        # so switching sort_mode back to "rank" doesn't require re-querying.
        "last_items": [], "sort_mode": "rank",
        # query_url: whatever's currently shown in the query-image panel --
        # tracked separately here (updated wherever set_query_image()
        # is called) rather than read back off the widget, so a history
        # snapshot doesn't depend on NiceGUI's Image element exposing its
        # current src as a readable property.
        "query_url": None,
        # Box search (see set_query_image): query_full is the full-resolution
        # file behind the query (query_path can be the 512px proxy), which a
        # box is cropped from. crop is the current box as an
        # {x0, y0, x1, y1} dict of 0-1 fractions, or None for the whole image.
        # last_apply_refine: whether the last image search was "Find
        # matches" (refine text applied), so a box re-search repeats it.
        "query_full": None, "crop": None, "last_apply_refine": False,
        # history/future: back/forward navigation stacks (see push_history/
        # go_back/go_forward below). Each entry is a full snapshot of a past
        # view -- query image, its cached result list, sort mode, and the
        # text-search box contents -- so going back/forward never re-queries
        # FAISS/the DB, just redraws from what's already in memory.
        "history": [], "future": [],
    }

    # Tightens every ui.markdown() on the page (title + both stats lines
    # below) -- NiceGUI wraps markdown output in a plain <p>, which picks up
    # the browser's default ~1em top/bottom paragraph margin. That default
    # was most of the header block's height; this + swapping the title from
    # an "## " h2 (2em font, its own margin on top of the <p>'s) down to
    # inline bold text is what gets the ~75% reduction requested.
    ui.add_head_html("<style>.nicegui-markdown p { margin: 0; }</style>")

    with ui.row().classes("w-full items-center justify-between py-1"):
        ui.markdown(f"**Image Match** — `{config.CATEGORY}` catalog").classes("text-base")
        with ui.row().classes("items-center gap-4"):
            stats_md = ui.markdown(index_stats()).classes("text-xs")
            trash_stats_md = ui.markdown(trash_stats()).classes("text-xs")

            def refresh_all_stats():
                stats_md.set_content(index_stats())
                trash_stats_md.set_content(trash_stats())

            ui.button("↻ Refresh stats", on_click=refresh_all_stats).props("outline size=sm")

            # The actual on/off state lives client-side (window.__imageMatchHoverEnabled,
            # checked by the mouseover listener below) since that's where the
            # decision needs to be made on every hover with no server round
            # trip -- this button just flips that flag and relabels itself.
            # state["hover_preview_enabled"] is only the server-side mirror of
            # it, so the label is right on first render.
            def toggle_hover_preview():
                state["hover_preview_enabled"] = not state["hover_preview_enabled"]
                enabled = state["hover_preview_enabled"]
                hover_preview_btn.set_text(f"\U0001F5BC️ Hover preview: {'ON' if enabled else 'OFF'}")
                js_bool = "true" if enabled else "false"
                ui.run_javascript(
                    f"window.__imageMatchHoverEnabled = {js_bool};"
                    f"if (!{js_bool}) {{ const o = document.getElementById('image-match-hover-preview'); if (o) o.style.display = 'none'; }}"
                )

            hover_preview_btn = ui.button("\U0001F5BC️ Hover preview: ON", on_click=toggle_hover_preview).props("outline size=sm")

            # Same client-side-source-of-truth pattern as hover_preview_enabled
            # above (window.__imageMatchPreviewHighRes, checked in resolveUrl()
            # below) -- defaults to the proxy (small, fast, already loaded --
            # no extra request) rather than the /img-preview resize (sharp, but
            # a real decode+resize+network round trip per hover), since that's
            # the safer default over a slower tailnet link. Users on a fast
            # local connection can flip it.
            def toggle_preview_quality():
                state["preview_highres"] = not state["preview_highres"]
                highres = state["preview_highres"]
                preview_quality_btn.set_text(f"\U0001F50D Preview quality: {'High-res' if highres else 'Proxy'}")
                js_bool = "true" if highres else "false"
                ui.run_javascript(f"window.__imageMatchPreviewHighRes = {js_bool};")

            preview_quality_btn = ui.button("\U0001F50D Preview quality: Proxy", on_click=toggle_preview_quality).props("outline size=sm")

            # Re-sorts whatever's currently displayed -- reads from
            # state["last_items"] (the untouched, original match-rank order
            # from the last query) rather than state["result_items_full"]
            # (which render_gallery overwrites with whatever order was last
            # *displayed*), so toggling back to "Match rank" is always
            # exact, not a re-sort of an already-sorted list.
            SORT_MODES = ["rank", "name", "date"]
            SORT_LABELS = {"rank": "Match rank", "name": "Filename", "date": "Date (newest first)"}

            def sorted_for_display():
                items = state["last_items"]
                if state["sort_mode"] == "name":
                    return sorted(items, key=lambda it: Path(it[0]).name.lower())
                if state["sort_mode"] == "date":
                    # mtime is the DB's own recorded file mtime (see
                    # search_engine.py), not anything parsed out of the
                    # filename -- unlike the "Filename" sort, this sorts
                    # correctly even for files that don't follow a
                    # date-prefixed naming convention. Descending (newest
                    # first), unlike "Filename"'s ascending order.
                    return sorted(items, key=lambda it: it[3], reverse=True)
                return items

            def toggle_sort():
                next_mode = SORT_MODES[(SORT_MODES.index(state["sort_mode"]) + 1) % len(SORT_MODES)]
                state["sort_mode"] = next_mode
                sort_btn.set_text(f"⇅ Sort: {SORT_LABELS[next_mode]}")
                render_gallery(sorted_for_display())

            sort_btn = ui.button("⇅ Sort: Match rank", on_click=toggle_sort).props("outline size=sm")

            # Back/Forward: browser-history-style navigation across query
            # "hops" -- image A's results -> double-click image B to search
            # similar to it -> Back returns to A's results (no re-query,
            # restored straight from state["history"]) -> Forward returns to
            # B's results again. Covers both image-similarity chains and
            # text searches, since both funnel through push_history() below
            # before they replace what's on screen.
            def capture_snapshot():
                return {
                    "query_url": state["query_url"],
                    "query_path": state["query_path"],
                    "query_full": state["query_full"],
                    "crop": state["crop"],
                    "last_items": state["last_items"],
                    "sort_mode": state["sort_mode"],
                    "keyword": keyword_input.value,
                    "literal": literal_checkbox.value,
                    "refine_text": refine_input.value,
                    "refine_weight": refine_weight_slider.value,
                }

            def restore_snapshot(snap):
                state["query_url"] = snap["query_url"]
                state["query_path"] = snap["query_path"]
                state["query_full"] = snap["query_full"]
                if snap["query_url"]:
                    set_query_image(snap["query_url"], snap["crop"])
                state["last_items"] = snap["last_items"]
                state["sort_mode"] = snap["sort_mode"]
                sort_btn.set_text(f"⇅ Sort: {SORT_LABELS[state['sort_mode']]}")
                keyword_input.set_value(snap["keyword"] or "")
                literal_checkbox.set_value(bool(snap["literal"]))
                refine_input.set_value(snap.get("refine_text") or "")
                refine_weight_slider.set_value(snap.get("refine_weight", 0.6))
                render_gallery(sorted_for_display())
                update_nav_buttons()

            def update_nav_buttons():
                # Dim via opacity rather than Quasar's actual `disable` prop --
                # `disable` is a boolean HTML/Quasar attribute, where *presence*
                # means disabled regardless of the value assigned to it,
                # including a literal "false" string. That made an earlier
                # version of this (toggling it via .enable()/.disable()) get
                # stuck permanently disabled -- not-allowed cursor, unclickable
                # -- the moment it was set once. go_back()/go_forward() already
                # no-op safely on an empty stack, so an actual disable was only
                # ever cosmetic; opacity gets the same visual without the trap.
                back_btn.set_text(f"◀ Back ({len(state['history'])})")
                forward_btn.set_text(f"▶ Forward ({len(state['future'])})")
                if state["history"]:
                    back_btn.classes(remove="opacity-40")
                else:
                    back_btn.classes(add="opacity-40")
                if state["future"]:
                    forward_btn.classes(remove="opacity-40")
                else:
                    forward_btn.classes(add="opacity-40")

            def push_history():
                # Called at the top of every action that's about to replace
                # the current view with a genuinely new search (fresh
                # upload, "search similar", or a text search) -- NOT called
                # by go_back/go_forward themselves, which move between
                # already-captured snapshots instead of creating new ones.
                # Skips the very first search (nothing on screen yet worth
                # preserving). Clears "future" the same way a browser does:
                # navigating somewhere new invalidates the old forward path.
                if state["last_items"] or state["query_url"]:
                    state["history"].append(capture_snapshot())
                    state["future"].clear()
                    update_nav_buttons()

            def go_back():
                if not state["history"]:
                    return
                state["future"].append(capture_snapshot())
                restore_snapshot(state["history"].pop())

            def go_forward():
                if not state["future"]:
                    return
                state["history"].append(capture_snapshot())
                restore_snapshot(state["future"].pop())

            back_btn = ui.button("◀ Back (0)", on_click=go_back).props("outline size=sm").classes("opacity-40")
            forward_btn = ui.button("▶ Forward (0)", on_click=go_forward).props("outline size=sm").classes("opacity-40")

            ui.button("\U0001F5D1️ Empty trash", color="negative", on_click=lambda: empty_trash_dialog.open()).props("outline size=sm")

    # flex-col md:flex-row (not a permanent row) -- stacks the sidebar above
    # the results grid on narrow/mobile viewports instead of forcing both
    # into a squeezed side-by-side row; switches to side-by-side at Tailwind's
    # md breakpoint (768px) same as the sidebar column's own width class below.
    with ui.row().classes("w-full items-start gap-6 flex-col md:flex-row"):
        with ui.column().classes("w-full md:w-80 md:shrink-0 gap-2"):
            # Fixed at the old default size (320x320) -- no longer
            # resizable, since the old adjustable slider was removed; 320
            # is what settled as the size that actually gets used.
            # Box search: drag on the query image to search only that part
            # of it (Yandex-style). A plain <img> inside a wrapper sized to
            # the image itself, rather than ui.image (Quasar QImg), so the
            # box's percentages line up exactly with the picture and not
            # with letterboxing. The box is a selection on top of the
            # query, never a replacement: the original stays loaded, and
            # every move/resize re-searches the same photo. All dragging
            # happens client-side (script below); only the finished box
            # (0-1 fractions) comes back, via the crop_changed event.
            ui.add_head_html("""
            <style>
            .im-crop-frame { width: 320px; height: 320px; display: flex; align-items: center; justify-content: center; }
            .im-crop-wrap { position: relative; display: inline-block; line-height: 0; overflow: hidden;
                            touch-action: none; cursor: crosshair; user-select: none; }
            .im-crop-wrap img { display: block; max-width: 320px; max-height: 320px; -webkit-user-drag: none; }
            .im-crop-wrap img:not([src]) { display: none; }
            .im-crop-box { position: absolute; display: none; box-sizing: border-box; cursor: move;
                           border: 2px solid #fff; outline: 1px solid rgba(0,0,0,0.6);
                           box-shadow: 0 0 0 9999px rgba(0,0,0,0.5); }
            .im-crop-handle { position: absolute; width: 8px; height: 8px; background: #fff;
                              outline: 1px solid rgba(0,0,0,0.6); }
            .im-crop-handle[data-handle=nw] { left: 0; top: 0; cursor: nwse-resize; }
            .im-crop-handle[data-handle=ne] { right: 0; top: 0; cursor: nesw-resize; }
            .im-crop-handle[data-handle=sw] { left: 0; bottom: 0; cursor: nesw-resize; }
            .im-crop-handle[data-handle=se] { right: 0; bottom: 0; cursor: nwse-resize; }
            </style>
            """)
            with ui.element("div").classes("im-crop-frame bg-grey-9 rounded"):
                with ui.element("div").classes("im-crop-wrap"):
                    query_img = ui.element("img").props("draggable=false")
                    with ui.element("div").classes("im-crop-box"):
                        for corner in ("nw", "ne", "sw", "se"):
                            ui.element("div").classes("im-crop-handle").props(f"data-handle={corner}")
            ui.add_body_html(r"""
            <script>
            (function() {
                if (window.__imCropBound) return;
                window.__imCropBound = true;
                window.__imCropBox = null;  // {x0, y0, x1, y1} as 0-1 fractions, or null
                let drag = null;

                function parts() {
                    const wrap = document.querySelector('.im-crop-wrap');
                    return wrap && {wrap, box: wrap.querySelector('.im-crop-box'), img: wrap.querySelector('img')};
                }
                function draw() {
                    const p = parts();
                    if (!p) return;
                    const b = window.__imCropBox;
                    if (!b) { p.box.style.display = 'none'; return; }
                    Object.assign(p.box.style, {
                        display: 'block', left: b.x0 * 100 + '%', top: b.y0 * 100 + '%',
                        width: (b.x1 - b.x0) * 100 + '%', height: (b.y1 - b.y0) * 100 + '%',
                    });
                }
                // Python sets/clears the box with this (new query image, Back/Forward).
                window.__imCropSet = function(b) { window.__imCropBox = b; draw(); };

                const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
                function point(p, ev) {
                    const r = p.wrap.getBoundingClientRect();
                    return {x: clamp((ev.clientX - r.left) / r.width, 0, 1), y: clamp((ev.clientY - r.top) / r.height, 0, 1)};
                }
                function rect(a, b) {
                    return {x0: Math.min(a.x, b.x), y0: Math.min(a.y, b.y), x1: Math.max(a.x, b.x), y1: Math.max(a.y, b.y)};
                }

                // Small images are scaled up to fill the 320px frame, same as the old QImg did.
                document.addEventListener('load', (ev) => {
                    const img = ev.target;
                    if (!(img instanceof HTMLImageElement) || !img.closest('.im-crop-wrap')) return;
                    const s = Math.min(320 / img.naturalWidth, 320 / img.naturalHeight);
                    img.style.width = Math.round(img.naturalWidth * s) + 'px';
                    img.style.height = Math.round(img.naturalHeight * s) + 'px';
                }, true);

                document.addEventListener('pointerdown', (ev) => {
                    const p = parts();
                    if (!p || !p.wrap.contains(ev.target) || !p.img.getAttribute('src') || ev.button !== 0) return;
                    ev.preventDefault();
                    const at = point(p, ev);
                    const b = window.__imCropBox;
                    const handle = ev.target.dataset && ev.target.dataset.handle;
                    if (b && handle) {
                        // the corner opposite the grabbed one stays put
                        const fixed = {x: handle.includes('w') ? b.x1 : b.x0, y: handle.includes('n') ? b.y1 : b.y0};
                        drag = {mode: 'resize', fixed, prev: b};
                    } else if (b && p.box.contains(ev.target)) {
                        drag = {mode: 'move', start: at, prev: b};
                    } else {
                        drag = {mode: 'new', start: at, prev: b};
                    }
                    p.wrap.setPointerCapture(ev.pointerId);
                });
                document.addEventListener('pointermove', (ev) => {
                    if (!drag) return;
                    const p = parts();
                    const at = point(p, ev);
                    if (drag.mode === 'new') {
                        window.__imCropBox = rect(drag.start, at);
                    } else if (drag.mode === 'resize') {
                        window.__imCropBox = rect(drag.fixed, at);
                    } else {
                        const o = drag.prev;
                        const dx = clamp(at.x - drag.start.x, -o.x0, 1 - o.x1);
                        const dy = clamp(at.y - drag.start.y, -o.y0, 1 - o.y1);
                        window.__imCropBox = {x0: o.x0 + dx, y0: o.y0 + dy, x1: o.x1 + dx, y1: o.y1 + dy};
                    }
                    draw();
                });
                function finish() {
                    if (!drag) return;
                    const d = drag;
                    drag = null;
                    const p = parts();
                    const b = window.__imCropBox;
                    const r = p.wrap.getBoundingClientRect();
                    // A plain click (or a sliver) isn't a box -- put back whatever was there.
                    if (!b || (b.x1 - b.x0) * r.width < 10 || (b.y1 - b.y0) * r.height < 10) {
                        window.__imCropSet(d.prev);
                        return;
                    }
                    const same = d.prev && ['x0', 'y0', 'x1', 'y1'].every(k => Math.abs(d.prev[k] - b[k]) < 1e-4);
                    if (!same) emitEvent('crop_changed', b);
                }
                document.addEventListener('pointerup', finish);
                document.addEventListener('pointercancel', finish);
            })();
            </script>
            """)

            ui.label("Drag on the image to search just that part of it.").classes("text-xs text-grey")
            whole_image_btn = ui.button("⬚ Clear box (search whole image)").props("outline size=sm").classes("w-full")

            def set_query_image(url, crop=None):
                """Show url in the query panel and set (or clear) its box."""
                query_img.props["src"] = url
                query_img.update()
                state["crop"] = crop
                ui.run_javascript(f"window.__imCropSet && window.__imCropSet({json.dumps(crop)})")

            # Replaced Quasar's QUploader widget with a plain button that
            # triggers the browser's native file-picker dialog -- three
            # rounds of fighting QUploader's internal styling (text color,
            # then box height, neither of which actually took effect) made
            # clear it wasn't worth it for what's fundamentally just "pick a
            # file". The hidden <input type="file"> + button.click() pattern
            # below reuses the exact same upload pipeline already built for
            # whole-page drag-and-drop (/api/drop-upload -> emitEvent
            # page_drop -> handle_page_drop, see uploadFile() in the script
            # block further down) -- selecting a file this way and dropping
            # one anywhere on the page now go through identical code.
            # color="primary" + text-color=black (not "outline") matches
            # find_btn/text_btn further down -- every full-width primary
            # action button in the sidebar (this one, Search similar,
            # Copy, Find matches, Search text) now shares one consistent
            # filled style. The header row's small utility toggles
            # (Refresh stats, Hover preview, Sort, Back/Forward, etc.) stay
            # outline on purpose -- a distinct, more compact "toolbar"
            # category, not a primary action.
            ui.button("\U0001F4C1 Select Image", color="primary", on_click=lambda: ui.run_javascript(
                "document.getElementById('image-match-file-input').click()"
            )).classes("w-full").props("text-color=black")

            def handle_page_drop(e):
                # Payload comes from the JS uploadFile() helper below via
                # emitEvent -- {"path": ..., "url": ...} from drop_upload(),
                # fired by either a page-wide drop or the "Select Image"
                # button's file input. Not a push_history() point -- this
                # only stages a new query image, it doesn't touch what's in
                # the results grid yet (that only happens on "Find
                # matches", which is where the actual navigation happens).
                state["query_path"] = e.args["path"]
                state["query_full"] = e.args["path"]
                state["query_url"] = e.args["url"]
                set_query_image(state["query_url"])
                # A fresh drop/upload replaces the query outright -- clear
                # any previously clicked gallery result too, or "Search
                # similar to selected" (which checks state["selected"]
                # before state["query_path"]) keeps chaining off the old
                # click instead of the file that was just dropped.
                state["selected"] = None
                selected_path_box.set_value("")

            ui.on("page_drop", handle_page_drop)

            # navigator.clipboard is only defined in "secure contexts"
            # (HTTPS, or http://localhost) -- this app is served plain HTTP
            # over the Tailscale IP, so navigator.clipboard is undefined
            # there and the old writeText() call threw immediately, before
            # ever reaching the clipboard. It failed silently because
            # ui.run_javascript() is fire-and-forget from the Python side,
            # so the "Path copied" toast fired regardless of whether the JS
            # actually succeeded. document.execCommand('copy') is
            # deprecated but still works from plain HTTP as long as it runs
            # inside a real user-gesture handler (a button click, which
            # this always is), so it's used unconditionally as a reliable
            # fallback rather than only when the Clipboard API is missing.
            ui.add_body_html("""
            <script>
            window.__imageMatchCopyText = function(text) {
                if (window.isSecureContext && navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(text).catch(() => window.__imageMatchCopyFallback(text));
                } else {
                    window.__imageMatchCopyFallback(text);
                }
            };
            window.__imageMatchCopyFallback = function(text) {
                const ta = document.createElement('textarea');
                ta.value = text;
                ta.style.position = 'fixed';
                ta.style.left = '-9999px';
                document.body.appendChild(ta);
                ta.focus();
                ta.select();
                try { document.execCommand('copy'); } catch (e) { /* nothing more to try */ }
                document.body.removeChild(ta);
            };
            </script>
            """)

            # Whole-page drag-and-drop, plus the hidden native file input
            # the "Select Image" button above triggers -- both funnel
            # through the same uploadFile() helper: POST to our own
            # /api/drop-upload endpoint, then emitEvent (a NiceGUI-provided
            # JS helper) to trigger handle_page_drop above with the result.
            # That's NiceGUI's documented JS-calls-Python pattern, distinct
            # from ui.run_javascript (Python-calls-JS), which is what the
            # button uses to open the file picker.
            # TEMPORARY console.log calls at every step -- purely to find
            # out exactly where this breaks if it doesn't work, rather than
            # guessing blind again. Remove once confirmed working.
            ui.add_body_html("""
            <input type="file" id="image-match-file-input" accept="image/*" style="display:none">
            <script>
            (function() {
                console.log('upload-js: script block ran');
                if (window.__imageMatchDropBound) { console.log('upload-js: already bound, skipping'); return; }
                window.__imageMatchDropBound = true;
                console.log('upload-js: listeners bound');

                async function uploadFile(file) {
                    if (!file) { console.log('upload-js: no file given'); return; }
                    console.log('upload-js: uploading file', file.name);
                    const formData = new FormData();
                    formData.append('file', file);
                    const resp = await fetch('/api/drop-upload', {method: 'POST', body: formData});
                    console.log('upload-js: upload response status', resp.status);
                    const data = await resp.json();
                    console.log('upload-js: emitting page_drop with', data);
                    emitEvent('page_drop', data);
                }

                document.addEventListener('dragover', (e) => e.preventDefault());
                document.addEventListener('drop', (e) => {
                    console.log('upload-js: drop event fired on', e.target);
                    e.preventDefault();
                    uploadFile(e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]);
                });

                document.getElementById('image-match-file-input').addEventListener('change', (e) => {
                    console.log('upload-js: file input changed');
                    uploadFile(e.target.files && e.target.files[0]);
                    e.target.value = '';  // reset so picking the same file again still fires 'change'
                });
            })();
            </script>
            """)

            # Hover-preview: pause over any gallery thumbnail for 1s and see
            # it at its real (non-square) aspect ratio -- the grid crops
            # everything into a square via CSS (aspect-square object-contain
            # letterboxes it, doesn't reflow the layout), which is fine for
            # scanning a wall of results but hides the actual shape. One
            # delegated listener on `document` (not per-card) so this keeps
            # working across every results_grid.clear()+rebuild without
            # re-binding -- matches the pattern used for whole-page drop
            # above. Purely visual/client-side: reuses whatever URL is
            # already loaded in the hovered <img>, no extra network fetch,
            # no round trip to Python. pointer-events:none on the overlay
            # means it can only ever be dismissed by moving the mouse off
            # the thumbnail (or scrolling/clicking) -- there's nothing to
            # click on it since it's not meant to be interactive.
            ui.add_body_html(r"""
            <style>
            #image-match-hover-preview {
                position: fixed; inset: 0; display: none;
                align-items: center; justify-content: center;
                background: rgba(0,0,0,0.75); z-index: 9999; pointer-events: none;
            }
            #image-match-hover-preview img {
                width: 90vw; height: 90vh; object-fit: contain;
                border-radius: 8px; box-shadow: 0 8px 40px rgba(0,0,0,0.6);
            }
            #image-match-hover-preview .hp-caption {
                position: absolute; bottom: 4vh; color: #ccc; font-size: 0.85rem;
                text-align: center; max-width: 80vw; overflow-wrap: break-word;
            }
            </style>
            <div id="image-match-hover-preview">
                <img id="image-match-hover-preview-img">
                <div class="hp-caption" id="image-match-hover-preview-caption"></div>
            </div>
            <script>
            (function() {
                console.log('hover-js: script block ran');
                if (window.__imageMatchHoverBound) { console.log('hover-js: already bound, skipping'); return; }
                window.__imageMatchHoverBound = true;
                // Only set a default the first time -- toggle_hover_preview()
                // (Python side, via ui.run_javascript) may have already set
                // this before this IIFE re-runs on a later page load, and we
                // don't want to stomp a value that isn't actually a fresh
                // page load's default.
                if (window.__imageMatchHoverEnabled === undefined) window.__imageMatchHoverEnabled = true;
                // Fixed at 2s -- the adjustable delay slider was removed
                // (the on/off toggle above covers the main use case for
                // wanting to change this), so this constant is now the only
                // place the delay is ever set.
                if (window.__imageMatchHoverDelayMs === undefined) window.__imageMatchHoverDelayMs = 2000;
                // Matches the "Preview quality" button's default (Proxy) --
                // see toggle_preview_quality() (Python side).
                if (window.__imageMatchPreviewHighRes === undefined) window.__imageMatchPreviewHighRes = false;
                console.log('hover-js: listeners bound');

                const overlay = document.getElementById('image-match-hover-preview');
                const overlayImg = document.getElementById('image-match-hover-preview-img');
                const overlayCaption = document.getElementById('image-match-hover-preview-caption');
                let showTimer = null;
                let hideTimer = null;
                let currentTarget = null;

                // Robust against NiceGUI/Quasar rendering the thumbnail as a
                // literal <img> OR as a wrapper div around one OR (older
                // Quasar QImg behavior) as a div using a CSS background-image
                // instead of a real <img> at all -- try all three rather
                // than assuming which one this NiceGUI version does.
                function resolveUrl(el) {
                    // data-preview-src (set per-card in render_gallery) points at
                    // the on-the-fly resize of the true source, not the small
                    // search-thumbnail proxy -- only used when the "Preview
                    // quality" button is set to High-res; Proxy (the default)
                    // falls through to the thumbnail's own already-loaded src,
                    // no extra request at all.
                    if (window.__imageMatchPreviewHighRes && el.dataset && el.dataset.previewSrc) return el.dataset.previewSrc;
                    if (el.tagName === 'IMG') return el.currentSrc || el.src || null;
                    const inner = el.querySelector('img');
                    if (inner && (inner.currentSrc || inner.src)) return inner.currentSrc || inner.src;
                    const bgHost = el.matches('[style*="background-image"]') ? el : el.querySelector('[style*="background-image"]');
                    if (bgHost) {
                        const m = /url\((['"]?)(.*?)\1\)/.exec(bgHost.style.backgroundImage || '');
                        if (m) return m[2];
                    }
                    return null;
                }

                function show(el) {
                    const url = resolveUrl(el);
                    console.log('hover-js: show() resolved url =', url, 'from element', el);
                    if (!url) return;
                    overlayImg.src = url;
                    const card = el.closest('.cursor-pointer');
                    const label = card ? card.querySelector('.text-xs') : null;
                    overlayCaption.textContent = label ? label.textContent : '';
                    overlay.style.display = 'flex';
                }
                function hide() {
                    overlay.style.display = 'none';
                    overlayImg.src = '';
                }

                document.addEventListener('mouseover', (e) => {
                    if (!window.__imageMatchHoverEnabled) return;
                    const el = e.target.closest('.hover-thumb');
                    if (!el) return;
                    // Cancel any pending grace-period hide -- this is what
                    // absorbs the spurious mouseout+mouseover pairs fired
                    // when the cursor crosses onto Quasar's internal image
                    // element. Turned out that element isn't reliably
                    // detectable as a DOM descendant via relatedTarget/
                    // contains() (confirmed empirically, logged false for
                    // what looked like a nested <img>), so rather than keep
                    // chasing the exact DOM relationship, just don't act on
                    // any single mouseout immediately -- only a leave that
                    // sticks for hideDelayMs with no follow-up mouseover
                    // counts as a real leave.
                    clearTimeout(hideTimer);
                    if (el === currentTarget) return;  // already hovering this one
                    currentTarget = el;
                    hide();  // don't keep showing the previous thumbnail's preview while this one's delay elapses
                    clearTimeout(showTimer);
                    showTimer = setTimeout(() => show(el), window.__imageMatchHoverDelayMs || 2000);
                });
                document.addEventListener('mouseout', (e) => {
                    const el = e.target.closest('.hover-thumb');
                    if (!el || el !== currentTarget) return;
                    clearTimeout(showTimer);
                    const hideDelayMs = 200;
                    hideTimer = setTimeout(() => {
                        currentTarget = null;
                        hide();
                    }, hideDelayMs);
                });
                // Don't let a stale preview linger through unrelated interactions.
                document.addEventListener('scroll', hide, true);
                document.addEventListener('click', hide, true);
            })();
            </script>
            """)

            ui.label("Max results").classes("text-xs text-grey")
            top_k_slider = ui.slider(min=300, max=1000, value=500, step=50).props("label-always")

            ui.label("Near-duplicate sensitivity (lower = stricter)").classes("text-xs text-grey")
            ham_slider = ui.slider(min=0, max=20, value=config.NEAR_DUP_HAMMING_THRESHOLD, step=1).props("label-always")

            # Moved here from its old spot right below the Selected Image
            # box -- on_click is still wired up at the bottom of
            # main_page() alongside the other buttons, once
            # use_selected_as_query_and_search is actually defined. Same
            # filled color="primary" + text-color=black style as
            # find_btn/text_btn/Select Image above -- see the comment on
            # Select Image for why.
            similar_btn = ui.button("\U0001F50E Search similar to selected", color="primary").classes("w-full").props("text-color=black")

            ui.separator()
            # Optional: nudges the image search toward a plain-language
            # detail as well -- e.g. a house photo + "blue sky" biases
            # ranking toward similar houses that also look sunny. Blends
            # the image and text CLIP vectors into one query vector (see
            # search_engine.search_combined) rather than filtering results
            # after the fact -- a soft nudge, not a guarantee every result
            # matches the text, since CLIP's shared embedding space doesn't
            # give a hard AND. Only applied when this box is non-empty AND
            # "Find matches" is the button clicked -- "Search similar to
            # selected" (and its double-click shortcut) deliberately ignore
            # whatever's sitting in this box, since that's a different
            # action the user didn't ask to blend.
            refine_input = ui.input(label="Refine with text (optional)", placeholder="e.g. blue sky").classes("w-full")
            ui.label("Image weight vs. refine text (left = more text, right = more image)").classes("text-xs text-grey")
            refine_weight_slider = ui.slider(min=0.0, max=1.0, value=0.6, step=0.05).props("label-always")

            find_btn = ui.button("Find matches", color="primary").classes("w-full").props("text-color=black")

            ui.separator()
            keyword_input = ui.input(label="Search by description or filename/path").classes("w-full")
            literal_checkbox = ui.checkbox("Match literally by filename/path (instead of by description)").classes("text-grey")
            text_btn = ui.button("Search text", color="primary").classes("w-full").props("text-color=black")

            ui.separator()
            selected_path_box = ui.textarea(label="Selected result — full path").props("readonly outlined rows=3").classes("w-full")

            def copy_selected_path():
                if state["selected"]:
                    ui.run_javascript(f"window.__imageMatchCopyText({json.dumps(state['selected'][0])})")
                    ui.notify("Path copied", type="positive")

            ui.button("\U0001F4CB Copy", color="primary", on_click=copy_selected_path).classes("w-full").props("text-color=black")

        with ui.column().classes("w-full flex-1 min-w-0"):
            results_area = ui.scroll_area().classes("w-full h-[85vh]")
            with results_area:
                # columns=5 (fixed) replaced with responsive Tailwind
                # grid-cols-* breakpoints instead -- a hardcoded 5-wide grid
                # left almost nothing per thumbnail on a phone-width screen.
                results_grid = ui.grid().classes("w-full gap-2 grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5")

    # Shared delete-confirmation dialog, reused for every thumbnail -- one
    # dialog + one pending-index slot in state, rather than a dialog per
    # card. "Delete" here is a soft delete: the source file is moved into
    # TRASH_DIR (config.py), not unlinked, so it's recoverable by hand
    # until "Empty trash" is used. The proxy copy (if any) and the catalog
    # row are removed immediately though -- proxies are cheap/regenerable,
    # and removing the row is what actually makes the file stop surfacing
    # in search, which is the whole point of "delete" here. This does NOT
    # touch the FAISS index: HNSW vectors can't be removed in place anyway
    # (see db.delete_by_path), so the vector is simply left as an inert
    # orphan that can never surface again once its row is gone (search
    # results are always joined against the files table). No reindex/
    # rebuild needed for any of this to take effect immediately.
    with ui.dialog() as confirm_dialog, ui.card():
        confirm_label = ui.label().classes("text-sm")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=confirm_dialog.close).props("flat")
            ui.button("Move to trash", color="negative", on_click=lambda: do_delete()).props("flat")

    # Separate dialog for the header's "Empty trash" button -- this one IS
    # the irreversible step (permanently unlinks everything in TRASH_DIR).
    with ui.dialog() as empty_trash_dialog, ui.card():
        ui.label("Permanently erase every file currently in the trash? This cannot be undone.").classes("text-sm")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=empty_trash_dialog.close).props("flat")
            ui.button("Empty trash", color="negative", on_click=lambda: do_empty_trash()).props("flat")

    def render_gallery(items):
        """items: list of (true_path, proxy_path_or_None, caption, mtime)
        tuples, already in the order they should display -- both search()
        and text_search() results and plain db rows get normalized to this
        shape by their respective callers below before this runs. mtime
        rides along purely for sorted_for_display()'s date mode -- it's
        never shown in the card itself."""
        state["result_items"] = [(p, pp) for p, pp, _, _ in items]
        state["result_items_full"] = items  # keeps captions around so delete can re-render without re-querying
        results_grid.clear()
        with results_grid:
            for idx, (true_path, proxy_path, caption, _mtime) in enumerate(items):
                thumb_url = _public_url(proxy_path or true_path)
                preview_url = _preview_url(true_path)
                with ui.card().tight().classes("cursor-pointer hover:shadow-lg") as card:
                    ui.image(thumb_url).classes("w-full aspect-square object-contain bg-grey-9 hover-thumb") \
                        .props(f'data-preview-src="{preview_url}"')
                    ui.label(caption).classes("text-xs p-1 break-words")
                    with ui.context_menu():
                        ui.menu_item("\U0001F50E Search similar", on_click=lambda i=idx: search_similar_to(i))
                        ui.menu_item("\U0001F4CB Copy path", on_click=lambda i=idx: copy_path_for(i))
                        ui.menu_item("\U0001F5BC️ Open full-size original", on_click=lambda i=idx: open_full_image(i))
                        ui.menu_item("⬇️ Download source", on_click=lambda i=idx: download_source(i))
                        ui.menu_item("⬇️ Download medium (1600px)", on_click=lambda i=idx: download_resized(i, config.DOWNLOAD_MEDIUM_MAX_SIDE, "medium"))
                        ui.menu_item("⬇️ Download small (800px)", on_click=lambda i=idx: download_resized(i, config.DOWNLOAD_SMALL_MAX_SIDE, "small"))
                        ui.separator()
                        ui.menu_item("\U0001F5D1️ Move to trash...", on_click=lambda i=idx: request_delete(i)).classes("text-red")
                card.on("click", lambda i=idx: select_item(i))
                card.on("dblclick", lambda i=idx: search_similar_to(i))
        results_area.scroll_to(percent=0)  # land on the closest match, no manual scrolling needed

    def select_item(i):
        if i >= len(state["result_items"]):
            return
        path, proxy_path = state["result_items"][i]
        state["selected"] = (path, proxy_path)
        selected_path_box.set_value(path)

    def copy_path_for(i):
        select_item(i)
        copy_selected_path()

    def open_full_image(i):
        # Opens the true source file (not the downsized proxy) in a new
        # browser tab. An in-page dialog was tried instead but the image
        # never rendered inside it; reverted since results already scale
        # well to the browser window, so a dedicated full-size view isn't
        # pulling much weight anyway.
        if i >= len(state["result_items"]):
            return
        true_path, _ = state["result_items"][i]
        ui.run_javascript(f"window.open({json.dumps(_public_url(true_path))}, '_blank')")

    def download_source(i):
        # ui.download reads the local file server-side and triggers a real
        # browser "Save As" -- unlike open_full_image, which just opens the
        # source's URL (browsers often display an image inline rather than
        # downloading it when navigated to directly). filename= keeps the
        # true original name, not whatever the proxy/upload path looks like.
        if i >= len(state["result_items"]):
            return
        true_path, _ = state["result_items"][i]
        ui.download(true_path, filename=Path(true_path).name)

    def download_resized(i, max_side, suffix):
        # Same idea as download_source, but resized via the same decode path
        # the hover-preview route uses (embedder.load_and_downsample --
        # JPEG-draft-mode decode, EXIF-safe, never upscales). The resized
        # copy has to exist as a real file for ui.download to serve it, so
        # it's written to a uniquely-named temp file first under
        # _DOWNLOAD_TMP_DIR; nothing under SOURCE_DIR or PROXY_DIR is
        # touched. Left on disk after serving -- swept at next container
        # startup by _sweep_temp_dir, not deleted per-request (deleting
        # right after ui.download() would race the browser's fetch).
        if i >= len(state["result_items"]):
            return
        true_path, _ = state["result_items"][i]
        img = embedder.load_and_downsample(true_path, max_side=max_side)
        stem = Path(true_path).stem
        tmp_path = _DOWNLOAD_TMP_DIR / f"{stem}_{suffix}_{uuid.uuid4().hex}.jpg"
        img.save(tmp_path, format="JPEG", quality=config.DOWNLOAD_QUALITY)
        ui.download(str(tmp_path), filename=f"{stem}_{suffix}.jpg")

    def request_delete(i):
        if i >= len(state["result_items"]):
            return
        true_path, _ = state["result_items"][i]
        state["pending_delete_idx"] = i
        confirm_label.set_text(
            f"Move this file to the trash folder? It will stop appearing in search immediately, "
            f"but the original stays on disk under _deleted/ until you use \"Empty trash\".\n\n{true_path}"
        )
        confirm_dialog.open()

    def do_delete():
        confirm_dialog.close()
        i = state.get("pending_delete_idx")
        if i is None or i >= len(state["result_items"]):
            return
        true_path, proxy_path = state["result_items"][i]
        try:
            dest = config.TRASH_DIR / f"{uuid.uuid4().hex}_{Path(true_path).name}"
            shutil.move(true_path, dest)  # same volume as SOURCE_DIR -- fast rename, not a copy
            if proxy_path:
                Path(proxy_path).unlink(missing_ok=True)  # disposable/regenerable, no need to keep
            with db.connect() as conn:
                db.delete_by_path(conn, true_path)
        except OSError as e:
            ui.notify(f"Move to trash failed: {e}", type="negative")
            return
        if state["selected"] and state["selected"][0] == true_path:
            state["selected"] = None
            selected_path_box.set_value("")
        state["last_items"] = [it for it in state["last_items"] if it[0] != true_path]
        # A trashed file should stop appearing everywhere, not just the
        # current view -- also strip it from every cached back/forward
        # snapshot, or Back could later resurrect a thumbnail whose proxy
        # was just unlinked above (broken image) or whose original is now
        # sitting in _deleted/ instead of where the snapshot thinks it is.
        for stack in (state["history"], state["future"]):
            for snap in stack:
                snap["last_items"] = [it for it in snap["last_items"] if it[0] != true_path]
        render_gallery(sorted_for_display())
        refresh_all_stats()
        ui.notify(f"Moved {Path(true_path).name} to trash", type="positive")

    def do_empty_trash():
        empty_trash_dialog.close()
        n = 0
        for p in config.TRASH_DIR.iterdir():
            if p.is_file():
                p.unlink(missing_ok=True)
                n += 1
        refresh_all_stats()
        ui.notify(f"Permanently deleted {n} file(s) from trash", type="positive")

    def run_image_search(apply_refine=False):
        if not state["query_path"]:
            ui.notify("Upload a query image first", type="warning")
            return
        # apply_refine gates the refine-text blend to only the "Find
        # matches" button (find_matches_clicked passes True) -- without
        # this, whatever was left sitting in the refine box kept silently
        # riding along on every other search too (Search similar to
        # selected, double-click, even Back/Forward's re-render), which is
        # surprising: the box reads as "the next Find matches", not "every
        # search from now until I clear this field".
        state["last_apply_refine"] = apply_refine
        refine_text = (refine_input.value or "").strip() if apply_refine else ""
        # A box is cut from the full-resolution file, so small boxes keep
        # their detail; the whole-image search keeps using the fast proxy.
        crop = state["crop"]
        query_path = (state["query_full"] or state["query_path"]) if crop else state["query_path"]
        box = (crop["x0"], crop["y0"], crop["x1"], crop["y1"]) if crop else None
        if refine_text:
            results = search_combined(
                _store, query_path, refine_text, image_weight=float(refine_weight_slider.value),
                top_k=int(top_k_slider.value), hamming_threshold=int(ham_slider.value), crop=box,
            )
        else:
            results = search(
                _store, query_path, top_k=int(top_k_slider.value), hamming_threshold=int(ham_slider.value), crop=box,
            )
        items = []
        for r in results:
            tag = "\U0001F7E9 DUPLICATE" if r["near_duplicate"] else "\U0001F537 similar"
            caption = f"{tag}  sim={r['similarity']:.3f}  {r['filename']}"
            if refine_text:
                caption = f"[+\"{refine_text}\"]  {caption}"
            items.append((r["path"], r["proxy_path"], caption, r["mtime"]))
        state["last_items"] = items
        render_gallery(sorted_for_display())

    def use_selected_as_query_and_search():
        if state["selected"]:
            # A gallery result was clicked -- chain off it as the new query.
            push_history()  # this is the "search similar" hop -- preserve the view being left behind
            path, proxy_path = state["selected"]
            query_source = proxy_path or path  # prefer the fast proxy copy
            state["query_path"] = query_source
            state["query_full"] = path if Path(path).is_file() else query_source
            state["query_url"] = _public_url(query_source)
            set_query_image(state["query_url"])
        elif state["query_path"]:
            # Nothing clicked yet, but there's already a query image loaded
            # (fresh upload/drop) -- this button sits directly under
            # "Selected Image" now, so "search similar to selected" should
            # also work off *that* rather than forcing a redundant click on
            # a result first. Same effect as "Find matches" in this case.
            push_history()
        else:
            ui.notify("Upload or select an image first", type="warning")
            return
        run_image_search()

    def search_similar_to(i):
        # Double-click shortcut: select this item, then run the same
        # action as the "Search similar to selected" button.
        select_item(i)
        use_selected_as_query_and_search()

    def run_text_search():
        term = (keyword_input.value or "").strip()
        if not term:
            return
        push_history()
        if literal_checkbox.value:
            with db.connect() as conn:
                rows = db.search_by_text(conn, term)
            items = [(row["path"], row["proxy_path"], row["filename"], row["mtime"]) for row in rows]
        else:
            results = text_search(_store, term, top_k=int(top_k_slider.value))
            items = [
                (r["path"], r["proxy_path"], f"\U0001F50E sim={r['similarity']:.3f}  {r['filename']}", r["mtime"])
                for r in results
            ]
        state["last_items"] = items
        render_gallery(sorted_for_display())

    def find_matches_clicked():
        # "Find matches" is a navigation point too (a fresh upload replacing
        # whatever results are currently shown), separate from
        # run_image_search() itself since that's also called internally by
        # use_selected_as_query_and_search() -- which already does its own
        # push_history() -- and pushing twice for one hop would leave a
        # stale duplicate entry in the back stack.
        push_history()
        run_image_search(apply_refine=True)

    def on_crop_changed(e):
        # Finished drag from the box-search script (fractions of the shown
        # image). Re-runs the current image search on just that part --
        # replaces the results in place, no history entry per box move.
        try:
            box = {k: min(1.0, max(0.0, float(e.args[k]))) for k in ("x0", "y0", "x1", "y1")}
        except (KeyError, TypeError, ValueError):
            return
        if box["x1"] <= box["x0"] or box["y1"] <= box["y0"]:
            return
        state["crop"] = box
        if state["query_path"]:
            run_image_search(apply_refine=state["last_apply_refine"])

    def clear_box():
        if not state["crop"]:
            return
        state["crop"] = None
        ui.run_javascript("window.__imCropSet && window.__imCropSet(null)")
        if state["query_path"]:
            run_image_search(apply_refine=state["last_apply_refine"])

    ui.on("crop_changed", on_crop_changed)
    whole_image_btn.on_click(clear_box)
    find_btn.on_click(find_matches_clicked)
    text_btn.on_click(run_text_search)
    keyword_input.on("keydown.enter", lambda: run_text_search())
    similar_btn.on_click(use_selected_as_query_and_search)


def run():
    ui.run(
        host=config.UI_HOST, port=config.UI_PORT, title=f"Image Match — {config.CATEGORY}",
        reload=False, show=False, favicon="\U0001F50D",
        # NiceGUI's default (3s) is tuned for typical web apps, not a single
        # synchronous function rebuilding up to top_k=300 gallery cards (each
        # ~9 elements) in one shot -- see render_gallery(). A render that
        # blocks the event loop past the reconnect window looks to the
        # browser like a dead connection, and past this timeout NiceGUI just
        # discards that client's server-side state instead of reconnecting
        # it -- which is what "the page clears and I have to start over"
        # actually is. Raised again (30s -> 120s) for a metered/throttled
        # connection, where a slow stretch on the wire itself -- not just a
        # big render -- can leave the websocket unacknowledged for longer
        # than 30s. This is a private single-user tool, so there's no real
        # cost to being generous here.
        reconnect_timeout=120.0,
    )


if __name__ in {"__main__", "__mp_main__"}:
    run()
