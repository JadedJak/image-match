"""Local web UI: drop in an image, get back a ranked gallery of matches
from this instance's index, each captioned with match type, similarity
score, and filename. Click a result to select it -- shows/copies its
full source path -- then optionally hit "Search similar to selected" to
drill into that cluster with a fresh search. Selecting and searching are
deliberately separate actions: a click alone never discards your current
results.

The text box does one of two kinds of search, chosen explicitly by the
"Match literally" checkbox rather than guessed automatically: CLIP
semantic search by default (a description like "dog on a beach"), or a
literal filename/path substring match when the checkbox is on (for when
you already know roughly what the file is called). Auto-falling-back to
literal matching used to be the default, but that meant any search term
that happened to coincidentally match part of some file's name/path would
silently hijack a semantic search -- surprising and hard to predict."""
import gradio as gr

from . import config, db, embedder
from .faiss_store import FaissStore
from .search_engine import search, text_search

db.init_db()
_store = FaissStore(embedder.dim())


def _fresh_gallery(gallery):
    """Wrap a new result list so the Gallery always redraws as the full
    grid, scrolled/starting at the top-most match -- without this, Gradio
    carries over whatever item index was previously selected/expanded
    (from browsing an earlier result set or from the "search similar"
    click that triggered this new search) onto the *new* list, which
    looks like it's popped open a random unrelated match instead of
    showing the fresh results."""
    return gr.update(value=gallery, selected_index=None)


def run_search(image_path, top_k, hamming_threshold):
    if image_path is None:
        return _fresh_gallery([]), []
    results = search(_store, image_path, top_k=int(top_k), hamming_threshold=int(hamming_threshold))
    gallery, paths = [], []
    for r in results:
        tag = "\U0001F7E9 DUPLICATE" if r["near_duplicate"] else "\U0001F537 similar"
        caption = f"{tag}  sim={r['similarity']:.3f}  {r['filename']}"
        gallery.append((r["proxy_path"] or r["path"], caption))
        paths.append((r["path"], r["proxy_path"]))
    return _fresh_gallery(gallery), paths


def run_keyword_search(term, top_k, literal):
    if not term or not term.strip():
        return _fresh_gallery([]), []
    term = term.strip()

    if literal:
        with db.connect() as conn:
            rows = db.search_by_text(conn, term)
        gallery, paths = [], []
        for row in rows:
            gallery.append((row["proxy_path"] or row["path"], row["filename"]))
            paths.append((row["path"], row["proxy_path"]))
        return _fresh_gallery(gallery), paths

    # CLIP text-to-image semantic search -- the default.
    results = text_search(_store, term, top_k=int(top_k))
    gallery, paths = [], []
    for r in results:
        caption = f"\U0001F50E sim={r['similarity']:.3f}  {r['filename']}"
        gallery.append((r["proxy_path"] or r["path"], caption))
        paths.append((r["path"], r["proxy_path"]))
    return _fresh_gallery(gallery), paths


def on_select(evt: gr.SelectData, paths):
    """A click only selects -- shows/copies the true path and remembers
    (path, proxy_path) for the "search similar to selected" button below.
    Deliberately does *not* trigger a new search by itself: the first
    click opens Gradio's built-in lightbox preview too, and forcing every
    single click to blow away the current result set would rule out ever
    using selection for anything else (just inspecting/copying a path,
    future actions, etc)."""
    if not paths or evt.index >= len(paths):
        return "", None
    path, proxy_path = paths[evt.index]
    return path, (path, proxy_path)


def use_selected_as_query(selected):
    """Explicit "search similar to selected" action -- reads back whatever
    on_select last stashed and hands the proxy (fast NVMe) off as the new
    query image; the .then() chain in the Blocks below fires the actual
    re-search once this sets the query Image component."""
    if not selected:
        return gr.update()
    path, proxy_path = selected
    return proxy_path or path


def index_stats():
    with db.connect() as conn:
        n = db.count(conn)
    return (
        f"**{n}** images indexed in `{config.CATEGORY}` -- {_store.ntotal()} vectors loaded "
        f"(restart this container after reindexing to pick up newly added ones)"
    )


with gr.Blocks(title=f"Image Match — {config.CATEGORY}") as demo:
    gr.Markdown(f"## Image Match — `{config.CATEGORY}` catalog")
    with gr.Row():
        stats_md = gr.Markdown(index_stats())
        refresh_btn = gr.Button("↻ Refresh stats", size="sm", scale=0)

    result_paths = gr.State([])  # [(true_path, proxy_path_or_None), ...] in gallery order
    selected_source = gr.State(None)  # (true_path, proxy_path_or_None) of the last-clicked result

    with gr.Row():
        with gr.Column(scale=1):
            query = gr.Image(
                type="filepath", label="Query image", height=320, width=320,
                # "webcam" is Gradio's default third source alongside
                # upload/clipboard -- on a headless/no-camera box it just
                # gets stuck repeatedly prompting for camera permission.
                # Not needed here, so drop it.
                sources=["upload", "clipboard"],
            )
            thumb_size = gr.Slider(80, 640, value=320, step=20, label="Query image preview size (px)")
            top_k = gr.Slider(5, 300, value=config.TOP_K, step=5, label="Max results")
            hamming_threshold = gr.Slider(
                0, 20, value=config.NEAR_DUP_HAMMING_THRESHOLD, step=1,
                label="Near-duplicate sensitivity (lower = stricter)",
            )
            search_btn = gr.Button("Find matches", variant="primary")

            gr.Markdown("---")
            keyword = gr.Textbox(
                label="Search by description or filename/path",
                placeholder="e.g. dog on a beach  --or--  IMG_2024",
            )
            literal_match = gr.Checkbox(
                label="Match literally by filename/path (instead of by description)", value=False,
            )
            keyword_btn = gr.Button("Search text")

            gr.Markdown("---")
            selected_path = gr.Textbox(
                label="Selected result — full path", show_copy_button=True, interactive=False,
                lines=3,  # tall enough that a long wrapped path isn't clipped
            )
            similar_btn = gr.Button("\U0001F50E Search similar to selected", elem_id="similar-btn")

        with gr.Column(scale=3):
            results = gr.Gallery(
                elem_id="results-gallery",
                label="Matches (click a thumbnail to select it and search similar, most alike first)",
                columns=5, height="auto", object_fit="contain",
                # Lightbox back off. The Escape-keypress-dispatch trick to
                # auto-close it after a search didn't reliably fire (nor,
                # per that same failure, did the plain scrollIntoView call
                # bundled in the same JS step) -- rather than keep guessing
                # blind at Gradio's internal event-chaining behavior with no
                # way to test it live, go back to the version that's
                # structurally guaranteed correct: no lightbox means no
                # stale-image state to accidentally leave open in the first
                # place, full stop.
                allow_preview=False,
            )

    # Scrolls to the top of the page after a search finishes, so the
    # closest match is immediately visible without the user manually
    # scrolling back up from wherever the page happened to be.
    #
    # Confirmed via console.log that the JS itself runs and correctly
    # finds #results-gallery either way this gets wired up (chained
    # .then(), or an independent fn=None listener) -- but scrollIntoView()
    # targeting that specific element produced no visible scroll in
    # practice either time, for reasons not diagnosable without directly
    # inspecting the live page. Falling back to the coarser, harder-to-get-
    # wrong window.scrollTo(0,0): scroll the whole page to its absolute
    # top rather than a specific element into view. Less precise (lands on
    # the sidebar controls too, not just the gallery) but about as close
    # to foolproof as browser scrolling gets.
    _scroll_to_results_js = "() => { window.scrollTo({top: 0, left: 0, behavior: 'instant'}); }"

    thumb_size.change(lambda v: gr.update(height=v, width=v), inputs=thumb_size, outputs=query)
    search_btn.click(
        run_search, inputs=[query, top_k, hamming_threshold], outputs=[results, result_paths],
    )
    search_btn.click(None, None, None, js=_scroll_to_results_js)
    keyword_btn.click(
        run_keyword_search, inputs=[keyword, top_k, literal_match], outputs=[results, result_paths],
    )
    keyword_btn.click(None, None, None, js=_scroll_to_results_js)
    keyword.submit(
        run_keyword_search, inputs=[keyword, top_k, literal_match], outputs=[results, result_paths],
    )
    keyword.submit(None, None, None, js=_scroll_to_results_js)
    results.select(on_select, inputs=result_paths, outputs=[selected_path, selected_source])
    similar_btn.click(use_selected_as_query, inputs=selected_source, outputs=query).then(
        run_search, inputs=[query, top_k, hamming_threshold], outputs=[results, result_paths],
    )
    similar_btn.click(None, None, None, js=_scroll_to_results_js)
    refresh_btn.click(index_stats, outputs=stats_md)
    demo.load(index_stats, outputs=stats_md)

    # Double-click a thumbnail as a shortcut for "search similar to
    # selected" -- Gradio's Gallery has no native double-click event, so
    # this binds a plain JS listener that just clicks the real button for
    # us once bound. The first click of the double-click has already
    # fired .select() (updating selected_source) by the time the second
    # click resolves into "dblclick", so the button reads the right item.
    # Single-click still only selects, unchanged -- this is purely an
    # added shortcut, not a replacement for the button.
    demo.load(None, js="""
    () => {
        const gallery = document.getElementById('results-gallery');
        if (gallery && !gallery.dataset.dblclickBound) {
            gallery.dataset.dblclickBound = '1';
            gallery.addEventListener('dblclick', () => {
                document.getElementById('similar-btn')?.click();
            });
        }
    }
    """)

def run():
    # Gradio 5.x refuses to serve/cache a file from outside its cwd, the
    # system temp dir, or an explicit allowed_paths entry -- both the
    # source volume and the proxy cache live outside all of those (bind
    # mounts at /data/source, /data/proxy), so gallery thumbnails/images
    # need an explicit allow-list or every result 500s with InvalidPathError.
    allowed = [str(config.SOURCE_DIR)]
    if config.PROXY_DIR:
        allowed.append(str(config.PROXY_DIR))
    demo.launch(server_name=config.UI_HOST, server_port=config.UI_PORT, allowed_paths=allowed)


if __name__ == "__main__":
    run()
