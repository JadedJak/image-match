import torch
from PIL import Image, ImageOps
import open_clip

from . import config

# Pillow refuses to decode images above ~179M pixels by default, as a guard
# against decompression-bomb DoS from untrusted uploads. Everything indexed
# here comes from the user's own encrypted personal/general source drives,
# not untrusted input, and modern phone panoramas/high-res shots legitimately
# exceed that (e.g. a real 200MP photo hit this as "failed to process" during
# indexing). Disabled outright rather than raised to a bigger number, since
# there's no untrusted-input scenario in this project to guard against.
Image.MAX_IMAGE_PIXELS = None

_model = None
_preprocess = None
_device = None
_dtype = None
_tokenizer = None

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _pick_device(want):
    if want == "cuda" and torch.cuda.is_available():
        return "cuda"
    if want == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def _load():
    global _model, _preprocess, _device, _dtype, _tokenizer
    if _model is not None:
        return
    _device = _pick_device(config.DEVICE)
    _dtype = _DTYPES[config.PRECISION] if _device != "cpu" else torch.float32
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED
    )
    model.eval().to(_device, dtype=_dtype)
    # visible in indexer output and container logs -- a requested GPU that
    # silently fell back to cpu would otherwise only show up as slowness
    print(f"embedder: {config.CLIP_MODEL} on {_device} "
          f"({str(_dtype).replace('torch.', '')}; requested DEVICE={config.DEVICE} PRECISION={config.PRECISION})",
          flush=True)
    _model = model
    _preprocess = preprocess
    _tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL)


def load_and_downsample(path, max_side=None):
    """Open an image and shrink it in-memory only -- the file on disk (and
    on the source VeraCrypt volume) is never modified. See config.MAX_SIDE.
    max_side defaults to config.MAX_SIDE (indexing/embedding use); the
    on-the-fly hover-preview route (see ui_nicegui.py) passes
    config.PREVIEW_MAX_SIDE instead to get a bigger, still-bounded resize
    from the same decode path without a second copy of this function.

    Uses JPEG draft mode where available: libjpeg can decode directly at a
    reduced DCT scale (~1/2, 1/4, 1/8) instead of decoding full resolution
    and shrinking afterward. For 10MB+ originals being reduced to ~512px,
    this skips decoding the vast majority of pixels that would just be
    thrown away by the resize anyway."""
    max_side = max_side or config.MAX_SIDE
    img = _open_rgb(path, (max_side, max_side))
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def load_region(path, box, max_side=None):
    """Like load_and_downsample, but only the part of the image inside box
    (x0, y0, x1, y1 as 0-1 fractions of the upright image -- the same
    coordinates the UI's box overlay reports, since browsers and the
    proxies both show photos EXIF-rotated). Used for box search: the crop
    is cut from the full-resolution file before shrinking, so a small box
    keeps its detail. Draft mode still applies, just at a scale where the
    cropped part stays at least max_side across."""
    max_side = max_side or config.MAX_SIDE
    x0, y0, x1, y1 = box
    # min over both sides, since draft works on the stored (pre-rotation) axes
    frac = max(min(x1 - x0, y1 - y0), 0.01)
    need = int(max_side / frac)
    img = _open_rgb(path, (need, need))
    w, h = img.size
    img = img.crop((round(x0 * w), round(y0 * h), max(round(x1 * w), round(x0 * w) + 1), max(round(y1 * h), round(y0 * h) + 1)))
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def _open_rgb(path, draft_size):
    """Decode to an upright RGB image, letting JPEG draft mode skip
    resolution beyond draft_size. Shared by load_and_downsample/load_region."""
    img = Image.open(path)
    try:
        img.draft("RGB", draft_size)
    except Exception:
        pass  # not all formats support draft mode (e.g. PNG) -- falls through fine
    img = ImageOps.exif_transpose(img)  # respect camera/phone rotation metadata
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        # A plain .convert("RGB") on a transparent image doesn't composite
        # -- it just drops the alpha channel and keeps whatever RGB values
        # happen to be stored underneath the transparent pixels, which are
        # normally hidden and can be anything (some exporters flatten them
        # to a flat fill color). That surfaced as small transparent-icon
        # PNGs embedding/thumbnailing as solid gray squares. Only images
        # that actually carry transparency take this extra path -- the
        # common case (opaque JPEGs/PNGs/scans, the bulk of a real
        # catalog) is untouched, same single convert("RGB") call as before.
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    else:
        img = img.convert("RGB")
    return img


def save_proxy(img: Image.Image, proxy_path):
    """Write the already-downsampled image out as a small JPEG cache copy.
    Called with the same in-memory image used for embedding -- no extra
    decode. Disposable: only used as embed input + UI thumbnail source,
    never as the authoritative file location."""
    proxy_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(proxy_path, format="JPEG", quality=config.PROXY_QUALITY)


@torch.no_grad()
def embed(img: Image.Image):
    return embed_batch([img])[0]


@torch.no_grad()
def embed_batch(imgs):
    _load()
    tensors = torch.stack([_preprocess(img) for img in imgs]).to(_device, dtype=_dtype)
    # normalize in float32 regardless of PRECISION -- FAISS stores float32
    features = _model.encode_image(tensors).float()
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype("float32")


@torch.no_grad()
def embed_text(text: str):
    """CLIP's text and image encoders are trained into one shared
    embedding space, so a plain-language prompt ("dog on a beach") can be
    searched against the same FAISS index of image vectors directly --
    no separate captioning/tagging pass needed over the catalog."""
    _load()
    tokens = _tokenizer([text]).to(_device)
    features = _model.encode_text(tokens).float()
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype("float32")[0]


def dim():
    _load()
    return getattr(_model.visual, "output_dim", 512)
