"""Image loading and model-input preparation.

Policy: preserve visual detail. Original bytes are passed through untouched unless the image
exceeds the configured size/byte caps, in which case it is downscaled with high-quality
resampling. The raw dataset on disk is never modified.
"""

from __future__ import annotations

import io
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import ImageInputConfig
from ..exceptions import InvalidImageError

try:  # HEIC files (iPhone photos) appear in the client dataset with a ".jpg" extension
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover - dependency is declared, but degrade gracefully
    pillow_heif = None

Image.MAX_IMAGE_PIXELS = 200_000_000  # guard against decompression bombs, allow big phone photos

_MIME_BY_FORMAT = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
# Formats the providers do not accept natively: always converted to JPEG for model input.
# MPO ("Multi-Picture Object") is a JPEG container variant written by some phone/camera makers.
_CONVERT_FORMATS = {"HEIF", "HEIC", "MPO"}


@dataclass(frozen=True)
class PreparedImage:
    data: bytes
    mime_type: str
    width: int
    height: int
    resized: bool
    original_bytes: int


def guess_mime(path: Path) -> str | None:
    mime, _ = mimetypes.guess_type(str(path))
    return mime


def open_image(path: Path) -> Image.Image:
    """Open, fully decode, apply EXIF orientation, return RGB. Raises InvalidImageError."""
    path = Path(path)
    if not path.is_file():
        raise InvalidImageError(f"image not found: {path.name}")
    try:
        with Image.open(path) as im:
            im.load()
            im = ImageOps.exif_transpose(im)
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"cannot decode image {path.name}: {type(exc).__name__}") from exc


def prepare_for_model(
    path: Path,
    cfg: ImageInputConfig,
    allowed_mime_types: list[str] | None = None,
) -> PreparedImage:
    """Return model-ready bytes. Passes the original through when it is within limits."""
    path = Path(path)
    if not path.is_file():
        raise InvalidImageError(f"image not found: {path.name}")
    raw = path.read_bytes()
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            fmt = probe.format
            probe.load()
            w, h = probe.size
            orientation = probe.getexif().get(274, 1)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"cannot decode image {path.name}: {type(exc).__name__}") from exc

    convert = (fmt or "") in _CONVERT_FORMATS
    mime = "image/jpeg" if convert else _MIME_BY_FORMAT.get(fmt or "")
    if mime is None or (allowed_mime_types is not None and mime not in allowed_mime_types):
        raise InvalidImageError(f"unsupported image format {fmt!r} for {path.name}")

    needs_resize = max(w, h) > cfg.max_side_px or len(raw) > cfg.max_bytes
    needs_rotation = orientation not in (None, 1)
    if not needs_resize and not needs_rotation and not convert:
        return PreparedImage(raw, mime, w, h, False, len(raw))

    im = open_image(path)  # orientation applied
    if max(im.size) > cfg.max_side_px:
        scale = cfg.max_side_px / max(im.size)
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
    quality = cfg.jpeg_quality
    while True:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() <= cfg.max_bytes or quality <= 60:
            break
        quality -= 8
    if buf.tell() > cfg.max_bytes:
        raise InvalidImageError(f"image {path.name} cannot be reduced below {cfg.max_bytes} bytes")
    return PreparedImage(buf.getvalue(), "image/jpeg", im.width, im.height, True, len(raw))


def normalize_for_tuning(path: Path, cfg: ImageInputConfig) -> PreparedImage:
    """Resolution-normalized JPEG: long side == ``tuning_normalize_long_side_px`` (scaled up OR down), orientation
    applied, identical encoding for every image, no EXIF. The original file is never modified."""
    if not cfg.tuning_normalize_long_side_px:
        raise ValueError("tuning_normalize_long_side_px is not set")
    path = Path(path)
    im = open_image(path)
    w, h = im.size
    scale = cfg.tuning_normalize_long_side_px / max(w, h)
    target = (max(1, round(w * scale)), max(1, round(h * scale)))
    if target != im.size:
        im = im.resize(target, Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=cfg.tuning_jpeg_quality, subsampling=2, optimize=False)  # 4:2:0, no EXIF
    return PreparedImage(buf.getvalue(), "image/jpeg", im.width, im.height, True, path.stat().st_size)


def prepare_for_tuned_model(
    path: Path, cfg: ImageInputConfig, allowed_mime_types: list[str] | None = None
) -> PreparedImage:
    """The input for tuning examples AND for the tuned model at inference (train/serve must match)."""
    if cfg.tuning_normalize_long_side_px:
        return normalize_for_tuning(path, cfg)
    return prepare_for_model(path, cfg, allowed_mime_types)
