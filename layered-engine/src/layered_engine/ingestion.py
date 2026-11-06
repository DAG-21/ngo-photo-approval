"""Image ingestion: the original bytes are sent as they are (no resizing or recompression); we only read facts about the file."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

MAX_BYTES = 18 * 1024 * 1024  # below the ~20 MB inline request limit; a larger file is refused, never silently shrunk
MIME_BY_SUFFIX = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".heic": "image/heic", ".heif": "image/heif"}


class IngestionError(ValueError):
    pass


@dataclass
class LoadedImage:
    path: Path
    data: bytes
    mime_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def load_image(path: Path | str) -> LoadedImage:
    path = Path(path)
    if not path.exists():
        raise IngestionError(f"image not found: {path}")
    data = path.read_bytes()
    if not data:
        raise IngestionError(f"empty file: {path}")
    if len(data) > MAX_BYTES:
        raise IngestionError(f"{path.name} is {len(data) / 1e6:.1f} MB, over the {MAX_BYTES / 1e6:.0f} MB limit; it is not resized automatically")
    mime = MIME_BY_SUFFIX.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0]
    if not mime or not mime.startswith("image/"):
        raise IngestionError(f"{path.name}: unsupported file type")
    img = LoadedImage(path=path, data=data, mime_type=mime)
    img.metadata = extract_metadata(data, len(data), img.warnings)
    return img


def extract_metadata(data: bytes, size: int, warnings: Optional[list[str]] = None) -> dict[str, Any]:
    """width, height, aspect ratio, file size, format and EXIF capture time where available."""
    import io

    meta: dict[str, Any] = {"file_size": size}
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            meta.update(width=im.width, height=im.height, aspect_ratio=round(im.width / im.height, 4), format=im.format)
            try:
                exif = im.getexif()
                meta["exif_datetime"] = exif.get(36867) or exif.get(306)
            except Exception:  # noqa: BLE001
                meta["exif_datetime"] = None
    except Exception as exc:  # noqa: BLE001  (e.g. HEIC without a plugin: the provider may still read it)
        if warnings is not None:
            warnings.append(f"could not read image dimensions locally ({type(exc).__name__})")
    return meta
