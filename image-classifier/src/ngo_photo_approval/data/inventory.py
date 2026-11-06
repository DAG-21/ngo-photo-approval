"""Dataset inventory: one manifest row per file with label, metadata, hashes and quality metrics."""

from __future__ import annotations

import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from ..config import DatasetConfig
from ..exceptions import DataError
from .preprocessing import guess_mime

logger = logging.getLogger(__name__)

# `<program>-<submission index>[ (<copy>)].jpg`, plus ` - 2026-10-29T151507.346` browser-download suffixes.
_NAME_RE = re.compile(r"^(?P<program>[A-Za-z_]+?)-(?P<sub>\d+)(?:\s*\((?P<copy>\d+)\))?(?:\s*-\s*\d{4}-\d{2}-\d{2}T[\d.]+)?$")

MEDIA_IMAGE = "image"
MEDIA_UNSUPPORTED = "unsupported_media"
MEDIA_UNKNOWN = "unknown"

MANIFEST_COLUMNS = [
    "image_id", "rel_path", "folder", "original_label", "label", "program", "submission_idx",
    "copy_idx", "filename_group", "extension", "mime_type", "pil_format", "extension_mismatch",
    "media_kind", "file_size", "mtime_ns", "sha256", "readable", "error", "width", "height", "aspect_ratio", "megapixels",
    "exif_datetime", "blur_laplacian_var", "brightness", "contrast", "dhash",
]


def parse_filename(stem: str) -> dict[str, Any]:
    """Extract (program, submission_idx, copy_idx) from a client filename stem; None if unmatched."""
    m = _NAME_RE.match(stem)
    if not m:
        return {"program": None, "submission_idx": None, "copy_idx": None}
    return {
        "program": m.group("program"),
        "submission_idx": int(m.group("sub")),
        "copy_idx": int(m.group("copy")) if m.group("copy") else None,
    }


def make_image_id(rel_path: str) -> str:
    """Stable ID derived from the relative path (which includes the label folder).

    File names alone are NOT unique: the same name occurs in both `approved` and `disapproved`.
    """
    return "img_" + hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def dhash64(gray: np.ndarray) -> str:
    """64-bit difference hash as 16 hex chars."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:016x}"


def quality_metrics(gray: np.ndarray) -> dict[str, float]:
    return {
        "blur_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
    }


_EXT_FORMATS = {".jpg": {"JPEG", "MPO"}, ".jpeg": {"JPEG", "MPO"}, ".png": {"PNG"}, ".webp": {"WEBP"}}


def sniff_format(path: Path) -> str | None:
    """True container format from the file header (cheap; no full decode)."""
    try:
        with Image.open(path) as im:
            return im.format
    except Exception:  # noqa: BLE001
        return None


def _decode_for_metrics(path: Path, max_side: int = 1024) -> tuple[Image.Image, int, int, Any]:
    """Decode (with JPEG draft mode for speed). Returns (RGB image, true_w, true_h, exif)."""
    with Image.open(path) as im:
        exif = im.getexif()
        orientation = exif.get(274, 1)
        true_w, true_h = im.size
        if im.format == "JPEG":
            im.draft("RGB", (max_side, max_side))
        im.load()
        im = ImageOps.exif_transpose(im).convert("RGB")
    if orientation in (5, 6, 7, 8):
        true_w, true_h = true_h, true_w
    return im, true_w, true_h, exif


def _exif_datetime(exif: Any) -> str | None:
    try:
        # 36867 DateTimeOriginal lives in the Exif IFD (0x8769); 306 DateTime is in IFD0.
        sub = exif.get_ifd(0x8769)
        return sub.get(36867) or exif.get(306)
    except Exception:  # noqa: BLE001 - metadata is best-effort
        return None


def build_row(path: Path, data_dir: Path, cfg: DatasetConfig, legacy_rel: str | None = None) -> dict[str, Any]:
    """``legacy_rel``: the file's ORIGINAL relative path if it was renamed (see data/renaming.py). The image id and
    the program/submission metadata are derived from it so they stay stable across the rename."""
    rel = path.relative_to(data_dir).as_posix()
    folder = path.relative_to(data_dir).parts[0]
    ext = path.suffix.lower()
    stat = path.stat()
    label = cfg.folder_label_map.get(folder)
    row: dict[str, Any] = {c: None for c in MANIFEST_COLUMNS}
    row.update(
        image_id=make_image_id(legacy_rel or rel), rel_path=rel, folder=folder, original_label=folder,
        label=label, extension=ext, mime_type=guess_mime(path), file_size=stat.st_size,
        mtime_ns=stat.st_mtime_ns, readable=False, error=None,
        **parse_filename(Path(legacy_rel).stem if legacy_rel else path.stem),
    )
    if ext in cfg.known_non_image_extensions:
        row.update(media_kind=MEDIA_UNSUPPORTED, error=f"non-image media ({ext})")
    elif ext in cfg.supported_image_extensions:
        row["media_kind"] = MEDIA_IMAGE
    else:
        row.update(media_kind=MEDIA_UNKNOWN, error=f"unrecognized extension ({ext})")

    try:
        row["sha256"] = sha256_file(path)
    except OSError as exc:
        row["error"] = f"read error: {exc}"
        return row
    if row["media_kind"] != MEDIA_IMAGE:
        return row

    fmt = sniff_format(path)
    row["pil_format"] = fmt
    row["extension_mismatch"] = bool(fmt) and fmt not in _EXT_FORMATS.get(ext, {fmt})
    try:
        im, w, h, exif = _decode_for_metrics(path)
        gray = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
        row.update(
            readable=True, width=w, height=h, aspect_ratio=round(w / h, 4),
            megapixels=round(w * h / 1e6, 3), exif_datetime=_exif_datetime(exif),
            dhash=dhash64(gray), **quality_metrics(gray),
        )
    except Exception as exc:  # noqa: BLE001 - any decode failure marks the file unreadable
        # Exception text can contain absolute local paths; keep only the type.
        row["error"] = f"decode error: {type(exc).__name__}"
    return row


def _group_key(row: dict[str, Any]) -> str | None:
    if row["program"] is None or row["submission_idx"] is None:
        return None
    # Label is part of the key: identical `program-N` names exist in several label folders and
    # there is no evidence they are the same event (see docs/DATASET_AUDIT.md).
    return f"{row['label']}|{row['program']}|{row['submission_idx']}"


def build_inventory(
    data_dir: Path,
    cfg: DatasetConfig,
    cache: pd.DataFrame | None = None,
    workers: int = 8,
    legacy: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Scan ``data_dir`` and return the manifest. Reuses rows from ``cache`` when
    (rel_path, file_size, mtime_ns) are unchanged. Never modifies the dataset.

    ``legacy`` maps current rel_path -> original rel_path for renamed files (data/manifests/rename_map.csv)."""
    legacy = legacy or {}
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise DataError(f"dataset directory not found: {data_dir} (set NGO_DATA_DIR)")

    known_dirs = set(cfg.folder_label_map)
    paths: list[Path] = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name not in known_dirs:
            logger.warning("ignoring unmapped folder %r (not in folder_label_map)", child.name)
            continue
        if cfg.folder_label_map[child.name] == "pending" and not cfg.inventory_pending:
            continue
        paths.extend(p for p in sorted(child.rglob("*")) if p.is_file())
    if not paths:
        raise DataError(f"no files found under {data_dir} for folders {sorted(known_dirs)}")

    cached: dict[str, dict[str, Any]] = {}
    if cache is not None and len(cache):
        cached = {r["rel_path"]: r for r in cache.to_dict("records")}

    def task(p: Path) -> dict[str, Any]:
        rel = p.relative_to(data_dir).as_posix()
        st = p.stat()
        legacy_rel = legacy.get(rel)
        hit = cached.get(rel) or (cached.get(legacy_rel) if legacy_rel else None)
        fresh = (hit is not None and hit.get("file_size") == st.st_size
                 and hit.get("mtime_ns") == st.st_mtime_ns)
        # Previously-unreadable images are always retried (decoder support may have changed).
        if fresh and (hit.get("readable") is True or hit.get("media_kind") != MEDIA_IMAGE):
            row = {k: (None if pd.isna(v) else v) for k, v in hit.items()}
            row["rel_path"] = rel
            if hit.get("media_kind") == MEDIA_IMAGE and row.get("pil_format") is None:
                fmt = sniff_format(p)
                row["pil_format"] = fmt
                row["extension_mismatch"] = bool(fmt) and fmt not in _EXT_FORMATS.get(row["extension"], {fmt})
            return row
        return build_row(p, data_dir, cfg, legacy_rel)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(task, paths))

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    df["filename_group"] = [_group_key(r) for r in df.to_dict("records")]
    for col in ("file_size", "mtime_ns", "width", "height", "submission_idx", "copy_idx"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["readable"] = df["readable"].fillna(False).astype(bool)
    if df["image_id"].duplicated().any():
        raise DataError("image_id collision in inventory")
    return df
