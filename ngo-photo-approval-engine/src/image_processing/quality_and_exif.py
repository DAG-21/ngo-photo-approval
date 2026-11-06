"""
Extracts image dimensions, aspect ratio, file size, format, EXIF capture date,
camera metadata, SHA-256 exact hash, and difference/perceptual hash (dHash)
for duplicate and burst detection.
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from PIL import Image, ExifTags


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(img: Image.Image, hash_size: int = 8) -> str:
    """Computes a 64-bit difference perceptual hash (dHash) in hex."""
    gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    bits = []
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            bits.append(1 if pixels[row_start + col] > pixels[row_start + col + 1] else 0)
    val = 0
    for b in bits:
        val = (val << 1) | b
    return f"{val:016x}"


def hamming_distance_hex(hash1: str, hash2: str) -> int:
    """Calculates bitwise Hamming distance between two 64-bit hex hashes."""
    return bin(int(hash1, 16) ^ int(hash2, 16)).count("1")


def extract_exif_metadata(img: Image.Image) -> Dict[str, Any]:
    exif_data: Dict[str, Any] = {}
    try:
        raw_exif = img.getexif()
        if not raw_exif:
            return exif_data
        for tag_id, value in raw_exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            if tag_name in {"DateTimeOriginal", "DateTime", "Make", "Model", "Orientation"}:
                exif_data[tag_name] = str(value)
        # DateTimeOriginal (true capture time) lives in the Exif sub-IFD, not the top-level IFD;
        # the top-level `DateTime` is only the last-modified time.
        sub_ifd = raw_exif.get_ifd(ExifTags.IFD.Exif)
        if sub_ifd.get(ExifTags.Base.DateTimeOriginal):
            exif_data["DateTimeOriginal"] = str(sub_ifd[ExifTags.Base.DateTimeOriginal])
    except Exception:
        pass
    return exif_data


def parse_capture_datetime(exif_dict: Dict[str, Any]) -> Optional[str]:
    raw_dt = exif_dict.get("DateTimeOriginal") or exif_dict.get("DateTime")
    if not raw_dt:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(raw_dt.strip(), fmt)
            return dt.isoformat()
        except ValueError:
            continue
    return str(raw_dt)


def extract_image_metadata(file_path: Path) -> Dict[str, Any]:
    """Extracts complete technical and EXIF metadata for a single image."""
    stat = file_path.stat()
    sha256 = compute_sha256(file_path)
    with Image.open(file_path) as img:
        width, height = img.size
        img_format = img.format or file_path.suffix.lstrip(".").upper()
        aspect_ratio = round(width / height, 4) if height > 0 else 0.0
        dhash = compute_dhash(img)
        exif = extract_exif_metadata(img)
        capture_date = parse_capture_datetime(exif)

    return {
        "image_width": width,
        "image_height": height,
        "aspect_ratio": aspect_ratio,
        "file_size_bytes": stat.st_size,
        "file_format": img_format,
        "sha256": sha256,
        "dhash": dhash,
        "capture_date": capture_date,
        "camera_make": exif.get("Make"),
        "camera_model": exif.get("Model"),
        "exif_json": str(exif) if exif else "{}",
    }
