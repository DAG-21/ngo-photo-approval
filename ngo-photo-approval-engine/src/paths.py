"""
Portable image-path handling.
Image paths in `image_metadata.csv` are stored relative to the project root (POSIX separators)
so the dataset works across machines. `resolve_image_path` turns a stored path back into a usable
absolute path, and also repairs legacy absolute paths written on another machine by re-rooting
everything from the `data/` segment onwards.
"""

from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def to_stored_path(path: Union[str, Path]) -> str:
    """Project-relative POSIX path if the image lives under the project root, else absolute."""
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)


def resolve_image_path(stored: Union[str, Path]) -> Path:
    """Returns an absolute path for a stored image path, re-rooting stale foreign paths."""
    p = Path(str(stored))
    if not p.is_absolute() and not str(stored).startswith(("/", "\\")):
        return PROJECT_ROOT / p
    if p.exists():
        return p
    parts = str(stored).replace("\\", "/").split("/")
    if "data" in parts:
        return PROJECT_ROOT.joinpath(*parts[parts.index("data"):])
    return p
