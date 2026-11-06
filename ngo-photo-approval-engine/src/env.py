"""
Minimal `.env` loader (no extra dependency).
Reads KEY=VALUE lines into `os.environ` without overriding variables that are already set, and
never logs or returns values, only the names it set.
"""

import os
from pathlib import Path
from typing import List, Optional

from src.paths import PROJECT_ROOT


def load_env_file(path: Optional[Path] = None) -> List[str]:
    """Loads `<project>/.env` (or `path`) and returns the names of variables newly set."""
    env_path = Path(path) if path else PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return []

    newly_set: List[str] = []
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if name and name not in os.environ:
            os.environ[name] = value
            newly_set.append(name)
    return newly_set
