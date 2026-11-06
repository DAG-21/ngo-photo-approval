"""Object storage. A small interface with a local-disk implementation; a cloud (GCS or S3) implementation can replace it without touching the rest.

Images are stored once, content-addressed, inside the organisation's own prefix: `<org_id>/images/<sha256><ext>`. The same bytes uploaded
twice by one organisation are one object. Two organisations never share an object, even for identical bytes (isolation beats de-duplication).
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Protocol

_SAFE = re.compile(r"^[A-Za-z0-9._/\-]+$")


class StorageError(RuntimeError):
    pass


class ObjectStorage(Protocol):
    def put(self, key: str, data: bytes) -> str: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...


class LocalStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if not _SAFE.match(key) or ".." in key.split("/") or key.startswith("/"):
            raise StorageError("invalid storage key")
        p = (self.root / key).resolve()
        if self.root not in p.parents:
            raise StorageError("invalid storage key")
        return p

    def put(self, key: str, data: bytes) -> str:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():  # content-addressed keys are write-once; raw-response keys are unique per evaluation
            return key
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)
        return key

    def get(self, key: str) -> bytes:
        p = self._path(key)
        if not p.exists():
            raise StorageError("object not found")
        return p.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_key(organisation_id: str, file_hash: str, ext: str) -> str:
    return f"{organisation_id}/images/{file_hash}{ext}"


def raw_response_key(organisation_id: str, evaluation_id: str) -> str:
    return f"{organisation_id}/raw/{evaluation_id}.json"
