"""Image embeddings, used only to find similar photographs. Computed on this machine: nothing is sent anywhere, and nothing is ever downloaded.

* `DinoV2Embedder` and `ClipEmbedder` load weights that are already in the local Hugging Face cache (offline); if they are not there they say so
  instead of downloading. The cache here has the weights but not the preprocessing files, so the standard preprocessing is done by hand.
* `ThumbnailEmbedder` is a tiny deterministic fallback (an 8x8 colour thumbnail). It needs no model and is what the tests use.

Every embedder has a `name`; an embedding is only ever compared with embeddings of the same name.
"""

from __future__ import annotations

import io
import logging
import os
import threading
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
from PIL import Image

try:  # the NGO's photographs include HEIC files
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pass

log = logging.getLogger("ngo_agent.embeddings")


class EmbedderUnavailable(RuntimeError):
    """The embedding model cannot be loaded here (not installed or not in the local cache). Never triggers a download."""


class Embedder(Protocol):
    name: str

    def embed(self, image_bytes: bytes) -> list[float]: ...


def unit(v: np.ndarray) -> list[float]:
    n = float(np.linalg.norm(v))
    return [float(x) for x in (v / n if n > 0 else v)]


def load_rgb(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as im:
        im.load()
        return im.convert("RGB")


def _prepare(image: Image.Image, mean: tuple[float, float, float], std: tuple[float, float, float], short: int, crop: int = 224) -> np.ndarray:
    """Resize the short side to `short` (bicubic), centre-crop to `crop`, scale to 0..1 and normalise: the standard preprocessing of these models."""
    w, h = image.size
    s = short / min(w, h)
    image = image.resize((max(crop, round(w * s)), max(crop, round(h * s))), Image.BICUBIC)
    w, h = image.size
    left, top = (w - crop) // 2, (h - crop) // 2
    a = np.asarray(image.crop((left, top, left + crop, top + crop)), dtype=np.float32) / 255.0
    return ((a - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)).transpose(2, 0, 1)[None]


def _hub_dir() -> Path:
    return Path(os.environ.get("HF_HUB_CACHE") or (Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"))


class ThumbnailEmbedder:
    name = "thumbnail-8x8-v1"

    def embed(self, image_bytes: bytes) -> list[float]:
        a = np.asarray(load_rgb(image_bytes).resize((8, 8), Image.LANCZOS), dtype=np.float32).ravel() / 255.0 - 0.5
        return unit(a)


class _TorchEmbedder:
    name = ""

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def _load(self):  # pragma: no cover  (needs the real weights)
        raise NotImplementedError

    def _get(self):
        with self._lock:
            if self._model is None:
                try:
                    import torch  # noqa: F401
                    import transformers  # noqa: F401
                except ImportError as exc:
                    raise EmbedderUnavailable("torch and transformers are not installed") from exc
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                try:
                    self._model = self._load()
                except Exception as exc:  # noqa: BLE001
                    raise EmbedderUnavailable(f"{self.name}: the model is not available in the local cache ({type(exc).__name__}); nothing is downloaded") from exc
            return self._model


class DinoV2Embedder(_TorchEmbedder):
    """DINOv2 (base): good at the visual similarity of scenes. The CLS embedding after the final layer norm."""

    name = "dinov2-base"
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def _load(self):  # pragma: no cover
        from transformers import AutoModel

        return AutoModel.from_pretrained("facebook/dinov2-base", local_files_only=True).eval()

    def embed(self, image_bytes: bytes) -> list[float]:  # pragma: no cover
        import torch

        model = self._get()
        x = torch.from_numpy(_prepare(load_rgb(image_bytes), self.MEAN, self.STD, short=256))
        with torch.no_grad():
            out = model(pixel_values=x)
        return unit(out.pooler_output[0].numpy())


class ClipEmbedder(_TorchEmbedder):
    """CLIP ViT-B/32: also captures what the photograph is about in words. Weights from the local cache snapshot that has both config and weights."""

    name = "clip-vit-b32"
    MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

    def _snapshot(self) -> Path:  # pragma: no cover
        root = _hub_dir() / "models--openai--clip-vit-base-patch32" / "snapshots"
        for d in sorted(root.glob("*")):
            if (d / "config.json").exists() and ((d / "pytorch_model.bin").exists() or (d / "model.safetensors").exists()):
                return d
        raise FileNotFoundError("no complete CLIP snapshot in the local cache")

    def _load(self):  # pragma: no cover
        from transformers import CLIPModel

        return CLIPModel.from_pretrained(str(self._snapshot()), local_files_only=True).eval()

    def embed(self, image_bytes: bytes) -> list[float]:  # pragma: no cover
        import torch

        model = self._get()
        x = torch.from_numpy(_prepare(load_rgb(image_bytes), self.MEAN, self.STD, short=224))
        with torch.no_grad():
            f = model.get_image_features(pixel_values=x)
        f = f if torch.is_tensor(f) else f.pooler_output
        return unit(f[0].numpy())


def make_embedder(kind: Optional[str]) -> Optional[Embedder]:
    """`clip` (default), `dinov2`, `thumbnail`, or `none`. Loading is lazy: nothing heavy happens until the first photograph is embedded.

    CLIP is the default because it did best, though not significantly, on the 50 pilot photographs (scripts/measure_embedders.py: the nearest neighbour had
    the same staff decision 68% of the time against 50% for DINOv2 and 50% by chance; every interval includes chance). Measure again on more decisions."""
    k = (kind or "clip").strip().lower()
    if k in ("none", "off", ""):
        return None
    table = {"dinov2": DinoV2Embedder, "clip": ClipEmbedder, "thumbnail": ThumbnailEmbedder}
    if k not in table:
        raise ValueError(f"unknown embedder {kind!r}; choose one of {sorted(table)} or 'none'")
    return table[k]()
