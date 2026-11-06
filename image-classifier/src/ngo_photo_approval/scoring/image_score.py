"""Learned image score: CLIP embedding of the resolution-normalized photo -> logistic regression on the human labels.

Stage 1 of the combiner. The preprocessing is identical to what produced the training embeddings
(``scripts/embedding_probe.py``): the photo goes through ``normalize_for_tuning`` (long side N px, identical JPEG
encoding), is squashed to 224x224 (no center crop, so framing/cropping stays visible) and normalized with CLIP's
mean/std. Torch/transformers are imported lazily, only when a real embedder is used.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image

from ..config import ImageInputConfig
from ..data.preprocessing import normalize_for_tuning
from ..exceptions import ModelArtifactError

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def tensor_from_image(im: Image.Image) -> np.ndarray:
    """(3, 224, 224) float32 CLIP input from a PIL image (squash, no crop)."""
    a = np.asarray(im.convert("RGB").resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
    return ((a - _MEAN) / _STD).transpose(2, 0, 1)


def tensor_from_normalized_file(path: Path) -> np.ndarray:
    """Input from an already-normalized copy (what the training embeddings were computed from)."""
    with Image.open(path) as im:
        return tensor_from_image(im)


def tensor_from_original(path: Path, cfg: ImageInputConfig) -> np.ndarray:
    """Input for a NEW photo: normalize exactly like the tuning/training copies, then embed."""
    prep = normalize_for_tuning(path, cfg)
    with Image.open(io.BytesIO(prep.data)) as im:
        return tensor_from_image(im)


class ClipEmbedder:
    """Lazy CLIP image embedder (needs torch + transformers; see the `probe` extra)."""

    def __init__(self, model_name: str = CLIP_MODEL_NAME, threads: int = 4) -> None:
        self.model_name, self.threads = model_name, threads
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                import torch
                from transformers import CLIPModel
            except ImportError as exc:
                raise ModelArtifactError("the image score needs torch and transformers (pip install -e .[probe])") from exc
            torch.set_num_threads(max(1, self.threads))
            self._model = CLIPModel.from_pretrained(self.model_name).eval()
        return self._model

    def embed_tensors(self, tensors: np.ndarray) -> np.ndarray:
        import torch

        model = self._load()
        with torch.no_grad():
            f = model.get_image_features(pixel_values=torch.from_numpy(tensors))
            f = f if isinstance(f, torch.Tensor) else f.pooler_output
            return torch.nn.functional.normalize(f, dim=-1).numpy()


class ImageScorer:
    """P(approved) from the photo alone. ``embed`` maps a (1, 3, 224, 224) array to a (1, d) embedding."""

    def __init__(self, model: Any, metadata: dict[str, Any], embed: Callable[[np.ndarray], np.ndarray],
                 image_cfg: ImageInputConfig) -> None:
        self._model, self.metadata, self._embed, self._cfg = model, metadata, embed, image_cfg

    @classmethod
    def load(cls, directory: Path, image_cfg: ImageInputConfig,
             embedder: ClipEmbedder | None = None) -> ImageScorer:
        directory = Path(directory)
        if not (directory / "model.joblib").exists() or not (directory / "metadata.json").exists():
            raise ModelArtifactError("image score model not found; run scripts/build_combiner.py")
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if meta.get("normalize_long_side_px") != image_cfg.tuning_normalize_long_side_px:
            raise ModelArtifactError(
                "image normalization changed since the image score model was trained "
                f"({meta.get('normalize_long_side_px')} -> {image_cfg.tuning_normalize_long_side_px}); retrain it")
        emb = embedder or ClipEmbedder(meta.get("embedding_model", CLIP_MODEL_NAME))
        return cls(joblib.load(directory / "model.joblib"), meta, emb.embed_tensors, image_cfg)

    @property
    def version(self) -> str:
        return str(self.metadata["model_version"])

    def score_embedding(self, embedding: np.ndarray) -> float:
        return float(self._model.predict_proba(np.asarray(embedding, dtype=float).reshape(1, -1))[0, 1])

    def score(self, image_path: Path) -> float:
        tensor = tensor_from_original(image_path, self._cfg)[None, ...]
        return self.score_embedding(self._embed(tensor)[0])
