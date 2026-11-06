"""
Phase 5 (Step 5.2): Historical Few-Shot Vector Retrieval Engine
- Stores visual embeddings from OpenCLIP (default ViT-B-32 / laion2b_s34b_b79k, see
  `few_shot_retrieval` in evaluation_config.yaml). If `open_clip`/`torch` or the weights are
  unavailable, falls back (with a logged warning) to a deterministic spatial-color-texture vector.
  The backend is chosen once per retriever so index and query vectors always share one space.
- Strictly excludes TEST partition images from the reference library
- When retrieving few-shot examples for a target image:
  1. Excludes the target image itself and any image from the same `event_id` or burst
  2. Avoids near-duplicate references (enforces minimum distance between selected examples)
  3. Retrieves a balanced set of 2-3 historically APPROVED and 2-3 historically REJECTED examples
  4.Clearly distinguishes `confirmed_rejection_reasons` (human-verified) from `potential_rejection_factors` (model-observed)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import yaml
from PIL import Image, ImageOps

from src.paths import resolve_image_path

logger = logging.getLogger(__name__)

BACKEND_OPENCLIP = "openclip"
BACKEND_FALLBACK = "fallback"


class HistoricalFewShotRetriever:
    def __init__(
        self,
        embedding_dim: int = 512,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
        use_openclip: bool = True,
        num_approved: int = 3,
        num_rejected: int = 3,
        min_cosine_distance_dedup: float = 0.02,
    ):
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.use_openclip = use_openclip
        self.num_approved = num_approved
        self.num_rejected = num_rejected
        self.min_cosine_distance_dedup = min_cosine_distance_dedup
        self.backend: Optional[str] = None
        self.entries: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self._clip_model = None
        self._clip_preprocess = None
        self._torch = None

    @classmethod
    def from_config(cls, eval_config_path: Path, **overrides: Any) -> "HistoricalFewShotRetriever":
        """Builds a retriever from the `few_shot_retrieval` block of evaluation_config.yaml."""
        cfg = yaml.safe_load(Path(eval_config_path).read_text(encoding="utf-8")).get("few_shot_retrieval", {})
        kwargs: Dict[str, Any] = {
            "model_name": cfg.get("embedding_model", "ViT-B-32"),
            "pretrained": cfg.get("pretrained", "laion2b_s34b_b79k"),
            "use_openclip": cfg.get("enabled", True),
            "num_approved": cfg.get("num_approved_examples", 3),
            "num_rejected": cfg.get("num_rejected_examples", 3),
            "min_cosine_distance_dedup": cfg.get("min_cosine_distance_dedup", 0.02),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def _resolve_backend(self) -> str:
        """Loads OpenCLIP once; on any failure pins the retriever to the fallback descriptor."""
        if self.backend is not None:
            return self.backend
        if not self.use_openclip:
            self.backend = BACKEND_FALLBACK
            return self.backend
        try:
            import open_clip
            import torch

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained, device=device
            )
            model.eval()
            self._clip_model, self._clip_preprocess, self._torch = model, preprocess, torch
            self.device = device
            self.embedding_dim = int(model.visual.output_dim)
            self.backend = BACKEND_OPENCLIP
            logger.info(
                "Using OpenCLIP %s (%s) on %s, dim=%d", self.model_name, self.pretrained, device, self.embedding_dim
            )
        except Exception as exc:
            logger.warning(
                "OpenCLIP unavailable (%s: %s); falling back to handcrafted color/texture embeddings. "
                "Few-shot retrieval will match on color/texture, not semantic content.",
                type(exc).__name__,
                exc,
            )
            self.backend = BACKEND_FALLBACK
        return self.backend

    def _fallback_embedding(self, image_path: Path) -> np.ndarray:
        """Multi-scale spatial color/gradient descriptor padded/truncated to `embedding_dim`."""
        with Image.open(image_path) as img:
            rgb = img.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0

        # Spatial grid features (8x8x3 = 192) + channel histograms (3x32 = 96) + gradient energy
        small = np.asarray(
            Image.fromarray((arr * 255).astype(np.uint8)).resize((8, 8), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).flatten() / 255.0
        hist_r, _ = np.histogram(arr[:, :, 0], bins=32, range=(0, 1), density=True)
        hist_g, _ = np.histogram(arr[:, :, 1], bins=32, range=(0, 1), density=True)
        hist_b, _ = np.histogram(arr[:, :, 2], bins=32, range=(0, 1), density=True)
        gx = np.abs(np.diff(arr, axis=1)).mean(axis=(0, 1))
        gy = np.abs(np.diff(arr, axis=0)).mean(axis=(0, 1))

        raw_vec = np.concatenate([small, hist_r, hist_g, hist_b, gx, gy])
        if len(raw_vec) < self.embedding_dim:
            pad = np.zeros(self.embedding_dim - len(raw_vec), dtype=np.float32)
            vec = np.concatenate([raw_vec, pad])
        else:
            vec = raw_vec[: self.embedding_dim]

        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 1e-8 else vec

    def compute_visual_embeddings(self, image_paths: List[Path], batch_size: int = 32) -> np.ndarray:
        """Returns an (N, dim) array of L2-normalised embeddings from the active backend."""
        if self._resolve_backend() == BACKEND_FALLBACK:
            if not image_paths:
                return np.zeros((0, self.embedding_dim), dtype=np.float32)
            return np.vstack([self._fallback_embedding(p) for p in image_paths]).astype(np.float32)

        torch = self._torch
        chunks: List[np.ndarray] = []
        for start in range(0, len(image_paths), batch_size):
            batch = []
            for p in image_paths[start : start + batch_size]:
                with Image.open(p) as img:
                    batch.append(self._clip_preprocess(ImageOps.exif_transpose(img).convert("RGB")))
            with torch.no_grad():
                feats = self._clip_model.encode_image(torch.stack(batch).to(self.device))
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            chunks.append(feats.cpu().numpy().astype(np.float32))
        if not chunks:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return np.vstack(chunks)

    def compute_visual_embedding(self, image_path: Path) -> np.ndarray:
        """Single-image convenience wrapper around `compute_visual_embeddings`."""
        return self.compute_visual_embeddings([image_path])[0]

    def build_index(
        self,
        enriched_jsonl_path: Path,
        metadata_df_path: Path,
        allowed_partitions: Tuple[str, ...] = ("TRAIN",),
    ) -> int:
        """
        Indexes historical images strictly from `allowed_partitions` (default: TRAIN only)
        so that TEST images are completely isolated from few-shot retrieval.
        """
        import pandas as pd

        meta_df = pd.read_csv(metadata_df_path).set_index("image_id").to_dict(orient="index")
        self.entries = []
        img_paths: List[Path] = []

        with open(enriched_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                img_id = rec["image_id"]
                meta = meta_df.get(img_id)
                if not meta or meta.get("partition") not in allowed_partitions:
                    continue
                if meta.get("is_exact_duplicate"):
                    continue

                img_path = resolve_image_path(meta["original_filepath"])
                img_paths.append(img_path)
                self.entries.append({
                    "image_id": img_id,
                    "filepath": str(img_path),
                    "historical_decision": rec["historical_decision"],
                    "event_id": rec["metadata"].get("event_id"),
                    "all_tags": (
                        rec["technical_tags"]
                        + rec["composition_tags"]
                        + rec["editorial_tags"]
                        + rec["brand_tags"]
                        + rec["safety_tags"]
                        + rec["context_tags"]
                    ),
                    "potential_rejection_factors": rec["potential_rejection_factors"],
                    "human_validation": rec["human_validation"],
                })

        self.embeddings = self.compute_visual_embeddings(img_paths)
        return len(self.entries)

    def retrieve_examples(
        self,
        target_image_path: Path,
        target_image_id: Optional[str] = None,
        target_event_id: Optional[str] = None,
        num_approved: Optional[int] = None,
        num_rejected: Optional[int] = None,
        min_cosine_distance_dedup: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves `num_approved` historically APPROVED and `num_rejected` historically REJECTED
        reference photographs while excluding same-event/burst images and near-duplicates.
        Unspecified limits default to the values the retriever was configured with.
        """
        num_approved = self.num_approved if num_approved is None else num_approved
        num_rejected = self.num_rejected if num_rejected is None else num_rejected
        min_cosine_distance_dedup = (
            self.min_cosine_distance_dedup if min_cosine_distance_dedup is None else min_cosine_distance_dedup
        )

        if self.embeddings is None or len(self.entries) == 0:
            return []

        q_emb = self.compute_visual_embedding(target_image_path)
        sims = np.dot(self.embeddings, q_emb)
        ranked_indices = np.argsort(-sims)

        selected_approved: List[Dict[str, Any]] = []
        selected_rejected: List[Dict[str, Any]] = []
        selected_vecs: List[np.ndarray] = []

        for idx in ranked_indices:
            cand = self.entries[int(idx)]
            if target_image_id and cand["image_id"] == target_image_id:
                continue
            if target_event_id and cand.get("event_id") == target_event_id:
                continue

            c_vec = self.embeddings[int(idx)]
            # Deduplicate against already chosen few-shot references
            if any((1.0 - float(np.dot(c_vec, sv))) < min_cosine_distance_dedup for sv in selected_vecs):
                continue

            human_val = cand["human_validation"]
            confirmed_reasons = human_val.get("confirmed_rejection_reasons", [])
            reason_status = (
                f"Human-confirmed reasons: {', '.join(confirmed_reasons)}"
                if human_val.get("reviewed") and confirmed_reasons
                else (
                    f"Unconfirmed model-observed potential factors: {', '.join(cand['potential_rejection_factors'])} "
                    "(Original historical rejection reason unknown)"
                    if cand["historical_decision"] == "REJECTED"
                    else "Historically approved benchmark"
                )
            )

            formatted_ref = {
                "reference_image_id": cand["image_id"],
                "filepath": cand["filepath"],
                "historical_decision": cand["historical_decision"],
                "similarity_score": round(float(sims[int(idx)]), 4),
                "observed_tags": cand["all_tags"],
                "reason_attribution": reason_status,
                "human_confirmed": bool(human_val.get("reviewed")),
            }

            if cand["historical_decision"] == "APPROVED" and len(selected_approved) < num_approved:
                selected_approved.append(formatted_ref)
                selected_vecs.append(c_vec)
            elif cand["historical_decision"] == "REJECTED" and len(selected_rejected) < num_rejected:
                selected_rejected.append(formatted_ref)
                selected_vecs.append(c_vec)

            if len(selected_approved) >= num_approved and len(selected_rejected) >= num_rejected:
                break

        return selected_approved + selected_rejected
