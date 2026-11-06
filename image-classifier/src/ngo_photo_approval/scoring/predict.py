"""Inference-time probability estimation with honest labelling of what the number means."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel

from ..exceptions import ModelArtifactError
from ..schemas import ProbabilityType


class ProbabilityEstimate(BaseModel):
    raw_classifier_probability: float
    calibrated_probability: float | None
    probability: float
    probability_type: ProbabilityType
    scoring_model_version: str


class ProbabilityScorer:
    """Loads artifacts written by ``scoring.train``. ``calibrated`` is only reported when the
    training metadata says calibration was justified (``calibration_status == "calibrated"``)."""

    def __init__(self, bundle: dict[str, Any], metadata: dict[str, Any]) -> None:
        self._bundle, self.metadata = bundle, metadata
        self.feature_names: list[str] = list(metadata["feature_names"])

    @classmethod
    def load(cls, directory: Path) -> ProbabilityScorer:
        model_path, meta_path = Path(directory) / "model.joblib", Path(directory) / "metadata.json"
        if not model_path.exists() or not meta_path.exists():
            raise ModelArtifactError(
                "probability model artifacts not found; run scripts/train_probability_model.py")
        try:
            return cls(joblib.load(model_path), json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            raise ModelArtifactError(f"cannot load probability model: {type(exc).__name__}") from exc

    @property
    def version(self) -> str:
        return str(self.metadata["model_version"])

    def check_features(self, expected: list[str]) -> None:
        if list(expected) != self.feature_names:
            raise ModelArtifactError(
                "probability model was trained with a different feature set / rule set; retrain it")

    def score(self, features: dict[str, float]) -> ProbabilityEstimate:
        missing = [n for n in self.feature_names if n not in features]
        if missing:
            raise ModelArtifactError(f"missing features: {missing[:3]}")
        x = np.array([[features[n] for n in self.feature_names]], dtype=float)
        raw = float(self._bundle["base"].predict_proba(x)[0, 1])
        calibrated_ok = self.metadata.get("calibration_status") == "calibrated"
        cal = float(self._bundle["calibrated"].predict_proba(x)[0, 1]) if calibrated_ok else None
        p = cal if cal is not None else raw
        return ProbabilityEstimate(
            raw_classifier_probability=min(1.0, max(0.0, raw)),
            calibrated_probability=None if cal is None else min(1.0, max(0.0, cal)),
            probability=min(1.0, max(0.0, p)),
            probability_type=ProbabilityType.CALIBRATED if cal is not None else ProbabilityType.UNCALIBRATED_CLASSIFIER,
            scoring_model_version=self.version,
        )
