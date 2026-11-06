"""Probability calibration (scikit-learn) and the rule deciding whether we may call it calibrated.

Calibration is fitted by *cross-fitting on the training split* (group-aware folds), so the
validation split stays untouched for honest threshold selection / ECE measurement, and the test
split is never seen.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold

from ..config import CalibrationConfig
from ..exceptions import DataError


def fit_calibrated(
    base: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    method: str,
    n_splits: int = 5,
    seed: int = 42,
) -> CalibratedClassifierCV:
    """Cross-fit ``base`` + calibrator on (X, y). Folds never split a group."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        raise DataError("need both classes to fit a calibrated model")
    min_class = int(min((y == 0).sum(), (y == 1).sum()))
    k = max(2, min(n_splits, min_class, len(np.unique(groups))))
    folds = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y, groups))
    cal = CalibratedClassifierCV(estimator=base, method=method, cv=folds)
    cal.fit(X, y)
    return cal


def calibration_status(
    n_train: int, n_pos: int, n_neg: int, validation_ece: float | None, cfg: CalibrationConfig,
    validation_auc: float | None = None,
) -> tuple[str, list[str]]:
    """Return ("calibrated" | "uncalibrated_classifier", reasons).

    Only claim calibration if there is enough data, measured validation ECE is acceptable, and the model
    actually discriminates (a no-skill model that outputs the base rate can have a tiny ECE but is not a
    usable approval probability).
    """
    reasons: list[str] = []
    if validation_auc is not None and validation_auc < cfg.min_validation_roc_auc:
        reasons.append(f"validation ROC-AUC {validation_auc:.3f} < {cfg.min_validation_roc_auc} (no useful discrimination)")
    if n_train < cfg.min_samples_for_calibration:
        reasons.append(f"only {n_train} training samples (< {cfg.min_samples_for_calibration})")
    if min(n_pos, n_neg) < cfg.min_samples_per_class:
        reasons.append(f"minority class has {min(n_pos, n_neg)} samples (< {cfg.min_samples_per_class})")
    if validation_ece is None or np.isnan(validation_ece):
        reasons.append("validation ECE unavailable")
    elif validation_ece > cfg.max_ece_for_calibrated_claim:
        reasons.append(f"validation ECE {validation_ece:.3f} > {cfg.max_ece_for_calibrated_claim}")
    return ("calibrated" if not reasons else "uncalibrated_classifier"), reasons


def describe(meta: dict[str, Any]) -> str:
    return f"{meta['calibration_status']} ({'; '.join(meta['calibration_reasons']) or 'meets all criteria'})"
