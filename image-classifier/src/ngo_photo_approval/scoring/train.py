"""Train the interpretable probability model (logistic regression) + calibration.

Inputs are the TRAIN and VALIDATION splits only; there is no way to pass test data in.
Label convention: y = 1 for human ``approved``, 0 for ``not_approved``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import CalibrationConfig
from ..evaluation.calibration_metrics import calibration_summary, plot_reliability
from ..exceptions import DataError
from .calibration import calibration_status, fit_calibrated

logger = logging.getLogger(__name__)

DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)


def make_base(c: float, seed: int) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),  # fitted on training folds only (inside the pipeline)
        ("lr", LogisticRegression(C=c, max_iter=5000, random_state=seed)),
    ])


def train_probability_model(
    X_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feature_names: list[str], calib_cfg: CalibrationConfig, out_dir: Path,
    seed: int = 42, c_grid: tuple[float, ...] = DEFAULT_C_GRID,
    extra_meta: dict[str, Any] | None = None, calibration_dir: Path | None = None,
) -> dict[str, Any]:
    for name, y in (("train", y_train), ("validation", y_val)):
        if len(np.unique(y)) < 2:
            raise DataError(f"{name} split must contain both classes")
    if X_train.shape[1] != len(feature_names):
        raise DataError("feature matrix / names mismatch")

    # 1) choose C by validation Brier of the cross-fitted calibrated model
    scores: dict[float, float] = {}
    for c in c_grid:
        cal = fit_calibrated(make_base(c, seed), X_train, y_train, groups_train, calib_cfg.method, seed=seed)
        scores[c] = float(brier_score_loss(y_val, cal.predict_proba(X_val)[:, 1]))
    best_c = min(scores, key=scores.get)  # type: ignore[arg-type]

    # 2) final models: raw (all train) and cross-fitted calibrated
    base = make_base(best_c, seed).fit(X_train, y_train)
    cal = fit_calibrated(make_base(best_c, seed), X_train, y_train, groups_train, calib_cfg.method, seed=seed)

    raw_val = base.predict_proba(X_val)[:, 1]
    cal_val = cal.predict_proba(X_val)[:, 1]
    raw_summary = calibration_summary(y_val, raw_val)
    cal_summary = calibration_summary(y_val, cal_val)

    n_pos, n_neg = int((y_train == 1).sum()), int((y_train == 0).sum())
    val_auc = float(roc_auc_score(y_val, cal_val))
    status, reasons = calibration_status(len(y_train), n_pos, n_neg, cal_summary["ece"], calib_cfg, val_auc)

    meta: dict[str, Any] = {
        "model_version": "prob-" + hashlib.sha256(
            json.dumps([feature_names, best_c, seed, len(y_train)]).encode()).hexdigest()[:10],
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "algorithm": "logistic_regression", "C": best_c, "c_grid_validation_brier": {str(k): v for k, v in scores.items()},
        "calibration_method": calib_cfg.method, "calibration_status": status, "calibration_reasons": reasons,
        "calibration_fit": "cross-fitted on train split with group-aware folds",
        "feature_names": feature_names, "seed": seed,
        "n_train": int(len(y_train)), "n_train_approved": n_pos, "n_train_not_approved": n_neg,
        "n_validation": int(len(y_val)),
        "validation": {
            "raw": {k: raw_summary[k] for k in ("brier", "ece", "mce")},
            "calibrated": {k: cal_summary[k] for k in ("brier", "ece", "mce")},
            "roc_auc": val_auc,
            "pr_auc": float(average_precision_score(y_val, cal_val)),
            "approved_base_rate": float(np.mean(y_val)),
        },
        "uses_test_data": False,
        **(extra_meta or {}),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"base": base, "calibrated": cal, "feature_names": feature_names}, out_dir / "model.joblib")
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    cal_dir = calibration_dir or out_dir
    cal_dir.mkdir(parents=True, exist_ok=True)
    (cal_dir / "validation_reliability.json").write_text(
        json.dumps({"raw": raw_summary, "calibrated": cal_summary}, indent=2), encoding="utf-8")
    plot_reliability({"raw (validation)": (y_val, raw_val), "calibrated (validation)": (y_val, cal_val)},
                     cal_dir / "reliability_validation.png")
    logger.info("trained %s: C=%s status=%s val_brier=%.4f val_ece=%.4f",
                meta["model_version"], best_c, status, cal_summary["brier"], cal_summary["ece"])
    return meta
