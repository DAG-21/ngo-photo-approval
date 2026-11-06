"""Evaluation metrics. Convention: positive class = human ``approved``.

* false approval rate (FAR)   = P(system approves | human not_approved)   <- primary business risk
* false rejection rate (FRR)  = P(system rejects  | human approved)
* approval precision          = P(human approved | system approves)

Predictions that are not APPROVE/REJECT (REVIEW, provider/validation errors) are *unresolved*.
Two views are reported: ``resolved`` (unresolved dropped) and ``strict`` (unresolved counted as
REJECT, i.e. nothing is approved unless the system explicitly approved it).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ..evaluation.calibration_metrics import expected_calibration_error


def _safe_div(a: float, b: float) -> float | None:
    return float(a / b) if b else None


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """y_*: 1 = approved, 0 = not_approved."""
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    n = len(y_true)
    if n == 0:
        return {"n": 0}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return {
        "n": int(n),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                             "layout": "rows=human[not_approved,approved], cols=system[reject,approve]"},
        "accuracy": float((tp + tn) / n),
        "approval_precision": _safe_div(tp, tp + fp),
        "approval_recall": _safe_div(tp, tp + fn),
        "approval_f1": float(f[1]),
        "false_approval_rate": _safe_div(fp, fp + tn),
        "false_rejection_rate": _safe_div(fn, fn + tp),
        "per_class": {
            "not_approved": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0]), "support": int(s[0])},
            "approved": {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f[1]), "support": int(s[1])},
        },
    }


def probability_metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y_true, p = np.asarray(y_true, int), np.asarray(p, float)
    out: dict[str, Any] = {"n": int(len(y_true))}
    if len(y_true) == 0:
        return out
    out["brier"] = float(brier_score_loss(y_true, p))
    out["ece"] = expected_calibration_error(y_true, p)
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, p))
        out["pr_auc"] = float(average_precision_score(y_true, p))
    else:
        out["roc_auc"] = out["pr_auc"] = None
    return out


def evaluate_predictions(df: pd.DataFrame) -> dict[str, Any]:
    """``df`` needs: label (approved/not_approved), decision (APPROVE/REJECT/REVIEW/ERROR),
    optional probability (0-1), optional latency_s, cost_usd."""
    d = df.copy()
    d["y"] = (d["label"] == "approved").astype(int)
    resolved = d["decision"].isin(["APPROVE", "REJECT"])
    total = len(d)
    out: dict[str, Any] = {
        "n_total": int(total),
        "processing": {
            "automatic": int(resolved.sum()),
            "automatic_rate": _safe_div(resolved.sum(), total),
            "review": int((d["decision"] == "REVIEW").sum()),
            "errors": int((d["decision"] == "ERROR").sum()),
            "unresolved_rate": _safe_div((~resolved).sum(), total),
        },
        "resolved": classification_metrics(d.loc[resolved, "y"], (d.loc[resolved, "decision"] == "APPROVE").astype(int)),
        "strict": classification_metrics(d["y"], (d["decision"] == "APPROVE").astype(int)),
    }
    if "probability" in d and d["probability"].notna().any():
        m = d["probability"].notna()
        out["probability"] = probability_metrics(d.loc[m, "y"], d.loc[m, "probability"])
    if "latency_s" in d and d["latency_s"].notna().any():
        lat = d["latency_s"].dropna()
        out["latency_s"] = {"mean": float(lat.mean()), "p50": float(lat.quantile(0.5)), "p95": float(lat.quantile(0.95))}
    if "cost_usd" in d and d["cost_usd"].notna().any():
        out["cost_usd"] = {"total": float(d["cost_usd"].sum()), "per_image": float(d["cost_usd"].mean())}
    else:
        out["cost_usd"] = "unknown (pricing not configured or usage not returned)"
    return out


def select_threshold(
    y_true: np.ndarray, p: np.ndarray, max_far: float, min_recall: float = 0.0, step: float = 0.01
) -> dict[str, Any]:
    """Pick the approval threshold on VALIDATION data.

    Chooses the threshold with the highest approval recall subject to false-approval-rate <=
    ``max_far`` and recall >= ``min_recall``. If no threshold satisfies both, returns the
    threshold with the lowest false-approval rate and ``feasible=False`` so the caller can
    escalate instead of silently relaxing the client's risk limit.
    """
    y_true, p = np.asarray(y_true, int), np.asarray(p, float)
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 6)
    rows = []
    for t in grid:
        pred = (p + 1e-9 >= t).astype(int)
        m = classification_metrics(y_true, pred)
        rows.append({"threshold": float(t), "far": m["false_approval_rate"], "recall": m["approval_recall"],
                     "precision": m["approval_precision"]})
    feasible = [r for r in rows if r["far"] is not None and r["far"] <= max_far
                and r["recall"] is not None and r["recall"] >= min_recall]
    if feasible:
        best = max(feasible, key=lambda r: (r["recall"], r["threshold"]))
        return {"threshold": best["threshold"], "feasible": True, "validation_far": best["far"],
                "validation_recall": best["recall"], "validation_precision": best["precision"],
                "max_false_approval_rate": max_far, "min_recall": min_recall}
    best = min((r for r in rows if r["far"] is not None), key=lambda r: (r["far"], -r["recall"]))
    return {"threshold": best["threshold"], "feasible": False, "validation_far": best["far"],
            "validation_recall": best["recall"], "validation_precision": best["precision"],
            "max_false_approval_rate": max_far, "min_recall": min_recall,
            "note": "no threshold meets the false-approval limit on validation data"}


def rule_level_metrics(predicted: pd.DataFrame, verified: pd.DataFrame) -> dict[str, Any]:
    """Per-rule agreement against INDEPENDENTLY VERIFIED labels only.

    ``predicted``: image_id, rule_id, status. ``verified``: image_id, rule_id, status, verified.
    """
    v = verified[verified["verified"].astype(str).str.lower().isin({"true", "1", "yes"})]
    m = predicted.merge(v[["image_id", "rule_id", "status"]], on=["image_id", "rule_id"], suffixes=("_pred", "_true"))
    out: dict[str, Any] = {}
    for rid, g in m.groupby("rule_id"):
        out[rid] = {
            "n": int(len(g)),
            "accuracy": float((g["status_pred"] == g["status_true"]).mean()),
            "uncertain_rate": float((g["status_pred"] == "UNCERTAIN").mean()),
            "pass_precision": _safe_div(((g["status_pred"] == "PASS") & (g["status_true"] == "PASS")).sum(),
                                        (g["status_pred"] == "PASS").sum()),
            "fail_recall": _safe_div(((g["status_pred"] == "FAIL") & (g["status_true"] == "FAIL")).sum(),
                                     (g["status_true"] == "FAIL").sum()),
            "confusion": pd.crosstab(g["status_true"], g["status_pred"]).to_dict("index"),
        }
    return out
