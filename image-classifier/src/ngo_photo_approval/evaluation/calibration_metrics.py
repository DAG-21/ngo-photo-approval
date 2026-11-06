"""Calibration diagnostics: reliability bins, ECE/MCE, Brier score, reliability diagram."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss


def reliability_bins(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    y_true, p = np.asarray(y_true, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            bins.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                         "mean_predicted": float(p[m].mean()), "observed_rate": float(y_true[m].mean())})
    return bins


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    n = len(p)
    if n == 0:
        return float("nan")
    return float(sum(b["n"] / n * abs(b["mean_predicted"] - b["observed_rate"]) for b in reliability_bins(y_true, p, n_bins)))


def max_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = reliability_bins(y_true, p, n_bins)
    return float(max((abs(b["mean_predicted"] - b["observed_rate"]) for b in bins), default=float("nan")))


def calibration_summary(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    return {
        "n": int(len(p)),
        "brier": float(brier_score_loss(y_true, p)) if len(p) else None,
        "ece": expected_calibration_error(y_true, p, n_bins),
        "mce": max_calibration_error(y_true, p, n_bins),
        "bins": reliability_bins(y_true, p, n_bins),
    }


def plot_reliability(series: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, n_bins: int = 10) -> None:
    """Save a reliability diagram. ``series`` maps a name to (y_true, probabilities)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    for name, (y, p) in series.items():
        bins = reliability_bins(y, p, n_bins)
        ax.plot([b["mean_predicted"] for b in bins], [b["observed_rate"] for b in bins], "o-", label=name)
    ax.set_xlabel("Mean predicted approval probability")
    ax.set_ylabel("Observed approval rate")
    ax.legend()
    ax.set_title("Reliability diagram")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
