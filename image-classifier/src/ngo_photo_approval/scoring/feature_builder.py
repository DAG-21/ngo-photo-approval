"""Feature construction for the probability model.

The SAME function is used for training, evaluation and inference. Inputs are only things that
exist at deployment time: rule-engine outputs on VLM results, and deterministic image-quality
measures. Human decisions and ground-truth rule annotations are never features.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..data.inventory import quality_metrics
from ..data.preprocessing import open_image
from ..schemas import CheckStatus, PolicyResult, RuleSet

FEATURE_VERSION = "features-v1"
FEATURE_SETS = ("quality", "rules", "rules+quality", "image", "image+rules")
# "image": logit of the learned image score (scoring/image_score.py: CLIP embedding -> logistic regression trained on the
# human labels). Must come from OUT-OF-FOLD predictions when training the combiner, never from a model that saw the photo.
IMAGE_FEATURES = ["img_logit"]
# NOTE: resolution/file-size are deliberately NOT features. In this dataset they separate the labels almost as well as the
# whole quality model (file size alone: validation ROC-AUC ~0.80) because of image provenance, not rule compliance.
QUALITY_FEATURES = ["q_log_blur", "q_brightness", "q_contrast", "q_aspect_ratio"]


def compute_quality(path: Path, max_side: int = 1024) -> dict[str, float]:
    """Deterministic quality metrics, computed exactly as in the dataset inventory
    (decoded at <= max_side), so train and inference see the same distribution."""
    im = open_image(path)
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    small = im if scale == 1.0 else im.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    gray = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2GRAY)
    q = quality_metrics(gray)
    q["megapixels"] = w * h / 1e6
    q["aspect_ratio"] = w / h
    return q


def quality_features(q: Mapping[str, float]) -> dict[str, float]:
    return {
        "q_log_blur": math.log1p(max(float(q["blur_laplacian_var"]), 0.0)),
        "q_brightness": float(q["brightness"]) / 255.0,
        "q_contrast": float(q["contrast"]) / 255.0,
        "q_aspect_ratio": float(q["aspect_ratio"]),
    }


def image_logit(p: float, eps: float = 1e-4) -> float:
    """logit of a probability, clipped away from 0/1 so the feature stays finite."""
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def rule_feature_names(rule_set: RuleSet) -> list[str]:
    names: list[str] = []
    for r in rule_set.active():
        names += [f"{r.rule_id}_pass", f"{r.rule_id}_fail", f"{r.rule_id}_uncertain", f"{r.rule_id}_missing"]
    return names + ["n_fail", "n_uncertain", "n_missing", "frac_pass"]


def feature_names(rule_set: RuleSet, feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
    names: list[str] = []
    if "image" in feature_set:
        names += IMAGE_FEATURES
    if "rules" in feature_set:
        names += rule_feature_names(rule_set)
    if "quality" in feature_set:
        names += QUALITY_FEATURES
    return names


def rule_features(rule_set: RuleSet, policy_result: PolicyResult | None) -> dict[str, float]:
    """One-hot status per active rule. ``None`` (analysis failed) => every rule is 'missing'."""
    active = rule_set.active()
    by_id = {c.rule_id: c.status for c in (policy_result.checks if policy_result else [])}
    feats: dict[str, float] = {}
    n_fail = n_unc = n_miss = n_pass = 0
    for r in active:
        st = by_id.get(r.rule_id)
        # A duplicated rule id in the result is ambiguous: treat as missing.
        if policy_result is not None and r.rule_id in policy_result.missing_rules:
            st = None
        feats[f"{r.rule_id}_pass"] = float(st == CheckStatus.PASS)
        feats[f"{r.rule_id}_fail"] = float(st == CheckStatus.FAIL)
        feats[f"{r.rule_id}_uncertain"] = float(st == CheckStatus.UNCERTAIN)
        feats[f"{r.rule_id}_missing"] = float(st is None)
        n_pass += st == CheckStatus.PASS
        n_fail += st == CheckStatus.FAIL
        n_unc += st == CheckStatus.UNCERTAIN
        n_miss += st is None
    n = max(len(active), 1)
    feats.update(n_fail=float(n_fail), n_uncertain=float(n_unc), n_missing=float(n_miss), frac_pass=n_pass / n)
    return feats


def build_features(
    rule_set: RuleSet,
    feature_set: str,
    policy_result: PolicyResult | None = None,
    quality: Mapping[str, float] | None = None,
    image_score: float | None = None,
) -> dict[str, float]:
    feats: dict[str, Any] = {}
    if "image" in feature_set:
        if image_score is None:
            raise ValueError("an image score is required for this feature set")
        feats["img_logit"] = image_logit(image_score)
    if "rules" in feature_set:
        feats.update(rule_features(rule_set, policy_result))
    if "quality" in feature_set:
        if quality is None:
            raise ValueError("quality metrics required for this feature set")
        feats.update(quality_features(quality))
    return {k: feats[k] for k in feature_names(rule_set, feature_set)}


def to_matrix(rows: Sequence[Mapping[str, float]], names: Sequence[str]) -> np.ndarray:
    return np.array([[r[n] for n in names] for r in rows], dtype=float)


def heuristic_score(policy_result: PolicyResult | None, n_active: int) -> float | None:
    """Fraction of active rules passed, in [0, 1]. A heuristic, NOT a probability."""
    if policy_result is None or n_active == 0:
        return None
    passed = sum(1 for c in policy_result.checks if c.status == CheckStatus.PASS and c.rule_id not in policy_result.missing_rules)
    return min(1.0, passed / n_active)
