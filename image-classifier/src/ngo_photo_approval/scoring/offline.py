"""Offline (dataset-level) helpers that mirror inference exactly.

Features for training/evaluation come from the same ``assess`` / ``build_features`` code path as
live inference, and from *saved model outputs* (never from human annotations).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import PolicyConfig, ProjectConfig
from ..exceptions import DataError
from ..rules.evaluator import evaluate_rules
from ..schemas import RuleCheck, RuleSet
from ..scoring.feature_builder import build_features, feature_names

QUALITY_COLUMNS = ["blur_laplacian_var", "brightness", "contrast", "megapixels", "aspect_ratio"]


def load_split(project: ProjectConfig, name: str) -> pd.DataFrame:
    path = project.output_dir / "splits" / f"{name}.csv"
    if not path.exists():
        raise DataError(f"{path.name} not found; run scripts/prepare_dataset.py")
    return pd.read_csv(path)


def load_manifest(project: ProjectConfig) -> pd.DataFrame:
    path = project.output_dir / "manifests" / "image_manifest.csv"
    if not path.exists():
        raise DataError("image_manifest.csv not found; run scripts/audit_dataset.py")
    return pd.read_csv(path, low_memory=False)


def quality_lookup(manifest: pd.DataFrame) -> dict[str, dict[str, float]]:
    m = manifest.dropna(subset=QUALITY_COLUMNS).set_index("image_id")
    return {i: {c: float(r[c]) for c in QUALITY_COLUMNS} for i, r in m[QUALITY_COLUMNS].iterrows()}


def load_analyses(path: Path) -> dict[str, dict[str, Any]]:
    """image_id -> last AnalysisResult record from a JSONL file (later lines supersede earlier)."""
    out: dict[str, dict[str, Any]] = {}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[rec["image_id"]] = rec
    return out


def checks_of(record: Mapping[str, Any]) -> list[RuleCheck] | None:
    if not record.get("ok") or not record.get("analysis"):
        return None
    return [RuleCheck.model_validate(c) for c in record["analysis"]["checks"]]


def feature_table(
    split: pd.DataFrame,
    rule_set: RuleSet,
    policy: PolicyConfig,
    feature_set: str,
    quality: Mapping[str, Mapping[str, float]],
    analyses: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, int]]:
    """Return (X, y, groups, image_ids, drop_counts). y: 1 = human approved.

    Images lacking a successful analysis (rules feature sets) or quality metrics are DROPPED from
    the table and counted - they are infrastructure failures, not training signal.
    """
    names = feature_names(rule_set, feature_set)
    rows, y, groups, ids = [], [], [], []
    drops = {"no_analysis": 0, "analysis_failed": 0, "no_quality": 0}
    for r in split.itertuples():
        policy_result = None
        if "rules" in feature_set:
            rec = (analyses or {}).get(r.image_id)
            if rec is None:
                drops["no_analysis"] += 1
                continue
            checks = checks_of(rec)
            if checks is None:
                drops["analysis_failed"] += 1
                continue
            policy_result = evaluate_rules(rule_set, checks, policy, getattr(r, "program", None))
        q = quality.get(r.image_id)
        if "quality" in feature_set and q is None:
            drops["no_quality"] += 1
            continue
        f = build_features(rule_set, feature_set, policy_result, q)
        rows.append([f[n] for n in names])
        y.append(int(r.label == "approved"))
        groups.append(r.group_id)
        ids.append(r.image_id)
    return np.array(rows, dtype=float), np.array(y), np.array(groups), ids, drops


# --------------------------------------------------------------------------- image embeddings (stage 1 of the combiner)


def embedding_cache_path(project: ProjectConfig) -> Path:
    return project.output_dir / "cache" / "embeddings_clip_b32.npz"


def unique_trainval(project: ProjectConfig) -> pd.DataFrame:
    """One row per unique photo content in train + validation. The TEST split is never included."""
    sp = pd.concat([load_split(project, s) for s in ("train", "validation")])
    man = load_manifest(project)
    d = sp.drop(columns=["file_size"], errors="ignore").merge(
        man[["image_id", "megapixels", "file_size"]], on="image_id", how="left")
    return d.sort_values("image_id").drop_duplicates("sha256").reset_index(drop=True)


def load_embeddings(project: ProjectConfig) -> dict[str, np.ndarray]:
    path = embedding_cache_path(project)
    if not path.exists():
        raise DataError("no embeddings cached; run `python scripts/embedding_probe.py embed`")
    z = np.load(path, allow_pickle=False)
    return dict(zip(z["ids"].tolist(), z["emb"], strict=True))
