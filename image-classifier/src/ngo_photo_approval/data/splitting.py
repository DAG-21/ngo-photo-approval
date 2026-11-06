"""Reproducible, group-aware, class-stratified train/validation/test splitting.

Only images with final labels (approved / not_approved) that passed the audit are split.
`pending` images never appear in any split.

Group = connected component over
  * filename group (label|program|submission index; assumed to be one submission/event),
  * exact duplicates (sha256),
  * near duplicates (dHash cluster).
All members of a group land in the same split, which prevents duplicate / burst leakage.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..config import SplitConfig
from ..exceptions import DataError, LeakageError
from .duplicates import UnionFind

SPLITS = ("train", "validation", "test")


def assign_groups(df: pd.DataFrame, use_filename_groups: bool = True) -> pd.Series:
    """Return a Series (index = df.index) of stable group ids."""
    ids = df["image_id"].tolist()
    uf = UnionFind(ids)

    def link(col: str) -> None:
        sub = df[df[col].notna()]
        for _, members in sub.groupby(col)["image_id"]:
            members = list(members)
            for m in members[1:]:
                uf.union(members[0], m)

    if use_filename_groups and "filename_group" in df:
        link("filename_group")
    if "sha256" in df:
        link("sha256")
    if "duplicate_cluster_id" in df:
        link("duplicate_cluster_id")

    comps = uf.components()
    # Stable ids: ordered by the smallest member id.
    ordered = sorted(comps.values(), key=lambda m: min(m))
    gid = {m: f"grp_{n:05d}" for n, members in enumerate(ordered, start=1) for m in members}
    return df["image_id"].map(gid)


def grouped_stratified_split(
    df: pd.DataFrame, ratios: dict[str, float], seed: int
) -> pd.Series:
    """Greedy group assignment that tracks per-class target counts in each split.

    Groups are processed largest-first (random tiebreak from ``seed``); each goes to the split
    where it most reduces the squared relative deviation from the per-class targets.
    Deterministic for a given (data, ratios, seed).
    """
    if not {"group_id", "label"} <= set(df.columns):
        raise DataError("df needs group_id and label columns")
    classes = sorted(df["label"].unique())
    totals = df["label"].value_counts().to_dict()
    targets = {s: {c: totals[c] * ratios[s] for c in classes} for s in SPLITS}
    current = {s: {c: 0 for c in classes} for s in SPLITS}

    gcounts = df.groupby("group_id")["label"].value_counts().unstack(fill_value=0).reindex(columns=classes, fill_value=0)
    rng = np.random.default_rng(seed)
    order = gcounts.index.to_numpy()
    tiebreak = rng.permutation(len(order))
    sizes = gcounts.sum(axis=1).to_numpy()
    idx = sorted(range(len(order)), key=lambda i: (-sizes[i], tiebreak[i]))

    assignment: dict[str, str] = {}
    for i in idx:
        g = order[i]
        add = gcounts.loc[g].to_dict()
        best, best_delta = None, None
        for s in SPLITS:
            delta = 0.0
            for c in classes:
                t = max(targets[s][c], 1e-9)
                before = ((current[s][c] - targets[s][c]) / t) ** 2
                after = ((current[s][c] + add[c] - targets[s][c]) / t) ** 2
                delta += (after - before) * (totals[c] / sum(totals.values()))
            # prefer the split furthest below target overall on ties
            if best_delta is None or delta < best_delta - 1e-12:
                best, best_delta = s, delta
        assert best is not None
        assignment[g] = best
        for c in classes:
            current[best][c] += add[c]
    return df["group_id"].map(assignment)


def verify_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Raise LeakageError if any image, group, or byte-identical file crosses splits."""
    names = list(splits)
    for col in ("image_id", "group_id", "sha256", "duplicate_cluster_id"):
        sets = {n: set(splits[n][col].dropna()) for n in names if col in splits[n]}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if a in sets and b in sets:
                    inter = sets[a] & sets[b]
                    if inter:
                        raise LeakageError(f"{col} shared between {a} and {b}: {sorted(inter)[:5]} (+{len(inter)} total)")


def verify_no_pending(splits: dict[str, pd.DataFrame]) -> None:
    for n, d in splits.items():
        if (d["label"] == "pending").any() or not d["label"].isin(["approved", "not_approved"]).all():
            raise LeakageError(f"non-final label present in split {n}")


def make_splits(
    manifest: pd.DataFrame, audit: pd.DataFrame, cfg: SplitConfig
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build split frames and a JSON-serializable report."""
    merged = manifest.merge(
        audit[["image_id", "duplicate_cluster_id", "eligible_for_training", "exclusion_reason"]], on="image_id"
    )
    elig = merged[merged["eligible_for_training"]].copy().reset_index(drop=True)
    if elig.empty:
        raise DataError("no eligible images to split")
    elig["group_id"] = assign_groups(elig, cfg.use_filename_groups)
    ratios = {"train": cfg.train, "validation": cfg.validation, "test": cfg.test}
    elig["split"] = grouped_stratified_split(elig, ratios, cfg.seed)

    cols = ["image_id", "rel_path", "label", "program", "group_id", "split", "sha256",
            "duplicate_cluster_id", "width", "height", "file_size"]
    splits = {s: elig[elig["split"] == s][cols].sort_values("image_id").reset_index(drop=True) for s in SPLITS}
    verify_no_leakage(splits)
    verify_no_pending(splits)

    report: dict[str, Any] = {
        "seed": cfg.seed, "ratios": ratios, "use_filename_groups": cfg.use_filename_groups,
        "method": "union-find groups (filename group + sha256 + near-dup cluster); greedy class-stratified group assignment",
        "n_eligible": int(len(elig)), "n_groups": int(elig["group_id"].nunique()),
        "excluded": {k: int(v) for k, v in merged["exclusion_reason"].value_counts().items()},
        "splits": {},
    }
    for s, d in splits.items():
        counts = d["label"].value_counts().to_dict()
        n = len(d)
        report["splits"][s] = {
            "images": int(n), "groups": int(d["group_id"].nunique()),
            "approved": int(counts.get("approved", 0)), "not_approved": int(counts.get("not_approved", 0)),
            "approved_share": round(counts.get("approved", 0) / n, 4) if n else None,
            "programs": {k: int(v) for k, v in d["program"].fillna("unknown").value_counts().items()},
        }
    return splits, report


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2)
