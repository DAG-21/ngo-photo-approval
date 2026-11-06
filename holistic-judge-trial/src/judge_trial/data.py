"""Photo pools for the trial: examples, dev, confirm, validation.

Rules that keep the numbers honest:
* Only train + validation photos are ever used. The test split is never read.
* Each unique photo content is counted once (exact duplicates share one result).
* No photo that shares a submission group, duplicate cluster or exact content with an example is ever evaluated,
  so a worked example cannot leak into the score it is being tested on.
* dev and confirm are drawn from the train split and are group-disjoint from each other.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import TrialConfig, ensure_main_importable

POOL_COLUMNS = ["image_id", "label", "program", "group_id", "sha256", "split"]
POOL_NAMES = ("dev", "confirm", "validation")


def load_main(cfg: TrialConfig):
    """(project_config, full train+validation rows including duplicates, unique-by-content rows)."""
    ensure_main_importable(cfg)
    from ngo_photo_approval.config import load_project_config
    from ngo_photo_approval.scoring.offline import load_split, unique_trainval

    project = load_project_config()
    full = pd.concat([load_split(project, "train"), load_split(project, "validation")], ignore_index=True)
    return project, full, unique_trainval(project)


def image_path(cfg: TrialConfig, image_id: str) -> Path:
    return cfg.normalized_dir / f"{image_id}.jpg"


MAX_SUBMISSION_GROUP = 25  # larger groups are not real submissions (see below) and are never used to exclude photos


def related_mask(frame: pd.DataFrame, full: pd.DataFrame, anchor_ids: set[str],
                 max_group_size: int = MAX_SUBMISSION_GROUP) -> pd.Series:
    """True for rows of ``frame`` that are the same photo as, or related to, any anchor photo.

    Related = identical content, same duplicate cluster, or in the same SMALL submission group. The project's
    ``group_id`` is a connected component over (label, program, submission index) and can be enormous (one component
    holds 512 of the 695 disapproved training photos), so a group above ``max_group_size`` says nothing about two
    photos being the same occasion; excluding it wholesale would silently remove a whole program from the pools.
    """
    sizes = full.groupby("group_id")["image_id"].transform("size")
    small = set(full.loc[sizes <= max_group_size, "group_id"].dropna())
    anchors = full[full["image_id"].isin(anchor_ids)]
    groups = set(anchors["group_id"].dropna()) & small
    clusters = set(anchors["duplicate_cluster_id"].dropna())
    shas = set(anchors["sha256"])
    for _ in range(3):  # widen until stable: a copy in a related group can pull in more related content
        rel = full[full["group_id"].isin(groups) | full["duplicate_cluster_id"].isin(clusters) | full["sha256"].isin(shas)]
        new = (set(rel["group_id"].dropna()) & small, set(rel["duplicate_cluster_id"].dropna()), set(rel["sha256"]))
        if new == (groups, clusters, shas):
            break
        groups, clusters, shas = new
    return (frame["group_id"].isin(groups) | frame["duplicate_cluster_id"].isin(clusters)
            | frame["sha256"].isin(shas) | frame["image_id"].isin(anchor_ids))


def load_pool_embeddings(cfg: TrialConfig) -> dict[str, np.ndarray]:
    """CLIP embeddings cached by the main project (read-only)."""
    ensure_main_importable(cfg)
    from ngo_photo_approval.config import load_project_config
    from ngo_photo_approval.scoring.offline import load_embeddings

    return load_embeddings(load_project_config())


def similar_mask(frame: pd.DataFrame, emb: dict[str, np.ndarray], anchor_ids: set[str], threshold: float) -> pd.Series:
    """True for photos whose CLIP cosine similarity to any anchor photo is >= threshold (near-identical scenes)."""
    def unit(ids):
        x = np.stack([emb[i] for i in ids]).astype("float64")
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    if not anchor_ids or frame.empty:
        return pd.Series(False, index=frame.index)
    sim = unit(frame["image_id"]) @ unit(sorted(anchor_ids)).T
    return pd.Series(sim.max(axis=1) >= threshold, index=frame.index)


def stratified_sample(frame: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Seeded sample of up to ``per_class`` photos per label (deterministic for a given frame)."""
    rng = np.random.RandomState(seed)
    parts = []
    for label in ("approved", "not_approved"):
        g = frame[frame["label"] == label].sort_values("image_id")
        n = min(per_class, len(g))
        parts.append(g.iloc[rng.permutation(len(g))[:n]])
    return pd.concat(parts).sort_values("image_id").reset_index(drop=True)


def build_pools(cfg: TrialConfig, example_ids: set[str]) -> dict[str, pd.DataFrame]:
    _, full, unique = load_main(cfg)
    unique = unique[unique["label"].isin(["approved", "not_approved"])]
    if example_ids:
        unique = unique[~related_mask(unique, full, example_ids)]
        if cfg.pools.example_similarity_max is not None:
            unique = unique[~similar_mask(unique, load_pool_embeddings(cfg), example_ids, cfg.pools.example_similarity_max)]
    train = unique[unique["split"] == "train"]
    dev = stratified_sample(train, cfg.pools.dev_size_per_class, cfg.pools.seed)
    rest = train[~related_mask(train, full, set(dev["image_id"]))]
    confirm = stratified_sample(rest, cfg.pools.confirm_size_per_class, cfg.pools.seed + 1)
    validation = unique[unique["split"] == "validation"].sort_values("image_id").reset_index(drop=True)
    return {"dev": dev[POOL_COLUMNS], "confirm": confirm[POOL_COLUMNS], "validation": validation[POOL_COLUMNS]}


def save_pools(cfg: TrialConfig, pools: dict[str, pd.DataFrame]) -> None:
    for name, df in pools.items():
        path = cfg.pool_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


def load_pool(cfg: TrialConfig, name: str, example_ids: set[str] | None = None) -> pd.DataFrame:
    if name not in POOL_NAMES:
        raise ValueError(f"unknown pool {name!r}; choose from {POOL_NAMES}")
    path = cfg.pool_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/build_pools.py first")
    df = pd.read_csv(path)
    clash = set(df["image_id"]) & (example_ids or set())
    if clash:
        raise ValueError(f"pool {name!r} contains example photos {sorted(clash)[:3]}; rebuild with scripts/build_pools.py")
    return df
