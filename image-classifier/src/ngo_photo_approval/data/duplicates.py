"""Exact (SHA-256) and near (64-bit dHash) duplicate detection and clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def components(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for i in self.parent:
            out.setdefault(self.find(i), []).append(i)
        return out


def hamming_matrix(hashes_hex: list[str]) -> np.ndarray:
    """Pairwise Hamming distances between 64-bit hex hashes (int16 matrix, n x n)."""
    arr = np.array([int(h, 16) for h in hashes_hex], dtype=np.uint64)
    bits = np.unpackbits(arr.view(np.uint8).reshape(-1, 8), axis=1).astype(np.float32)
    agree = bits @ bits.T + (1 - bits) @ (1 - bits).T
    return (64 - agree).astype(np.int16)


def near_duplicate_pairs(
    df: pd.DataFrame, max_hamming: int, max_aspect_diff: float = 0.05
) -> list[tuple[str, str, int]]:
    """Pairs of readable images whose dHash differs by <= max_hamming bits and whose aspect
    ratios are similar (low-texture screenshots can collide on dHash alone)."""
    sub = df[df["readable"] & df["dhash"].notna()].reset_index(drop=True)
    if len(sub) < 2:
        return []
    dist = hamming_matrix(sub["dhash"].tolist())
    ar = sub["aspect_ratio"].astype(float).to_numpy()
    ids = sub["image_id"].tolist()
    iu, ju = np.where(np.triu(dist <= max_hamming, k=1))
    pairs = []
    for i, j in zip(iu, ju, strict=True):
        if abs(ar[i] - ar[j]) <= max_aspect_diff:
            pairs.append((ids[i], ids[j], int(dist[i, j])))
    return pairs


def exact_duplicate_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """sha256 -> image_ids for hashes shared by more than one file."""
    sub = df[df["sha256"].notna()]
    groups = sub.groupby("sha256")["image_id"].apply(list)
    return {h: ids for h, ids in groups.items() if len(ids) > 1}


def build_duplicate_report(df: pd.DataFrame, max_hamming: int) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return (duplicate_report, image_id -> duplicate_cluster_id).

    Clusters merge exact and near duplicates. ``label_conflict`` marks clusters whose members
    carry more than one *final* label (approved + not_approved); ``exact_conflict`` marks the
    stricter case where byte-identical files carry conflicting final labels.
    """
    uf = UnionFind(df["image_id"].tolist())
    exact = exact_duplicate_groups(df)
    exact_member: dict[str, str] = {}
    for h, ids in exact.items():
        for i in ids[1:]:
            uf.union(ids[0], i)
        for i in ids:
            exact_member[i] = h
    pairs = near_duplicate_pairs(df, max_hamming)
    near_member: set[str] = set()
    for a, b, _ in pairs:
        uf.union(a, b)
        near_member.update((a, b))

    comps = {r: m for r, m in uf.components().items() if len(m) > 1}
    by_id = df.set_index("image_id")
    rows = []
    cluster_of: dict[str, str] = {}
    for n, (_root, members) in enumerate(sorted(comps.items(), key=lambda kv: kv[0]), start=1):
        cid = f"dup_{n:05d}"
        labels = sorted({by_id.at[m, "label"] for m in members if by_id.at[m, "label"] != "pending"})
        exact_labels: dict[str, set[str]] = {}
        for m in members:
            if m in exact_member and by_id.at[m, "label"] != "pending":
                exact_labels.setdefault(exact_member[m], set()).add(by_id.at[m, "label"])
        exact_conflict = any(len(s) > 1 for s in exact_labels.values())
        for m in members:
            cluster_of[m] = cid
            rows.append({
                "cluster_id": cid, "image_id": m, "rel_path": by_id.at[m, "rel_path"],
                "label": by_id.at[m, "label"], "cluster_size": len(members),
                "is_exact_duplicate": m in exact_member, "is_near_duplicate": m in near_member,
                "cluster_labels": "|".join(labels),
                "label_conflict": len(labels) > 1, "exact_label_conflict": exact_conflict,
            })
    cols = ["cluster_id", "image_id", "rel_path", "label", "cluster_size", "is_exact_duplicate",
            "is_near_duplicate", "cluster_labels", "label_conflict", "exact_label_conflict"]
    return pd.DataFrame(rows, columns=cols), cluster_of
