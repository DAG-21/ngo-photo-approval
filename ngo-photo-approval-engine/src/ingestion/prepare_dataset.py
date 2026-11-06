"""
Phase 0: Dataset Preparation
- Step 0.1 & 0.2: Scans the labelled class folders under `original_dir` (see `FOLDER_LABELS`) and
  assigns randomized, label-blind unique IDs (`NGO_IMG_000001` ...). Folders not listed there
  (for example `pending-approval/`, which is unlabelled) are ignored.
- Step 0.3: Extracts dimensions, aspect ratio, EXIF capture timestamps, SHA-256 exact duplicates,
  and perceptual dHash near-duplicates; groups same-submission, near-duplicate and burst photos together.
- Step 0.4: Creates leak-free, event-grouped, class-stratified dataset splits
  (`training_ids.csv` ~70%, `validation_ids.csv` ~15%, `test_ids.csv` ~15%).

Event grouping uses the original upload names when `<original_dir>/_rename_map.csv` is present
(`arham_aahar-13 (35).jpg` -> event hint `arham_aahar-13`), otherwise the filename prefix before `_`.
"""

import argparse
import csv
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

from src.image_processing.quality_and_exif import extract_image_metadata
from src.paths import resolve_image_path, to_stored_path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Class folder name -> historical decision. Anything else (e.g. `pending-approval`) is skipped.
FOLDER_LABELS = {
    "approved": "APPROVED",
    "rejected": "REJECTED",
    "disapproved": "REJECTED",
}

RENAME_MAP_NAME = "_rename_map.csv"
_ORIGINAL_NAME_RE = re.compile(r"^(?P<hint>.+?)(?:\s*\(\d+\))?$")


def load_event_hints_from_rename_map(original_dir: Path) -> Dict[str, str]:
    """
    Maps `new_rel_path` (relative to `original_dir`, POSIX) -> submission-level event hint
    parsed from the original upload filename, e.g. `arham_aahar-13 (35).jpg` -> `arham_aahar-13`.
    Returns an empty dict when no rename map exists.
    """
    map_path = original_dir / RENAME_MAP_NAME
    if not map_path.exists():
        return {}
    hints: Dict[str, str] = {}
    with open(map_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            old_stem = Path(row["old_rel_path"]).stem
            m = _ORIGINAL_NAME_RE.match(old_stem)
            hints[row["new_rel_path"].replace("\\", "/")] = (m.group("hint") if m else old_stem).strip()
    return hints


def _popcount64(values: np.ndarray) -> np.ndarray:
    return np.unpackbits(values.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def assign_burst_and_event_groups(
    records: List[Dict],
    phash_threshold: int = 6,
    burst_window_seconds: int = 60,
) -> List[Dict]:
    """
    Groups images into event/burst clusters so that near-duplicate or same-event photos
    never leak across training, validation, and testing partitions. A cluster is the union of:
      1. exact SHA-256 duplicates,
      2. photos sharing an `event_hint` (same submission),
      3. perceptual near-duplicates (dHash Hamming distance <= `phash_threshold`, any event),
      4. photos whose EXIF capture times are within `burst_window_seconds` of each other.
    Independent of record order.
    """
    n = len(records)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # 1. Exact SHA-256 duplicates
    sha_seen: Dict[str, int] = {}
    for idx, rec in enumerate(records):
        sha = rec["sha256"]
        if sha in sha_seen:
            rec["is_exact_duplicate"] = True
            rec["duplicate_of_index"] = sha_seen[sha]
            union(idx, sha_seen[sha])
        else:
            rec["is_exact_duplicate"] = False
            rec["duplicate_of_index"] = None
            sha_seen[sha] = idx

    # 2. Same submission / explicit event hint
    hint_seen: Dict[str, int] = {}
    for idx, rec in enumerate(records):
        rec["is_near_duplicate"] = False
        hint = rec.get("event_hint")
        if hint:
            if hint in hint_seen:
                union(idx, hint_seen[hint])
            else:
                hint_seen[hint] = idx

    # 3. Perceptual near-duplicates across all images (vectorised pairwise Hamming distance)
    if n > 1:
        hashes = np.array([int(r["dhash"], 16) for r in records], dtype=np.uint64)
        for i in range(n - 1):
            dist = _popcount64(hashes[i + 1:] ^ hashes[i])
            for j in np.nonzero(dist <= phash_threshold)[0] + i + 1:
                records[int(j)]["is_near_duplicate"] = True
                union(i, int(j))

    # 4. EXIF capture-time bursts
    # A burst is a run of shots from one physical camera, so photos are only linked when they share
    # a camera make/model; without that, unrelated uploads from different people merge into giant clusters.
    timed_by_camera: Dict[Tuple[str, str], List[Tuple[datetime, int]]] = defaultdict(list)
    for idx, rec in enumerate(records):
        make, model = rec.get("camera_make"), rec.get("camera_model")
        if rec.get("capture_date") and isinstance(model, str) and model.strip():
            try:
                timed_by_camera[(str(make), model)].append((datetime.fromisoformat(rec["capture_date"]), idx))
            except ValueError:
                pass
    for timed in timed_by_camera.values():
        timed.sort()
        # Anchored windows: a burst spans at most `burst_window_seconds` from its first shot. Chaining
        # consecutive gaps instead would merge a whole day of continuous shooting into one cluster.
        anchor_time, anchor_idx = None, None
        for t, idx in timed:
            if anchor_time is not None and (t - anchor_time).total_seconds() <= burst_window_seconds:
                union(anchor_idx, idx)
            else:
                anchor_time, anchor_idx = t, idx

    cluster_map: Dict[int, str] = {}
    cluster_counter = 1
    for i in range(n):
        root = find(i)
        if root not in cluster_map:
            cluster_map[root] = f"EVT_{cluster_counter:04d}"
            cluster_counter += 1
        records[i]["event_id"] = cluster_map[root]

    return records


def create_event_grouped_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into Train (~70%), Validation (~15%), and Test (~15%)
    while keeping all images sharing an `event_id` in the same partition and
    keeping the APPROVED:REJECTED ratio close to the overall ratio in every partition.

    Groups are placed largest-first into whichever partition's per-class squared deviation from
    its target grows the least, so a few very large clusters cannot swamp a small partition.
    """
    rng = random.Random(seed)
    classes = ["APPROVED", "REJECTED"]
    ratios = [train_ratio, val_ratio, 1.0 - train_ratio - val_ratio]
    totals = {c: int((df["historical_decision"] == c).sum()) for c in classes}
    targets = [{c: totals[c] * r for c in classes} for r in ratios]
    counts = [{c: 0 for c in classes} for _ in ratios]

    groups = []
    for _, g in df.groupby("event_id"):
        groups.append((len(g), rng.random(), g, {c: int((g["historical_decision"] == c).sum()) for c in classes}))
    groups.sort(key=lambda x: (-x[0], x[1]))

    def deviation(p: int, add: Dict[str, int]) -> float:
        return sum(
            ((counts[p][c] + add[c] - targets[p][c]) / max(totals[c], 1)) ** 2 for c in classes
        )

    no_add = {c: 0 for c in classes}
    parts: List[List[pd.DataFrame]] = [[], [], []]
    for _, _, g, add in groups:
        best = min(range(3), key=lambda p: deviation(p, add) - deviation(p, no_add))
        parts[best].append(g)
        for c in classes:
            counts[best][c] += add[c]

    def _cat(frames: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=df.columns)

    return _cat(parts[0]), _cat(parts[1]), _cat(parts[2])


def prepare_dataset_inventory(
    original_dir: Path,
    processed_dir: Path,
    evaluation_dir: Path,
    seed: int = 42,
    max_workers: int = 8,
    phash_threshold: int = 6,
    burst_window_seconds: int = 10,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> pd.DataFrame:
    """
    End-to-end Phase 0 execution:
    1. Discovers labelled images (folders named in `FOLDER_LABELS`).
    2. Randomizes ordering before assigning `NGO_IMG_XXXXXX` so image IDs do not leak historical labels.
    3. Extracts image & EXIF metadata, duplicate flags, and event groupings; unreadable files are
       skipped and listed in `unreadable_files.csv` instead of aborting the run.
    4. Saves `image_metadata.csv` and partition CSVs (`training_ids.csv`, `validation_ids.csv`, `test_ids.csv`).
    """
    original_dir = Path(original_dir)
    event_hints = load_event_hints_from_rename_map(original_dir)

    raw_items = []
    for folder in sorted(p for p in original_dir.iterdir() if p.is_dir()):
        label = FOLDER_LABELS.get(folder.name.lower())
        if label is None:
            continue
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                rel = p.relative_to(original_dir).as_posix()
                raw_items.append({
                    "original_filename": p.name,
                    "original_filepath": to_stored_path(p),
                    "historical_decision": label,
                    "event_hint": event_hints.get(rel) or (p.stem.split("_")[0] if "_" in p.stem else None),
                })

    if not raw_items:
        raise FileNotFoundError(
            f"No images found under {original_dir} in folders {sorted(FOLDER_LABELS)}"
        )

    # Step 0.3: Extract metadata (I/O bound, so threaded); keep failures out of the dataset
    def _extract(item: Dict) -> Tuple[Dict, Optional[str]]:
        try:
            return extract_image_metadata(resolve_image_path(item["original_filepath"])), None
        except Exception as err:
            return {}, f"{type(err).__name__}: {err}"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_extract, raw_items))

    readable, unreadable = [], []
    for item, (meta, err) in zip(raw_items, results):
        if err is None:
            item.update(meta)
            readable.append(item)
        else:
            unreadable.append({"original_filepath": item["original_filepath"],
                               "historical_decision": item["historical_decision"], "error": err})
    raw_items = readable
    if not raw_items:
        raise RuntimeError("Every candidate image failed metadata extraction.")

    raw_items = assign_burst_and_event_groups(
        raw_items, phash_threshold=phash_threshold, burst_window_seconds=burst_window_seconds
    )

    # Step 0.2: Randomize order before assigning NGO_IMG_XXXXXX so ID sequence never leaks class label
    rng = random.Random(seed)
    rng.shuffle(raw_items)
    for idx, item in enumerate(raw_items, start=1):
        item["image_id"] = f"NGO_IMG_{idx:06d}"

    df = pd.DataFrame(raw_items)
    cols = [
        "image_id",
        "original_filename",
        "original_filepath",
        "historical_decision",
        "event_id",
        "capture_date",
        "image_width",
        "image_height",
        "aspect_ratio",
        "file_size_bytes",
        "file_format",
        "sha256",
        "dhash",
        "is_exact_duplicate",
        "is_near_duplicate",
        "camera_make",
        "camera_model",
    ]
    df = df[cols].sort_values("image_id").reset_index(drop=True)

    processed_dir = Path(processed_dir)
    evaluation_dir = Path(evaluation_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = create_event_grouped_splits(
        df, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )
    df["partition"] = "TRAIN"
    df.loc[df["image_id"].isin(val_df["image_id"]), "partition"] = "VALIDATION"
    df.loc[df["image_id"].isin(test_df["image_id"]), "partition"] = "TEST"

    df.to_csv(processed_dir / "image_metadata.csv", index=False)
    train_df[["image_id", "event_id", "historical_decision"]].to_csv(evaluation_dir / "training_ids.csv", index=False)
    val_df[["image_id", "event_id", "historical_decision"]].to_csv(evaluation_dir / "validation_ids.csv", index=False)
    test_df[["image_id", "event_id", "historical_decision"]].to_csv(evaluation_dir / "test_ids.csv", index=False)
    if unreadable:
        pd.DataFrame(unreadable).to_csv(processed_dir / "unreadable_files.csv", index=False)

    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 0: build image_metadata.csv and event-grouped splits.")
    ap.add_argument("--original-dir", type=Path, required=True,
                    help="Folder containing approved/ and disapproved/ (or rejected/) subfolders")
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument("--evaluation-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "config/evaluation_config.yaml",
                    help="Source of dataset_splits settings (ratios, burst window, dHash threshold)")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")).get("dataset_splits", {})
    out = prepare_dataset_inventory(
        args.original_dir,
        args.processed_dir,
        args.evaluation_dir,
        seed=args.seed,
        phash_threshold=cfg.get("phash_duplicate_threshold", 6),
        burst_window_seconds=cfg.get("burst_time_window_seconds", 10),
        train_ratio=cfg.get("train_ratio", 0.70),
        val_ratio=cfg.get("val_ratio", 0.15),
    )
    print(out["partition"].value_counts().to_dict())
