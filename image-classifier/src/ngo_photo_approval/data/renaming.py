"""Rename dataset files to ``<class_folder>_<n><ext>`` safely and reversibly.

Design goals
* **Nothing is lost:** files are only renamed in place (``os.rename`` never overwrites on Windows and we also
  pre-check every target), and every old->new pair is recorded in a cumulative map BEFORE any rename, so a crash
  can be resumed and the whole operation can be reverted.
* **Stable identity:** the map records the old path, so image ids (hashes of the ORIGINAL relative path) and the
  program / submission metadata parsed from the original names stay valid after the rename.
* **Cumulative:** re-running only names files that are not yet in the map, continuing each class's numbering.
"""

from __future__ import annotations

import csv
import re
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..exceptions import DataError
from .inventory import make_image_id

TMP_SUFFIX = ".__renaming__"  # parking name used while resolving rename chains
MAP_COLUMNS = ["class_folder", "index", "old_rel_path", "new_rel_path", "size_bytes", "mtime_ns",
               "legacy_image_id"]


@dataclass(frozen=True)
class RenameEntry:
    class_folder: str
    index: int
    old_rel: str
    new_rel: str
    size: int
    mtime_ns: int


def natural_key(text: str) -> list[object]:
    """Sort key that orders 'x-2' before 'x-10' (case-insensitive)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text.lower())]


# --------------------------------------------------------------------------- map file


def load_map(path: Path) -> list[RenameEntry]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [RenameEntry(r["class_folder"], int(r["index"]), r["old_rel_path"], r["new_rel_path"],
                            int(r["size_bytes"]), int(r["mtime_ns"])) for r in csv.DictReader(f)]


def write_map(path: Path, entries: Iterable[RenameEntry]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(MAP_COLUMNS)
        for e in entries:
            w.writerow([e.class_folder, e.index, e.old_rel, e.new_rel, e.size, e.mtime_ns, make_image_id(e.old_rel)])


def legacy_lookup(entries: Iterable[RenameEntry]) -> dict[str, str]:
    """new_rel_path -> original rel_path (used by the inventory to keep ids/metadata stable)."""
    return {e.new_rel: e.old_rel for e in entries}


# --------------------------------------------------------------------------- planning


def _class_files(data_dir: Path, folder: str) -> list[str]:
    root = data_dir / folder
    return sorted((p.relative_to(data_dir).as_posix() for p in root.rglob("*") if p.is_file() and not p.name.endswith(TMP_SUFFIX)),
                  key=natural_key)


def plan_new_entries(data_dir: Path, class_folders: Iterable[str], existing: list[RenameEntry]) -> list[RenameEntry]:
    """Entries for files that are not in the map yet. Numbering continues after the highest existing index."""
    data_dir = Path(data_dir)
    known_new = {e.new_rel for e in existing}
    known_old = {e.old_rel for e in existing}
    planned: list[RenameEntry] = []
    for folder in class_folders:
        if not (data_dir / folder).is_dir():
            continue
        next_idx = max((e.index for e in existing if e.class_folder == folder), default=0) + 1
        for rel in _class_files(data_dir, folder):
            if rel in known_new or rel in known_old:
                continue
            p = data_dir / rel
            new_rel = (Path(rel).parent / f"{folder}_{next_idx}{p.suffix}").as_posix()
            st = p.stat()
            planned.append(RenameEntry(folder, next_idx, rel, new_rel, st.st_size, st.st_mtime_ns))
            next_idx += 1
    check_no_collisions(data_dir, planned, existing)
    return planned


def check_no_collisions(data_dir: Path, planned: list[RenameEntry], existing: list[RenameEntry]) -> None:
    """Abort if any target name is already taken by a file that is not part of the rename."""
    all_entries = [*existing, *planned]
    olds = {e.old_rel for e in all_entries}
    news = [e.new_rel for e in all_entries]
    if len(set(news)) != len(news):
        raise DataError("rename plan produces duplicate target names")
    for e in planned:
        if (data_dir / e.new_rel).exists() and e.new_rel not in olds:
            raise DataError(f"target already exists and is not part of the plan: {e.new_rel}")


def _tmp(rel: str) -> str:
    return rel + TMP_SUFFIX


def pending(data_dir: Path, entries: Iterable[RenameEntry]) -> list[RenameEntry]:
    """Entries still to execute: the new name is absent and the file is at its old name or parked at a temp name."""
    data_dir = Path(data_dir)
    return [e for e in entries
            if not (data_dir / e.new_rel).exists()
            and ((data_dir / e.old_rel).exists() or (data_dir / _tmp(e.new_rel)).exists())]


def _move_all(data_dir: Path, pairs: list[tuple[str, str]], sizes: Mapping[str, int] | None = None) -> int:
    """Rename every (src, dst) pair in place, safely and resumably.

    * never overwrites: a destination that exists and is not itself being moved aborts the run;
    * name chains/cycles (a destination that is another pair's source) go through a temporary name
      ``<dst>.__renaming__`` in two phases, so no file is ever clobbered;
    * a file already parked at its temp name (crashed run) is completed first.
    """
    data_dir = Path(data_dir)
    srcs = {s for s, _ in pairs}
    n = 0
    todo: list[tuple[str, str]] = []
    for src, dst in pairs:
        if (data_dir / _tmp(dst)).exists() and not (data_dir / dst).exists():
            (data_dir / _tmp(dst)).rename(data_dir / dst)  # finish a crashed phase 2
            n += 1
        elif (data_dir / src).exists():
            todo.append((src, dst))
    for src, dst in todo:
        if (data_dir / dst).exists() and dst not in srcs:
            raise DataError(f"refusing to overwrite existing file: {dst}")
        if sizes is not None and (data_dir / src).stat().st_size != sizes[src]:
            raise DataError(f"file changed since the plan was made: {src}")
    dsts = {d for _, d in todo}
    chained = [(s, d) for s, d in todo if d in srcs or s in dsts]
    for src, dst in todo:
        if (src, dst) not in chained:
            (data_dir / src).rename(data_dir / dst)
            n += 1
    for src, dst in chained:  # phase 1: park at temp names (all sources vacate their old names)
        if (data_dir / _tmp(dst)).exists():
            raise DataError(f"temporary name already in use: {_tmp(dst)}")
        (data_dir / src).rename(data_dir / _tmp(dst))
    for _src, dst in chained:  # phase 2: temp -> final
        if (data_dir / dst).exists():
            raise DataError(f"refusing to overwrite existing file: {dst}")
        (data_dir / _tmp(dst)).rename(data_dir / dst)
        n += 1
    return n


def apply_renames(data_dir: Path, entries: Iterable[RenameEntry]) -> int:
    """Rename old -> new for every entry (resumable). Returns the number of files renamed."""
    entries = list(entries)
    return _move_all(data_dir, [(e.old_rel, e.new_rel) for e in entries], {e.old_rel: e.size for e in entries})


def revert_renames(data_dir: Path, entries: Iterable[RenameEntry]) -> int:
    """Rename new -> old for every entry (same safety rules as apply)."""
    return _move_all(data_dir, [(e.new_rel, e.old_rel) for e in entries])


def verify(data_dir: Path, entries: Iterable[RenameEntry]) -> dict[str, int]:
    data_dir = Path(data_dir)
    out = {"renamed_ok": 0, "still_old_name": 0, "missing": 0, "size_mismatch": 0}
    for e in entries:
        new, old = data_dir / e.new_rel, data_dir / e.old_rel
        if new.exists():
            out["renamed_ok" if new.stat().st_size == e.size else "size_mismatch"] += 1
        elif old.exists():
            out["still_old_name"] += 1
        else:
            out["missing"] += 1
    return out


# --------------------------------------------------------------------------- derived tables


def patch_table(path: Path, column: str, mapping: Mapping[str, str], backup_dir: Path | None = None) -> int:
    """Rewrite ``column`` of a CSV through ``mapping`` (string-preserving: no other cell is touched).
    Returns the number of cells changed. Missing files/columns are skipped."""
    path = Path(path)
    if not path.exists():
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    if column not in fields:
        return 0
    changed = 0
    for r in rows:
        new = mapping.get(r[column])
        if new is not None and new != r[column]:
            r[column] = new
            changed += 1
    if changed:
        if backup_dir is not None:
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_dir / path.name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    return changed


def patch_derived_tables(
    output_dir: Path, data_dir: Path, entries: Iterable[RenameEntry], reverse: bool = False
) -> dict[str, int]:
    """Point every derived table at the new (or, with ``reverse``, the old) file names."""
    entries = list(entries)
    rel_map = {(e.new_rel if reverse else e.old_rel): (e.old_rel if reverse else e.new_rel) for e in entries}
    abs_map = {str(Path(data_dir) / k): str(Path(data_dir) / v) for k, v in rel_map.items()}
    backup = Path(output_dir) / "manifests" / f"backup_before_rename_{datetime.now():%Y%m%d_%H%M%S}"
    plan = [
        (Path(output_dir) / "manifests" / "image_manifest.csv", "rel_path", rel_map),
        (Path(output_dir) / "manifests" / "dataset_audit.csv", "rel_path", rel_map),
        (Path(output_dir) / "manifests" / "duplicate_report.csv", "rel_path", rel_map),
        (Path(output_dir) / "splits" / "train.csv", "rel_path", rel_map),
        (Path(output_dir) / "splits" / "validation.csv", "rel_path", rel_map),
        (Path(output_dir) / "splits" / "test.csv", "rel_path", rel_map),
        (Path(output_dir) / "tuning" / "upload_plan.csv", "local_path", abs_map),
    ]
    return {p.name: patch_table(p, col, m, backup) for p, col, m in plan}
