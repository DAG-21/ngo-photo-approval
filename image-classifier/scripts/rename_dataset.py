"""Rename every file in each class folder to ``<class_folder>_<n><ext>`` (n = 1..count of files in that class).

    python scripts/rename_dataset.py             # DRY RUN: shows the plan, changes nothing
    python scripts/rename_dataset.py --apply     # renames in place, records the map, updates derived tables
    python scripts/rename_dataset.py --revert    # restores every original name from the map

Safety
* Files are renamed in place (never moved, copied, converted or overwritten); extensions are kept as they are.
* The old->new map is written to data/manifests/rename_map.csv (and copied to <data_dir>/_rename_map.csv)
  BEFORE any rename. A crashed run is resumed by re-running --apply; --revert undoes everything.
* Image ids and program/submission metadata are derived from the ORIGINAL names via that map, so splits,
  tuning exports and manifests stay valid; derived tables (manifests, splits, upload plan) are patched to the new
  names, with backups in data/manifests/backup_before_rename_*.
* Numbering is in natural order of the original relative path (e.g. 'x-2' before 'x-10'). Re-running later
  only names new files, continuing each class's numbering.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter

from ngo_photo_approval.config import load_project_config, setup_logging
from ngo_photo_approval.data.renaming import (
    apply_renames,
    load_map,
    patch_derived_tables,
    pending,
    plan_new_entries,
    revert_renames,
    verify,
    write_map,
)
from ngo_photo_approval.exceptions import NgoPhotoError


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="perform the rename")
    mode.add_argument("--revert", action="store_true", help="restore the original names")
    args = ap.parse_args()

    project = load_project_config()
    setup_logging(project.logging["level"])
    data_dir = project.data_dir.resolve()
    out = project.output_dir
    map_path = out / "manifests" / "rename_map.csv"
    classes = list(project.dataset.folder_label_map)

    try:
        existing = load_map(map_path)
        if args.revert:
            if not existing:
                print("nothing to revert: no rename map found")
                return 1
            n = revert_renames(data_dir, existing)
            patched = patch_derived_tables(out, data_dir, existing, reverse=True)
            map_path.rename(map_path.with_name("rename_map.reverted.csv"))
            (data_dir / "_rename_map.csv").unlink(missing_ok=True)
            print(f"reverted {n} files to their original names; tables patched: {patched}")
            return 0

        planned = plan_new_entries(data_dir, classes, existing)
        todo = pending(data_dir, [*existing, *planned])
        per_class = Counter(e.class_folder for e in planned)
        print(f"data dir: {data_dir}")
        print(f"already in map: {len(existing)}   new files to name: {len(planned)}   renames still to execute: {len(todo)}")
        for c in classes:
            print(f"  {c:<18} new={per_class.get(c, 0):<5} (numbering continues at "
                  f"{max((e.index for e in existing if e.class_folder == c), default=0) + 1})")
        for e in (planned or todo)[:4] + (planned or todo)[-2:]:
            print(f"    {e.old_rel}  ->  {e.new_rel}")
        if not todo:
            print("nothing to do.")
            return 0
        if not args.apply:
            print("\nDRY RUN: nothing was changed. Re-run with --apply to rename (reversible with --revert).")
            return 0

        write_map(map_path, [*existing, *planned])  # journal first, then act
        shutil.copy2(map_path, data_dir / "_rename_map.csv")
        n = apply_renames(data_dir, todo)
        v = verify(data_dir, [*existing, *planned])
        patched = patch_derived_tables(out, data_dir, todo)
        print(f"\nrenamed {n} files. verification: {v}")
        print(f"derived tables updated (cells changed): {patched}")
        print(f"map: {map_path} (copy: {data_dir / '_rename_map.csv'})")
        return 0 if v["missing"] == 0 and v["size_mismatch"] == 0 and v["still_old_name"] == 0 else 1
    except NgoPhotoError as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
