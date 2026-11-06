"""Excel sheet for humans to tag WHY photos were disapproved; and the importer for the returned sheet.

    python scripts/make_tagging_sheet.py build              # -> data/annotations/disapproval_tagging_sheet.xlsx
    python scripts/make_tagging_sheet.py import FILE.xlsx   # -> data/annotations/disapproval_reasons.csv + summary

Which photos: not_approved photos from the TRAIN + VALIDATION splits, one row per unique picture (identical copies are
counted, not repeated). Test-split photos and photos whose identical copies carry both labels are left out. The order is a
seeded random shuffle so that any prefix is a fair sample. Needs openpyxl (`pip install -e .[tagging]`).
Nothing is sent anywhere. The workbook contains thumbnails of real photos: share it only with authorised taggers.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ngo_photo_approval.config import load_model_config, load_project_config, setup_logging
from ngo_photo_approval.data.preprocessing import open_image
from ngo_photo_approval.data.tagging import TagRow, build_workbook, iter_codes, read_tags, reason_options
from ngo_photo_approval.exceptions import NgoPhotoError
from ngo_photo_approval.rules.loader import load_rule_set
from ngo_photo_approval.scoring.offline import load_manifest, unique_trainval

THUMB_LONG_SIDE = 420  # stored resolution
BOX_W, BOX_H = 285, 255  # display box in pixels (fits the 195pt rows)


def select_rows(project, seed: int) -> tuple[list[TagRow], dict[str, int]]:
    d = unique_trainval(project)
    d = d[d.label == "not_approved"].copy()
    man = load_manifest(project)
    na = man[man.label == "not_approved"]
    copies = na.groupby("sha256").size()
    d["copies"] = d.sha256.map(copies).fillna(1).astype(int)
    order = np.random.default_rng(seed).permutation(len(d))
    d = d.iloc[order].reset_index(drop=True)
    rows = [TagRow(order=i + 1, image_id=r.image_id, file=Path(r.rel_path).name,
                   program=str(r.program) if pd.notna(r.program) else "", copies=int(r.copies))
            for i, r in enumerate(d.itertuples())]
    stats = {
        "not_approved_files": int(len(na)),
        "unique_not_approved_pictures": int(na.sha256.nunique()),
        "sheet_rows": len(rows),
        "left_out_test_or_conflicting": int(na.sha256.nunique()) - len(rows),
    }
    return rows, stats


def make_thumb_fn(project, model_cfg):
    norm = model_cfg.image_input.tuning_normalize_long_side_px
    img_dir = project.output_dir / "derived" / f"tuning_images_norm{norm}"
    man = load_manifest(project).set_index("image_id")

    def thumb(row: TagRow) -> tuple[bytes, int, int]:
        p = img_dir / f"{row.image_id}.jpg"
        im = Image.open(p).convert("RGB") if p.exists() else open_image(project.data_dir / man.loc[row.image_id, "rel_path"])
        im.thumbnail((THUMB_LONG_SIDE, THUMB_LONG_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=72)
        scale = min(BOX_W / im.width, BOX_H / im.height)
        return buf.getvalue(), max(1, round(im.width * scale)), max(1, round(im.height * scale))

    return thumb


def cmd_build(args, project, model_cfg, rule_set) -> int:
    rows, stats = select_rows(project, args.seed)
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out) if args.out else project.output_dir / "annotations" / "disapproval_tagging_sheet.xlsx"
    if out.exists() and not args.force:
        try:
            tagged = len(read_tags(out, rule_set)[0])
        except Exception:  # noqa: BLE001 - unreadable/locked: be careful and refuse
            tagged = -1
        if tagged != 0:
            why = "could not be read (is it open in Excel?)" if tagged < 0 else f"already has {tagged} tagged rows"
            print(f"REFUSING to overwrite {out.name}: it {why}. Use --out NEW.xlsx to build a separate copy, "
                  "or --force if you are sure.", file=sys.stderr)
            return 2
    try:
        build_workbook(rows, make_thumb_fn(project, model_cfg), rule_set, out)
    except PermissionError:
        print(f"Cannot write {out}: the file is open in another program (Excel?). Close it, or use --out NEW.xlsx.",
              file=sys.stderr)
        return 2
    mb = out.stat().st_size / 1e6
    print(f"not_approved files: {stats['not_approved_files']}  ->  unique pictures: {stats['unique_not_approved_pictures']}")
    print(f"rows in the sheet: {len(rows)} (train+validation only; test-split and contradictory-label pictures left out)")
    print(f"reason options: {len(reason_options(rule_set))} | file: {out} ({mb:.1f} MB)")
    return 0


def cmd_import(args, project, model_cfg, rule_set) -> int:
    df, problems = read_tags(Path(args.file), rule_set)
    out = project.output_dir / "annotations" / "disapproval_reasons.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"tagged photos: {len(df)}  -> {out}")
    if len(df):
        counts = pd.Series(list(iter_codes(df))).value_counts()
        print("reasons mentioned (primary + second + third):")
        for code, n in counts.items():
            print(f"  {code:<4} {n}")
        print("primary reason only:", df.primary.value_counts().to_dict())
    for p in problems:
        print("PROBLEM:", p)
    return 2 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--limit", type=int, default=None, help="only the first N rows (testing)")
    b.add_argument("--out", default=None)
    b.add_argument("--force", action="store_true", help="overwrite even if the existing sheet has tags")
    i = sub.add_parser("import")
    i.add_argument("file")
    args = ap.parse_args()
    project = load_project_config()
    setup_logging(project.logging["level"])
    try:
        fn = cmd_build if args.command == "build" else cmd_import
        return fn(args, project, load_model_config(), load_rule_set())
    except NgoPhotoError as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
