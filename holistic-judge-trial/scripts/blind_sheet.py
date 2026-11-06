"""Blind tagging sheet: build it, import the returned copy, analyse it (all offline; nothing is sent anywhere).

    python scripts/blind_sheet.py build                      # -> blind_sheet/blind_tagging_sheet.xlsx (give this to taggers)
                                                             #    and blind_sheet/key.csv (KEEP PRIVATE: it has the labels)
    python scripts/blind_sheet.py import RETURNED.xlsx       # -> blind_sheet/blind_reasons.csv
    python scripts/blind_sheet.py analyse                    # prints the comparison, saves blind_sheet/report.md

The photos are the 82 of the tagged-and-paired test (41 disapproved that were tagged in the first round + 41 approved from the
same programmes), shuffled with a fixed seed. Needs openpyxl.
"""

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from judge_trial.blind_sheet import analyse, build_workbook, read_returned, render, reason_list, select_rows
from judge_trial.config import TRIAL_ROOT, ensure_main_importable, load_trial_config
from judge_trial.data import image_path

OUT = TRIAL_ROOT / "blind_sheet"


def cmd_build(args) -> int:
    cfg = load_trial_config()
    ensure_main_importable(cfg)
    from ngo_photo_approval.data.tagging import EXTRA_NON_SOP, reason_options
    from ngo_photo_approval.rules.loader import load_rule_set

    paired = pd.read_csv(TRIAL_ROOT / "ids" / "paired_tagged_dev.csv")
    rows = select_rows(paired, args.seed)
    # same reason list the first-round taggers used (without the optional extra codes), 'looks fine' reworded neutrally
    options = reason_list(reason_options(load_rule_set()), {c for c, _, _ in EXTRA_NON_SOP})
    OUT.mkdir(exist_ok=True)
    sheet = build_workbook(rows, options, OUT / "blind_tagging_sheet.xlsx", lambda i: image_path(cfg, i))
    rows[["row", "image_id", "label", "program", "tagged_primary"]].to_csv(OUT / "key.csv", index=False, encoding="utf-8")
    print(f"wrote {sheet} ({len(rows)} photos, {int((rows.label == 'approved').sum())} approved / {int((rows.label != 'approved').sum())} disapproved, shuffled)")
    print(f"wrote {OUT / 'key.csv'}  <- contains the labels; do NOT share with taggers")
    return 0


def cmd_import(args) -> int:
    df = read_returned(Path(args.file))
    df.to_csv(OUT / "blind_reasons.csv", index=False, encoding="utf-8")
    print(f"imported {len(df)} tagged photos -> {OUT / 'blind_reasons.csv'}")
    print("rows tagged (first/last):", int(df.row.min()) if len(df) else None, int(df.row.max()) if len(df) else None)
    return 0


def cmd_analyse(_args) -> int:
    cfg = load_trial_config()
    blind = pd.read_csv(OUT / "blind_reasons.csv")
    key = pd.read_csv(OUT / "key.csv")
    earlier_path = cfg.main_root / "data" / "annotations" / "disapproval_reasons.csv"
    earlier = pd.read_csv(earlier_path).set_index("image_id") if earlier_path.exists() else None
    report = render(analyse(blind, key, earlier))
    print(report)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=11)
    b.set_defaults(fn=cmd_build)
    i = sub.add_parser("import")
    i.add_argument("file")
    i.set_defaults(fn=cmd_import)
    sub.add_parser("analyse").set_defaults(fn=cmd_analyse)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
