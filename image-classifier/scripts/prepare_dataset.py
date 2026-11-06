"""Phase 4: create reproducible, leakage-resistant train/validation/test split manifests.

Reads data/manifests/{image_manifest,dataset_audit}.csv (run scripts/audit_dataset.py first).
Writes data/splits/{train,validation,test}.csv and artifacts/reports/split_report.json.

Usage:
    python scripts/prepare_dataset.py [--seed 42] [--train 0.7 --validation 0.15 --test 0.15]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ngo_photo_approval.config import SplitConfig, load_project_config, setup_logging
from ngo_photo_approval.data.splitting import make_splits, report_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--train", type=float, default=None)
    ap.add_argument("--validation", type=float, default=None)
    ap.add_argument("--test", type=float, default=None)
    ap.add_argument("--no-filename-groups", action="store_true")
    args = ap.parse_args()

    cfg = load_project_config()
    setup_logging(cfg.logging["level"])
    s = cfg.splits.model_dump()
    for k in ("seed", "train", "validation", "test"):
        if getattr(args, k) is not None:
            s[k] = getattr(args, k)
    if args.no_filename_groups:
        s["use_filename_groups"] = False
    split_cfg = SplitConfig(**s)

    man_dir = cfg.output_dir / "manifests"
    manifest = pd.read_csv(man_dir / "image_manifest.csv", low_memory=False)
    audit = pd.read_csv(man_dir / "dataset_audit.csv", low_memory=False)

    splits, report = make_splits(manifest, audit, split_cfg)
    out = cfg.output_dir / "splits"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(out / f"{name}.csv", index=False)
    rep_dir = cfg.artifacts_dir / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "split_report.json").write_text(report_json(report), encoding="utf-8")

    print(f"seed={report['seed']} eligible={report['n_eligible']} groups={report['n_groups']}")
    for name, r in report["splits"].items():
        print(f"  {name:<10} images={r['images']:<5} groups={r['groups']:<5} approved={r['approved']:<5} "
              f"not_approved={r['not_approved']:<5} approved_share={r['approved_share']}")
    print(f"wrote {Path(out)} and split_report.json (no leakage detected)")


if __name__ == "__main__":
    main()
