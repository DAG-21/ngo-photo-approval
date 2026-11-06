"""Phase 7: build Gemini supervised-tuning JSONL from the saved split manifests (DRY RUN ONLY).

This script never uploads anything. It writes data/tuning/{train,validation}.jsonl, upload_plan.csv,
skipped_examples.csv and export_report.json. Uploading and job submission live in
scripts/manage_tuning_job.py and require explicit authorization.

Usage:
    python scripts/build_tuning_data.py --track A [--balance downsample]
    python scripts/build_tuning_data.py --track B   # needs verified rule annotations
"""

from __future__ import annotations

import argparse

from ngo_photo_approval.config import load_model_config, load_project_config, setup_logging
from ngo_photo_approval.rules.loader import load_rule_set
from ngo_photo_approval.tuning.dataset_builder import build_tuning_files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=["A", "B"], default=None, help="default: tuning.task_track from model.yaml")
    ap.add_argument("--balance", choices=["none", "downsample"], default="none")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    project = load_project_config()
    setup_logging(project.logging["level"])
    model_cfg = load_model_config()
    track = args.track or model_cfg.tuning.task_track
    rep = build_tuning_files(project, model_cfg, load_rule_set(), track=track, balance=args.balance,
                             seed=args.seed, workers=args.workers, dry_run=True)
    print(f"track={rep.track} balance={rep.balance} bucket={rep.bucket}")
    for split, files in rep.files.items():
        print(f"  {split:<10} examples={files['examples']:<5} bytes={files['bytes']:<10} labels={rep.counts[split]}")
    print(f"  skipped={len(rep.skipped)} derived_image_copies={rep.derived_images} upload_plan_rows={rep.upload_plan_rows}")
    for w in rep.warnings:
        print("  WARNING:", w)
    print("Nothing was uploaded. See data/tuning/export_report.json")


if __name__ == "__main__":
    main()
