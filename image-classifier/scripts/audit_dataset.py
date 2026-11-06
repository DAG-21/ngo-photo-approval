"""Phase 1/3: inventory + audit the raw dataset (read-only) and write manifests and report.

Usage (from project root):
    python scripts/audit_dataset.py [--data-dir PATH] [--workers 8] [--no-cache]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from ngo_photo_approval.config import load_project_config, setup_logging
from ngo_photo_approval.data.audit import build_audit, render_markdown, summarize
from ngo_photo_approval.data.duplicates import build_duplicate_report
from ngo_photo_approval.data.inventory import build_inventory
from ngo_photo_approval.data.renaming import legacy_lookup, load_map

logger = logging.getLogger("audit_dataset")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None, help="Override NGO_DATA_DIR / project.yaml")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-cache", action="store_true", help="Recompute every row")
    args = ap.parse_args()

    cfg = load_project_config()
    setup_logging(cfg.logging["level"])
    data_dir = (args.data_dir or cfg.data_dir).resolve()
    out = cfg.output_dir / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "image_manifest.csv"

    cache = None
    if manifest_path.exists() and not args.no_cache:
        cache = pd.read_csv(manifest_path, low_memory=False)
        logger.info("using %d cached manifest rows", len(cache))

    logger.info("scanning %s", data_dir)
    rename_map = out / "rename_map.csv"  # written by scripts/rename_dataset.py; keeps ids/metadata stable
    legacy = legacy_lookup(load_map(rename_map)) if rename_map.exists() else None
    manifest = build_inventory(data_dir, cfg.dataset, cache=cache, workers=args.workers, legacy=legacy)
    manifest.to_csv(manifest_path, index=False)

    dup_report, cluster_of = build_duplicate_report(manifest, cfg.dataset.near_duplicate_hamming_threshold)
    dup_report.to_csv(out / "duplicate_report.csv", index=False)

    audit = build_audit(manifest, dup_report, cluster_of, cfg.dataset)
    audit.to_csv(out / "dataset_audit.csv", index=False)

    summary = summarize(manifest, audit, dup_report)
    reports = cfg.artifacts_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "dataset_audit_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = render_markdown(summary, stamp, data_dir.name)
    docs = cfg.resolve(cfg.paths.docs_dir)
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "DATASET_AUDIT.md").write_text(md, encoding="utf-8")

    print(f"files={summary['total_files']}  per_label={summary['files_per_label']}")
    print(f"eligible={summary['eligible_per_label']}  exclusions={summary['exclusions']}")
    print(f"duplicates={summary['duplicates']}")
    print(f"wrote {manifest_path}, duplicate_report.csv, dataset_audit.csv, docs/DATASET_AUDIT.md")


if __name__ == "__main__":
    main()
