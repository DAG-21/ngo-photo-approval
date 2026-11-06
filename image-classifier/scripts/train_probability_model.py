"""Phase 10: train the probability model (logistic regression + calibration) and select a threshold.

Uses ONLY the train and validation splits. The test split is never read here.

Feature sets:
  quality         deterministic image-quality measures only (offline, no cloud needed)
  rules           rule-engine features from saved VLM analyses (needs data/results/vlm/<slug>/<split>.jsonl)
  rules+quality   both

Usage:
    python scripts/train_probability_model.py --feature-set quality --name technical_quality
    python scripts/train_probability_model.py --feature-set rules --analysis-model gemini-3.5-flash-lite --name vlm_gemini-3.5-flash-lite
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from ngo_photo_approval.config import load_policy_config, load_project_config, setup_logging
from ngo_photo_approval.evaluation.metrics import classification_metrics, select_threshold
from ngo_photo_approval.exceptions import NgoPhotoError
from ngo_photo_approval.rules.loader import load_rule_set, rule_set_fingerprint
from ngo_photo_approval.scoring.feature_builder import FEATURE_VERSION, feature_names
from ngo_photo_approval.scoring.offline import feature_table, load_analyses, load_manifest, load_split, quality_lookup
from ngo_photo_approval.scoring.predict import ProbabilityScorer
from ngo_photo_approval.scoring.train import train_probability_model


def slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature-set", choices=["quality", "rules", "rules+quality"], default="quality")
    ap.add_argument("--name", default=None, help="artifact name under artifacts/probability_model/ (default: feature set)")
    ap.add_argument("--analysis-model", default=None, help="model id whose saved analyses supply rule features")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    project = load_project_config()
    setup_logging(project.logging["level"])
    policy, rule_set = load_policy_config(), load_rule_set()
    name = args.name or args.feature_set.replace("+", "_")
    out_dir = project.artifacts_dir / "probability_model" / name
    quality = quality_lookup(load_manifest(project))

    tables = {}
    try:
        for split in ("train", "validation"):  # NOTE: "test" is intentionally absent
            df = load_split(project, split)
            analyses = None
            if "rules" in args.feature_set:
                if not args.analysis_model:
                    print("--analysis-model is required for rule-based feature sets", file=sys.stderr)
                    return 2
                analyses = load_analyses(project.output_dir / "results" / "vlm" / slug(args.analysis_model) / f"{split}.jsonl")
            tables[split] = feature_table(df, rule_set, policy, args.feature_set, quality, analyses)
            X, y, _, _, drops = tables[split]
            print(f"{split}: {len(y)} rows ({int(y.sum())} approved) dropped={drops}")
        (Xtr, ytr, gtr, _, dtr), (Xva, yva, _, idva, _) = tables["train"], tables["validation"]
        if len(ytr) == 0 or len(yva) == 0:
            print("No usable rows. Run scripts/run_baseline.py --mode vlm first for rule-based features.", file=sys.stderr)
            return 2
        names = feature_names(rule_set, args.feature_set)
        meta = train_probability_model(
            Xtr, ytr, gtr, Xva, yva, names, policy.calibration, out_dir, seed=args.seed,
            calibration_dir=project.artifacts_dir / "calibration" / name,
            extra_meta={"feature_set": args.feature_set, "feature_version": FEATURE_VERSION, "name": name,
                        "rule_set_version": rule_set.rule_set_version, "rule_set_fingerprint": rule_set_fingerprint(rule_set),
                        "rule_set_confirmed": rule_set.rule_set_confirmed, "analysis_model": args.analysis_model,
                        "train_dropped": dtr},
        )
    except NgoPhotoError as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2

    scorer = ProbabilityScorer.load(out_dir)
    pva = np.array([scorer.score(dict(zip(names, row, strict=True))).probability for row in Xva])
    ts = policy.threshold_selection
    sel = select_threshold(yva, pva, ts.max_false_approval_rate, ts.min_validation_recall, ts.candidate_grid_step)
    sel["selected_on"] = "validation"
    sel["probability_type"] = scorer.score(dict(zip(names, Xva[0], strict=True))).probability_type.value
    (out_dir / "threshold_selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    at_cfg = classification_metrics(yva, (pva + 1e-9 >= policy.approval_threshold).astype(int))
    at_sel = classification_metrics(yva, (pva + 1e-9 >= sel["threshold"]).astype(int))

    print(f"\nmodel {meta['model_version']}  C={meta['C']}  calibration: {meta['calibration_status']}")
    for r in meta["calibration_reasons"]:
        print("  - not claimed calibrated:", r)
    v = meta["validation"]
    print(f"validation: brier raw/cal = {v['raw']['brier']:.4f}/{v['calibrated']['brier']:.4f}  "
          f"ECE raw/cal = {v['raw']['ece']:.4f}/{v['calibrated']['ece']:.4f}  ROC-AUC={v['roc_auc']:.3f}  PR-AUC={v['pr_auc']:.3f} "
          f"(approved base rate {v['approved_base_rate']:.3f})")
    print(f"threshold in thresholds.yaml = {policy.approval_threshold}: FAR={at_cfg['false_approval_rate']}, recall={at_cfg['approval_recall']}")
    print(f"validation-selected threshold = {sel['threshold']} (feasible={sel['feasible']} for max FAR {ts.max_false_approval_rate}): "
          f"FAR={at_sel['false_approval_rate']}, recall={at_sel['approval_recall']}")
    print(f"artifacts: {out_dir}  (images used: {len(idva)} validation; test split untouched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
