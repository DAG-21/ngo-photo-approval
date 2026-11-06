"""Phase 5: baselines.

Modes
  technical       Offline, free. Deterministic image-quality features -> logistic regression; evaluated
                  on VALIDATION (the test split is never touched here).
  vlm             Untuned (or tuned rule-level) Gemini rule analysis. DRY RUN unless --execute is given;
                  --execute sends images to Google Cloud and therefore also needs
                  NGO_ALLOW_CLOUD_INFERENCE=true plus GOOGLE_CLOUD_PROJECT.
  tuned-decision  Track A tuned model (overall approved/not_approved label). Same gating as vlm.

Examples
    python scripts/run_baseline.py --mode technical
    python scripts/run_baseline.py --mode vlm --splits validation --sample 100            # dry run
    python scripts/run_baseline.py --mode vlm --splits validation --sample 100 --execute  # sends 100 images
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from ngo_photo_approval.config import load_model_config, load_project_config, setup_logging
from ngo_photo_approval.exceptions import NgoPhotoError
from ngo_photo_approval.models.baseline_analyzer import RuleAnalyzer
from ngo_photo_approval.models.batch_runner import run_batch
from ngo_photo_approval.models.cloud_client import GeminiBackend
from ngo_photo_approval.models.tuned_model import TunedDecisionClassifier, resolve_tuned_model_id
from ngo_photo_approval.rules.loader import config_warnings, load_rule_set
from ngo_photo_approval.scoring.offline import load_split

HERE = Path(__file__).parent


def slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def select_items(project, splits: list[str], sample: int | None, seed: int) -> list[tuple[str, Path, str | None, str]]:
    items = []
    for split in splits:
        df = load_split(project, split)
        if sample and sample < len(df):
            total = len(df)
            df = df.groupby("label", group_keys=False).apply(
                lambda g, total=total: g.sample(n=max(1, round(sample * len(g) / total)), random_state=seed))
        for r in df.itertuples():
            items.append((r.image_id, project.data_dir / r.rel_path, getattr(r, "program", None), split))
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["technical", "vlm", "tuned-decision"], required=True)
    ap.add_argument("--splits", nargs="+", default=["validation"], choices=["train", "validation", "test"])
    ap.add_argument("--sample", type=int, default=None, help="stratified random sample size PER split")
    ap.add_argument("--model", choices=["primary", "comparison", "tuned"], default="primary")
    ap.add_argument("--model-id", default=None, help="explicit model id (overrides --model)")
    ap.add_argument("--execute", action="store_true", help="actually call the cloud API (default: dry run)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.mode == "technical":
        for cmd in (["train_probability_model.py", "--feature-set", "quality", "--name", "technical_quality"],
                    ["evaluate_models.py", "--split", "validation", "--candidates", "technical_quality"]):
            rc = subprocess.run([sys.executable, str(HERE / cmd[0]), *cmd[1:]]).returncode
            if rc:
                return rc
        return 0

    project = load_project_config()
    setup_logging(project.logging["level"])
    model_cfg, rule_set = load_model_config(), load_rule_set()
    if "test" in args.splits:
        print("NOTE: scoring the TEST split is only appropriate for the final comparative evaluation.")
    try:
        if args.model_id:
            model_id = args.model_id
        elif args.model == "tuned":
            model_id = resolve_tuned_model_id(model_cfg)
        else:
            model_id = model_cfg.models[args.model].id
            if not model_cfg.models[args.model].verified:
                print(f"WARNING: model id {model_id!r} is not verified against the docs; the API may reject it.")
        items = select_items(project, args.splits, args.sample, args.seed)
        n = len(items)
        k = len(rule_set.active())
        # Calibrated on a real text-only call (16 rules: 1,969 prompt tokens; 647 output + 166 thinking): ~120 tokens/rule
        # in plus ~300 for the image, ~50/rule out. Image token cost is a guess until a real photo call is measured.
        est_in, est_out = n * (300 + 120 * k), n * (40 + 50 * k)
        print(f"model={model_id} images={n} splits={args.splits} sample={args.sample} mode={args.mode}")
        print(f"rule set {rule_set.rule_set_version}: {len(rule_set.active())} active rules")
        for w in config_warnings(rule_set)[:3]:
            print("  note:", w)
        print(f"rough token estimate: ~{est_in:,} input / ~{est_out:,} output (cost unknown until pricing is set in model.yaml)")
        cred = (f"api key {'present' if os.environ.get('GEMINI_API_KEY') else 'NOT SET'}"
                if project.cloud.auth_mode == "api_key" else f"project {'set' if project.cloud.project_id else 'NOT SET'}")
        print(f"authorization: cloud_inference={project.cloud.allow_cloud_inference} auth_mode={project.cloud.auth_mode} {cred}")
        if not args.execute:
            print("\nDRY RUN: no image was sent. Re-run with --execute (and authorization) to proceed.")
            return 0

        backend = GeminiBackend(project.cloud)  # raises AuthorizationError / ConfigError if not permitted
        if args.mode == "vlm":
            analyzer = RuleAnalyzer(backend, rule_set, model_cfg, model_id)
            call = lambda iid, path, prog: analyzer.analyze(path, iid, prog)  # noqa: E731
            out_dir = project.output_dir / "results" / "vlm" / slug(model_id)
        else:
            clf = TunedDecisionClassifier(backend, model_cfg, model_id)
            call = lambda iid, path, prog: clf.classify(path, iid)  # noqa: E731
            out_dir = project.output_dir / "results" / "tuned_decision" / slug(model_id)
        for split in args.splits:
            sub = [(i, p, g) for i, p, g, s in items if s == split]
            stats = run_batch(sub, call, out_dir / f"{split}.jsonl", workers=args.workers)
            print(f"{split}: {stats} -> {out_dir / (split + '.jsonl')}")
    except NgoPhotoError as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
