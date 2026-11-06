"""Phase 12: single-image and batch inference through the shared PredictionService.

Sends images to Google Cloud, so it requires NGO_ALLOW_CLOUD_INFERENCE=true and GOOGLE_CLOUD_PROJECT.

Examples
    python scripts/run_inference.py --image path/to/photo.jpg
    python scripts/run_inference.py --dir path/to/folder --out data/results/batch.jsonl --csv data/results/batch.csv
    python scripts/run_inference.py --manifest list.csv --out out.jsonl      # columns: path[,image_id,program]
    python scripts/run_inference.py --image photo.jpg --use-tuned --scoring-model vlm_tuned
Re-running a batch command resumes it: finished images are skipped, failed ones retried.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ngo_photo_approval.config import load_model_config, load_policy_config, load_project_config, setup_logging
from ngo_photo_approval.exceptions import NgoPhotoError
from ngo_photo_approval.rules.loader import load_rule_set
from ngo_photo_approval.service.prediction_service import (
    anonymous_image_id,
    build_service,
    items_from_directory,
    load_batch_results,
)


def slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path)
    src.add_argument("--dir", type=Path)
    src.add_argument("--manifest", type=Path)
    ap.add_argument("--program", default=None, help="program name for --image (rules may be program-specific)")
    ap.add_argument("--out", type=Path, default=None, help="batch JSONL output (default data/results/batch_predictions.jsonl)")
    ap.add_argument("--csv", type=Path, default=None, help="also write a flat CSV summary of the batch")
    ap.add_argument("--use-tuned", action="store_true", help="use the tuned model from configs/model.yaml / registry")
    ap.add_argument("--scoring-model", default=None, help="probability-model artifact name (default: vlm_<model slug>)")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    project = load_project_config()
    setup_logging(project.logging["level"])
    model_cfg, policy, rule_set = load_model_config(), load_policy_config(), load_rule_set()
    try:
        model_id = model_cfg.tuned_model.endpoint if args.use_tuned else model_cfg.models["primary"].id
        scoring = args.scoring_model or f"vlm_{slug(model_id or 'tuned')}"
        service = build_service(project, model_cfg, policy, rule_set, scoring_model=scoring, use_tuned=args.use_tuned)
        for w in service.readiness()["warnings"][:2]:
            print("note:", w, file=sys.stderr)

        if args.image:
            resp = service.predict_path(args.image, program=args.program)
            print(resp.model_dump_json(indent=2))
            return 0

        if args.dir:
            items = items_from_directory(args.dir, project.dataset.supported_image_extensions)
        else:
            m = pd.read_csv(args.manifest)
            col = "path" if "path" in m else "rel_path"
            items = [(r["image_id"] if "image_id" in m and pd.notna(r.get("image_id")) else anonymous_image_id(Path(r[col])),
                      Path(r[col]) if Path(r[col]).is_absolute() else project.data_dir / r[col],
                      r["program"] if "program" in m and pd.notna(r.get("program")) else None)
                     for r in m.to_dict("records")]
        out = args.out or project.output_dir / "results" / "batch_predictions.jsonl"
        stats = service.predict_batch(items, out, resume=not args.no_resume)
        print(json.dumps({"output": str(out), **stats}))
        if args.csv:
            recs = load_batch_results(out)
            flat = [{k: (",".join(v) if isinstance(v, list) and all(isinstance(x, str) for x in v) else v)
                     for k, v in r.items() if k != "checks"} for r in recs]
            pd.DataFrame(flat).to_csv(args.csv, index=False)
            print("csv:", args.csv)
    except NgoPhotoError as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
