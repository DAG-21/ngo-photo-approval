#!/usr/bin/env python3
"""
Phase 1 pilot: label-blind Gemini annotation of 100 approved + 100 rejected historical photos.

DRY RUN BY DEFAULT. Without `--execute` this only selects the sample, writes the queue files and
prints what would be sent; no Gemini call is made. With `--execute` it annotates the queue,
resuming from existing annotations in `data/processed/annotations/`.

    python run_pilot.py                      # plan only
    python run_pilot.py --execute --limit 3  # smoke test on 3 images
    python run_pilot.py --execute            # full 100 + 100 pilot
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.annotation.annotator import HistoricalDatasetAnnotator
from src.env import load_env_file
from src.gemini.client import GeminiConfigError, GeminiVisionClient


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-approved", type=int, default=100)
    ap.add_argument("--n-rejected", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="Only annotate the first N queued images (smoke test)")
    ap.add_argument("--execute", action="store_true", help="Actually call the Gemini API (default: dry run)")
    args = ap.parse_args()

    load_env_file()
    metadata = pd.read_csv(PROJECT_DIR / "data/processed/image_metadata.csv")
    pilot_dir = PROJECT_DIR / "outputs/pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = PROJECT_DIR / "data/processed/annotations"

    client = GeminiVisionClient(
        config_path=PROJECT_DIR / "config/gemini_config.yaml",
        raw_output_dir=PROJECT_DIR / "outputs/raw_gemini_responses",
    )
    annotator = HistoricalDatasetAnnotator(
        client=client,
        rules_path=PROJECT_DIR / "config/ngo_rules.json",
        prompt_template_path=PROJECT_DIR / "prompts/image_annotation_prompt.txt",
        annotations_dir=annotations_dir,
    )

    queue = annotator.select_pilot_sample(
        metadata, n_approved=args.n_approved, n_rejected=args.n_rejected, seed=args.seed
    )
    assert "historical_decision" not in queue.columns, "Pilot queue must stay label-blind!"

    # The queue is what the annotator sees; the label key is kept in a separate file for later analysis.
    queue.to_csv(pilot_dir / "pilot_queue.csv", index=False)
    key = metadata[metadata["image_id"].isin(queue["image_id"])][["image_id", "historical_decision", "partition"]]
    key.to_csv(pilot_dir / "pilot_label_key.csv", index=False)

    # Resume only from annotations made with the current prompt version; older ones are redone.
    done = set()
    for p in annotations_dir.glob("NGO_IMG_*.json"):
        try:
            version = json.loads(p.read_text(encoding="utf-8")).get("_audit_metadata", {}).get("prompt_version")
        except (OSError, json.JSONDecodeError):
            continue
        if version == annotator.prompt_version:
            done.add(p.stem)
    todo = queue[~queue["image_id"].isin(done)]
    if args.limit is not None:
        todo = todo.head(args.limit)

    sizes = metadata.set_index("image_id").loc[todo["image_id"], "file_size_bytes"]
    print("=== Pilot plan ===")
    print(f"model              : {client.model_name}")
    print(f"API key            : {'found' if client.api_key else 'MISSING'} (value never printed)")
    print(f"selected           : {len(queue)} images "
          f"({int((key['historical_decision'] == 'APPROVED').sum())} approved, "
          f"{int((key['historical_decision'] == 'REJECTED').sum())} rejected; "
          f"partitions: {key['partition'].value_counts().to_dict()})")
    print(f"already annotated  : {len(queue) - len(queue[~queue['image_id'].isin(done)])}")
    print(f"to send this run   : {len(todo)} images, ~{sizes.sum() / 1e6:.0f} MB of source files")
    print(f"queue file         : {pilot_dir / 'pilot_queue.csv'}")

    if not args.execute:
        print("\nDRY RUN: no Gemini calls were made. Re-run with --execute to annotate.")
        return
    if not client.api_key:
        sys.exit(f"Cannot execute: {client.api_key_env} is not set in the environment or .env")
    if not (client.api_key.startswith("AIza") and len(client.api_key) == 39):
        print(f"WARNING: {client.api_key_env} does not look like a Gemini key "
              f"(expected 39 characters starting with 'AIza', got {len(client.api_key)}); the API will likely reject it.")
    try:
        from google import genai  # noqa: F401
    except ImportError:
        sys.exit("Cannot execute: the `google-genai` package is not installed (pip install google-genai)")
    if todo.empty:
        print("Nothing to do: every selected image already has an annotation.")
        return

    start = time.time()
    try:
        statuses = annotator.run_batch(todo[["image_id", "original_filepath", "event_id"]])
    except GeminiConfigError as err:
        sys.exit(f"Aborted, nothing was annotated: {err}")
    failed = {k: v for k, v in statuses.items() if v != "COMPLETED"}
    ledger = [e for e in client.usage_ledger if e.get("status") == "SUCCESS"]
    summary = {
        "requested": len(todo),
        "completed": len(statuses) - len(failed),
        "failed": failed,
        "total_tokens": sum(e.get("total_tokens", 0) for e in ledger),
        "elapsed_seconds": round(time.time() - start, 1),
        "model": client.model_name,
    }
    (pilot_dir / "pilot_run_status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "failed"}, indent=2))
    if failed:
        print(f"{len(failed)} failures, see {pilot_dir / 'pilot_run_status.json'}")


if __name__ == "__main__":
    main()
