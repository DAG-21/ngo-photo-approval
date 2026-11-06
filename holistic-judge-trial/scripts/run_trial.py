"""Plan, and (only with --execute) run, a judging pass.

    python scripts/run_trial.py --variant D --pool dev                  # DRY RUN: prints the plan, calls nothing
    python scripts/run_trial.py --variant D --pool dev --limit 20       # dry run for a 20-photo pilot
    python scripts/run_trial.py --variant D --pool dev --limit 20 --execute   # real calls (only when you say so)
    python scripts/run_trial.py --variant D --pool dev --random 50 --seed 7   # a seeded random sample instead of --limit

A real run additionally needs the main project's own authorization (NGO_ALLOW_CLOUD_INFERENCE=true and credentials).
Results are appended to runs/<pool>/<variant>__<model>[__tag]/results.jsonl, so an interrupted run resumes.
"""

import argparse
import json
import sys
from dataclasses import asdict

import _bootstrap  # noqa: F401
import pandas as pd

from judge_trial.config import load_trial_config
from judge_trial.data import image_path, load_pool
from judge_trial.examples import load_examples
from judge_trial.prompt import build_for_variant
from judge_trial.runner import RunRefused, execute, make_plan, run_dir_for, write_plan


def show_plan(plan, execute_flag: bool) -> None:
    e = plan.estimate
    print(f"variant {plan.variant}  pool {plan.pool}  model {plan.model_id}  prompt hash {plan.prompt_hash}")
    print(f"photos in this run: {plan.n_photos} ({plan.n_done} already done, {plan.n_todo} to do); "
          f"{plan.n_examples} worked examples in the prompt")
    print(f"API calls this would make: {plan.n_calls_planned}"
          + ("  (one of them creates the context cache)" if e["caching_expected"] else "  (no caching: prompt sent with every photo)"))
    print(f"estimated tokens: fixed part about {e['prefix_tokens']}, each photo about {e['per_photo_tokens']} new input "
          f"+ about {e['per_photo_output_tokens']} output")
    print(f"  total input without cache: {e['input_tokens_no_cache']:,}")
    if e["caching_expected"]:
        print(f"  with cache: {e['input_tokens_with_cache_fresh']:,} full-price + {e['input_tokens_with_cache_cached_reads']:,} cached reads")
    print(f"  cost: {e['usd'] if e['usd'] else 'unknown - fill the prices in config/trial.yaml (cost section)'}")
    print("  (estimates; tokens per photo is an assumption until the first real call reports measured usage)")
    print("MODE: " + ("EXECUTE - photos will be sent to Gemini" if execute_flag else "DRY RUN - no API call is made"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, help="A-D for the holistic judge (v1-v3); K for the checklist version")
    ap.add_argument("--pool", required=True, choices=["dev", "confirm", "validation"])
    ap.add_argument("--limit", type=int, default=None, help="pilot: number of photos (half per class)")
    ap.add_argument("--random", type=int, default=None, dest="random_n", help="a random sample of this many photos (seeded)")
    ap.add_argument("--seed", type=int, default=42, help="seed for --random")
    ap.add_argument("--ids-file", default=None, help="CSV with an image_id column: run exactly these photos (must be in the pool)")
    ap.add_argument("--tag", default=None, help="separate run folder, e.g. for a repeat run")
    ap.add_argument("--version", default="v1", help="prompt version folder")
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--execute", action="store_true", help="actually call Gemini (default: dry run)")
    args = ap.parse_args()

    cfg = load_trial_config()
    prompt = build_for_variant(cfg, args.variant, args.version)
    example_ids = {e["image_id"] for e in load_examples(cfg)}
    pool = load_pool(cfg, args.pool, example_ids)
    run_dir = run_dir_for(cfg, args.pool, args.variant, args.tag)
    try:
        plan, todo = make_plan(cfg, prompt, args.pool, pool, run_dir, args.limit, args.random_n, args.seed,
                              ids=pd.read_csv(args.ids_file)["image_id"].tolist() if args.ids_file else None)
    except RunRefused as exc:
        print(f"REFUSED: {exc}")
        return 2
    missing = [i for i in list(todo["image_id"]) + [e.image_id for e in prompt.examples] if not image_path(cfg, i).exists()]
    if missing:
        print(f"REFUSED: {len(missing)} normalized photos are missing (e.g. {missing[:3]}); run the main project's tuning-data build.")
        return 2
    show_plan(plan, args.execute)
    write_plan(plan, run_dir)
    if not args.execute:
        return 0
    if plan.n_todo == 0:
        print("nothing to do: every photo in this run is already scored.")
        return 0
    from judge_trial.backend import GeminiJudgeBackend  # imported late so dry runs never touch the SDK

    backend = GeminiJudgeBackend(cfg)  # fails closed unless the main project's authorization is in place
    try:
        summary = execute(cfg, prompt, todo, backend, run_dir, args.max_calls or cfg.run.max_calls_default, args.workers)
    except RunRefused as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(json.dumps(summary))
    print(f"results: {run_dir / 'results.jsonl'}\nnext: python scripts/evaluate_trial.py report --variant {args.variant} --pool {args.pool}"
          + (f" --tag {args.tag}" if args.tag else ""))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
