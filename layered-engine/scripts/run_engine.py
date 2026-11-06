"""Run the layered engine. Every command is a DRY RUN unless --execute is given, and --execute also needs the cloud switch and the API key.

    python scripts/run_engine.py prompt     [--profile P]                       # print the effective prompt and its hash (offline)
    python scripts/run_engine.py schemas    [--profile P]                       # (re)write config/schemas/*.json (offline)
    python scripts/run_engine.py historical LIST.csv --run-name NAME [--execute --max-calls N]
    python scripts/run_engine.py new        IMAGE --image-id ID [--metadata FILE.json] [--execute]
    python scripts/run_engine.py export     RUN_DIR [--labels labels.csv] [--human human.csv]

LIST.csv has the columns image_id,path (no label column is accepted: labels are joined afterwards by `export`).
Authorization: NGO_ALLOW_CLOUD_INFERENCE=true and GEMINI_API_KEY in the environment, or --env-file PATH (only those two keys are read).
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from layered_engine.annotation import HISTORICAL, NEW_PHOTO, build_engine, evaluate_one, plan_run, run_batch
from layered_engine.export import build_enriched
from layered_engine.providers import AuthorizationError, GeminiProvider, load_gemini_config, read_env_file
from layered_engine.rulebook import ROOT
from layered_engine.schema_export import export_schemas

RUNS = ROOT / "runs"


def provider_for(cfg, env_file):
    env = dict(os.environ)
    if env_file:
        env.update(read_env_file(Path(env_file), (cfg["auth"]["api_key_env"], cfg["auth"]["authorization_env"])))
    return GeminiProvider(cfg, env=env)


def read_list(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        cols = [c.lower() for c in (rd.fieldnames or [])]
        if "image_id" not in cols or "path" not in cols:
            raise SystemExit("the list needs the columns image_id,path")
        if any(c in ("label", "historical_label", "historical_decision", "decision") for c in cols):
            raise SystemExit("the list must not contain a label column: historical labels are joined afterwards, never shown to the model")
        base = path.parent
        return [{"image_id": r["image_id"], "path": (base / r["path"]) if not Path(r["path"]).is_absolute() else Path(r["path"])} for r in rd]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--profile", default=None)
    common.add_argument("--execute", action="store_true")
    common.add_argument("--max-calls", type=int, default=None)
    common.add_argument("--env-file", default=None)
    sub.add_parser("prompt", parents=[common])
    sub.add_parser("schemas", parents=[common])
    h = sub.add_parser("historical", parents=[common])
    h.add_argument("list")
    h.add_argument("--run-name", required=True)
    n = sub.add_parser("new", parents=[common])
    n.add_argument("image")
    n.add_argument("--image-id", required=True)
    n.add_argument("--metadata", default=None)
    e = sub.add_parser("export")
    e.add_argument("run_dir")
    e.add_argument("--labels", default=None)
    e.add_argument("--human", default=None)
    args = ap.parse_args()

    cfg = load_gemini_config()
    if args.cmd == "export":
        run = Path(args.run_dir)
        print(build_enriched(run / "results.jsonl", run / "enriched", Path(args.labels) if args.labels else None, Path(args.human) if args.human else None))
        return 0
    engine = build_engine(cfg, None, args.profile)
    if args.cmd == "prompt":
        print(engine.prompt.system_text, "\n\n--- user message ---\n", engine.prompt.user_text,
              f"\n\n--- profile {engine.profile.name}, rulebook {engine.profile.rulebook_version}, template {engine.prompt.template_version}, hash {engine.prompt.hash}, "
              f"{len(engine.prompt.criterion_ids)} criteria asked, about {len(engine.prompt.system_text) // 4} prompt tokens (estimate) ---")
        return 0
    if args.cmd == "schemas":
        for p in export_schemas(engine.profile, engine.prompt.criterion_ids):
            print("wrote", p)
        return 0

    mode = HISTORICAL if args.cmd == "historical" else NEW_PHOTO
    if args.cmd == "historical":
        entries = read_list(Path(args.list))
        out = RUNS / args.run_name
    else:
        meta = json.loads(Path(args.metadata).read_text(encoding="utf-8")) if args.metadata else {}
        entries = [{"image_id": args.image_id, "path": Path(args.image), "metadata": meta}]
        out = RUNS / f"new_{args.image_id}"
    plan, todo = plan_run(engine, entries, out, mode, args.max_calls)
    print(f"mode {plan.mode} | profile {plan.profile} | model {plan.model_id} | photos {plan.n_photos} ({plan.n_done} already done, {plan.n_todo} to do) | calls {plan.n_calls} (cap {plan.max_calls})")
    if not args.execute:
        print("DRY RUN: no API call is made. Add --execute to send photos to Gemini.")
        return 0
    if plan.n_todo == 0:
        print("nothing to do")
        return 0
    try:
        engine.provider = provider_for(cfg, args.env_file)  # fails closed without authorization
    except AuthorizationError as exc:
        print(f"REFUSED: {exc}")
        return 2
    if args.cmd == "historical":
        print(run_batch(engine, entries, out, mode, args.max_calls), "->", out / "results.jsonl")
    else:
        row = evaluate_one(engine, args.image_id, args.image, mode, entries[0]["metadata"], out / "raw")
        out.mkdir(parents=True, exist_ok=True)
        with (out / "results.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row.get("decision") or {"error": row.get("error")}, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
