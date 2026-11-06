"""Score finished runs against the human labels (offline; reads results.jsonl only).

    python scripts/evaluate_trial.py report   --variant D --pool dev [--tag t]
    python scripts/evaluate_trial.py ablation --pool dev                 # one row per variant that has been run
    python scripts/evaluate_trial.py noise    --variant D --pool dev --tags a b   # how stable are repeated runs?
"""

import argparse

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from judge_trial.config import load_trial_config
from judge_trial.data import load_pool
from judge_trial.evaluate import baseline_scores, evaluate_frame, load_run_frame, render_report, save_report
from judge_trial.examples import load_examples
from judge_trial.runner import read_results, run_dir_for


def evaluate_one(cfg, pool_name: str, variant: str, tag: str | None, n_boot: int):
    pool = load_pool(cfg, pool_name, {e["image_id"] for e in load_examples(cfg)})
    run_dir = run_dir_for(cfg, pool_name, variant, tag)
    df, meta = load_run_frame(run_dir, pool)
    res = evaluate_frame(df, meta, baseline_scores(cfg.main_root, df["image_id"]), n_boot)
    return run_dir, df, res


def cmd_report(cfg, args) -> int:
    run_dir, df, res = evaluate_one(cfg, args.pool, args.variant, args.tag, args.boot)
    save_report(run_dir, res, df)
    print(render_report(res))
    print(f"\nsaved: {run_dir / 'report.md'}, metrics.json, false_approvals.csv")
    return 0


def cmd_ablation(cfg, args) -> int:
    rows = []
    for v in "ABCD":
        try:
            _, _, r = evaluate_one(cfg, args.pool, v, args.tag, args.boot)
        except FileNotFoundError:
            continue
        if "auc" not in r:
            continue
        op, bs = r["operating_point_5pct_far"], r["bootstrap"]
        rows.append({"variant": v, "photos": r["n_scored"], "AUC": round(r["auc"], 3),
                     "AUC 95% range": f"{bs['auc_ci95'][0]:.2f}-{bs['auc_ci95'][1]:.2f}" if bs["auc_ci95"] else "n/a",
                     "approved accepted @5% FAR": f"{op['approved_recall']:.1%}", "FP": op["false_positives"], "FN": op["false_negatives"],
                     "failed": r["failed"], "embeddings AUC": round(r["baseline_embeddings"]["auc"], 3) if "baseline_embeddings" in r else None})
    print(pd.DataFrame(rows).to_string(index=False) if rows else "no finished runs for this pool yet")
    return 0


def cmd_noise(cfg, args) -> int:
    a, b = (pd.DataFrame(read_results(run_dir_for(cfg, args.pool, args.variant, t))) for t in args.tags)
    m = a[a.ok == True].merge(b[b.ok == True], on="image_id", suffixes=("_a", "_b"))  # noqa: E712
    if m.empty:
        print("no photos scored in both runs")
        return 1
    d = (m.score_a - m.score_b).abs()
    print(f"photos in both runs: {len(m)}\nspearman correlation of scores: {m.score_a.corr(m.score_b, method='spearman'):.3f}\n"
          f"mean absolute score difference: {d.mean():.1f} points (max {d.max():.0f})\n"
          f"decision flips: {(m.decision_a != m.decision_b).mean():.1%}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("report", "ablation", "noise"):
        p = sub.add_parser(name)
        p.add_argument("--pool", required=True, choices=["dev", "confirm", "validation"])
        p.add_argument("--boot", type=int, default=500, help="bootstrap resamples")
        if name != "ablation":
            p.add_argument("--variant", required=True, choices=list("ABCD"))
        if name == "noise":
            p.add_argument("--tags", nargs=2, required=True)
        else:
            p.add_argument("--tag", default=None)
    args = ap.parse_args()
    cfg = load_trial_config()
    return {"report": cmd_report, "ablation": cmd_ablation, "noise": cmd_noise}[args.cmd](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
