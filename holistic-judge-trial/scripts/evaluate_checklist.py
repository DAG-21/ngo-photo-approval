"""Analyse a finished checklist run (offline; reads results.jsonl, the pool CSV and the imported human tags).

    python scripts/evaluate_checklist.py --pool dev --tag pilot
    python scripts/evaluate_checklist.py --pool dev --tag tagged-paired --version checklist1 --variant K

Prints a report (also saved next to the run as report.md) and writes features.csv: one row per photo with every check's
answer, each rule's fraction of checks met and its PASS/FAIL label, ready for a model. Labels are read here, never sent.
"""

import argparse
import sys

import _bootstrap  # noqa: F401
import pandas as pd

from judge_trial.checklist import load_checks, rule_summary
from judge_trial.checklist_eval import badness, cv_auc, check_table, load_answers, render_report, tag_agreement
from judge_trial.config import load_trial_config
from judge_trial.data import load_pool
from judge_trial.examples import load_examples
from judge_trial.runner import run_dir_for


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, choices=["dev", "confirm", "validation"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--version", default="checklist1")
    ap.add_argument("--variant", default="K")
    args = ap.parse_args()

    cfg = load_trial_config()
    spec = load_checks(args.version)
    run_dir = run_dir_for(cfg, args.pool, args.variant, args.tag)
    answers = load_answers(run_dir, spec)
    if answers.empty:
        print(f"no successful results in {run_dir}")
        return 2
    pool = load_pool(cfg, args.pool, {e["image_id"] for e in load_examples(cfg)}).set_index("image_id")
    pool = pool.loc[answers.index]
    y = (pool["label"] == "approved").astype(int)

    tags_path = cfg.main_root / "data" / "annotations" / "disapproval_reasons.csv"
    tags = pd.read_csv(tags_path).set_index("image_id") if tags_path.exists() else pd.DataFrame(columns=["primary", "second", "third"])

    tbl = check_table(answers, y, spec, pool["program"])
    cv = cv_auc(badness(answers, spec), y, pool["group_id"])
    agree = tag_agreement(answers, y, tags, spec)
    n = {"photos": len(answers), "approved": int(y.sum()), "disapproved": int((1 - y).sum())}
    report = render_report(tbl, cv, agree, n)
    print(report)
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    feats = answers.copy()
    rows = []
    for _, row in answers.iterrows():
        rs = rule_summary({c: {"answer": a, "evidence": ""} for c, a in row.items()}, spec)
        rows.append({k: v for r, s in rs.items() for k, v in ((f"{r}_fraction_met", s["fraction_met"]), (f"{r}_status", s["status"]))})
    feats = pd.concat([feats, pd.DataFrame(rows, index=answers.index)], axis=1)
    feats.insert(0, "human_label", pool["label"])
    feats.insert(1, "program", pool["program"])
    feats.to_csv(run_dir / "features.csv", encoding="utf-8")
    print(f"\nsaved {run_dir / 'report.md'} and {run_dir / 'features.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
