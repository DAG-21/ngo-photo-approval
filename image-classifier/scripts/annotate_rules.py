"""Track B support: create a human-reviewable rule-annotation template, and validate completed annotations.

  python scripts/annotate_rules.py template --splits train validation --sample 300
  python scripts/annotate_rules.py validate

The template has ONE ROW PER (image, active rule). Humans fill `status` (PASS/FAIL/UNCERTAIN), optional
`evidence`, `annotator`, and set `verified` to true. If saved VLM analyses exist, they are shown ONLY in the
separate `vlm_suggestion` column to speed up review; they are never copied into `status` and rows are never
treated as ground truth until a human sets `verified=true`.

Completed annotations go to data/annotations/rule_annotations.csv (git-ignored: it describes client images).
Test-split images are excluded from templates so rule labels cannot leak into model selection by accident.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from ngo_photo_approval.config import load_project_config, setup_logging
from ngo_photo_approval.rules.loader import load_rule_set
from ngo_photo_approval.scoring.offline import load_analyses, load_split

COLUMNS = ["image_id", "rel_path", "program", "human_label", "rule_id", "rule_name", "rule_description",
           "vlm_suggestion", "status", "evidence", "annotator", "verified", "notes"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["template", "validate"])
    ap.add_argument("--splits", nargs="+", default=["train", "validation"], choices=["train", "validation"])
    ap.add_argument("--sample", type=int, default=300, help="images per split (stratified by human label)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suggestions-model", default=None, help="model id whose saved analyses fill vlm_suggestion")
    args = ap.parse_args()

    project = load_project_config()
    setup_logging(project.logging["level"])
    rules = load_rule_set().active()
    ann_dir = project.output_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "template":
        rows = []
        for split in args.splits:
            df = load_split(project, split)
            if args.sample < len(df):
                total = len(df)
                df = df.groupby("label", group_keys=False).apply(
                    lambda g, total=total: g.sample(n=max(1, round(args.sample * len(g) / total)), random_state=args.seed))
            sugg = {}
            if args.suggestions_model:
                sugg = load_analyses(project.output_dir / "results" / "vlm" / args.suggestions_model.replace("/", "_")
                                     / f"{split}.jsonl")
            for r in df.itertuples():
                by_rule = {c["rule_id"]: c["status"] for c in (sugg.get(r.image_id, {}).get("analysis") or {}).get("checks", [])}
                for rule in rules:
                    rows.append(dict(image_id=r.image_id, rel_path=r.rel_path, program=r.program, human_label=r.label,
                                     rule_id=rule.rule_id, rule_name=rule.name, rule_description=rule.description,
                                     vlm_suggestion=by_rule.get(rule.rule_id, ""), status="", evidence="",
                                     annotator="", verified="", notes=""))
        path = ann_dir / "rule_annotation_template.csv"
        pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
        print(f"wrote {path} ({len(rows)} rows, {len({r['image_id'] for r in rows})} images x {len(rules)} rules)")
        print("Fill status/evidence/annotator and set verified=true, then save as rule_annotations.csv in the same folder.")
        return 0

    path = ann_dir / "rule_annotations.csv"
    if not path.exists():
        print(f"{path} not found: no verified rule annotations exist yet (Track B is blocked on this).")
        return 1
    df = pd.read_csv(path)
    problems = []
    need = {"image_id", "rule_id", "status", "verified"}
    if not need <= set(df.columns):
        print(f"missing columns: {sorted(need - set(df.columns))}")
        return 2
    known = {r.rule_id for r in load_rule_set().rules}
    df["status_n"] = df["status"].astype(str).str.upper().str.strip()
    df["verified_b"] = df["verified"].astype(str).str.lower().isin({"true", "1", "yes"})
    if (~df["status_n"].isin({"PASS", "FAIL", "UNCERTAIN"}) & df["verified_b"]).any():
        problems.append("verified rows with a status other than PASS/FAIL/UNCERTAIN")
    if (~df["rule_id"].isin(known)).any():
        problems.append("unknown rule_id values")
    if df.duplicated(["image_id", "rule_id"]).any():
        problems.append("duplicate (image_id, rule_id) rows")
    test_ids = set(load_split(project, "test").image_id)
    if df["image_id"].isin(test_ids).any():
        problems.append("annotations include TEST-split images (not allowed for model selection/tuning)")
    v = df[df["verified_b"]]
    complete = v.groupby("image_id")["rule_id"].nunique().eq(len(rules)).sum()
    print(f"rows={len(df)} verified={len(v)} images_with_all_{len(rules)}_rules_verified={complete}")
    print(v.groupby("rule_id")["status_n"].value_counts().unstack(fill_value=0).to_string() if len(v) else "no verified rows")
    for p in problems:
        print("PROBLEM:", p)
    return 2 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
