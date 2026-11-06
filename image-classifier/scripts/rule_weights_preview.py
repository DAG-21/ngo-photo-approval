"""Exploratory: which rules does a model LEARN to rely on when predicting the human approve / not-approve decision?

Fits an L2 logistic regression on per-rule results (saved by scripts/rule_check.py) against the human labels,
using TRAIN + VALIDATION split images only (the test split is never touched), with group-aware cross-validation.
Runs locally; makes no API call.

    python scripts/rule_weights_preview.py [--per-image PATH] [--c 0.3]

Per rule it has three indicator features: applicable-PASS, FAIL, UNCERTAIN (N/A is the baseline).
Coefficients are on standardized features, so their sizes are comparable. This is a preview on a partial sample made
with an older prompt, not the final model (that is scripts/train_probability_model.py on the final analyses).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ngo_photo_approval.config import load_project_config

ACTIVITY = ["R006", "R014"]  # "is the seva shown": also fails for bills / screenshots


def load(per_image: Path, project) -> tuple[pd.DataFrame, list[str]]:
    t = pd.read_csv(per_image)
    t = t[t.analyzed].copy()
    splits = pd.concat([pd.read_csv(project.output_dir / "splits" / f"{s}.csv") for s in ("train", "validation")])
    t = t.merge(splits[["image_id", "group_id"]], on="image_id")  # drops test images and excluded duplicates
    t = t.sort_values("image_id").drop_duplicates("sha256")
    rules = [c for c in t.columns if c.startswith("R") and len(c) == 4]
    return t, rules


def features(t: pd.DataFrame, rules: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    cols = {}
    for r in rules:
        cols[f"{r}_pass"] = (t[r] == "PASS").astype(float)
        cols[f"{r}_fail"] = (t[r] == "FAIL").astype(float)
        cols[f"{r}_unc"] = (t[r] == "UNCERTAIN").astype(float)
    return pd.DataFrame(cols, index=t.index), (t.label == "approved").astype(int).to_numpy()


def cv_auc(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, c: float, seed: int = 0) -> tuple[float, np.ndarray]:
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000)).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return float(roc_auc_score(y, oof)), oof


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-image", type=Path, default=None)
    ap.add_argument("--c", type=float, default=0.3)
    args = ap.parse_args()
    project = load_project_config()
    path = args.per_image or next(iter(sorted((project.output_dir / "results" / "rule_check").glob("*/*/per_image.csv"))))
    t, rules = load(path, project)
    X, y = features(t, rules)
    groups = t.group_id.to_numpy()
    print(f"source: {path.relative_to(project.output_dir.parent)}")
    print(f"images used: {len(t)} unique (train+validation only): {int(y.sum())} approved / {int((1 - y).sum())} not approved; test split excluded\n")

    auc_all, oof = cv_auc(X, y, groups, args.c)
    sel = lambda keep: X[[c for c in X.columns if c.split("_")[0] in keep]]  # noqa: E731
    auc_act, _ = cv_auc(sel(ACTIVITY), y, groups, args.c)
    auc_rest, _ = cv_auc(sel([r for r in rules if r not in ACTIVITY]), y, groups, args.c)
    print("grouped 5-fold cross-validated ROC-AUC (approved vs not approved):")
    print(f"  all {len(rules)} rules                      : {auc_all:.3f}")
    print(f"  only R006+R014 (seva shown)            : {auc_act:.3f}")
    print(f"  all rules EXCEPT R006+R014             : {auc_rest:.3f}\n")

    m = make_pipeline(StandardScaler(), LogisticRegression(C=args.c, max_iter=2000)).fit(X, y)
    coef = pd.Series(m[-1].coef_[0], index=X.columns)
    rows = []
    for r in rules:
        rows.append({"rule": r, "pass_weight": coef[f"{r}_pass"], "fail_weight": coef[f"{r}_fail"],
                     "uncertain_weight": coef[f"{r}_unc"],
                     "importance": abs(coef[f"{r}_pass"]) + abs(coef[f"{r}_fail"]) + abs(coef[f"{r}_unc"])})
    w = pd.DataFrame(rows).sort_values("importance", ascending=False)
    print("learned weights (standardized; + raises the odds of 'approved', - lowers them):")
    print(w.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))

    ids = t.reset_index(drop=True)
    for rel in ("disapproved/disapproved_3.jpg",):
        man = pd.read_csv(project.output_dir / "manifests" / "image_manifest.csv", low_memory=False).set_index("rel_path")
        iid = man.loc[rel, "image_id"]
        hit = ids.index[ids.image_id == iid]
        if len(hit):
            print(f"\nout-of-fold P(approved) for {rel}: {oof[hit[0]]:.2f}  (human label: not approved)")
        else:
            print(f"\n{rel} is not in the fitted sample (test split or a duplicate-label image), so it is not scored here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
