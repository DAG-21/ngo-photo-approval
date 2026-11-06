"""Build and evaluate the combiner: learned image score + rule results -> final approval probability.

Stage 1  image score: CLIP embedding of the resolution-normalized photo -> logistic regression on the human labels.
         Out-of-fold (OOF) scores are produced with group-aware cross-validation, so a photo's score never used its
         own label (the combiner must not learn from leaked, over-confident scores).
Stage 2  combiner: logistic regression on [logit(OOF image score)] + the rule indicators. It learns, from your labels,
         how much the image signal and each rule count. It is then calibrated (cross-fitted).

Only TRAIN + VALIDATION photos are used; the test split is never touched. Local only: nothing is sent anywhere.
Because rule results currently exist for only ~20% of the photos (and come from an older prompt), the combined model is
saved as PROVISIONAL and the comparison is made with repeated grouped cross-validation on the overlap.

    python scripts/build_combiner.py [--rule-version rules-v3__1.1-sop] [--repeats 3]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ngo_photo_approval.config import load_model_config, load_policy_config, load_project_config, setup_logging
from ngo_photo_approval.evaluation.calibration_metrics import expected_calibration_error, plot_reliability
from ngo_photo_approval.evaluation.metrics import select_threshold
from ngo_photo_approval.rules.loader import load_rule_set, rule_set_fingerprint
from ngo_photo_approval.scoring.calibration import calibration_status, fit_calibrated
from ngo_photo_approval.scoring.feature_builder import FEATURE_VERSION, feature_names, image_logit
from ngo_photo_approval.scoring.image_score import CLIP_MODEL_NAME
from ngo_photo_approval.scoring.offline import (
    feature_table,
    load_analyses,
    load_embeddings,
    load_manifest,
    unique_trainval,
)
from ngo_photo_approval.scoring.train import make_base

STAGE1_C = 0.01  # chosen in scripts/embedding_probe.py (best of 0.001 / 0.01 / 0.1)
STAGE2_GRID = (0.03, 0.1, 0.3, 1.0)


def slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def recall_at_far(y: np.ndarray, p: np.ndarray, max_far: float = 0.05) -> float:
    """Approved recall at <=5% false approvals, threshold taken from these (out-of-fold) scores: optimistic."""
    thr = np.quantile(p[y == 0], 1 - max_far)
    return float((p[y == 1] > thr).mean())


def summarize(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "recall_at_far_5pct": recall_at_far(y, p), "brier": float(brier_score_loss(y, np.clip(p, 0, 1)))}


def folds(y, g, seed):
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, g)


def stage1_oof(X: np.ndarray, y: np.ndarray, g: np.ndarray, seed: int = 0) -> np.ndarray:
    oof = np.zeros(len(y))
    for tr, te in folds(y, g, seed):
        oof[te] = make_base(STAGE1_C, seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof


def cv_stage2(X, y, g, c, seed, method=None):
    """Out-of-fold probabilities: raw, and (if ``method``) cross-fitted calibrated."""
    raw, cal = np.zeros(len(y)), np.zeros(len(y)) if method else None
    for tr, te in folds(y, g, seed):
        raw[te] = make_base(c, seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        if method:
            cal[te] = fit_calibrated(make_base(c, seed), X[tr], y[tr], g[tr], method, seed=seed).predict_proba(X[te])[:, 1]
    return raw, cal


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule-version", default="rules-v3__1.1-sop", help="rule_check results folder to take rule results from")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    project = load_project_config()
    setup_logging(project.logging["level"])
    model_cfg, policy, rule_set = load_model_config(), load_policy_config(), load_rule_set()
    model_id = args.model_id or model_cfg.models["primary"].id
    art = project.artifacts_dir / "probability_model"

    # ------------------------------------------------------------------ stage 1: image score
    emb = load_embeddings(project)
    d = unique_trainval(project)
    d = d[d.image_id.isin(emb)].reset_index(drop=True)
    Xe = np.stack([emb[i] for i in d.image_id])
    y1 = (d.label == "approved").astype(int).to_numpy()
    g1 = d.group_id.to_numpy()
    oof1 = stage1_oof(Xe, y1, g1, args.seed)
    auc1 = float(roc_auc_score(y1, oof1))
    final1 = make_base(STAGE1_C, args.seed).fit(Xe, y1)
    d1 = art / "image_score"
    d1.mkdir(parents=True, exist_ok=True)
    joblib.dump(final1, d1 / "model.joblib")
    meta1 = {
        "model_version": "img-" + hashlib.sha256(json.dumps([CLIP_MODEL_NAME, STAGE1_C, len(y1)]).encode()).hexdigest()[:10],
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"), "embedding_model": CLIP_MODEL_NAME,
        "algorithm": "logistic_regression on L2-normalized CLIP image embeddings", "C": STAGE1_C,
        "normalize_long_side_px": model_cfg.image_input.tuning_normalize_long_side_px,
        "n_photos": int(len(y1)), "n_approved": int(y1.sum()), "n_not_approved": int((1 - y1).sum()),
        "oof_roc_auc": auc1, "trained_on": "unique train+validation photos (test split excluded)", "uses_test_data": False,
    }
    (d1 / "metadata.json").write_text(json.dumps(meta1, indent=2), encoding="utf-8")
    print(f"stage 1 image score: {len(y1)} photos, out-of-fold ROC-AUC {auc1:.3f}  -> {d1}")
    oof_by_id = dict(zip(d.image_id, oof1, strict=True))

    # calibrated image-only model on all photos (the best model so far): LR on the OOF logit, cross-fitted calibration
    Xi = np.array([[image_logit(p)] for p in oof1])
    raw_i, cal_i_oof = cv_stage2(Xi, y1, g1, 1.0, args.seed, policy.calibration.method)
    ece_i, auc_i = expected_calibration_error(y1, cal_i_oof), float(roc_auc_score(y1, cal_i_oof))
    status_i, reasons_i = calibration_status(len(y1), int(y1.sum()), int((1 - y1).sum()), ece_i, policy.calibration, auc_i)
    tsel = policy.threshold_selection
    sel_i = select_threshold(y1, cal_i_oof, tsel.max_false_approval_rate, tsel.min_validation_recall, tsel.candidate_grid_step)
    sel_i["selected_on"] = "cross-validated out-of-fold predictions (provisional)"
    d0 = art / "image_only"
    d0.mkdir(parents=True, exist_ok=True)
    base_i = make_base(1.0, args.seed).fit(Xi, y1)
    joblib.dump({"base": base_i, "feature_names": ["img_logit"], "calibrated": fit_calibrated(
        make_base(1.0, args.seed), Xi, y1, g1, policy.calibration.method, seed=args.seed)}, d0 / "model.joblib")
    (d0 / "metadata.json").write_text(json.dumps({
        "model_version": "prob-" + hashlib.sha256(json.dumps(["image_only", meta1["model_version"], len(y1)]).encode()).hexdigest()[:10],
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"), "algorithm": "logistic_regression on the image-score logit",
        "C": 1.0, "calibration_method": policy.calibration.method, "calibration_status": status_i, "calibration_reasons": reasons_i,
        "calibration_fit": "cross-fitted with group-aware folds", "feature_names": ["img_logit"], "feature_set": "image",
        "feature_version": FEATURE_VERSION, "image_score_model": "image_score", "image_score_version": meta1["model_version"],
        "n_train": int(len(y1)), "n_train_approved": int(y1.sum()), "n_train_not_approved": int((1 - y1).sum()),
        "cv": {"roc_auc": auc_i, "ece": float(ece_i)}, "rules_used": False,
        "note": "ignores the rules entirely; do not use as a decision model while any rule is mandatory", "uses_test_data": False,
    }, indent=2), encoding="utf-8")
    (d0 / "threshold_selection.json").write_text(json.dumps(sel_i, indent=2), encoding="utf-8")
    print(f"image-only calibrated model: CV ROC-AUC {auc_i:.3f}, ECE {ece_i:.3f}, status = {status_i}"
          f"; threshold for <=5% false approvals {sel_i['threshold']} (feasible={sel_i['feasible']}, "
          f"approved recall {sel_i['validation_recall']:.1%})  -> {d0}")

    # ------------------------------------------------------------------ stage 2: rules + image
    slug_dir = project.output_dir / "results" / "rule_check" / slug(model_id) / args.rule_version
    analyses = load_analyses(slug_dir / "analyses.jsonl")
    man = load_manifest(project).set_index("image_id")
    by_sha: dict[str, dict] = {}
    for iid, rec in analyses.items():
        if rec.get("ok") and iid in man.index:
            by_sha[man.loc[iid, "sha256"]] = rec
    per_photo = {r.image_id: by_sha[r.sha256] for r in d.itertuples() if r.sha256 in by_sha}
    Xr, y, g, ids, drops = feature_table(d[d.image_id.isin(per_photo)], rule_set, policy, "rules", {}, per_photo)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    print(f"stage 2: {len(y)} photos have both an image score and rule results ({n_pos} approved / {n_neg} not approved) "
          f"from '{args.rule_version}'; dropped {drops}")
    img = np.array([image_logit(oof_by_id[i]) for i in ids])[:, None]
    names_rules = feature_names(rule_set, "rules")
    names_comb = feature_names(rule_set, "image+rules")
    assert names_comb == ["img_logit", *names_rules]
    Xc = np.hstack([img, Xr])

    image_only = [summarize(y, np.array([oof_by_id[i] for i in ids]))] * args.repeats  # fixed OOF scores
    results = {"image_only": image_only}
    best_c: dict[str, float] = {}
    summary_cols = [names_rules.index(n) for n in ("n_fail", "n_uncertain", "frac_pass")]
    Xs = np.hstack([img, Xr[:, summary_cols]])  # image + only 3 rule-summary numbers (far fewer parameters)
    for label, X in (("rules_only", Xr), ("combined", Xc), ("combined_summary", Xs)):
        best = None
        for c in STAGE2_GRID:
            runs = [summarize(y, cv_stage2(X, y, g, c, args.seed + r)[0]) for r in range(args.repeats)]
            m = float(np.mean([x["roc_auc"] for x in runs]))
            if best is None or m > best[0]:
                best = (m, c, runs)
        best_c[label] = best[1]
        results[label] = best[2]
    mean = lambda lab, k: float(np.mean([x[k] for x in results[lab]]))  # noqa: E731
    sd = lambda lab, k: float(np.std([x[k] for x in results[lab]]))  # noqa: E731
    gain = [c["roc_auc"] - i["roc_auc"] for c, i in zip(results["combined"], results["image_only"], strict=True)]

    # calibrated out-of-fold predictions for the combiner (ECE, threshold, calibration claim)
    c2 = best_c["combined"]
    raw_oof, cal_oof = cv_stage2(Xc, y, g, c2, args.seed, policy.calibration.method)
    ece_cal = expected_calibration_error(y, cal_oof)
    auc_cal = float(roc_auc_score(y, cal_oof))
    status, reasons = calibration_status(len(y), n_pos, n_neg, ece_cal, policy.calibration, auc_cal)
    ts = policy.threshold_selection
    sel = select_threshold(y, cal_oof, ts.max_false_approval_rate, ts.min_validation_recall, ts.candidate_grid_step)
    sel["selected_on"] = "cross-validated out-of-fold predictions (provisional)"

    # ------------------------------------------------------------------ final fit + artifact
    base = make_base(c2, args.seed).fit(Xc, y)
    cal = fit_calibrated(make_base(c2, args.seed), Xc, y, g, policy.calibration.method, seed=args.seed)
    d2 = art / "image_rules"
    d2.mkdir(parents=True, exist_ok=True)
    joblib.dump({"base": base, "calibrated": cal, "feature_names": names_comb}, d2 / "model.joblib")
    meta2 = {
        "model_version": "prob-" + hashlib.sha256(json.dumps([names_comb, c2, len(y), meta1["model_version"]]).encode()).hexdigest()[:10],
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "algorithm": "logistic_regression stacked on the image score", "C": c2,
        "calibration_method": policy.calibration.method, "calibration_status": status, "calibration_reasons": reasons,
        "calibration_fit": "cross-fitted on the training photos with group-aware folds",
        "feature_names": names_comb, "feature_set": "image+rules", "feature_version": FEATURE_VERSION,
        "image_score_model": "image_score", "image_score_version": meta1["model_version"],
        "rule_set_version": rule_set.rule_set_version, "rule_set_fingerprint": rule_set_fingerprint(rule_set),
        "rule_set_confirmed": rule_set.rule_set_confirmed, "rule_results_source": args.rule_version,
        "beats_image_only_in_cv": bool(np.mean(gain) > 0),
        "provisional": True,
        "provisional_reason": (f"fitted on only {len(y)} photos with rule results from an older prompt/rule set "
                               f"({args.rule_version}); refit when rule results exist for all train photos"),
        "warning": ("does NOT beat the image-only model in cross-validation with the current rule results; prefer image_only"
                    if not np.mean(gain) > 0 else ""),
        "n_train": int(len(y)), "n_train_approved": n_pos, "n_train_not_approved": n_neg, "seed": args.seed,
        "cv": {k: {"roc_auc_mean": mean(k, "roc_auc"), "roc_auc_sd": sd(k, "roc_auc")} for k in results},
        "uses_test_data": False,
    }
    (d2 / "metadata.json").write_text(json.dumps(meta2, indent=2), encoding="utf-8")
    (d2 / "threshold_selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    plot_reliability({"combined, raw (CV)": (y, raw_oof), "combined, calibrated (CV)": (y, cal_oof)},
                     project.artifacts_dir / "calibration" / "image_rules" / "reliability_cv.png")

    # learned weights (standardized coefficients; sign = direction, size = strength)
    coef = pd.Series(base[-1].coef_[0], index=names_comb)
    rows = [{"feature": "IMAGE SCORE (logit)", "weight": coef["img_logit"]}]
    for r in rule_set.active():
        rows.append({"feature": f"{r.rule_id} {r.name}",
                     "weight": coef[f"{r.rule_id}_pass"], "fail": coef[f"{r.rule_id}_fail"], "uncertain": coef[f"{r.rule_id}_uncertain"]})
    wt = pd.DataFrame(rows)

    report = {
        "photos": {"stage1": int(len(y1)), "stage2_overlap": int(len(y)), "approved": n_pos, "not_approved": n_neg},
        "stage1_oof_roc_auc": auc1, "cv": {k: {m: {"mean": mean(k, m), "sd": sd(k, m)} for m in results[k][0]} for k in results},
        "best_C": best_c, "combined_minus_image_only_auc": {"mean": float(np.mean(gain)), "per_repeat": gain},
        "combined_summary_minus_image_only_auc": {"mean": float(np.mean(
            [c["roc_auc"] - i["roc_auc"] for c, i in zip(results["combined_summary"], results["image_only"], strict=True)]))},
        "calibrated_oof": {"ece": float(ece_cal), "roc_auc": auc_cal, "status": status, "reasons": reasons},
        "threshold_selection": sel,
        "weights": {r["feature"]: {k: float(v) for k, v in r.items() if k != "feature"} for r in rows},
    }
    out = project.output_dir / "results" / "combiner"
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\ngrouped 5-fold CV over {len(y)} photos, {args.repeats} repeats (mean +/- sd across repeats):")
    print(f"{'model':<14}{'ROC-AUC':>16}{'PR-AUC':>10}{'recall@5%FAR':>15}{'Brier':>9}")
    for k, label in (("image_only", "image only"), ("rules_only", "rules only"), ("combined", "image+all rules"),
                     ("combined_summary", "image+summary")):
        print(f"{label:<14}{mean(k, 'roc_auc'):>10.3f} +/-{sd(k, 'roc_auc'):.3f}{mean(k, 'pr_auc'):>10.3f}"
              f"{mean(k, 'recall_at_far_5pct'):>15.1%}{mean(k, 'brier'):>9.3f}")
    print(f"image+all rules minus image-only AUC: {np.mean(gain):+.3f} (per repeat: {', '.join(f'{x:+.3f}' for x in gain)})")
    gain_s = [c["roc_auc"] - i["roc_auc"] for c, i in zip(results["combined_summary"], results["image_only"], strict=True)]
    print(f"image+summary   minus image-only AUC: {np.mean(gain_s):+.3f} (per repeat: {', '.join(f'{x:+.3f}' for x in gain_s)})")
    print(f"approved base rate on this overlap: {y.mean():.2f}")
    print(f"calibrated combiner (out-of-fold): ECE {ece_cal:.3f}, status = {status}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    print(f"threshold for <=5% false approvals (CV): {sel['threshold']} (feasible={sel['feasible']}, recall {sel['validation_recall']:.1%})")
    print("\nlearned weights (standardized; + raises the odds of approved):")
    print(wt.sort_values("weight", key=lambda s: -s.abs()).head(8).to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    print(f"\nartifacts: {d1}  and  {d2}  (provisional)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
