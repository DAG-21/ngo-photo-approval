"""Phase 11: evaluate candidate architectures on the SAME split and write the comparison report.

Candidates (each only evaluated when its inputs exist; unavailable ones are listed, never faked):
  technical_quality           quality-feature logistic regression (artifacts/probability_model/technical_quality)
  vlm_rules:<model_id>        saved Gemini rule analyses (data/results/vlm/<slug>/<split>.jsonl)
                              + scoring artifact artifacts/probability_model/vlm_<slug> + rule engine
  tuned_decision:<model_id>   Track A tuned decision labels (data/results/tuned_decision/<slug>/<split>.jsonl)

Thresholds are the ones SELECTED ON VALIDATION (threshold_selection.json in each artifact directory).
The TEST split can only be scored with --final-test, which should be used once, at the very end.

Writes data/results/{metrics.json,predictions.csv,error_analysis.csv},
artifacts/reports/reliability_<split>.png and docs/EVALUATION_REPORT.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ngo_photo_approval.config import load_policy_config, load_project_config, setup_logging
from ngo_photo_approval.evaluation.calibration_metrics import plot_reliability
from ngo_photo_approval.evaluation.error_analysis import build_error_analysis, summarize_errors
from ngo_photo_approval.evaluation.metrics import evaluate_predictions, rule_level_metrics
from ngo_photo_approval.exceptions import NgoPhotoError
from ngo_photo_approval.rules.loader import load_rule_set
from ngo_photo_approval.scoring.feature_builder import build_features, feature_names
from ngo_photo_approval.scoring.offline import checks_of, load_analyses, load_manifest, load_split, quality_lookup
from ngo_photo_approval.scoring.predict import ProbabilityScorer
from ngo_photo_approval.service.prediction_service import assess


def slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def read_threshold(art_dir: Path, default: float) -> tuple[float, str]:
    p = art_dir / "threshold_selection.json"
    if p.exists():
        sel = json.loads(p.read_text(encoding="utf-8"))
        return float(sel["threshold"]), f"validation-selected (feasible={sel['feasible']})"
    return default, "configs/thresholds.yaml (no validation selection found)"


# ------------------------------------------------------------------ candidates


def cand_technical(ctx: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    art = ctx["project"].artifacts_dir / "probability_model" / "technical_quality"
    scorer = ProbabilityScorer.load(art)
    thr, thr_src = read_threshold(art, ctx["policy"].approval_threshold)
    names = feature_names(ctx["rules"], "quality")
    rows = []
    for r in ctx["split"].itertuples():
        f = build_features(ctx["rules"], "quality", None, ctx["quality"][r.image_id])
        est = scorer.score({k: f[k] for k in names})
        rows.append(dict(image_id=r.image_id, decision="APPROVE" if est.probability + 1e-9 >= thr else "REJECT",
                         probability=est.probability, probability_type=est.probability_type.value))
    ptype = rows[0]["probability_type"] if rows else "none"
    return pd.DataFrame(rows), {"threshold": thr, "threshold_source": thr_src, "probability_type": ptype,
                                "note": "No rule engine: quality-only baseline (cannot enforce mandatory rules)."}


def cand_vlm(ctx: dict[str, Any], model_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    project, s = ctx["project"], slug(model_id)
    path = project.output_dir / "results" / "vlm" / s / f"{ctx['split_name']}.jsonl"
    analyses = load_analyses(path)
    if not analyses:
        raise FileNotFoundError(f"no analyses at {path.relative_to(project.output_dir.parent)}")
    art = project.artifacts_dir / "probability_model" / f"vlm_{s}"
    scorer = ProbabilityScorer.load(art)
    fset = scorer.metadata["feature_set"]
    thr, thr_src = read_threshold(art, ctx["policy"].approval_threshold)
    rows = []
    for r in ctx["split"].itertuples():
        rec = analyses.get(r.image_id)
        base = dict(image_id=r.image_id)
        checks = checks_of(rec) if rec else None
        if rec is None or checks is None:
            kind = "no_analysis" if rec is None else (rec.get("error_type") or "failed")
            rows.append({**base, "decision": "ERROR", "failure_kind": kind})
            continue
        a = assess(ctx["rules"], ctx["policy"], checks, getattr(r, "program", None), scorer, fset,
                   ctx["quality"].get(r.image_id), threshold=thr)
        rows.append({**base, "decision": a.decision.value, "probability": a.estimate.probability if a.estimate else None,
                     "probability_type": a.estimate.probability_type.value if a.estimate else None,
                     "hard_rule_failures": "|".join(a.policy_result.hard_rule_failures),
                     "uncertain_rules": "|".join(a.policy_result.uncertain_rules),
                     "reason_codes": "|".join(c.value for c in a.reason_codes),
                     "latency_s": (rec.get("usage") or {}).get("latency_s"),
                     "cost_usd": (rec.get("usage") or {}).get("estimated_cost_usd")})
    info = {"threshold": thr, "threshold_source": thr_src, "analysis_model": model_id,
            "calibration_status": scorer.metadata["calibration_status"],
            "probability_type": "calibrated" if scorer.metadata["calibration_status"] == "calibrated" else "uncalibrated_classifier"}
    ann = project.output_dir / "annotations" / "rule_annotations.csv"
    if ann.exists():
        truth = pd.read_csv(ann)
        pred = [{"image_id": i, "rule_id": c["rule_id"], "status": c["status"]}
                for i, rec in analyses.items() if rec.get("ok") for c in rec["analysis"]["checks"]]
        info["rule_level"] = rule_level_metrics(pd.DataFrame(pred), truth) if pred else {}
    return pd.DataFrame(rows), info


def cand_tuned_decision(ctx: dict[str, Any], model_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    project = ctx["project"]
    path = project.output_dir / "results" / "tuned_decision" / slug(model_id) / f"{ctx['split_name']}.jsonl"
    recs = load_analyses(path)
    if not recs:
        raise FileNotFoundError(f"no decisions at {path.relative_to(project.output_dir.parent)}")
    rows = []
    for r in ctx["split"].itertuples():
        rec = recs.get(r.image_id)
        if rec is None or not rec.get("ok"):
            rows.append(dict(image_id=r.image_id, decision="ERROR",
                             failure_kind="no_result" if rec is None else rec.get("error_type")))
        else:
            rows.append(dict(image_id=r.image_id, decision="APPROVE" if rec["decision"] == "approved" else "REJECT",
                             latency_s=(rec.get("usage") or {}).get("latency_s"),
                             cost_usd=(rec.get("usage") or {}).get("estimated_cost_usd")))
    return pd.DataFrame(rows), {"threshold": None, "note": "Hard label only; no probability, no rule-level audit trail."}


# ------------------------------------------------------------------ driver


def discover(project, split_name: str) -> list[str]:
    cands = []
    if (project.artifacts_dir / "probability_model" / "technical_quality" / "metadata.json").exists():
        cands.append("technical_quality")
    for kind, folder in (("vlm_rules", "vlm"), ("tuned_decision", "tuned_decision")):
        root = project.output_dir / "results" / folder
        if root.exists():
            cands += [f"{kind}:{d.name}" for d in sorted(root.iterdir()) if (d / f"{split_name}.jsonl").exists()]
    return cands


def fmt(v: Any, pct: bool = True) -> str:
    if v is None:
        return "n/a"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def render_report(all_metrics: dict[str, Any], unavailable: dict[str, list[str]], max_far: float, min_auc: float = 0.6) -> str:
    L = ["# Evaluation Report\n",
         "_Auto-generated by `scripts/evaluate_models.py`. Positive class = human `approved`. "
         "**FAR** (false approval rate) = P(system approves | human not_approved) is the primary business risk._\n",
         "_`strict` counts unresolved outputs (REVIEW/ERROR) as REJECT. `resolved` ignores them. Thresholds were chosen on the validation split only._\n"]
    for split in ("validation", "test"):
        block = all_metrics.get(split)
        if not block:
            L.append(f"## {split.capitalize()} split\n\n_Not evaluated yet._\n")
            continue
        L.append(f"## {split.capitalize()} split ({block['n_images']} images, {block['approved_share']:.1%} approved)\n")
        L.append("| Candidate | Auto-processed | FAR (strict) | FRR (strict) | Approval precision | Approval recall | Accuracy | ROC-AUC | PR-AUC | Brier | ECE | Threshold |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, m in block["candidates"].items():
            s, p = m["metrics"]["strict"], m["metrics"].get("probability", {})
            L.append(f"| {name} | {fmt(m['metrics']['processing']['automatic_rate'])} | {fmt(s.get('false_approval_rate'))} | "
                     f"{fmt(s.get('false_rejection_rate'))} | {fmt(s.get('approval_precision'))} | {fmt(s.get('approval_recall'))} | "
                     f"{fmt(s.get('accuracy'))} | {fmt(p.get('roc_auc'), False)} | {fmt(p.get('pr_auc'), False)} | "
                     f"{fmt(p.get('brier'), False)} | {fmt(p.get('ece'), False)} | {m['info'].get('threshold')} |")
        L.append("")
        for name, m in block["candidates"].items():
            cm = m["metrics"]["strict"].get("confusion_matrix", {})
            L.append(f"- **{name}**: confusion (strict) TP={cm.get('tp')} FP={cm.get('fp')} FN={cm.get('fn')} TN={cm.get('tn')}; "
                     f"probabilities: {m['info'].get('probability_type', 'none')}; threshold {m['info'].get('threshold_source', 'n/a')}; latency {m['metrics'].get('latency_s', 'n/a')}; "
                     f"cost {m['metrics'].get('cost_usd')}")
            if m["info"].get("note"):
                L.append(f"  - {m['info']['note']}")
            if m.get("error_summary"):
                L.append(f"  - error tags: {json.dumps(m['error_summary'])}")
            if m["info"].get("rule_level"):
                L.append("  - rule-level accuracy vs verified labels: " + ", ".join(
                    f"{k}={v['accuracy']:.0%} (n={v['n']})" for k, v in m["info"]["rule_level"].items()))
        L.append("")
        no_skill = [n for n, m in block["candidates"].items()
                    if (m["metrics"].get("probability", {}).get("roc_auc") or 1.0) < min_auc]
        for n in no_skill:
            L.append(f"**WARNING - `{n}` has validation-style ROC-AUC below {min_auc} (no useful discrimination); "
                     "its apparent FAR/recall is noise and it is excluded from the recommendation.**")
        ok = [(n, m["metrics"]["strict"]) for n, m in block["candidates"].items()
              if n not in no_skill and (m["metrics"]["strict"].get("false_approval_rate") or 0) <= max_far]
        if ok:
            best = max(ok, key=lambda t: t[1].get("approval_recall") or 0)
            L.append(f"**Best candidate meeting FAR <= {max_far:.0%} on {split}:** `{best[0]}` "
                     f"(recall {fmt(best[1].get('approval_recall'))}).")
        else:
            L.append(f"**No candidate meets the FAR <= {max_far:.0%} limit on the {split} split.**")
        if split == "validation":
            L.append("_Validation results were used to pick thresholds, so they are optimistic; confirm on the test split._")
        L.append("")
    if unavailable:
        L.append("## Candidates not evaluated\n")
        for split, items in unavailable.items():
            for i in items:
                L.append(f"- ({split}) {i}")
        L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--final-test", action="store_true", help="REQUIRED to score the test split (use once, at the end)")
    ap.add_argument("--candidates", nargs="*", default=None)
    args = ap.parse_args()
    if args.split == "test" and not args.final_test:
        print("Refusing to score the TEST split without --final-test (the test set must stay untouched until the "
              "final comparison; never use it to pick thresholds or models).", file=sys.stderr)
        return 2

    project = load_project_config()
    setup_logging(project.logging["level"])
    policy, rules = load_policy_config(), load_rule_set()
    results_dir = project.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    split = load_split(project, args.split)
    quality = quality_lookup(load_manifest(project))
    audit = pd.read_csv(project.output_dir / "manifests" / "dataset_audit.csv", low_memory=False)
    ctx = dict(project=project, policy=policy, rules=rules, split=split, quality=quality, split_name=args.split)

    names = args.candidates if args.candidates else discover(project, args.split)
    cand_metrics: dict[str, Any] = {}
    frames, errs, unavailable = [], [], []
    rel_series = {}
    for name in names:
        try:
            if name == "technical_quality":
                pred, info = cand_technical(ctx)
            elif name.startswith("vlm_rules:"):
                pred, info = cand_vlm(ctx, name.split(":", 1)[1])
            elif name.startswith("tuned_decision:"):
                pred, info = cand_tuned_decision(ctx, name.split(":", 1)[1])
            else:
                unavailable.append(f"{name}: unknown candidate")
                continue
        except (FileNotFoundError, NgoPhotoError) as exc:
            unavailable.append(f"{name}: {exc}")
            continue
        df = split[["image_id", "label", "program"]].merge(pred, on="image_id", how="left")
        df["candidate"], df["split"] = name, args.split
        metrics = evaluate_predictions(df)
        e = build_error_analysis(df, audit)
        e.insert(0, "candidate", name)
        cand_metrics[name] = {"metrics": metrics, "info": info, "error_summary": summarize_errors(e) if len(e) else {}}
        frames.append(df)
        errs.append(e)
        if "probability" in df and df["probability"].notna().any():
            m = df["probability"].notna()
            rel_series[name] = ((df.loc[m, "label"] == "approved").astype(int).to_numpy(), df.loc[m, "probability"].to_numpy(float))
        print(f"[{args.split}] {name}: auto={metrics['processing']['automatic_rate']:.1%} "
              f"FAR(strict)={metrics['strict'].get('false_approval_rate')} recall={metrics['strict'].get('approval_recall')}")

    if not cand_metrics:
        print("No candidate could be evaluated:\n  " + "\n  ".join(unavailable), file=sys.stderr)
        return 1

    mp = results_dir / "metrics.json"
    all_metrics = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    all_metrics[args.split] = {"n_images": int(len(split)), "approved_share": float((split.label == "approved").mean()),
                               "candidates": cand_metrics}
    mp.write_text(json.dumps(all_metrics, indent=2, default=str), encoding="utf-8")

    pp, ep = results_dir / "predictions.csv", results_dir / "error_analysis.csv"
    new_pred = pd.concat(frames, ignore_index=True)
    new_err = pd.concat(errs, ignore_index=True)
    if pp.exists():
        old = pd.read_csv(pp)
        new_pred = pd.concat([old[old.split != args.split], new_pred], ignore_index=True)
    new_pred.to_csv(pp, index=False)
    new_err["split"] = args.split
    if ep.exists():
        old = pd.read_csv(ep)
        new_err = pd.concat([old[old.split != args.split], new_err], ignore_index=True)
    new_err.to_csv(ep, index=False)
    if rel_series:
        plot_reliability(rel_series, project.artifacts_dir / "reports" / f"reliability_{args.split}.png")

    report = render_report(all_metrics, {args.split: unavailable} if unavailable else {}, policy.threshold_selection.max_false_approval_rate,
                           policy.calibration.min_validation_roc_auc)
    (project.resolve(project.paths.docs_dir) / "EVALUATION_REPORT.md").write_text(report, encoding="utf-8")
    print(f"wrote {mp.name}, {pp.name}, {ep.name} and docs/EVALUATION_REPORT.md")
    for u in unavailable:
        print("  not evaluated:", u)
    return 0


if __name__ == "__main__":
    np.random.seed(0)
    sys.exit(main())
