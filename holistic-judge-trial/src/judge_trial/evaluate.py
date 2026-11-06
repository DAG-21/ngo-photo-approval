"""Scoring a finished run against the human labels.

y = 1 means the human reviewers APPROVED the photo. The judge's score is "likelihood of approval", so higher = more
approvable. Definitions used throughout:
  false positive (false approval) = a disapproved photo the judge would approve  (the costly error)
  false negative                  = an approved photo the judge would not approve
The 5% false-approval operating point accepts only scores strictly above the (floor(5% x negatives)+1)-th highest
disapproved score, so ties never push the false-approval rate over the limit (the judge's scores are coarse integers).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .runner import read_results

MIN_NEGATIVES_WARN = 50


def threshold_at_far(neg_scores: np.ndarray, max_far: float = 0.05) -> float:
    """Accept score > threshold. At most floor(max_far * n_neg) disapproved photos are accepted."""
    srt = np.sort(np.asarray(neg_scores, dtype=float))[::-1]
    allowed = int(np.floor(max_far * len(srt)))
    return -np.inf if allowed >= len(srt) else float(srt[allowed])


def recall_at_far(y: np.ndarray, s: np.ndarray, max_far: float = 0.05) -> dict:
    thr = threshold_at_far(s[y == 0], max_far)
    accept = s > thr
    return {"threshold": thr, "approved_recall": float(accept[y == 1].mean()), "false_approval_rate": float(accept[y == 0].mean()),
            "false_positives": int(accept[y == 0].sum()), "false_negatives": int((~accept[y == 1]).sum())}


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def calibrated_oof(y: np.ndarray, score: np.ndarray, groups: np.ndarray, seed: int = 0) -> np.ndarray | None:
    """Out-of-fold probabilities from a logistic fit on the logit of the 0-100 score (grouped CV)."""
    if min((y == 0).sum(), (y == 1).sum()) < 5:
        return None
    x = np.log(np.clip(score, 1, 99) / (100 - np.clip(score, 1, 99)))[:, None]
    out = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(x, y, groups):
        out[te] = LogisticRegression(C=1.0).fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    return out


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def group_bootstrap(y: np.ndarray, s: np.ndarray, base: np.ndarray | None, groups: np.ndarray, n: int = 500, seed: int = 0) -> dict:
    """Resample whole groups (same submission / duplicates move together) to get honest intervals."""
    rng = np.random.RandomState(seed)
    ug = pd.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ug}
    aucs, rec, diff = [], [], []
    for _ in range(n):
        pick = np.concatenate([members[g] for g in rng.choice(ug, len(ug), replace=True)])
        yy, ss = y[pick], s[pick]
        if len(set(yy)) < 2:
            continue
        aucs.append(_auc(yy, ss))
        rec.append(recall_at_far(yy, ss)["approved_recall"])
        if base is not None:
            diff.append(aucs[-1] - _auc(yy, base[pick]))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None  # noqa: E731
    return {"auc_ci95": ci(aucs), "recall_at_5pct_far_ci95": ci(rec), "auc_minus_baseline_ci95": ci(diff)}


def load_run_frame(run_dir: Path, pool: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = read_results(run_dir)
    if not rows:
        raise FileNotFoundError(f"no results in {run_dir}")
    hashes = {r["prompt_hash"] for r in rows}
    if len(hashes) != 1:
        raise ValueError(f"results mix several prompts {sorted(hashes)}; evaluate one prompt at a time")
    latest: dict[str, dict] = {}
    for r in rows:  # later lines supersede earlier ones (a retried photo)
        latest[r["image_id"]] = r
    res = pd.DataFrame(latest.values())
    meta = {"prompt_hash": hashes.pop(), "variant": rows[0]["variant"], "model_id": rows[0]["model_id"],
            "calls_logged": len(rows), "failed": int((~res["ok"].astype(bool)).sum()), "photos_in_pool": len(pool)}
    df = pool.merge(res, on="image_id", how="left")
    meta["not_attempted"] = int(df["ok"].isna().sum())
    return df, meta


def baseline_scores(main_root: Path, ids: pd.Series) -> np.ndarray | None:
    path = main_root / "data" / "results" / "embedding_probe" / "clip_b32+dinov2_base_oof.csv"
    if not path.exists():
        return None
    oof = pd.read_csv(path).set_index("image_id")["oof_p_approved"]
    vals = ids.map(oof)
    return vals.to_numpy(dtype=float) if vals.notna().all() else None


def evaluate_frame(df: pd.DataFrame, meta: dict, baseline: np.ndarray | None = None, n_boot: int = 500) -> dict:
    ok = df[df["ok"].fillna(False).astype(bool)].copy()
    cj = ok["cannot_judge"].fillna(False).astype(bool)
    y = (ok["label"] == "approved").astype(int).to_numpy()
    s = ok["score"].astype(float).to_numpy()
    groups = ok["group_id"].to_numpy()
    base = None
    if baseline is not None:
        base = pd.Series(baseline, index=df.index).loc[ok.index].to_numpy()
    n_neg = int((y == 0).sum())
    res: dict = {**meta, "n_scored": int(len(ok)), "approved": int(y.sum()), "not_approved": n_neg,
                 "cannot_judge": int(cj.sum()), "warnings": []}
    if len(set(y)) < 2:
        res["warnings"].append("only one class was scored; no ranking metrics")
        return res
    if n_neg < MIN_NEGATIVES_WARN:
        res["warnings"].append(f"only {n_neg} disapproved photos scored: every rate here is imprecise (see the intervals)")
    if meta["failed"] or meta["not_attempted"]:
        res["warnings"].append(f"{meta['failed']} calls failed and {meta['not_attempted']} photos were not attempted; "
                               "they are excluded from every number")
    res["auc"] = _auc(y, s)
    res["pr_auc"] = float(average_precision_score(y, s))
    res["operating_point_5pct_far"] = recall_at_far(y, s, 0.05)
    dec = ok["decision"].eq("APPROVED").to_numpy()
    res["as_decided"] = {"false_positives": int((dec & (y == 0)).sum()), "false_negatives": int((~dec & (y == 1)).sum()),
                         "false_approval_rate": float((dec & (y == 0)).sum() / max(n_neg, 1)),
                         "false_negative_rate": float((~dec & (y == 1)).sum() / max(int(y.sum()), 1))}
    res["score_distinct_values"] = int(pd.Series(s).nunique())
    p = calibrated_oof(y, s, groups)
    if p is not None:
        res["calibration"] = {"ece": expected_calibration_error(y, p), "brier": float(np.mean((p - y) ** 2)),
                              "auc_after_calibration": _auc(y, p)}
    res["bootstrap"] = group_bootstrap(y, s, base, groups, n=n_boot)
    if base is not None:
        res["baseline_embeddings"] = {"auc": _auc(y, base), "operating_point_5pct_far": recall_at_far(y, base, 0.05),
                                      "note": "CLIP+DINOv2 out-of-fold scores from the main project, same photos"}
    return res


def false_approvals(df: pd.DataFrame, threshold: float, top: int = 40) -> pd.DataFrame:
    ok = df[df["ok"].fillna(False).astype(bool) & (df["label"] == "not_approved")]
    cols = ["image_id", "program", "score", "decision", "observations", "reasons"] + (["reason"] if "reason" in df.columns else [])
    return ok.sort_values("score", ascending=False)[cols].head(top).assign(accepted_at_5pct=lambda d: d["score"] > threshold)


def render_report(res: dict) -> str:
    f = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"  # noqa: E731
    op = res.get("operating_point_5pct_far")
    lines = [f"# Run report: variant {res['variant']}, prompt {res['prompt_hash']}, model {res['model_id']}", "",
             f"Photos scored: {res['n_scored']} ({res['approved']} approved, {res['not_approved']} disapproved). "
             f"Failed calls: {res['failed']}. Not attempted: {res['not_attempted']}. Marked cannot-judge: {res['cannot_judge']}.", ""]
    lines += [f"> WARNING: {w}" for w in res["warnings"]] + ([""] if res["warnings"] else [])
    if "auc" not in res:
        return "\n".join(lines)
    bs = res["bootstrap"]
    ci = lambda v: "n/a" if not v else f"{v[0]:.3f} to {v[1]:.3f}"  # noqa: E731
    lines += ["| Measure | Judge | Embeddings (CLIP+DINOv2) |", "|---|---:|---:|",
              f"| ROC-AUC | {f(res['auc'])} (95% range {ci(bs['auc_ci95'])}) | {f(res.get('baseline_embeddings', {}).get('auc'))} |"]
    b = res.get("baseline_embeddings", {}).get("operating_point_5pct_far")
    lines += [f"| Approved photos accepted at <=5% false approvals | {op['approved_recall']:.1%} (95% range {ci(bs['recall_at_5pct_far_ci95'])}) | {b['approved_recall']:.1%} |" if b else
              f"| Approved photos accepted at <=5% false approvals | {op['approved_recall']:.1%} | n/a |",
              f"| False positives at that point | {op['false_positives']} of {res['not_approved']} | {b['false_positives'] if b else 'n/a'} |",
              f"| False negatives at that point | {op['false_negatives']} of {res['approved']} | {b['false_negatives'] if b else 'n/a'} |", ""]
    if bs.get("auc_minus_baseline_ci95"):
        lines.append(f"AUC difference, judge minus embeddings: 95% range {ci(bs['auc_minus_baseline_ci95'])} "
                     "(a range that includes 0 means the two are not clearly different).")
    d = res["as_decided"]
    lines += ["", f"Using the judge's own APPROVED/DISAPPROVED decision (no threshold chosen): {d['false_positives']} false positives "
              f"({d['false_approval_rate']:.1%} of disapproved) and {d['false_negatives']} false negatives "
              f"({d['false_negative_rate']:.1%} of approved).",
              f"The score takes {res['score_distinct_values']} distinct values (few values = ties = a coarse ranking)."]
    if "calibration" in res:
        c = res["calibration"]
        lines.append(f"After calibration (grouped cross-validation): ECE {c['ece']:.3f}, Brier {c['brier']:.3f}. "
                     "ECE and Brier depend on the class mix of this pool, which is not the real mix when the pool is balanced.")
    return "\n".join(lines)


def save_report(run_dir: Path, res: dict, df: pd.DataFrame) -> None:
    (run_dir / "metrics.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(res), encoding="utf-8")
    if "operating_point_5pct_far" in res:
        false_approvals(df, res["operating_point_5pct_far"]["threshold"]).to_csv(run_dir / "false_approvals.csv", index=False)
