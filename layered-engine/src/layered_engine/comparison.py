"""Offline comparison of evaluator versions on stored model answers. Nothing here calls a model.

Re-deciding a stored answer under another profile is exact for every change that only alters what Python does with the answers (hard or soft,
applicability, which components a gate aggregates). It cannot show the effect of a change in what the model is ASKED (new wording): the
stored answers were given to the old wording. Those changes need a new run, and a new run differs from an old one for two reasons at once
(the changed wording and ordinary run-to-run variation), which this module cannot separate.

Positive class = approved. A photo counts as APPROVED by the engine only when the engine lets it through automatically (APPROVE or
APPROVED_PENDING_PUBLICATION_CHECKS); REJECT, HUMAN_REVIEW and RESELECT are all "not auto-approved". False positive = a photo the humans
disapproved that the engine auto-approved (the costly error). False negative = a photo the humans approved that was not auto-approved.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .annotation import read_rows
from .decision_engine import evaluate_final_decision
from .rulebook import Rulebook
from .schemas import Decision, ModelAssessment, Status

AUTO = {"APPROVE", "APPROVED_PENDING_PUBLICATION_CHECKS"}


# ----------------------------------------------------------------------------------------------------------------- loading and re-deciding
def load_answers(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Latest valid row per image of a run folder (image_id -> row)."""
    rows = [r for r in read_rows(Path(run_dir) / "results.jsonl") if r.get("ok")]
    return {r["image_id"]: r for r in rows}


def redecide(rows: Mapping[str, Mapping[str, Any]], rb: Rulebook, profile: str, without: Iterable[str] = (), overlay: Any = None) -> dict[str, Decision]:
    """The decision each stored answer would have had under `profile` (optionally with some policy changes switched off, and an overlay of stored changes)."""
    prof = rb.resolve(profile, without=without, overlay=overlay)
    out: dict[str, Decision] = {}
    for iid, row in rows.items():
        assessment = ModelAssessment.model_validate(row["validated"])
        out[iid] = evaluate_final_decision(assessment, prof, row.get("metadata_used") or {})
    return out


# ----------------------------------------------------------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def auc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    if not len(pos) or not len(neg):
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def auc_range(pos: Sequence[float], neg: Sequence[float], n: int = 2000, seed: int = 0) -> tuple[Optional[float], Optional[float]]:
    if not len(pos) or not len(neg):
        return None, None
    rng = np.random.default_rng(seed)
    p, q = np.array(pos), np.array(neg)
    v = [auc(rng.choice(p, len(p)), rng.choice(q, len(q))) for _ in range(n)]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n if n else None), "ci_low": lo, "ci_high": hi}


# ------------------------------------------------------------------------------------------------------------------------ headline metrics
def headline(decisions: Mapping[str, Decision], approved: Mapping[str, bool]) -> dict[str, Any]:
    """Confusion matrix and the rates asked for. `approved[image_id]` is the historical label (True = approved)."""
    ids = list(decisions)
    app = [i for i in ids if approved[i]]
    dis = [i for i in ids if not approved[i]]
    rec = {i: decisions[i].recommendation.value for i in ids}
    auto = {i: rec[i] in AUTO for i in ids}
    tp = sum(auto[i] for i in app)
    fn = len(app) - tp
    fp = sum(auto[i] for i in dis)
    tn = len(dis) - fp
    rejected_app = sum(rec[i] == "REJECT" for i in app)
    rejected_dis = sum(rec[i] == "REJECT" for i in dis)
    n_rejected = rejected_app + rejected_dis
    quality = {i: decisions[i].quality.overall_score for i in ids}
    nf = {i: len(decisions[i].failed_gates) for i in ids}
    a_ = auc([-nf[i] for i in app], [-nf[i] for i in dis])
    a_lo, a_hi = auc_range([-nf[i] for i in app], [-nf[i] for i in dis])
    qa = auc([quality[i] for i in app if quality[i] is not None], [quality[i] for i in dis if quality[i] is not None])
    by_class = {"approved": dict(Counter(rec[i] for i in app)), "disapproved": dict(Counter(rec[i] for i in dis))}
    return {
        "n": len(ids), "n_approved": len(app), "n_disapproved": len(dis),
        "confusion": {"TP": tp, "FN": fn, "TN": tn, "FP": fp},
        "decisions_by_class": by_class,
        "ai_rejections": n_rejected, "ai_auto_approvals": tp + fp, "ai_human_review": sum(rec[i] == "HUMAN_REVIEW" for i in ids),
        "ai_reselect": sum(rec[i] == "RESELECT" for i in ids),
        "false_rejection_rate": _rate(fn, len(app)),                      # approved photos not auto-approved (rejected, reviewed or reselect)
        "hard_false_rejection_rate": _rate(rejected_app, len(app)),       # approved photos explicitly REJECTed
        "false_approval_rate": _rate(fp, len(dis)),                       # disapproved photos auto-approved
        "approval_precision": _rate(tp, tp + fp),
        "rejection_precision": _rate(rejected_dis, n_rejected),           # of the explicit REJECTs, how many were disapproved
        "rejection_recall": _rate(rejected_dis, len(dis)),                # of the disapproved photos, how many were explicitly REJECTed
        "agreement": _rate(tp + tn, len(ids)),
        "agreement_definitive_only": _rate(sum(1 for i in ids if rec[i] != "HUMAN_REVIEW" and (auto[i] == approved[i])),
                                           sum(1 for i in ids if rec[i] != "HUMAN_REVIEW")),
        "mean_failed_gates": {"approved": float(np.mean([nf[i] for i in app])) if app else None, "disapproved": float(np.mean([nf[i] for i in dis])) if dis else None},
        "auc_failed_gates": {"auc": a_, "ci_low": a_lo, "ci_high": a_hi},
        "auc_quality_score": qa,
        "mean_quality": {"approved": _mean([quality[i] for i in app]), "disapproved": _mean([quality[i] for i in dis]),
                         "false_rejections": _mean([quality[i] for i in app if not auto[i]]),
                         "false_approvals": _mean([quality[i] for i in dis if auto[i]])},
        "false_approval_ids": [i for i in dis if auto[i]],
        "false_rejection_ids": [i for i in app if not auto[i]],
    }


def _mean(v: Sequence[Optional[float]]) -> Optional[float]:
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


# ----------------------------------------------------------------------------------------------------------------------------- gate level
def gate_table(decisions: Mapping[str, Decision], approved: Mapping[str, bool]) -> list[dict[str, Any]]:
    """One row per gate. Denominators are the photos on which the gate APPLIES (status other than NOT_APPLICABLE), per historical class.

    `hard` says whether the gate can reject in this profile. For a soft gate the NO counts are still shown (what the model said), but they
    never reject. `sole_cause_*`: photos rejected for this gate alone, i.e. the photos that would stop being rejected if only this gate
    were removed.
    """
    ids = list(decisions)
    gate_ids = sorted({g.gate_id for d in decisions.values() for g in d.gates})
    rows: list[dict[str, Any]] = []
    for gid in gate_ids:
        st = {i: next(g for g in decisions[i].gates if g.gate_id == gid) for i in ids if any(g.gate_id == gid for g in decisions[i].gates)}
        hard = any(g.hard for g in st.values())
        row: dict[str, Any] = {"gate": gid, "hard": hard}
        for cls, flag in (("approved", True), ("disapproved", False)):
            grp = [i for i in ids if approved[i] == flag and i in st]
            c = Counter(st[i].status.value for i in grp)
            applicable = [i for i in grp if st[i].status is not Status.NOT_APPLICABLE]
            no = [i for i in grp if st[i].status is Status.NO]
            row[cls] = {"n": len(grp), "applicable": len(applicable), "yes": c["YES"], "no": c["NO"], "cannot_tell": c["CANNOT_TELL"], "not_applicable": c["NOT_APPLICABLE"],
                        "fail_rate_of_applicable": (len(no) / len(applicable) if applicable else None)}
        a, d = row["approved"], row["disapproved"]
        row["difference_in_fail_rate"] = (a["fail_rate_of_applicable"] - d["fail_rate_of_applicable"]) if None not in (a["fail_rate_of_applicable"], d["fail_rate_of_applicable"]) else None
        rejected = lambda i: decisions[i].recommendation.value == "REJECT"  # noqa: E731
        fails = lambda i: gid in decisions[i].failed_gates  # noqa: E731
        row["false_rejections_with_gate"] = sum(1 for i in ids if approved[i] and rejected(i) and fails(i))
        row["correct_rejections_with_gate"] = sum(1 for i in ids if (not approved[i]) and rejected(i) and fails(i))
        row["sole_cause_false_rejections"] = sum(1 for i in ids if approved[i] and rejected(i) and decisions[i].failed_gates == [gid])
        row["sole_cause_correct_rejections"] = sum(1 for i in ids if (not approved[i]) and rejected(i) and decisions[i].failed_gates == [gid])
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------------------------------- transitions
def transitions(before: Mapping[str, Decision], after: Mapping[str, Decision], approved: Mapping[str, bool]) -> dict[str, list[str]]:
    """Per photo: how the engine's verdict changed between two decision sets."""
    out: dict[str, list[str]] = {"rescued_approved": [], "new_false_approvals": [], "newly_rejected_approved": [], "newly_stopped_disapproved": [],
                                 "approved_still_not_auto": [], "disapproved_still_stopped": [], "approved_still_auto": [], "disapproved_still_auto": []}
    for i in before:
        b, a = before[i].recommendation.value in AUTO, after[i].recommendation.value in AUTO
        if approved[i]:
            key = {(False, True): "rescued_approved", (True, False): "newly_rejected_approved", (False, False): "approved_still_not_auto", (True, True): "approved_still_auto"}[(b, a)]
        else:
            key = {(False, True): "new_false_approvals", (True, False): "newly_stopped_disapproved", (False, False): "disapproved_still_stopped", (True, True): "disapproved_still_auto"}[(b, a)]
        out[key].append(i)
    return out


# ----------------------------------------------------------------------------------------------------------------------- what-if: remove gates
def without_gates(decisions: Mapping[str, Decision], removed: set[str], profile: Any) -> dict[str, str]:
    """An UPPER BOUND, not a proposal: what each photo's verdict would be if the gates in `removed` were deleted outright.

    It only follows the decision engine's own order (remaining failed gate -> REJECT; remaining unknown gate -> HUMAN_REVIEW; then quality;
    then publication) using what the stored decision already holds, so it answers "how many wrongly rejected photos is this gate responsible
    for, and how many correctly rejected ones would slip through". `profile` is the ResolvedProfile the decisions were made under.
    """
    out: dict[str, str] = {}
    th = profile.quality["thresholds"]
    q_decides = bool(profile.settings.get("quality", {}).get("affects_decision"))
    p_decides = bool(profile.settings.get("publication", {}).get("affects_decision"))
    for iid, d in decisions.items():
        failed = [g for g in d.failed_gates if g not in removed]
        unsure = [g for g in d.uncertain_gates if g not in removed]
        if failed:
            out[iid] = "REJECT"
            continue
        if unsure:
            out[iid] = "HUMAN_REVIEW"
            continue
        q = d.quality
        if q_decides:
            sev = q.severe_weakness
            if q.incomplete or q.overall_score is None:
                out[iid] = "HUMAN_REVIEW"
                continue
            if (sev and sev.startswith("UNUSABLE")) or q.overall_score < th["reselection"]:
                out[iid] = "RESELECT"
                continue
            if q.overall_score < th["review"] or (sev and sev.startswith("POOR")):
                out[iid] = "HUMAN_REVIEW"
                continue
        if p_decides and d.publication.status == "BLOCKED":
            out[iid] = "HUMAN_REVIEW"
        elif p_decides and d.publication.status == "PENDING_VERIFICATION":
            out[iid] = "APPROVED_PENDING_PUBLICATION_CHECKS"
        else:
            out[iid] = "APPROVE"
    return out


def confusion_from(verdicts: Mapping[str, str], approved: Mapping[str, bool]) -> dict[str, int]:
    tp = sum(1 for i, v in verdicts.items() if approved[i] and v in AUTO)
    fp = sum(1 for i, v in verdicts.items() if (not approved[i]) and v in AUTO)
    n_app = sum(1 for i in verdicts if approved[i])
    return {"TP": tp, "FN": n_app - tp, "FP": fp, "TN": (len(verdicts) - n_app) - fp}
