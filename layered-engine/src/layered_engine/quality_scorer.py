"""Layer 3: soft quality scoring. Python computes the score; the model only grades each dimension 0-4.

overall = sum over applicable dimensions of (score / 4) * weight * (100 / sum of applicable weights)

A high dimension never hides a severe weakness in another one, and one defect is never counted twice (a former hard gate that the
active profile treats as soft caps a single dimension once). Nothing here can override a failed mandatory gate: the decision engine
applies gates first.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .applicability import facts_dict
from .expressions import evaluate
from .rulebook import ResolvedProfile
from .schemas import CriterionResult, ModelAssessment, QualityResult, Status


def score_quality(assessment: ModelAssessment, results: Mapping[str, CriterionResult], profile: ResolvedProfile, metadata: Optional[Mapping[str, Any]] = None) -> QualityResult:
    q = profile.quality
    dims_cfg: dict[str, Any] = q["dimensions"]
    out = QualityResult(computed=True, calibrated=(q.get("status") != "UNCALIBRATED"))
    given = assessment.layer_3_quality_dimensions
    facts = facts_dict(assessment)
    model_answers = {a.criterion_id: a.status for a in assessment.layer_2_criteria}
    scores: dict[str, float | None] = {}
    contributions = 0.0
    applicable_weight = 0.0
    for name, cfg in dims_cfg.items():
        d = given.get(name)
        if d is None:
            out.incomplete = True
            out.warnings.append(f"dimension {name} is missing from the model response")
            scores[name] = None
            continue
        if not d.applicable:
            scores[name] = None
            continue
        s = float(d.score)
        # a soft cap uses the model's own answer plus the cap's applicability condition, independent of whether the criterion is a hard gate here
        caps = [c for c in q.get("soft_caps", []) if c["dimension"] == name and profile.name in c.get("profiles", [profile.name]) and model_answers.get(c["criterion_id"]) is Status.NO
                and evaluate({"ref": c.get("applicability", "always")}, facts, metadata or {}, profile.conditions) is not False]
        if caps:
            cap = min(c["max_score"] for c in caps)
            if s > cap:
                out.caps_applied.append(f"{name}: {s:g} -> {cap} ({', '.join(c['criterion_id'] for c in caps)})")
                s = float(cap)
        no_diag = [r for cid, r in results.items() if profile.criteria[cid].dimension == name and r.status is Status.NO
                   and profile.criteria[cid].decision_effect in ("DIAGNOSTIC", "SOFT_SCORE")]
        cw = q.get("consistency_warning", {})
        if cw and s >= cw["high_score"] and len(no_diag) >= cw["min_no_answers"]:
            out.warnings.append(f"{name} scored {s:g} although {len(no_diag)} of its diagnostics were answered NO ({', '.join(r.criterion_id for r in no_diag)})")
        scores[name] = s
        applicable_weight += float(cfg["weight"])
        contributions += (s / 4.0) * float(cfg["weight"])
    out.dimension_scores = scores
    out.applicable_weight = applicable_weight
    if applicable_weight > 0 and not out.incomplete:
        out.overall_score = round(contributions * (100.0 / applicable_weight), 2)
        out.band = band_for(out.overall_score, q["thresholds"])
    sev = q["severe_weakness"]
    for name, s in scores.items():
        if s is None:
            continue
        if s <= sev["unusable_score_at_or_below"]:
            out.severe_weakness = f"UNUSABLE:{name}"
            break
        if s <= sev["poor_score_at_or_below"] and out.severe_weakness is None:
            out.severe_weakness = f"POOR:{name}"
    if assessment.model_overall_score is not None and out.overall_score is not None and abs(assessment.model_overall_score - out.overall_score) > 5:
        out.warnings.append(f"the model reported an overall score of {assessment.model_overall_score:g}; the recomputed score is {out.overall_score:g} (the recomputed one is used)")
    return out


def band_for(score: float, thresholds: Mapping[str, float]) -> str:
    if score >= thresholds["strong_candidate"]:
        return "STRONG_CANDIDATE"
    if score >= thresholds["acceptable"]:
        return "ACCEPTABLE_CANDIDATE"
    if score >= thresholds["reselection"]:
        return "EDITORIAL_REVIEW"
    return "RESELECT"
