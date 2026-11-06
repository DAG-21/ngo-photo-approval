"""The authoritative decision. Python owns it; the model's own recommendation and aggregates are recorded but never used.

(With the profile setting `no_human_review` the HUMAN_REVIEW outcomes below become REJECT, or RESELECT for borderline quality: see the note in the code.)
Order (first match wins, every reason found is still reported):
  0. input validation (only if the active upload policy asks for it)
  1. any applicable HARD gate with NO            -> REJECT
  2. any applicable HARD gate with CANNOT_TELL   -> HUMAN_REVIEW       (unknown is never a pass)
  3. quality below the thresholds / severe weak. -> RESELECT / HUMAN_REVIEW   (only when the profile lets quality decide)
  4. publication BLOCKED                          -> HUMAN_REVIEW
  5. publication PENDING                          -> APPROVED_PENDING_PUBLICATION_CHECKS
  6. otherwise                                    -> APPROVE
NOT_APPLICABLE is skipped. A soft score can never override a failed gate because gates are checked first.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .applicability import facts_dict, validate_activity
from .gate_evaluator import evaluate_criteria, evaluate_gates
from .publication import evaluate_publication
from .quality_scorer import score_quality
from .rulebook import ResolvedProfile
from .schemas import Decision, GateResult, Metadata, ModelAssessment, QualityResult, Recommendation, Status, Tri

PROMPT_VERSION_KEY = "prompt_version"


def _check_metadata(metadata: Mapping[str, Any]) -> None:
    bad = [k for k in metadata if k.lower() in Metadata.FORBIDDEN_KEYS]
    if bad:
        raise ValueError(f"metadata must not carry historical labels: {bad}")


def evaluate_final_decision(assessment: ModelAssessment, profile: ResolvedProfile, metadata: Optional[Mapping[str, Any]] = None,
                            versions: Optional[Mapping[str, Any]] = None) -> Decision:
    metadata = dict(metadata or {})
    _check_metadata(metadata)
    warnings: list[str] = []
    reasons: list[str] = []
    questions: list[str] = []
    facts = facts_dict(assessment)

    bad_activity = validate_activity(assessment, profile)
    if bad_activity:
        warnings.append(bad_activity)

    results = evaluate_criteria(assessment, profile, metadata)
    gates = evaluate_gates(results, profile)
    hard = [g for g in gates if g.hard]
    soft = [g for g in gates if not g.hard]
    applicable_hard = [g for g in hard if g.status is not Status.NOT_APPLICABLE]

    failed = [g for g in hard if g.status is Status.NO]
    uncertain = [g for g in hard if g.status is Status.CANNOT_TELL]
    not_app = [g for g in hard if g.status is Status.NOT_APPLICABLE]
    # unknown_is_fail=false is supported for experiments only: unknown then stops being a review trigger. The default (and both shipped profiles) is true.
    if not profile.unknown_is_fail:
        uncertain = []
    for g in uncertain:
        for n in g.notes:
            if n.startswith("synthesized") or n.startswith("na_conflict"):
                warnings.append(f"{g.gate_id} {g.criterion_id}: {n}")

    quality = score_quality(assessment, results, profile, metadata)
    publication = evaluate_publication(assessment, metadata, profile)

    # ---- Step 0: input validation ---------------------------------------------------------------------------------------
    up = profile.settings.get("upload_policy", {})
    input_problem: Optional[str] = None
    input_unknown = False
    for flag, fact, text in (("require_photo", "f_photo", "not an actual photograph of an activity"), ("require_single", "f_single", "not a single photograph (collage)")):
        if up.get(flag):
            if facts[fact] == Tri.NO.value:
                input_problem = text
            elif facts[fact] == Tri.UNKNOWN.value:
                input_unknown = True

    # `no_human_review` (a profile setting, off everywhere before layered_final_v1): the AI always gives a verdict. Where the engine used to say HUMAN_REVIEW it now says REJECT
    # (a gate or fact it could not confirm, an incomplete quality assessment, a publication blocker) or RESELECT (quality below the cutoff). A person still decides every photograph.
    no_human_review = bool(profile.settings.get("no_human_review"))
    rec: Recommendation
    if input_problem:
        rec = Recommendation.REJECT
        reasons.append(f"INPUT_VALIDATION: {input_problem}")
    elif failed:
        rec = Recommendation.REJECT
    elif uncertain or input_unknown:
        rec = Recommendation.REJECT if no_human_review else Recommendation.HUMAN_REVIEW
    else:
        rec = Recommendation.APPROVE  # provisional; the next steps may change it

    for g in failed:
        reasons.append(f"GATE {g.gate_id} ({g.criterion_id}) failed: {g.failure_reason}" + (f" - {g.evidence}" if g.evidence else ""))
    for g in uncertain:
        reasons.append(f"GATE {g.gate_id} ({g.criterion_id}) cannot be established" + (f" - {g.evidence}" if g.evidence else "")
                       + (" - could not confirm, so the photo is not approved" if no_human_review else ""))
        questions.append(f"Is this true for the photo? {g.failure_reason}")
    if input_unknown:
        reasons.append("INPUT_VALIDATION: the upload policy requires a fact the model could not establish")

    # ---- Step 3: quality (only when the profile lets it decide, and only if nothing stronger already decided) -------------
    q_decides = bool(profile.settings.get("quality", {}).get("affects_decision"))
    if rec is Recommendation.APPROVE and q_decides:
        sev = quality.severe_weakness
        th = profile.quality["thresholds"]
        if quality.incomplete or quality.overall_score is None:
            rec = Recommendation.REJECT if no_human_review else Recommendation.HUMAN_REVIEW
            reasons.append("QUALITY: the quality assessment is incomplete" + (" - could not confirm, so the photo is not approved" if no_human_review else ""))
            questions.append("The quality assessment is incomplete. Please review the photograph.")
        elif sev and sev.startswith("UNUSABLE"):
            rec = Recommendation.RESELECT
            reasons.append(f"QUALITY: {sev} (a dimension is unusable)")
        elif quality.overall_score < th["reselection"]:
            rec = Recommendation.RESELECT
            reasons.append(f"QUALITY: score {quality.overall_score:g} is below the reselection threshold {th['reselection']}")
        elif quality.overall_score < th["review"] or (sev and sev.startswith("POOR")):
            if no_human_review:   # the AI must give a verdict: one cutoff at the review threshold, and a poor dimension cannot be approved either
                rec = Recommendation.RESELECT
                reasons.append(f"QUALITY: score {quality.overall_score:g}" + (f", {sev}" if sev else "") + f" is below the cutoff of {th['review']} (or a dimension is poor): re-select")
            else:
                rec = Recommendation.HUMAN_REVIEW
                reasons.append(f"QUALITY: score {quality.overall_score:g}" + (f", {sev}" if sev else "") + " needs an editorial look")
            questions.append("The photograph passes the mandatory gates but its quality is borderline. Is it good enough to publish?")
    elif rec is Recommendation.APPROVE and quality.overall_score is not None:
        reasons.append(f"quality score {quality.overall_score:g} reported only (the {profile.name} profile does not let quality decide)")

    # ---- Steps 4-6: publication readiness --------------------------------------------------------------------------------
    p_decides = bool(profile.settings.get("publication", {}).get("affects_decision"))
    if rec is Recommendation.APPROVE and p_decides:
        if publication.status == "BLOCKED":
            rec = Recommendation.REJECT if no_human_review else Recommendation.HUMAN_REVIEW
            reasons += [f"PUBLICATION BLOCKER: {b}" for b in publication.blockers]
            questions.append("A publication blocker was found. Please review it before publishing.")
        elif publication.status == "PENDING_VERIFICATION":
            rec = Recommendation.APPROVED_PENDING_PUBLICATION_CHECKS
            reasons.append("All mandatory gates pass; publication checks are still outstanding: " + "; ".join(publication.pending))
    elif rec in (Recommendation.REJECT, Recommendation.HUMAN_REVIEW, Recommendation.RESELECT) and p_decides and publication.blockers:
        reasons += [f"also: PUBLICATION BLOCKER: {b}" for b in publication.blockers]
    if rec is Recommendation.APPROVE and not reasons:
        reasons.append("every applicable mandatory gate passed")

    photo_status = {Recommendation.APPROVE: "APPROVED", Recommendation.APPROVED_PENDING_PUBLICATION_CHECKS: "APPROVED",
                    Recommendation.REJECT: "REJECTED", Recommendation.HUMAN_REVIEW: "REVIEW_REQUIRED", Recommendation.RESELECT: "RESELECT"}[rec]
    pub_status = publication.status if p_decides else "NOT_EVALUATED"
    if rec in (Recommendation.REJECT, Recommendation.RESELECT):
        pub_status = "NOT_APPLICABLE_REJECTED_PHOTO"
    warnings += quality.warnings
    changes = [c for c in profile.policy_changes if c.affects_mandatory]
    return Decision(
        recommendation=rec, photo_approval_status=photo_status, publication_readiness_status=pub_status, decision_reasons=reasons,
        human_review_questions=questions, gates=gates, failed_gates=[g.gate_id for g in failed], uncertain_gates=[g.gate_id for g in uncertain],
        not_applicable_gates=[g.gate_id for g in not_app], quality=quality, publication=publication, warnings=warnings,
        policy_validation_required=bool(changes) and any(c.status != "TEAM_APPROVED" for c in changes),
        versions={"profile": profile.name, "rulebook_version": profile.rulebook_version, **({"overlay": profile.overlay_ids} if profile.overlay_ids else {}), **(versions or {})},
    )
