"""Human review: open a review, record corrections, submit the final decision.

The person's submitted decision is final: there is no second validation step.

Rules this module enforces (each has a test):
  * the AI's value is read from the stored evaluation, never from the client;
  * nothing the AI said is ever changed: corrections are new rows, new revisions supersede older ones;
  * every decision needs a written reason, whether the person agrees with the AI or not (that written reason is the record);
  * approving a photograph whose evaluation lists publication blockers needs a written statement;
  * a submitted review becomes a learning example, inactive for retrieval, except when its reason is a policy change or a self-declared reviewer
    error, or when it approved over a publication blocker (a policy change goes through the rulebook approval process);
  * every query is scoped to the caller's organisation; another organisation's record is NotFound.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .enums import (AI_APPROVING, AI_DEFERRING, NOT_FOR_LEARNING, ROLE_RANK, Agreement, CorrectionReason, CriterionType, FeedbackEventType, HumanDecision,
                    PhotoStatus, ProcessingStatus, ReviewStatus, Role)
from .errors import Conflict, Forbidden, Invalid, NotFound
from .evaluation_service import canonical, log_lifecycle
from .evaluation_view import EVENT_FOR_TYPE, CorrectionError, addressable_items, effective_value, publication_blockers, resolve_ai_value, validate_corrected_value
from . import example_service
from .models import CriterionCorrection, EvaluationRun, FeedbackEvent, HumanReview, LearningExample, Organisation, Photo, User, now
from .reason_quality import is_low_information

_REASONS = {r.value for r in CorrectionReason}
_BEFORE_DECISION = {PhotoStatus.UPLOADED.value, PhotoStatus.PROCESSING.value, PhotoStatus.AI_EVALUATED.value, PhotoStatus.PENDING_REVIEW.value,
                    PhotoStatus.HUMAN_REVIEW_REQUIRED.value, PhotoStatus.REVIEW_IN_PROGRESS.value, PhotoStatus.APPROVED.value, PhotoStatus.REJECTED.value}


def need_role(user: User, minimum: Role) -> None:
    if ROLE_RANK[user.role] < ROLE_RANK[minimum.value]:
        raise Forbidden(f"this action needs the {minimum.value} role")


def record_event(s: Session, org_id: str, event_type: str, actor: Optional[User], review: Optional[HumanReview] = None, evaluation_id: Optional[str] = None,
                 photo_id: Optional[str] = None, payload: Optional[dict] = None) -> None:
    s.add(FeedbackEvent(organisation_id=org_id, review_id=review.review_id if review else None, evaluation_id=evaluation_id or (review.evaluation_id if review else None),
                        photo_id=photo_id or (review.photo_id if review else None), event_type=event_type, actor_id=actor.user_id if actor else None, payload=payload or {}))


def _reasons(raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not raw or not isinstance(raw, (list, tuple)):
        raise Invalid("at least one correction reason is required")
    bad = [r for r in raw if r not in _REASONS]
    if bad:
        raise Invalid(f"unknown correction reason(s) {bad}; allowed: {sorted(_REASONS)}")
    return list(dict.fromkeys(raw))


# ------------------------------------------------------------------------------------------------------------------------ scoped getters
def get_evaluation(s: Session, user: User, evaluation_id: str) -> EvaluationRun:
    run = s.scalar(select(EvaluationRun).where(EvaluationRun.evaluation_id == evaluation_id, EvaluationRun.organisation_id == user.organisation_id))
    if run is None:
        raise NotFound("evaluation not found")
    return run


def get_photo(s: Session, user: User, photo_id: str) -> Photo:
    p = s.scalar(select(Photo).where(Photo.photo_id == photo_id, Photo.organisation_id == user.organisation_id))
    if p is None:
        raise NotFound("photo not found")
    return p


def get_review(s: Session, user: User, review_id: str, for_update: bool = False) -> HumanReview:
    q = select(HumanReview).where(HumanReview.review_id == review_id, HumanReview.organisation_id == user.organisation_id)
    r = s.scalar(q.with_for_update() if for_update else q)
    if r is None:
        raise NotFound("review not found")
    return r


def review_corrections(s: Session, review: HumanReview) -> list[CriterionCorrection]:
    return list(s.scalars(select(CriterionCorrection).where(CriterionCorrection.review_id == review.review_id).order_by(CriterionCorrection.created_at, CriterionCorrection.revision)))


def latest_corrections(corrections: list[CriterionCorrection]) -> list[CriterionCorrection]:
    best: dict[tuple[str, str], CriterionCorrection] = {}
    for c in corrections:
        k = (c.criterion_type, c.criterion_id)
        if k not in best or c.revision > best[k].revision:
            best[k] = c
    return list(best.values())


# ---------------------------------------------------------------------------------------------------------------------------- review flow
def open_review(s: Session, user: User, evaluation_id: str) -> tuple[HumanReview, bool]:
    """Open my review of an evaluation, or resume the one I have already opened. Returns (review, created)."""
    need_role(user, Role.REVIEWER)
    run = get_evaluation(s, user, evaluation_id)
    if run.processing_status != ProcessingStatus.AI_EVALUATED.value:
        raise Conflict(f"this evaluation is {run.processing_status}; only a completed evaluation can be reviewed")
    existing = s.scalar(select(HumanReview).where(HumanReview.evaluation_id == run.evaluation_id, HumanReview.reviewer_id == user.user_id,
                                                  HumanReview.review_status == ReviewStatus.REVIEW_IN_PROGRESS.value))
    if existing:
        return existing, False
    review = HumanReview(photo_id=run.photo_id, evaluation_id=run.evaluation_id, organisation_id=run.organisation_id, reviewer_id=user.user_id)
    s.add(review)
    s.flush()
    photo = s.get(Photo, run.photo_id)
    log_lifecycle(s, photo, "HUMAN_REVIEW_IN_PROGRESS", PhotoStatus.REVIEW_IN_PROGRESS.value, run.evaluation_id, {"review_id": review.review_id, "reviewer_id": user.user_id},
                  change_status=photo.photo_status in (PhotoStatus.PENDING_REVIEW.value, PhotoStatus.HUMAN_REVIEW_REQUIRED.value, PhotoStatus.AI_EVALUATED.value))
    record_event(s, run.organisation_id, FeedbackEventType.REVIEW_OPENED.value, user, review)
    return review, True


def _my_open_review(s: Session, user: User, evaluation_id: str) -> HumanReview:
    run = get_evaluation(s, user, evaluation_id)
    r = s.scalar(select(HumanReview).where(HumanReview.evaluation_id == run.evaluation_id, HumanReview.reviewer_id == user.user_id,
                                           HumanReview.review_status == ReviewStatus.REVIEW_IN_PROGRESS.value))
    if r is None:
        raise Conflict("you have no open review of this evaluation: open one first")
    return r


def add_corrections(s: Session, user: User, evaluation_id: str, items: list[dict[str, Any]], activity_ids: Optional[set[str]] = None) -> list[CriterionCorrection]:
    """Record one or more corrections on my open review. Each item: criterion_type, criterion_id, human_corrected_value, correction_reason, reviewer_comment.

    Correcting the same item again creates a new revision that supersedes the previous one; both remain. All items are recorded or none are.
    """
    need_role(user, Role.REVIEWER)
    review = _my_open_review(s, user, evaluation_id)
    run = get_evaluation(s, user, evaluation_id)
    if not items:
        raise Invalid("no corrections were supplied")
    existing = review_corrections(s, review)
    out: list[CriterionCorrection] = []
    for it in items:
        ctype, cid = it.get("criterion_type"), it.get("criterion_id")
        if ctype not in {t.value for t in CriterionType}:
            raise Invalid(f"unknown criterion_type {ctype!r}")
        if ctype == CriterionType.FINAL_DECISION.value:
            raise Invalid("the final decision is recorded when the review is submitted, not as a separate correction")
        if not cid or not isinstance(cid, str) or len(cid) > 120:
            raise Invalid("criterion_id is required (at most 120 characters)")
        try:
            ai_value = resolve_ai_value(run, ctype, cid)  # read from the stored evaluation, never from the client
            value = validate_corrected_value(ctype, it.get("human_corrected_value"), activity_ids)
        except CorrectionError as exc:
            raise Invalid(str(exc)) from exc
        if value == ai_value and ctype not in (CriterionType.TAG_ADDED.value, CriterionType.TAG_REMOVED.value):
            raise Invalid(f"{cid}: the corrected value is the same as the AI's ({ai_value!r}); nothing to correct")
        reasons = _reasons(it.get("correction_reason"))
        prior = [c for c in existing + out if c.criterion_type == ctype and c.criterion_id == cid]
        last = max(prior, key=lambda c: c.revision) if prior else None
        c = CriterionCorrection(review_id=review.review_id, evaluation_id=run.evaluation_id, organisation_id=run.organisation_id, criterion_id=cid, criterion_type=ctype,
                                ai_predicted_value=ai_value, human_corrected_value=value, correction_reason=reasons, reviewer_comment=(it.get("reviewer_comment") or None),
                                revision=(last.revision + 1) if last else 1, supersedes_correction_id=last.correction_id if last else None, created_by=user.user_id)
        s.add(c)
        s.flush()
        out.append(c)
        record_event(s, run.organisation_id, EVENT_FOR_TYPE.get(ctype, FeedbackEventType.CRITERION_CORRECTED.value), user, review,
                     payload={"correction_id": c.correction_id, "criterion_type": ctype, "criterion_id": cid, "ai_value": ai_value, "human_value": value, "reasons": reasons,
                              "revision": c.revision})
    return out


def _agreement(ai_recommendation: Optional[str], decision: str) -> str:
    if ai_recommendation in AI_DEFERRING or ai_recommendation is None:
        return Agreement.AI_DEFERRED.value
    ai_says = HumanDecision.APPROVE.value if ai_recommendation in AI_APPROVING else HumanDecision.REJECT.value
    return Agreement.AGREE.value if ai_says == decision else Agreement.DISAGREE.value


MIN_BLOCKER_STATEMENT = 15
MAX_BLOCKER_STATEMENT = 2000
MIN_REASON = 10
MAX_REASON = 4000


def submit_review(s: Session, user: User, review_id: str, decision: str, reason_categories: Optional[list[str]] = None, comment: Optional[str] = None,
                  idempotency_key: Optional[str] = None, blocker_statement: Optional[str] = None) -> tuple[HumanReview, bool]:
    """Submit my review with the final decision. Returns (review, replayed). Safe to retry: the same submission again returns the stored result;
    a different submission of an already submitted review is a Conflict.

    The submission is final, and it always needs `comment`, a written reason (`reason_categories` is an optional extra): when the person disagrees it
    says what the AI got wrong; when the person agrees it still says why, because the AI may not have captured every point behind the approval or
    rejection. The agent that improves the rules and prompts learns from these reasons.

    Approving a photograph whose evaluation lists publication blockers needs `blocker_statement`: a written reason why approving is acceptable.
    It applies whatever the AI recommended (accepting an approving AI is still a human approval), is recorded as a BLOCKER_ACKNOWLEDGED event
    with the blockers as they were at the time, and a rejection never needs it."""
    need_role(user, Role.REVIEWER)
    if decision not in {d.value for d in HumanDecision}:
        raise Invalid("decision must be APPROVE or REJECT")
    review = get_review(s, user, review_id, for_update=True)
    if review.reviewer_id != user.user_id:
        raise Forbidden("only the reviewer who opened a review can submit it")
    statement = (blocker_statement or "").strip()
    sub_hash = hashlib.sha256(canonical({"decision": decision, "reasons": sorted(reason_categories or []), "comment": comment or "", "blocker_statement": statement}).encode()).hexdigest()
    if review.review_status == ReviewStatus.FEEDBACK_SUBMITTED.value:
        if review.submission_hash == sub_hash and (idempotency_key is None or review.idempotency_key in (None, idempotency_key)):
            return review, True
        raise Conflict("this review has already been submitted with a different decision")
    run = s.get(EvaluationRun, review.evaluation_id)
    blockers = publication_blockers(run) if decision == HumanDecision.APPROVE.value else []
    if blockers:
        if len(statement) < MIN_BLOCKER_STATEMENT:
            raise Invalid(f"this photograph has {len(blockers)} publication blocker(s): {'; '.join(blockers)}. Approving it needs a written statement "
                          f"(at least {MIN_BLOCKER_STATEMENT} characters) of why that is acceptable, for example what will be cropped, covered or verified before publishing")
        if len(statement) > MAX_BLOCKER_STATEMENT:
            raise Invalid(f"the blocker statement is too long (at most {MAX_BLOCKER_STATEMENT} characters)")
    agreement = _agreement(run.ai_recommendation, decision)
    reason_text = (comment or "").strip()
    if len(reason_text) < MIN_REASON:  # every decision says why, in words, whether or not the person agrees with the AI
        ai = run.ai_recommendation.replace("_", " ").lower()
        if agreement == Agreement.DISAGREE.value:
            raise Invalid(f"you are disregarding the AI's verdict ({ai}): write what it got wrong in the reason box (at least {MIN_REASON} characters)")
        if agreement == Agreement.AGREE.value:
            raise Invalid(f"you agree with the AI's verdict ({ai}): still write your reasons in the reason box (at least {MIN_REASON} characters), the AI may not have captured every point")
        raise Invalid(f"write the reasons for your decision in the reason box (at least {MIN_REASON} characters)")
    if len(reason_text) > MAX_REASON:
        raise Invalid(f"the reason is too long (at most {MAX_REASON} characters)")
    reasons = _reasons(reason_categories) if reason_categories else []
    review.human_final_decision, review.overall_agreement_with_ai = decision, agreement
    review.reason_categories, review.reviewer_comments = reasons, (reason_text or None)
    review.idempotency_key, review.submission_hash = idempotency_key, sub_hash
    review.review_status, review.reviewed_at = ReviewStatus.FEEDBACK_SUBMITTED.value, now()
    s.flush()
    if agreement == Agreement.DISAGREE.value:  # the override itself is a correction of the final decision, so downstream queries see it uniformly
        s.add(CriterionCorrection(review_id=review.review_id, evaluation_id=run.evaluation_id, organisation_id=run.organisation_id, criterion_id="final_decision",
                                  criterion_type=CriterionType.FINAL_DECISION.value, ai_predicted_value=run.ai_recommendation, human_corrected_value=decision, correction_reason=reasons,
                                  reviewer_comment=reason_text, revision=1, created_by=user.user_id))
    photo = s.get(Photo, review.photo_id)
    final_status = PhotoStatus.APPROVED.value if decision == HumanDecision.APPROVE.value else PhotoStatus.REJECTED.value
    log_lifecycle(s, photo, "HUMAN_OVERRIDES_AI" if agreement == Agreement.DISAGREE.value else "HUMAN_ACCEPTS_AI", None, run.evaluation_id,
                  {"review_id": review.review_id, "decision": decision, "agreement": agreement})
    log_lifecycle(s, photo, "FEEDBACK_RECORDED", final_status, run.evaluation_id, {"review_id": review.review_id})
    record_event(s, review.organisation_id, FeedbackEventType.AI_OVERRIDDEN.value if agreement == Agreement.DISAGREE.value else FeedbackEventType.AI_ACCEPTED.value, user, review,
                 payload={"ai_recommendation": run.ai_recommendation, "human_final_decision": decision, "agreement": agreement, "reasons": reasons, "reason": reason_text or None})
    if blockers:
        record_event(s, review.organisation_id, FeedbackEventType.BLOCKER_ACKNOWLEDGED.value, user, review,
                     payload={"blockers": blockers, "statement": statement, "ai_recommendation": run.ai_recommendation, "agreement": agreement})
    example, excluded = _record_learning_example(s, user, review, run, bool(blockers))
    record_event(s, review.organisation_id, FeedbackEventType.FEEDBACK_SUBMITTED.value, user, review,
                 payload={"corrections": len(review_corrections(s, review)), "learning_example_id": example.learning_example_id if example else None, "not_a_learning_example": excluded})
    return review, False


# ------------------------------------------------------------------------------------------------------------------------- learning example
def _verified_view(run: EvaluationRun, review: HumanReview, latest: list[CriterionCorrection]) -> dict[str, Any]:
    """The evaluation as the person leaves it: the AI's values with the person's latest corrections applied."""
    own = [c for c in latest if c.criterion_type != CriterionType.FINAL_DECISION.value]   # the final decision has its own field and reason below
    gates = {g["gate_id"]: g["status"] for g in (run.decision_output or {}).get("gates") or []}
    for c in own:
        if c.criterion_type == CriterionType.MANDATORY_GATE.value:
            gates[c.criterion_id] = c.human_corrected_value
    v = run.validated_response or {}
    tags = {o["text"]: "POSITIVE" for o in v.get("positive_attributes") or []} | {o["text"]: "CONCERN" for o in v.get("potential_rejection_factors") or []}
    for c in own:
        if c.criterion_type == CriterionType.TAG_REMOVED.value:
            tags.pop(c.criterion_id, None)
        elif c.criterion_type == CriterionType.TAG_ADDED.value:
            tags[c.criterion_id] = "ADDED_BY_REVIEWER"
    reasons = [{"criterion_type": c.criterion_type, "criterion_id": c.criterion_id, "reasons": c.correction_reason, "comment": c.reviewer_comment} for c in own]
    if review.reviewer_comments or review.reason_categories:
        reasons.append({"criterion_type": "FINAL_DECISION", "criterion_id": "final_decision", "reasons": review.reason_categories or [], "comment": review.reviewer_comments})
    return {"tags": [{"tag": t, "kind": k} for t, k in tags.items()], "gates": gates, "reasons": reasons}


def _record_learning_example(s: Session, user: User, review: HumanReview, run: EvaluationRun, approved_over_blocker: bool) -> tuple[Optional[LearningExample], Optional[str]]:
    """The person's submitted review is the validated feedback, so it becomes a learning example straight away. The example starts inactive for retrieval
    (Phase 2C decides when examples are used). Returns (example, None), or (None, why not).

    No example is made when the reason is a policy change (that goes through the rulebook approval process), when the person calls their own review a
    mistake, or when the photograph was approved over a publication blocker (that decision rests on context outside the photograph)."""
    latest = latest_corrections(review_corrections(s, review))
    if approved_over_blocker:
        return None, "approved over a publication blocker: that decision depends on what happens to the photograph before publishing, not on what it shows"
    bad = sorted({r for c in latest for r in c.correction_reason if r in NOT_FOR_LEARNING} | {r for r in (review.reason_categories or []) if r in NOT_FOR_LEARNING})
    if bad:
        return None, (f"reason {bad}: a policy change goes through the rulebook approval process, and a self-declared reviewer error is not trustworthy")
    view = _verified_view(run, review, latest)
    ex = LearningExample(organisation_id=review.organisation_id, photo_id=review.photo_id, original_evaluation_id=review.evaluation_id, human_review_id=review.review_id,
                         verified_decision=review.human_final_decision, verified_tags=view["tags"], verified_gate_results=view["gates"], verified_reasons=view["reasons"],
                         source="HUMAN_REVIEW", active_for_retrieval=bool(example_service.settings(s.get(Organisation, review.organisation_id))["enabled"]),
                         low_information=is_low_information(review.reviewer_comments))   # a generic reason ("looks good") is kept, but never shown to the model as an example
    s.add(ex)
    s.flush()
    record_event(s, review.organisation_id, FeedbackEventType.LEARNING_EXAMPLE_CREATED.value, user, review,
                 payload={"learning_example_id": ex.learning_example_id, "active_for_retrieval": ex.active_for_retrieval, "low_information": ex.low_information})
    log_lifecycle(s, s.get(Photo, review.photo_id), "LEARNING_EXAMPLE_CREATED", None, review.evaluation_id, {"learning_example_id": ex.learning_example_id}, change_status=False)
    return ex, None


# ------------------------------------------------------------------------------------------------------------------------------- queries
def list_feedback(s: Session, user: User, status: Optional[str] = None, limit: int = 100, offset: int = 0) -> list[HumanReview]:
    q = select(HumanReview).where(HumanReview.organisation_id == user.organisation_id).order_by(HumanReview.created_at.desc()).limit(min(limit, 500)).offset(offset)
    if status:
        q = q.where(HumanReview.review_status == status)
    return list(s.scalars(q))


def evaluation_history(s: Session, user: User, photo_id: Optional[str] = None, limit: int = 100, offset: int = 0) -> list[EvaluationRun]:
    q = select(EvaluationRun).where(EvaluationRun.organisation_id == user.organisation_id).order_by(EvaluationRun.created_at.desc()).limit(min(limit, 500)).offset(offset)
    if photo_id:
        q = q.where(EvaluationRun.photo_id == photo_id)
    return list(s.scalars(q))
