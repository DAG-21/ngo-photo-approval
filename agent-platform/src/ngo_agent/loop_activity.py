"""A plain account of what the learning loop did with each decided photograph, for the administrator's "Decisions" screen.

For every submitted decision, newest first: the photograph, what the AI said, what the reviewer decided and wrote, the points the reviewer disagreed with the AI about (if any), the rule each of
those points relates to, and what that rule was changed to (or why it was not). It only reads what is already stored (evaluations, reviews, the model-based gap analysis, the rule rewrites);
it calls no model and changes nothing.
"""

from __future__ import annotations

from typing import Any

from layered_engine.rulebook import Rulebook
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import rule_service
from .enums import ReviewStatus
from .models import EvaluationRun, GapAnalysis, HumanReview, Organisation, Photo, RuleEdit
from .reason_quality import redact
from .rule_drafter import rule_catalogue

MAX_POINTS = 8
DISPUTE_KINDS = ("CONTESTED_GATE", "CONTESTED_RULE", "CONTESTED_POINT")


def _ai_points(run: EvaluationRun, cat: dict[str, Any]) -> list[dict[str, Any]]:
    """What the AI found wrong or could not tell, one plain line each: the rule's question and the evidence the AI gave."""
    out = []
    for g in (run.decision_output or {}).get("gates") or []:
        if g.get("status") in ("NO", "CANNOT_TELL"):
            info = cat.get(g.get("criterion_id") or "")
            out.append({"status": g["status"], "gate_id": g.get("gate_id"), "rule_id": g.get("criterion_id"), "question": info.base if info else "", "evidence": redact(g.get("evidence"))})
    return out[:MAX_POINTS]


def _analysis(s: Session, review_id: str) -> GapAnalysis | None:
    rows = list(s.scalars(select(GapAnalysis).where(GapAnalysis.review_id == review_id, GapAnalysis.status == "DONE").order_by(GapAnalysis.created_at.desc())))
    return next((g for g in rows if g.analyzer.startswith("llm-gap")), rows[0] if rows else None)


def _change(edit: RuleEdit | None, status: str | None, mode_note: str) -> dict[str, Any]:
    """What happened to one rule after this decision, in a form the screen can show without interpreting anything."""
    if edit is None:
        return {"state": "NOT_CHANGED", "why": mode_note}
    base = {"edit_id": edit.edit_id, "before": edit.previous_text, "after": edit.new_text, "direction": edit.direction, "flags": list(edit.flags or []), "why": edit.rationale,
            "failed_checks": (edit.checks or {}).get("failed", [])}
    state = {"APPLIED": "CHANGED", "CONFIRMED": "CHANGED", "SUPERSEDED": "CHANGED_THEN_REPLACED", "WITHDRAWN": "CHANGED_THEN_WITHDRAWN", "REJECTED": "REJECTED", "NO_CHANGE": "NOT_NEEDED",
             "HELD": "HELD", "DRAFTED": "DRAFTED_ONLY", "NEEDS_CONFIRM": "WAITING_FOR_CONFIRMATION", "DECLINED": "DECLINED"}.get(status or "DRAFTED", "DRAFTED_ONLY")
    return {"state": state, **base}


def activity(s: Session, org: Organisation, rb: Rulebook, profile: str, limit: int = 30, offset: int = 0) -> dict[str, Any]:
    cat = rule_catalogue(rb, profile)
    by_gate = {i.gate_id: t for t, i in cat.items() if i.gate_id}
    cfg = rule_service.settings(org)
    off = cfg["mode"] == "OFF" or cfg["paused"]
    st = rule_service.status_map(s, org.organisation_id)
    where = (HumanReview.organisation_id == org.organisation_id, HumanReview.review_status == ReviewStatus.FEEDBACK_SUBMITTED.value)
    total = s.scalar(select(func.count()).select_from(HumanReview).where(*where)) or 0
    items = []
    for review in s.scalars(select(HumanReview).where(*where).order_by(HumanReview.reviewed_at.desc(), HumanReview.created_at.desc()).limit(min(max(limit, 1), 100)).offset(max(offset, 0))):
        run, photo = s.get(EvaluationRun, review.evaluation_id), s.get(Photo, review.photo_id)
        d = run.decision_output or {}
        edits = {e.target_id: e for e in s.scalars(select(RuleEdit).where(RuleEdit.review_id == review.review_id).order_by(RuleEdit.created_at))}
        gap = _analysis(s, review.review_id)
        shown: set[str] = set()
        disputes = []
        for g in (gap.gaps if gap else []) or []:
            if not (g.get("relation") == "CONTRADICTS_AI" or g.get("kind") in DISPUTE_KINDS):
                continue
            targets = g.get("targets") or [by_gate[x] for x in g.get("related_gates") or [] if x in by_gate]
            rules = []
            for t in targets:
                info = cat.get(t)
                if info is None:
                    continue
                shown.add(t)
                rules.append({"rule_id": t, "kind": info.kind, "gate_id": info.gate_id, "can_reject": info.hard, "wording": info.base,
                              "change": _change(edits.get(t), st.get(edits[t].edit_id) if t in edits else None,
                                                "Rule updates were off, so no rewrite was attempted." if off else "No rewrite was made for this rule.")})
            disputes.append({"point": g.get("summary") or "", "how_it_differs": g.get("rule_note") or "", "confidence": g.get("confidence"), "rules": rules})
        other = [{"rule_id": t, "wording": cat[t].base if t in cat else "", "change": _change(e, st.get(e.edit_id), "")} for t, e in edits.items() if t not in shown]
        items.append({
            "review_id": review.review_id, "photo_id": review.photo_id, "filename": photo.original_filename if photo else "", "decided_at": review.reviewed_at.isoformat() if review.reviewed_at else None,
            "ai": {"verdict": run.ai_recommendation, "points": _ai_points(run, cat), "reasons": [redact(r) for r in d.get("decision_reasons") or []]},
            "human": {"verdict": review.human_final_decision, "agrees_with_ai": review.overall_agreement_with_ai == "AGREE", "agreement": review.overall_agreement_with_ai, "reason": review.reviewer_comments or ""},
            "analysis": {"status": gap.status if gap else "NONE", "analyzer": gap.analyzer if gap else None, "model": gap.model_id if gap else None, "summary": gap.summary if gap else
                         "Not analysed: the written reason was missing, too generic, or the analysis has not run."},
            "disagreements": disputes, "other_changes": other})
    return {"total": total, "rule_updates": {"mode": cfg["mode"], "paused": cfg["paused"]}, "items": items}
