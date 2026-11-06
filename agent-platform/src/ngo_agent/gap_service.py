"""L3, gap analysis. After a person decides, compare what the AI said with what the person wrote and record the gaps, so the learning loop (L4) has
something concrete to count: "the AI keeps failing G004 where staff say the faces are fine", "staff keep approving for a reason the AI never mentions".

* It runs after the decision is committed, in the background, and can never change or block the decision: a failure is recorded as a FAILED analysis.
* The analyzer sees TEXT only (the AI's verdict and reasons, the person's written reasons, their corrections), never the photograph, and e-mail addresses,
  links and phone numbers are removed first. It is a pluggable `GapAnalyzer`.
* `BaselineGapAnalyzer` works offline with no model: it finds sentences of the person's reason that the AI's own text does not cover, and links them to
  the AI's failed or uncertain gates by shared words. It is a keyword heuristic and does not understand negation ("the faces are visible" shares words with
  "the faces are not visible"), so for a disagreement it records the contested gate and leaves the meaning to a person or to the model-based analyzer
  that replaces it once the NGO authorises model calls for this (`llm_gap_analyzer.py`: built, off by default, never run on a real model).
* Results are append-only (one row per review and analyzer).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .enums import AI_DEFERRING, Agreement, CriterionType, ReviewStatus
from .errors import NotFound
from .models import EvaluationRun, GapAnalysis, HumanReview, User
from .reason_quality import content_words, is_low_information, redact

log = logging.getLogger("ngo_agent.gaps")
MAX_SUMMARY = 300
MIN_SENTENCE_WORDS = 2
COVERED_ABOVE = 0.5          # a sentence is "covered" when at least this share of its content words appear in the AI's own text


@dataclass
class AnalyzerResult:
    gaps: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    model_id: Optional[str] = None
    cost_usd: Optional[float] = None


class GapAnalyzer(Protocol):
    name: str

    def analyze(self, payload: dict[str, Any]) -> AnalyzerResult: ...


# --------------------------------------------------------------------------------------------------------------------------------- the input
def build_payload(run: EvaluationRun, review: HumanReview, corrections: list[Any]) -> dict[str, Any]:
    """Everything an analyzer may see: text only, with contact details removed. No photograph, no names of people from the database."""
    d, v = run.decision_output or {}, run.validated_response or {}
    gates = [{"gate_id": g.get("gate_id"), "criterion_id": g.get("criterion_id"), "status": g.get("status"), "evidence": redact(g.get("evidence"))}
             for g in d.get("gates") or [] if g.get("status") in ("NO", "CANNOT_TELL")]
    gate_criteria = {g.get("criterion_id") for g in d.get("gates") or []}
    criteria = [{"criterion_id": c.get("criterion_id"), "status": c.get("status"), "evidence": redact(c.get("evidence"))}
                for c in v.get("layer_2_criteria") or [] if c.get("status") in ("NO", "CANNOT_TELL") and c.get("criterion_id") not in gate_criteria]   # the other rules the AI failed or doubted
    own = [c for c in corrections if c.criterion_type != CriterionType.FINAL_DECISION.value]
    return {
        "ai": {"recommendation": run.ai_recommendation, "reasons": [redact(r) for r in d.get("decision_reasons") or []], "failed_or_uncertain_gates": gates, "failed_or_uncertain_criteria": criteria,
               "positives": [redact(o.get("text")) for o in v.get("positive_attributes") or []], "concerns": [redact(o.get("text")) for o in v.get("potential_rejection_factors") or []]},
        "human": {"decision": review.human_final_decision, "agreement": review.overall_agreement_with_ai, "reason": redact(review.reviewer_comments),
                  "corrections": [{"criterion_id": c.criterion_id, "criterion_type": c.criterion_type, "from": c.ai_predicted_value, "to": c.human_corrected_value,
                                   "comment": redact(c.reviewer_comment)} for c in own]},
    }


# -------------------------------------------------------------------------------------------------------------------------- baseline analyzer
def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _words(text: Optional[str]) -> set[str]:
    return {_stem(w) for w in content_words(text)}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.;!?\n]+", text or "") if len(_words(s)) >= MIN_SENTENCE_WORDS]


class BaselineGapAnalyzer:
    """Offline keyword overlap. See the module docstring for what it can and cannot do."""

    name = "baseline-overlap-1"

    def analyze(self, payload: dict[str, Any]) -> AnalyzerResult:
        ai, human = payload["ai"], payload["human"]
        ai_text = " ".join([*ai["reasons"], *ai["positives"], *ai["concerns"], *(g.get("evidence") or "" for g in ai["failed_or_uncertain_gates"])])
        ai_words = _words(ai_text)
        gate_words = {g["gate_id"]: _words(" ".join(filter(None, [g.get("evidence"), (g.get("criterion_id") or "").replace("_", " ")]))) for g in ai["failed_or_uncertain_gates"]}
        disagree = human["agreement"] == Agreement.DISAGREE.value
        texts = [human["reason"]] + [c["comment"] for c in human["corrections"] if c.get("comment")]
        gaps: list[dict[str, Any]] = []
        seen: set[str] = set()
        for text in texts:
            for sent in _sentences(text):
                if sent.lower() in seen:
                    continue
                seen.add(sent.lower())
                w = _words(sent)
                covered = len(w & ai_words) / len(w)
                related = sorted((gid for gid, gw in gate_words.items() if w & gw), key=lambda gid: -len(w & gate_words[gid]))[:2]
                if disagree and related:
                    kind = "CONTESTED_GATE"          # the person disagreed and this is about a gate the AI failed or doubted
                elif covered < COVERED_ABOVE:
                    kind = "MISSED_POINT"            # most of what the person said is nowhere in the AI's own text
                else:
                    continue
                gaps.append({"kind": kind, "summary": sent[:MAX_SUMMARY], "related_gates": related, "covered": round(covered, 2)})
        n_missed = sum(g["kind"] == "MISSED_POINT" for g in gaps)
        n_contested = len(gaps) - n_missed
        summary = (f"{n_missed} point(s) the AI did not mention; {n_contested} contested gate(s)" if gaps else "nothing the person wrote is missing from the AI's own reasoning")
        return AnalyzerResult(gaps=gaps, summary=summary)


# --------------------------------------------------------------------------------------------------------------------------------- running it
def analyze_review(factory: sessionmaker[Session], review_id: str, analyzer: GapAnalyzer) -> Optional[str]:
    """Analyse one submitted review and store the result. Idempotent (one row per review and analyzer) and it never raises: a failure is stored as FAILED.
    The analyzer runs with no database transaction open, because a model-based one can be slow. Returns the status, or None if there was nothing to do."""
    try:
        with factory() as s:
            review = s.get(HumanReview, review_id)
            if review is None or review.review_status != ReviewStatus.FEEDBACK_SUBMITTED.value:
                return None
            if s.scalar(select(GapAnalysis).where(GapAnalysis.review_id == review_id, GapAnalysis.analyzer == analyzer.name)):
                return "EXISTS"
            from .review_service import latest_corrections, review_corrections  # lazy: review_service must not depend on this module

            run = s.get(EvaluationRun, review.evaluation_id)
            payload = build_payload(run, review, latest_corrections(review_corrections(s, review)))
            org, eid, pid, agreement, low = review.organisation_id, review.evaluation_id, review.photo_id, review.overall_agreement_with_ai, is_low_information(review.reviewer_comments)
        result, error = None, None
        try:
            result = analyzer.analyze(payload)
        except Exception as exc:  # noqa: BLE001  (an analyzer problem is recorded, never raised: it must not touch a decision)
            error = type(exc).__name__
            log.warning("gap analysis of review %s failed: %s", review_id, error)
        with factory() as s:
            if s.scalar(select(GapAnalysis).where(GapAnalysis.review_id == review_id, GapAnalysis.analyzer == analyzer.name)):
                return "EXISTS"
            s.add(GapAnalysis(organisation_id=org, review_id=review_id, evaluation_id=eid, photo_id=pid, analyzer=analyzer.name,
                              model_id=result.model_id if result else None, status="DONE" if result else "FAILED", agreement=agreement or "", low_information=low,
                              gaps=result.gaps if result else [], summary=result.summary if result else "", error=error, cost_usd=result.cost_usd if result else None))
            s.commit()
        return "DONE" if result else "FAILED"
    except Exception as exc:  # noqa: BLE001
        log.warning("gap analysis of review %s could not be stored: %s", review_id, type(exc).__name__)
        return None


def gaps_of_review(s: Session, review_id: str) -> list[GapAnalysis]:
    return list(s.scalars(select(GapAnalysis).where(GapAnalysis.review_id == review_id).order_by(GapAnalysis.created_at)))


def list_gaps(s: Session, user: User, limit: int = 100, offset: int = 0) -> list[GapAnalysis]:
    return list(s.scalars(select(GapAnalysis).where(GapAnalysis.organisation_id == user.organisation_id).order_by(GapAnalysis.created_at.desc()).limit(min(limit, 500)).offset(offset)))


def ser_gap(g: GapAnalysis) -> dict[str, Any]:
    return {"gap_id": g.gap_id, "review_id": g.review_id, "evaluation_id": g.evaluation_id, "photo_id": g.photo_id, "analyzer": g.analyzer, "model_id": g.model_id, "status": g.status,
            "agreement": g.agreement, "low_information": g.low_information, "gaps": g.gaps, "summary": g.summary, "created_at": g.created_at.isoformat() if g.created_at else None}


def require(s: Session, review_id: str, user: User) -> HumanReview:
    r = s.scalar(select(HumanReview).where(HumanReview.review_id == review_id, HumanReview.organisation_id == user.organisation_id))
    if r is None:
        raise NotFound("review not found")
    return r
