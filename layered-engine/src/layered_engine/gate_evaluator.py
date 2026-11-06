"""Layer 2: effective criterion statuses and mandatory gates.

Rules enforced here (each has a test):
  * applicability is decided by Python from the facts; the model's own N/A is advisory
  * a criterion that applies (or may apply: unknown facts) but is missing from the answer is CANNOT_TELL, never a pass
  * the model saying NOT_APPLICABLE for something Python says applies is CANNOT_TELL ("na_conflict"), never a pass
  * a NO on something Python says does not apply is ignored and noted
  * exceptions run after that, and only ever change a NO
"""

from __future__ import annotations

from typing import Any, Mapping

from .applicability import applicability_of, facts_dict
from .exception_handler import apply_exceptions
from .rulebook import ResolvedProfile
from .schemas import Confidence, CriterionResult, GateResult, ModelAssessment, Status


def evaluate_criteria(assessment: ModelAssessment, profile: ResolvedProfile, metadata: Mapping[str, Any]) -> dict[str, CriterionResult]:
    facts = facts_dict(assessment)
    answers = {a.criterion_id: a for a in assessment.layer_2_criteria}
    out: dict[str, CriterionResult] = {}
    for c in profile.asked_criteria():
        applicable = applicability_of(c, profile, facts, metadata)
        a = answers.get(c.criterion_id)
        notes: list[str] = []
        if applicable is False:
            if a is not None and a.status is Status.NO:
                notes.append("advisory_ignored: the model answered NO but the facts say this does not apply")
            out[c.criterion_id] = CriterionResult(criterion_id=c.criterion_id, applicable=False, model_status=a.status if a else None,
                                                  status=Status.NOT_APPLICABLE, notes=notes)
            continue
        if a is None:
            out[c.criterion_id] = CriterionResult(criterion_id=c.criterion_id, applicable=applicable, model_status=None, status=Status.CANNOT_TELL,
                                                  evidence="", notes=["synthesized: missing from the model response"])
            continue
        status = a.status
        if status is Status.NOT_APPLICABLE and not c.model_na_allowed:
            status = Status.CANNOT_TELL
            notes.append("na_conflict: the model said NOT_APPLICABLE but the facts say this applies")
        out[c.criterion_id] = CriterionResult(criterion_id=c.criterion_id, applicable=applicable, model_status=a.status, status=status,
                                              evidence=a.evidence, confidence=a.confidence, notes=notes)
    apply_exceptions(out, profile.exceptions, facts, metadata)
    return out


def _aggregate(statuses: list[Status]) -> Status:
    """Parent gate over its components: any NO fails; else any unknown; else all N/A -> N/A; else pass."""
    if Status.NO in statuses:
        return Status.NO
    if Status.CANNOT_TELL in statuses:
        return Status.CANNOT_TELL
    if statuses and all(s is Status.NOT_APPLICABLE for s in statuses):
        return Status.NOT_APPLICABLE
    return Status.YES


def evaluate_gates(results: Mapping[str, CriterionResult], profile: ResolvedProfile) -> list[GateResult]:
    gates: list[GateResult] = []
    for g in sorted(profile.gates, key=lambda x: x.gate_id):
        crit = profile.criteria[g.criterion_id]
        hard = profile.is_hard(g)
        if g.components:
            comps = [results[c] for c in g.components]
            status = _aggregate([c.status for c in comps])
            bad = [c for c in comps if c.status in (Status.NO, Status.CANNOT_TELL)]
            applicable = None if any(c.applicable is None for c in comps) else any(c.applicable for c in comps)
            gates.append(GateResult(
                gate_id=g.gate_id, criterion_id=g.criterion_id, source=g.source, parent_rule_id=g.parent_rule_id, hard=hard, status=status,
                applicable=applicable, evidence="; ".join(f"{c.criterion_id}: {c.evidence or c.status.value}" for c in bad), components=comps,
                confidence=min((c.confidence for c in comps), key=lambda x: ("HIGH", "MEDIUM", "LOW").index(x.value)),
                failure_reason=None if status in (Status.YES, Status.NOT_APPLICABLE) else crit.description_operational))
            continue
        r = results[g.criterion_id]
        gates.append(GateResult(
            gate_id=g.gate_id, criterion_id=g.criterion_id, source=g.source, parent_rule_id=g.parent_rule_id, hard=hard, status=r.status,
            applicable=r.applicable, evidence=r.evidence, confidence=r.confidence, notes=list(r.notes),
            failure_reason=None if r.status in (Status.YES, Status.NOT_APPLICABLE) else crit.description_operational))
    return gates
