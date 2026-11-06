"""Final decision policy: combines the rule outcome with the approval probability + threshold.

Mandatory-rule logic lives entirely in ``rules.evaluator``; a high probability can never
override a rule-engine rejection (and a rejection never rewrites the probability).
"""

from __future__ import annotations

from ..config import PolicyConfig
from ..schemas import Decision, PolicyResult, ProbabilityType, ReasonCode

_EPS = 1e-9


def to_percentage(probability: float) -> float:
    """0.80 -> 80.0 (never 0.80)."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    return round(probability * 100.0, 4)


def meets_threshold(probability: float, threshold: float) -> bool:
    """Inclusive comparison (p == threshold approves), robust to float noise."""
    return probability + _EPS >= threshold


def decide(
    policy_result: PolicyResult,
    probability: float | None,
    probability_type: ProbabilityType,
    policy: PolicyConfig,
    threshold: float | None = None,
) -> tuple[Decision, list[ReasonCode]]:
    """Return (decision, reason_codes).

    Order of precedence:
      1. rule engine reject            -> REJECT
      2. rule engine review            -> REVIEW if allowed, else REJECT
      3. probability missing/invalid   -> REVIEW if allowed, else REJECT
      4. probability >= threshold      -> APPROVE
      5. otherwise                     -> REJECT (BELOW_THRESHOLD)
    """
    thr = policy.approval_threshold if threshold is None else threshold
    codes = list(policy_result.reason_codes)

    if policy_result.rule_outcome == "reject":
        return Decision.REJECT, codes or [ReasonCode.RESULT_INVALID]
    if policy_result.rule_outcome == "review":
        if policy.allow_review_status:
            return Decision.REVIEW, codes
        return Decision.REJECT, codes

    if probability is None or probability_type == ProbabilityType.NONE or not 0.0 <= probability <= 1.0:
        codes.append(ReasonCode.PROBABILITY_UNAVAILABLE)
        return (Decision.REVIEW if policy.allow_review_status else Decision.REJECT), codes

    if meets_threshold(probability, thr):
        return Decision.APPROVE, [ReasonCode.APPROVED]
    return Decision.REJECT, [ReasonCode.BELOW_THRESHOLD]
