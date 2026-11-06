"""Deterministic rule engine.

Consumes already-parsed structured results (no provider calls, no natural-language logic) and
produces a PolicyResult. Invariants:

* An UNCERTAIN, missing, duplicated or unknown result is never a PASS for a mandatory rule.
* A failed mandatory rule forces ``rule_outcome == "reject"`` regardless of any score.
* Integrity problems (duplicate / unknown rule ids, no analysis at all) always reject, never review.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from ..config import PolicyConfig
from ..schemas import (
    CheckStatus,
    PolicyResult,
    ReasonCode,
    RequirementType,
    RuleCheck,
    RuleIssue,
    RuleSet,
    UncertainPolicy,
)

_STRICTNESS = {UncertainPolicy.SCORE_ONLY: 0, UncertainPolicy.REVIEW: 1, UncertainPolicy.REJECT: 2}


def _stricter(a: UncertainPolicy, b: UncertainPolicy) -> UncertainPolicy:
    return a if _STRICTNESS[a] >= _STRICTNESS[b] else b


def evaluate_rules(
    rule_set: RuleSet,
    checks: Sequence[RuleCheck] | None,
    policy: PolicyConfig,
    program: str | None = None,
) -> PolicyResult:
    """Apply the rule set to ``checks``. ``checks=None`` means the analysis failed entirely."""
    active = rule_set.active(program)
    active_ids = [r.rule_id for r in active]
    all_known_ids = {r.rule_id for r in rule_set.rules}
    issues: list[RuleIssue] = []
    hard: list[str] = []
    uncertain: list[str] = []
    missing: list[str] = []
    unmet: list[str] = []
    outcomes: list[str] = []  # "reject" | "review"

    if checks is None:
        issues.append(RuleIssue(rule_id=None, reason_code=ReasonCode.RESULT_INVALID, detail="no analysis available"))
        return PolicyResult(
            rule_set_version=rule_set.rule_set_version, checks=[], missing_rules=list(active_ids),
            issues=issues, rule_outcome="reject", reason_codes=[ReasonCode.RESULT_INVALID],
        )

    counts = Counter(c.rule_id for c in checks)
    integrity_failure = False
    for rid, n in sorted(counts.items()):
        if rid not in all_known_ids:
            issues.append(RuleIssue(rule_id=rid, reason_code=ReasonCode.RULE_RESULT_UNKNOWN, detail="unknown rule id"))
            integrity_failure = True
        elif n > 1:
            issues.append(RuleIssue(rule_id=rid, reason_code=ReasonCode.RULE_RESULT_DUPLICATED, detail=f"{n} results"))
            integrity_failure = True
        elif rid not in active_ids:
            # Known but not active for this program/config: ignore the result, but note it.
            issues.append(RuleIssue(rule_id=rid, reason_code=ReasonCode.RULE_RESULT_UNKNOWN,
                                    detail="result for a rule that is not active for this image"))
            integrity_failure = True
    if integrity_failure:
        outcomes.append("reject")

    by_id = {c.rule_id: c for c in checks if counts[c.rule_id] == 1}
    for rule in active:
        rid = rule.rule_id
        if counts.get(rid, 0) > 1:
            continue  # already flagged as duplicated
        res = by_id.get(rid)
        mandatory = rule.requirement_type == RequirementType.MANDATORY
        if res is None:
            missing.append(rid)
            issues.append(RuleIssue(rule_id=rid, reason_code=ReasonCode.RULE_RESULT_MISSING))
            # An incomplete analysis never passes: follow the configured policy for ANY missing rule.
            outcomes.append(policy.missing_rule_result_policy)
            if not mandatory:
                unmet.append(rid)
            continue
        if res.status == CheckStatus.PASS:
            continue
        if res.status == CheckStatus.FAIL:
            if mandatory:
                hard.append(rid)
                issues.append(RuleIssue(rule_id=rid, reason_code=ReasonCode.MANDATORY_RULE_FAILED, detail=res.evidence))
                if policy.enforce_mandatory_rules:
                    outcomes.append("reject")
            else:
                unmet.append(rid)
                issues.append(RuleIssue(rule_id=rid, reason_code=ReasonCode.SCORED_RULE_FAILED, detail=res.evidence))
            continue
        # UNCERTAIN
        uncertain.append(rid)
        eff = rule.uncertain_policy
        if mandatory:
            eff = _stricter(eff, UncertainPolicy(policy.uncertain_result_policy))
            issues.append(RuleIssue(rule_id=rid, reason_code=ReasonCode.MANDATORY_RULE_UNCERTAIN, detail=res.evidence))
        else:
            unmet.append(rid)
            issues.append(RuleIssue(rule_id=rid, reason_code=ReasonCode.SCORED_RULE_UNCERTAIN, detail=res.evidence))
        if eff == UncertainPolicy.REJECT:
            outcomes.append("reject")
        elif eff == UncertainPolicy.REVIEW:
            outcomes.append("review")

    if "reject" in outcomes:
        outcome = "reject"
    elif "review" in outcomes:
        outcome = "review"
    else:
        outcome = "pass"

    reason_codes = sorted({i.reason_code for i in issues if _is_blocking(i, outcome)}, key=lambda c: c.value)
    if outcome == "review" and ReasonCode.NEEDS_REVIEW not in reason_codes:
        reason_codes.append(ReasonCode.NEEDS_REVIEW)
    return PolicyResult(
        rule_set_version=rule_set.rule_set_version,
        checks=list(checks),
        hard_rule_failures=sorted(hard),
        uncertain_rules=sorted(uncertain),
        missing_rules=sorted(missing),
        unmet_scored_rules=sorted(unmet),
        issues=issues,
        rule_outcome=outcome,
        reason_codes=reason_codes,
    )


def _is_blocking(issue: RuleIssue, outcome: str) -> bool:
    """Reason codes reported with the decision: everything except benign scored-rule notes."""
    if outcome == "pass":
        return False
    return issue.reason_code not in {ReasonCode.SCORED_RULE_FAILED, ReasonCode.SCORED_RULE_UNCERTAIN}
