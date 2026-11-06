from __future__ import annotations

import pytest

from ngo_photo_approval.config import PolicyConfig
from ngo_photo_approval.rules.evaluator import evaluate_rules
from ngo_photo_approval.rules.policy import decide, meets_threshold, to_percentage
from ngo_photo_approval.schemas import (
    Decision,
    PredictionResponse,
    ProbabilityType,
    ReasonCode,
    RuleCheck,
)

CAL = ProbabilityType.CALIBRATED


def result(policy, **statuses):
    from ..conftest import make_rules

    checks = [RuleCheck(rule_id=k, status=v) for k, v in {"R001": "PASS", "R002": "PASS", "R003": "PASS",
                                                          "R004": "PASS", **statuses}.items()]
    return evaluate_rules(make_rules(), checks, policy)


def test_percentage_is_0_to_100_not_fraction():
    assert to_percentage(0.80) == 80.0
    assert to_percentage(1.0) == 100.0 and to_percentage(0.0) == 0.0
    with pytest.raises(ValueError):
        to_percentage(1.2)
    with pytest.raises(ValueError):
        to_percentage(-0.1)


def test_probability_exactly_at_threshold_approves(policy):
    d, codes = decide(result(policy), 0.80, CAL, policy)
    assert d == Decision.APPROVE and codes == [ReasonCode.APPROVED]


def test_float_noise_at_threshold_still_approves(policy):
    assert meets_threshold(0.7999999999999, 0.80)
    assert decide(result(policy), 0.1 + 0.7, CAL, policy)[0] == Decision.APPROVE


def test_below_threshold_rejects(policy):
    d, codes = decide(result(policy), 0.7999, CAL, policy)
    assert d == Decision.REJECT and codes == [ReasonCode.BELOW_THRESHOLD]


def test_above_threshold_approves(policy):
    assert decide(result(policy), 0.92, CAL, policy)[0] == Decision.APPROVE


def test_mandatory_failure_overrides_high_probability(policy):
    d, codes = decide(result(policy, R001="FAIL"), 0.91, CAL, policy)
    assert d == Decision.REJECT and ReasonCode.MANDATORY_RULE_FAILED in codes


def test_mandatory_failure_does_not_rewrite_probability(policy):
    """Probability and policy rejection are different concepts: the number is untouched."""
    resp = PredictionResponse(
        image_id="x", decision=Decision.REJECT, approval_probability=0.91, probability_type=CAL,
        threshold=0.8, hard_rule_failures=["R001"], rule_set_version="v")
    assert resp.approval_probability == 0.91 and resp.approval_percentage == 91.0


def _review_rules():
    from ngo_photo_approval.schemas import Rule, RuleSet

    return RuleSet(rule_set_version="v", rules=[
        Rule(rule_id="R001", name="a", description="d", requirement_type="mandatory", uncertain_policy="review")])


def _uncertain(pol):
    return evaluate_rules(_review_rules(), [RuleCheck(rule_id="R001", status="UNCERTAIN")], pol)


def test_uncertain_mandatory_rejects_by_default(policy):
    assert decide(result(policy, R002="UNCERTAIN"), 0.99, CAL, policy)[0] == Decision.REJECT
    # the rule asks for review, but the (stricter) global policy says reject
    assert decide(_uncertain(policy), 0.99, CAL, policy)[0] == Decision.REJECT


def test_uncertain_mandatory_review_when_policy_allows():
    lenient = PolicyConfig(uncertain_result_policy="review", allow_review_status=True)
    assert decide(_uncertain(lenient), 0.99, CAL, lenient)[0] == Decision.REVIEW


def test_review_disabled_falls_back_to_reject():
    pol = PolicyConfig(uncertain_result_policy="review", allow_review_status=False)
    assert decide(_uncertain(pol), 0.99, CAL, pol)[0] == Decision.REJECT


def test_missing_probability_never_approves(policy):
    d, codes = decide(result(policy), None, ProbabilityType.NONE, policy)
    assert d == Decision.REJECT and ReasonCode.PROBABILITY_UNAVAILABLE in codes
    pol = PolicyConfig(allow_review_status=True)
    assert decide(result(pol), None, ProbabilityType.NONE, pol)[0] == Decision.REVIEW


def test_threshold_override(policy):
    assert decide(result(policy), 0.6, CAL, policy, threshold=0.5)[0] == Decision.APPROVE
    assert decide(result(policy), 0.6, CAL, policy, threshold=0.7)[0] == Decision.REJECT


def test_response_probability_bounds_validated():
    with pytest.raises(ValueError):
        PredictionResponse(image_id="x", decision=Decision.APPROVE, approval_probability=1.5,
                           threshold=0.8, rule_set_version="v")
    with pytest.raises(ValueError):
        PredictionResponse(image_id="x", decision=Decision.APPROVE, approval_percentage=50,
                           threshold=0.8, rule_set_version="v")


def test_response_percentage_derived_from_probability():
    r = PredictionResponse(image_id="x", decision=Decision.APPROVE, approval_probability=0.87,
                           probability_type=CAL, threshold=0.8, rule_set_version="v")
    assert r.approval_percentage == 87.0 and 0 <= r.approval_percentage <= 100


def test_shipped_default_policy_is_conservative(real_policy):
    assert real_policy.approval_threshold == 0.80
    assert real_policy.enforce_mandatory_rules is True
    assert real_policy.missing_rule_result_policy == "reject"
    assert real_policy.uncertain_result_policy == "reject"
    assert real_policy.allow_review_status is False
