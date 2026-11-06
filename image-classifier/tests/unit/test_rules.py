from __future__ import annotations

import pytest
import yaml

from ngo_photo_approval.config import PolicyConfig
from ngo_photo_approval.exceptions import RuleConfigError
from ngo_photo_approval.rules.evaluator import evaluate_rules
from ngo_photo_approval.rules.loader import config_warnings, load_rule_set, rule_set_fingerprint
from ngo_photo_approval.schemas import ReasonCode, Rule, RuleCheck, RuleSet

from ..conftest import make_rules


def chk(**kw: str) -> list[RuleCheck]:
    return [RuleCheck(rule_id=k, status=v, evidence="e") for k, v in kw.items()]


ALL_PASS = dict(R001="PASS", R002="PASS", R003="PASS", R004="PASS")


def test_all_mandatory_pass(rule_set, policy):
    r = evaluate_rules(rule_set, chk(**ALL_PASS), policy)
    assert r.rule_outcome == "pass"
    assert r.hard_rule_failures == [] and r.reason_codes == []
    assert r.rule_set_version == "test-1.0"


def test_mandatory_fail_rejects(rule_set, policy):
    r = evaluate_rules(rule_set, chk(**{**ALL_PASS, "R001": "FAIL"}), policy)
    assert r.rule_outcome == "reject"
    assert r.hard_rule_failures == ["R001"]
    assert ReasonCode.MANDATORY_RULE_FAILED in r.reason_codes


def test_mandatory_uncertain_rejects_by_default(rule_set, policy):
    r = evaluate_rules(rule_set, chk(**{**ALL_PASS, "R002": "UNCERTAIN"}), policy)
    assert r.rule_outcome == "reject"
    assert r.uncertain_rules == ["R002"]
    assert ReasonCode.MANDATORY_RULE_UNCERTAIN in r.reason_codes


def test_uncertain_review_policy_gives_review(policy):
    rs = RuleSet(rule_set_version="v", rules=[
        Rule(rule_id="R001", name="a", description="d", requirement_type="mandatory", uncertain_policy="review")])
    # global policy "reject" is stricter than the rule's "review": the stricter wins
    assert evaluate_rules(rs, chk(R001="UNCERTAIN"), policy).rule_outcome == "reject"
    lenient = PolicyConfig(uncertain_result_policy="review")
    assert evaluate_rules(rs, chk(R001="UNCERTAIN"), lenient).rule_outcome == "review"


def test_missing_result_rejects_never_passes(rule_set, policy):
    r = evaluate_rules(rule_set, chk(R001="PASS", R002="PASS", R004="PASS"), policy)
    assert r.rule_outcome == "reject"
    assert r.missing_rules == ["R003"]
    assert ReasonCode.RULE_RESULT_MISSING in r.reason_codes


def test_missing_policy_review(rule_set):
    r = evaluate_rules(rule_set, chk(R001="PASS", R002="PASS", R004="PASS"),
                       PolicyConfig(missing_rule_result_policy="review"))
    assert r.rule_outcome == "review"


def test_duplicated_rule_id_is_integrity_failure(rule_set, policy):
    r = evaluate_rules(rule_set, chk(**ALL_PASS) + chk(R001="PASS"), policy)
    assert r.rule_outcome == "reject"
    assert ReasonCode.RULE_RESULT_DUPLICATED in r.reason_codes


def test_duplicate_never_reviewable(rule_set):
    lenient = PolicyConfig(missing_rule_result_policy="review", uncertain_result_policy="review")
    r = evaluate_rules(rule_set, chk(**ALL_PASS) + chk(R002="PASS"), lenient)
    assert r.rule_outcome == "reject"


def test_unknown_rule_id_rejects(rule_set, policy):
    r = evaluate_rules(rule_set, chk(**ALL_PASS) + chk(R999="PASS"), policy)
    assert r.rule_outcome == "reject"
    assert ReasonCode.RULE_RESULT_UNKNOWN in r.reason_codes


def test_scored_rule_failure_does_not_reject(rule_set, policy):
    r = evaluate_rules(rule_set, chk(**{**ALL_PASS, "R004": "FAIL"}), policy)
    assert r.rule_outcome == "pass"
    assert r.unmet_scored_rules == ["R004"] and r.hard_rule_failures == []


def test_scored_uncertain_score_only_passes(rule_set, policy):
    r = evaluate_rules(rule_set, chk(**{**ALL_PASS, "R004": "UNCERTAIN"}), policy)
    assert r.rule_outcome == "pass" and r.unmet_scored_rules == ["R004"]


def test_failed_analysis_rejects(rule_set, policy):
    r = evaluate_rules(rule_set, None, policy)
    assert r.rule_outcome == "reject" and ReasonCode.RESULT_INVALID in r.reason_codes


def test_enforce_mandatory_false_still_reports_failure(rule_set):
    r = evaluate_rules(rule_set, chk(**{**ALL_PASS, "R001": "FAIL"}), PolicyConfig(enforce_mandatory_rules=False))
    assert r.hard_rule_failures == ["R001"] and r.rule_outcome == "pass"


def test_disabled_rule_is_not_required(policy):
    rs = make_rules([dict(rule_id="R005", name="off", description="d", requirement_type="mandatory",
                          enabled=False)])
    assert evaluate_rules(rs, chk(**ALL_PASS), policy).rule_outcome == "pass"
    # but a result returned for it is out of contract
    assert evaluate_rules(rs, chk(**ALL_PASS) + chk(R005="PASS"), policy).rule_outcome == "reject"


def test_program_scoped_rule(policy):
    rs = make_rules([dict(rule_id="R005", name="birds", description="d", requirement_type="mandatory",
                          applies_to_programs=["kabutar_chan"])])
    assert [r.rule_id for r in rs.active("arham_aahar")] == ["R001", "R002", "R003", "R004"]
    assert "R005" in [r.rule_id for r in rs.active("kabutar_chan")]
    assert "R005" in [r.rule_id for r in rs.active(None)]  # unknown program: stricter
    assert evaluate_rules(rs, chk(**ALL_PASS), policy, program="arham_aahar").rule_outcome == "pass"


def test_results_carry_rule_set_version(policy):
    rs = RuleSet(rule_set_version="9.9", rules=make_rules().rules)
    assert evaluate_rules(rs, chk(**ALL_PASS), policy).rule_set_version == "9.9"


# ------------------------------------------------------------------ configuration validation


def test_invalid_status_rejected():
    with pytest.raises(ValueError):
        RuleCheck(rule_id="R001", status="MAYBE")


def test_status_is_case_normalized():
    assert RuleCheck(rule_id="R001", status=" pass ").status.value == "PASS"


def test_duplicate_rule_ids_in_config_invalid():
    r = dict(rule_id="R001", name="a", description="d", requirement_type="mandatory")
    with pytest.raises(ValueError):
        RuleSet(rule_set_version="v", rules=[Rule(**r), Rule(**{**r, "name": "b"})])


def test_mandatory_rule_cannot_be_score_only():
    with pytest.raises(ValueError):
        Rule(rule_id="R001", name="a", description="d", requirement_type="mandatory", uncertain_policy="score_only")


def test_loader_rejects_bad_files(tmp_path):
    bad = tmp_path / "rules.yaml"
    bad.write_text("rule_set_version: '1'\nrules: [{rule_id: X1}]\n")
    with pytest.raises(RuleConfigError):
        load_rule_set(bad)
    with pytest.raises(RuleConfigError):
        load_rule_set(tmp_path / "missing.yaml")
    bad.write_text(yaml.safe_dump({"rule_set_version": "1", "rules": []}))
    with pytest.raises(RuleConfigError):
        load_rule_set(bad)


def test_shipped_rules_load_in_sync_with_the_document():
    rs = load_rule_set()  # verifies against rules/photo-approval-requirement-rules.txt and raises if out of sync
    assert [r.rule_id for r in rs.rules] == [f"R{n:03d}" for n in range(1, 17)]
    assert all(r.confirmed and r.source_ref and r.original_text for r in rs.rules)
    assert len(rule_set_fingerprint(rs)) == 12
    assert any("NOT confirmed" in w for w in config_warnings(rs))  # restatements not yet reviewed by the client


def test_nothing_is_mandatory_because_the_document_never_says_so():
    rs = load_rule_set()
    assert [r.rule_id for r in rs.rules if r.requirement_type.value == "mandatory"] == []
    assert all(r.uncertain_policy.value == "score_only" and r.mandatory_status_confirmed for r in rs.rules)
    assert all(r.enabled for r in rs.rules) and len(rs.active()) == 16
    assert not any("mandatory status not confirmed" in w for w in config_warnings(rs))


def test_conditions_from_the_wording_survive():
    rs = load_rule_set()
    cond = {r.rule_id for r in rs.rules if r.applies_when}
    assert {"R001", "R002", "R003", "R004", "R008"} <= cond  # "If a banner is present", "wherever appropriate", ...


def test_with_no_mandatory_rules_failures_never_reject_but_gaps_do(policy):
    rs = load_rule_set()
    ids = [r.rule_id for r in rs.active()]

    def run(over: dict[str, str]):
        return evaluate_rules(rs, [RuleCheck(rule_id=i, status=over.get(i, "PASS")) for i in ids], policy)

    assert run({}).rule_outcome == "pass"
    r = run({"R002": "FAIL", "R009": "FAIL", "R001": "UNCERTAIN"})
    assert r.rule_outcome == "pass" and r.hard_rule_failures == []
    assert set(r.unmet_scored_rules) == {"R001", "R002", "R009"}  # recorded as features / evidence, not hard gates
    partial = evaluate_rules(rs, [RuleCheck(rule_id=i, status="PASS") for i in ids[:-1]], policy)
    assert partial.rule_outcome == "reject" and partial.missing_rules == [ids[-1]]  # incomplete analysis never passes


def test_guidance_never_alters_the_client_wording():
    """The camera-from-the-front clarification lives in additional_guidance; original_text stays verbatim."""
    rs = load_rule_set()  # also verifies original_text against the rules document
    for rid in ("R003", "R007", "R008", "R015", "R016"):
        r = rs.by_id(rid)
        assert "front" in r.additional_guidance and "behind" in r.additional_guidance
        assert "behind" not in r.original_text.lower()
    assert all(not r.additional_guidance for r in rs.rules if r.rule_id not in {"R003", "R007", "R008", "R015", "R016"})
