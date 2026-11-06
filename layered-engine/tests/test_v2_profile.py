"""Evaluator V2 (layered_proposed_v2): V1 stays frozen, every V2 change is recorded and switchable, and each one does what it says."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import PROPOSED, make_assessment, verified_metadata
from layered_engine.comparison import AUTO, auc, headline, transitions, wilson
from layered_engine.decision_engine import evaluate_final_decision
from layered_engine.prompt_builder import build_prompt
from layered_engine.schema_export import build_response_schema, check_provider_safe
from layered_engine.schemas import Recommendation

V2 = "layered_proposed_v2"
ALL = ["V2-01", "V2-02", "V2-03", "V2-04", "V2-05", "V2-06", "V2-07"]
FROZEN = Path(__file__).resolve().parents[1] / "evaluators" / "v1_frozen"


def decide(rb, profile, without=(), **kw):
    p = rb.resolve(profile, without=without)
    meta = kw.pop("metadata", verified_metadata())
    return evaluate_final_decision(make_assessment(p, **kw), p, meta)


# ------------------------------------------------------------------------------------------------------------------ V1 is frozen
def test_v1_prompt_hash_text_and_schema_are_byte_identical_to_the_frozen_snapshot(rb):
    p = rb.resolve(PROPOSED)
    bp = build_prompt(rb, p)
    assert bp.hash == "11427c79d96c569a"  # the hash recorded in every pilot-3 row
    assert bp.system_text == (FROZEN / "prompt_layered_proposed_v1_system.txt").read_text(encoding="utf-8")
    schema = json.dumps(build_response_schema(p, bp.criterion_ids), indent=1, sort_keys=True)
    assert schema == (FROZEN / "response_schema_layered_proposed_v1.json").read_text()
    assert "f_gaze_tag" not in schema and "POST_ACTIVITY" not in bp.system_text


def test_v1_and_legacy_gate_counts_are_unchanged(rb):
    assert len(rb.resolve("checklist5_legacy").hard_gates()) == 15
    assert len(rb.resolve(PROPOSED).hard_gates()) == 16


# ---------------------------------------------------------------------------------------------------------------- V2 is recorded
def test_every_v2_change_is_recorded_pending_ngo_approval_and_complete(rb):
    p = rb.profiles[V2]
    assert rb.change_ids(V2) == set(ALL)
    for key in ("overrides", "gate_overrides"):
        for tgt, ov in p[key].items():
            pc = ov["policy_change"]
            assert pc["status"] == "PROPOSED_PENDING_NGO_APPROVAL" and pc["affects_mandatory"] is True, tgt
            for field in ("original_rule", "proposed_rule", "previous_effect", "new_effect", "evidence", "reason"):
                assert pc[field], (tgt, field)
    assert rb.resolve(V2).settings["prompt_version"] == "photo_evaluation_prompt_v2"


def test_v2_decisions_carry_the_policy_validation_flag(rb):
    d = decide(rb, V2)
    assert d.policy_validation_required is True
    assert d.versions["profile"] == V2 and d.versions["rulebook_version"] == "2.0.0-proposed"


def test_unknown_change_id_is_refused(rb):
    with pytest.raises(KeyError):
        rb.resolve(V2, without=["V2-99"])


def test_switching_every_change_off_gives_v1_behaviour(rb):
    v2 = rb.resolve(V2, without=ALL)
    v1 = rb.resolve(PROPOSED)
    assert {g.gate_id for g in v2.hard_gates()} == {g.gate_id for g in v1.hard_gates()}
    assert {c: x.applicability for c, x in v2.criteria.items() if c in v1.criteria} == {c: x.applicability for c, x in v1.criteria.items()}
    assert {c: x.description_operational for c, x in v2.criteria.items() if c in v1.criteria} == {c: x.description_operational for c, x in v1.criteria.items()}
    assert v2.settings["beneficiary_visibility"]["subject_scope"] == "ALL_VISIBLE_BENEFICIARIES"


def test_v2_hard_gate_list(rb):
    assert [g.gate_id for g in rb.resolve(V2).hard_gates()] == ["G001", "G002", "G003", "G004", "G005", "G006", "G007", "G008", "G009", "G012", "G013", "G014", "G016"]
    assert not rb.resolve(V2).exceptions or all(e["id"] != "EXC_BACK_VIEW" for e in rb.resolve(V2).exceptions)  # the logo exception stays withdrawn


def test_v2_prompt_and_schema(rb):
    p = rb.resolve(V2)
    bp = build_prompt(rb, p)
    assert bp.template_version == "photo_evaluation_prompt_v2" and bp.hash != "11427c79d96c569a"
    schema = build_response_schema(p, bp.criterion_ids)
    assert not check_provider_safe(schema)
    facts = schema["properties"]["layer_1_scene_understanding"]["properties"]["scene_facts"]
    assert facts["properties"]["f_gaze_tag"]["enum"] == ["DIRECT_CAMERA_GAZE", "NATURAL_CANDID_GAZE", "INTENTIONAL_PORTRAIT", "UNCERTAIN_GAZE"]
    assert "f_gaze_tag" in facts["required"]
    assert "POST_ACTIVITY" in facts["properties"]["f_activity_type"]["enum"]
    assert "PRIMARY or FEATURED" in bp.system_text
    assert "approved photograph" not in bp.system_text.lower() and "disapproved" not in bp.system_text.lower()  # no historical decision in the prompt


# --------------------------------------------------------------------------------------------------------- each change behaves
def test_v2_01_direct_gaze_is_a_tag_not_a_rejection(rb):
    a = {"R004_BEN_GAZE": "NO"}
    assert decide(rb, PROPOSED, answers=a).recommendation is Recommendation.REJECT
    d = decide(rb, V2, answers=a)
    assert "G015" not in d.failed_gates and d.recommendation is not Recommendation.REJECT
    assert decide(rb, V2, without=["V2-01"], answers=a).recommendation is Recommendation.REJECT  # switching it off restores the hard gate


def test_v2_03_banner_folds_only_matter_for_a_prominent_banner(rb):
    a = {"R001_FLAT": "NO"}
    assert "G009" in decide(rb, V2, answers=a).failed_gates
    assert "G009" not in decide(rb, V2, answers=a, facts={"f_banner_prominent": "NO"}).failed_gates
    assert "G009" in decide(rb, V2, without=["V2-03"], answers=a, facts={"f_banner_prominent": "NO"}).failed_gates
    assert "G009" not in decide(rb, V2, answers=a, facts={"f_banner": "NO", "f_banner_prominent": "NO"}).failed_gates  # no banner is never a failure


def test_v2_04_banner_visibility_is_soft_and_caps_brand_once(rb):
    a = {"R001_VISIBLE": "NO", "R001_DISTINCT": "NO"}
    assert {"G017", "G018"} <= set(decide(rb, PROPOSED, answers=a).failed_gates)
    d = decide(rb, V2, answers=a)
    assert not {"G017", "G018"} & set(d.failed_gates)
    assert d.quality.dimension_scores["brand_presentation"] == 2.0 and len([c for c in d.quality.caps_applied if "brand_presentation" in c]) == 1
    assert decide(rb, V2, without=["V2-04"], answers=a).recommendation is Recommendation.REJECT
    # the banner soft cap exists only in V2
    assert decide(rb, PROPOSED, answers={"R001_VISIBLE": "NO"}).quality.dimension_scores["brand_presentation"] == 4.0


def test_v2_05_g016_is_the_volunteer_back_view_only(rb):
    ben_back = {"T_FRONT": "NO", "R015_BEN_FRONT": "NO"}
    assert "G016" in decide(rb, PROPOSED, answers=ben_back).failed_gates
    assert "G016" not in decide(rb, V2, answers=ben_back).failed_gates
    assert "G016" in decide(rb, V2, answers={"R008_BACK": "NO"}).failed_gates  # a volunteer's back still fails
    assert "G016" in decide(rb, V2, without=["V2-05"], answers=ben_back).failed_gates
    # no logo exception exists in V2: a volunteer's back with a logo, dress and visible beneficiaries still fails
    assert "G016" in decide(rb, V2, answers={"R008_BACK": "NO", "T_BACK_EXC_DRESS": "YES", "T_BACK_EXC_LOGO": "YES", "T_BACK_EXC_BENEFICIARIES": "YES"}).failed_gates


def test_v2_07_handover_expected_where_an_item_is_visibly_handed_over(rb):
    a = {"R015_BEN_RECEIVING": "NO"}
    teach = dict(f_activity_type="EDUCATION_SUPPORT", f_item_given="NO", f_handover="NO")
    book = dict(f_activity_type="EDUCATION_SUPPORT", f_item_given="YES", f_handover="YES")
    assert "G012" not in decide(rb, V2, answers=a, facts=teach).failed_gates   # teaching: no handover expected
    assert "G012" in decide(rb, V2, answers=a, facts=book).failed_gates        # a visible book handover: expected
    assert "G012" not in decide(rb, PROPOSED, answers=a, facts=book).failed_gates  # v1 skipped it because the activity was filed under education
    assert "G012" in decide(rb, V2, answers=a, facts=dict(f_activity_type="FOOD_DISTRIBUTION")).failed_gates
    assert "G012" not in decide(rb, V2, answers=a, facts=dict(f_activity_type="ANIMAL_CARE", f_item_given="NO", f_handover="NO", f_beneficiaries="NO", f_animals="YES")).failed_gates


def test_v2_still_rejects_what_must_be_rejected(rb):
    """Loosening must not open the mandatory failures that are not part of V2's changes."""
    for cid, gate in (("R003_FACE", "G004"), ("R003_CLEAR", "G005"), ("G001_NOTHING_HIDDEN", "G001"), ("R002_GIVER_WHITE", "G013"), ("R015_PERSON_PRESENT", "G014"),
                      ("R006_ITEM_SHARP", "G003"), ("R008_BACK", "G016")):
        d = decide(rb, V2, answers={cid: "NO"})
        assert gate in d.failed_gates and d.recommendation is Recommendation.REJECT, cid
    assert decide(rb, V2, answers={"R003_FACE": "CANNOT_TELL"}).recommendation is Recommendation.HUMAN_REVIEW  # unknown is never a pass
    assert decide(rb, V2, omit=("R003_FACE",)).recommendation is Recommendation.HUMAN_REVIEW  # a missing answer is never a pass


def test_a_soft_score_still_cannot_override_a_failed_gate_in_v2(rb):
    d = decide(rb, V2, answers={"R003_FACE": "NO"}, default_dim=4)
    assert d.recommendation is Recommendation.REJECT and d.quality.overall_score == 100


# --------------------------------------------------------------------------------------------------------------------- metrics
def test_headline_metrics_on_a_tiny_fixture(rb):
    p = rb.resolve(V2)
    good = evaluate_final_decision(make_assessment(p), p, verified_metadata())
    bad = evaluate_final_decision(make_assessment(p, answers={"R003_FACE": "NO"}), p, verified_metadata())
    dec = {"a1": good, "a2": bad, "d1": good, "d2": bad}
    approved = {"a1": True, "a2": True, "d1": False, "d2": False}
    h = headline(dec, approved)
    assert h["confusion"] == {"TP": 1, "FN": 1, "TN": 1, "FP": 1}
    assert h["false_rejection_rate"]["k"] == 1 and h["false_approval_rate"]["k"] == 1
    assert h["approval_precision"]["rate"] == 0.5 and h["rejection_precision"]["rate"] == 0.5 and h["rejection_recall"]["rate"] == 0.5
    assert h["false_approval_ids"] == ["d1"] and h["false_rejection_ids"] == ["a2"]
    t = transitions(dec, {"a1": good, "a2": good, "d1": bad, "d2": good}, approved)
    assert t["rescued_approved"] == ["a2"] and t["new_false_approvals"] == ["d2"] and t["newly_stopped_disapproved"] == ["d1"]


def test_statistics_helpers():
    assert auc([3, 4], [1, 2]) == 1.0 and auc([1, 1], [1, 1]) == 0.5 and auc([], [1]) is None
    lo, hi = wilson(0, 25)
    assert lo == 0.0 and 0.1 < hi < 0.2
    assert wilson(0, 0) == (None, None)
    assert "APPROVE" in AUTO
