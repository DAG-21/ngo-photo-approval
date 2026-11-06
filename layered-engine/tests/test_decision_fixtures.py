"""The fifteen edge-case photographs from the brief, as SYNTHETIC assessments (what a correct model would report) with the decision Python must reach.

These test the rule engine. They are not photographs, and they do not show that Gemini will report these assessments for a real photo.
"""

from __future__ import annotations

import pytest

from layered_engine.decision_engine import evaluate_final_decision
from layered_engine.schemas import Recommendation as R

from conftest import make_assessment, verified_metadata

EDU = dict(f_activity_type="EDUCATION_SUPPORT", f_item_given="NO", f_handover="NO", f_volunteer_giver="NO", f_banner="NO")
MED = dict(f_activity_type="HEALTHCARE_ASSISTANCE", f_medical="YES", f_item_given="NO", f_handover="NO", f_banner="NO")
ANIMAL = dict(f_activity_type="ANIMAL_CARE", f_beneficiaries="NO", f_animals="YES", f_item_given="NO", f_handover="NO", f_banner="NO")
NOBEN = dict(f_beneficiaries="NO", f_item_given="NO", f_handover="NO", f_banner="NO")
GOOD, MIN = verified_metadata(), verified_metadata()

# name -> (builder kwargs, metadata, expected decision in layered_proposed_v1)
CASES = {
    "01 clear food distribution": (dict(), GOOD, R.APPROVE),
    "02 education photo with no handover": (dict(facts=EDU, answers={"R015_BEN_RECEIVING": "NO"}), GOOD, R.APPROVE),
    "03 healthcare photo with masks (PPE only)": (dict(facts=MED, answers={"R003_CLEAR": "NO", "R003_OBSTRUCTION_PPE_ONLY": "YES", "R015_BEN_RECEIVING": "YES"}), GOOD, R.HUMAN_REVIEW),
    "04 animal-care photograph": (dict(facts=ANIMAL), GOOD, R.APPROVE),
    "05 no beneficiary (volunteers only)": (dict(facts=NOBEN), GOOD, R.APPROVE),
    "06 one of several beneficiaries has an obscured face": (dict(answers={"R003_FACE": "NO"}), GOOD, R.REJECT),
    "07 volunteers photographed from behind (a logo on their backs does not help)": (dict(facts={"f_orientation": "BACK"}, answers={"R008_BACK": "NO"}), GOOD, R.REJECT),
    "08 no banner": (dict(facts={"f_banner": "NO"}, answers={"R001_FLAT": "NO"}), GOOD, R.APPROVE),
    "09 badly crumpled prominent banner": (dict(answers={"R001_FLAT": "NO"}), GOOD, R.REJECT),
    "10 good composition but a blurred item being given": (dict(answers={"R006_ITEM_SHARP": "NO"}), GOOD, R.REJECT),
    "11 clear subject but the face is poorly lit": (dict(answers={"R009_FACE_LIT": "NO"}), GOOD, R.REJECT),
    "12 visible minor, consent not yet verified": (dict(flags=[{"flag_type": "POSSIBLE_MINOR", "evidence": "a child is identifiable", "confidence": "HIGH"}]), GOOD, R.APPROVED_PENDING_PUBLICATION_CHECKS),
    "13 exposes sensitive personal information": (dict(flags=[{"flag_type": "CONFIDENTIAL_INFORMATION", "evidence": "a readable ID card", "confidence": "HIGH"}]), GOOD, R.HUMAN_REVIEW),
    "14 technically excellent but degrading": (dict(flags=[{"flag_type": "DEGRADING_REPRESENTATION", "evidence": "a vulnerable person shown in distress", "confidence": "HIGH"}], default_dim=4), GOOD, R.HUMAN_REVIEW),
    "15 passes every gate but weak editorially": (dict(dims={"naturalness_storytelling": 1}, default_dim=3), GOOD, R.HUMAN_REVIEW),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_edge_case_decisions(proposed, name):
    kwargs, meta, expected = CASES[name]
    meta = dict(meta)
    if name.startswith("12"):
        meta["child_safeguarding_status"] = None  # not yet verified
    d = evaluate_final_decision(make_assessment(proposed, **kwargs), proposed, meta)
    assert d.recommendation is expected, (name, d.recommendation, d.decision_reasons)


def test_case_14_and_13_name_their_blocker_in_the_explanation(proposed):
    for name, word in (("13 exposes sensitive personal information", "NEW-003"), ("14 technically excellent but degrading", "NEW-004")):
        kwargs, meta, _ = CASES[name]
        d = evaluate_final_decision(make_assessment(proposed, **kwargs), proposed, meta)
        assert any(word in r for r in d.decision_reasons) and d.quality.overall_score == 100  # a perfect score did not rescue it


def test_the_same_fixtures_under_the_legacy_profile_differ_exactly_where_the_policy_differs(legacy):
    # legacy has no publication layer, no activity-dependent handover and no PPE exception
    outcome = {}
    for name in ("02 education photo with no handover", "03 healthcare photo with masks (PPE only)", "12 visible minor, consent not yet verified"):
        kwargs, meta, _ = CASES[name]
        outcome[name] = evaluate_final_decision(make_assessment(legacy, **kwargs), legacy, meta).recommendation
    assert outcome["02 education photo with no handover"] is R.REJECT       # strict original G12
    assert outcome["03 healthcare photo with masks (PPE only)"] is R.REJECT  # no PPE exception in legacy
    assert outcome["12 visible minor, consent not yet verified"] is R.APPROVE  # legacy does not evaluate publication readiness


def test_decisions_are_reproducible_and_carry_their_versions(proposed):
    a = make_assessment(proposed, answers={"R003_FACE": "NO"})
    d1 = evaluate_final_decision(a, proposed, {}, versions={"prompt_hash": "abc"})
    d2 = evaluate_final_decision(a, proposed, {}, versions={"prompt_hash": "abc"})
    assert d1.model_dump() == d2.model_dump() and d1.versions["profile"] == "layered_proposed_v1" and d1.versions["prompt_hash"] == "abc"
    assert d1.policy_validation_required  # the proposed profile contains unapproved changes to mandatory requirements


def test_input_validation_is_a_separate_policy(rb, tmp_path):
    import shutil

    import yaml

    from layered_engine.rulebook import RULEBOOK_DIR, Rulebook

    d = tmp_path / "rb"
    shutil.copytree(RULEBOOK_DIR, d)
    p = yaml.safe_load((d / "policy_profiles.yaml").read_text(encoding="utf-8"))
    p["profiles"]["layered_proposed_v1"]["upload_policy"] = {"require_photo": True, "require_single": True}
    (d / "policy_profiles.yaml").write_text(yaml.safe_dump(p), encoding="utf-8")
    strict = Rulebook(d).resolve("layered_proposed_v1")
    # by default F_PHOTO and F_SINGLE are facts, not publication-rejection rules
    assert evaluate_final_decision(make_assessment(strict, facts={"f_photo": "NO"}), strict, GOOD).recommendation is R.REJECT
    assert evaluate_final_decision(make_assessment(strict, facts={"f_single": "NO"}), strict, GOOD).recommendation is R.REJECT
    assert evaluate_final_decision(make_assessment(strict, facts={"f_photo": "UNKNOWN"}), strict, GOOD).recommendation is R.HUMAN_REVIEW
    default = rb.resolve("layered_proposed_v1")
    assert evaluate_final_decision(make_assessment(default, facts={"f_photo": "NO"}), default, GOOD).recommendation is R.APPROVE
