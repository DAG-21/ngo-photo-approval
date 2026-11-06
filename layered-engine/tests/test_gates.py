from __future__ import annotations

import shutil

import pytest
import yaml

from layered_engine.decision_engine import evaluate_final_decision
from layered_engine.rulebook import RULEBOOK_DIR, Rulebook
from layered_engine.schemas import Recommendation as R

from conftest import LEGACY, PROPOSED, make_assessment, verified_metadata

GATES = {  # gate id -> (criterion, a fact change that makes the gate NOT applicable, or None when it always applies)
    "G001": ("R007_ANGLE", None), "G002": ("R007_FACES_SEEN", {"f_beneficiaries": "NO"}), "G003": ("R006_ITEM_SHARP", {"f_item_given": "NO"}),
    "G004": ("R003_FACE", {"f_beneficiaries": "NO"}), "G005": ("R003_CLEAR", {"f_beneficiaries": "NO"}), "G006": ("R003_FACE_SHARP", {"f_beneficiaries": "NO"}),
    "G007": ("R003_FACE_SIZE", {"f_beneficiaries": "NO"}), "G008": ("R009_FACE_LIT", {"f_beneficiaries": "NO"}), "G009": ("R001_FLAT", {"f_banner": "NO"}),
    "G010": ("R006_COVERAGE", None), "G011": ("R006_ZOOMED_OUT", None), "G012": ("R015_BEN_RECEIVING", {"f_beneficiaries": "NO"}),
    "G013": ("R002_GIVER_WHITE", {"f_volunteer_giver": "NO"}), "G014": ("R015_PERSON_PRESENT", None), "G015": ("R004_BEN_GAZE", {"f_beneficiaries": "NO"}),
}


def decide(profile, assessment, meta=None):
    return evaluate_final_decision(assessment, profile, meta or {})


# ------------------------------------------------------------------------------------------- every gate, in the legacy profile
@pytest.mark.parametrize("gate", sorted(GATES))
def test_legacy_a_no_on_any_applicable_gate_rejects(legacy, gate):
    crit, _ = GATES[gate]
    d = decide(legacy, make_assessment(legacy, answers={crit: "NO"}))
    assert d.recommendation is R.REJECT and gate in d.failed_gates


@pytest.mark.parametrize("gate", sorted(GATES))
def test_legacy_cannot_tell_on_any_applicable_gate_goes_to_a_human(legacy, gate):
    crit, _ = GATES[gate]
    d = decide(legacy, make_assessment(legacy, answers={crit: "CANNOT_TELL"}))
    assert d.recommendation is R.HUMAN_REVIEW and gate in d.uncertain_gates and not d.failed_gates  # unknown is never a pass


@pytest.mark.parametrize("gate", [g for g in sorted(GATES) if GATES[g][1]])
def test_legacy_not_applicable_is_skipped_never_a_failure(legacy, gate):
    crit, facts = GATES[gate]
    d = decide(legacy, make_assessment(legacy, facts=facts, answers={crit: "NO"}))  # even a NO is ignored when the facts say it does not apply
    assert gate in d.not_applicable_gates and gate not in d.failed_gates
    g = next(x for x in d.gates if x.gate_id == gate)
    assert g.applicable is False and any("advisory_ignored" in n for n in g.notes)


def test_a_clean_photo_passes_every_gate(legacy, proposed):
    for prof in (legacy, proposed):
        d = decide(prof, make_assessment(prof), verified_metadata())
        assert d.recommendation is R.APPROVE and not d.failed_gates and not d.uncertain_gates


# ------------------------------------------------------------------------------------------- applicability is owned by Python
def test_unknown_facts_never_switch_a_gate_off(legacy):
    a = make_assessment(legacy, facts={"f_beneficiaries": "UNKNOWN"}, answers={"R003_FACE": "NO"})
    assert decide(legacy, a).recommendation is R.REJECT  # unknown applicability = applicable
    b = make_assessment(legacy, facts={"f_beneficiaries": "UNKNOWN"}, answers={"R003_FACE": "NOT_APPLICABLE"})
    d = decide(legacy, b)
    assert d.recommendation is R.HUMAN_REVIEW and any("na_conflict" in w for w in d.warnings)


def test_a_missing_gate_answer_is_never_a_pass(legacy):
    d = decide(legacy, make_assessment(legacy, omit=("R003_FACE",)))
    assert d.recommendation is R.HUMAN_REVIEW and "G004" in d.uncertain_gates and any("synthesized" in w for w in d.warnings)


def test_the_model_cannot_declare_an_applicable_gate_not_applicable(legacy):
    d = decide(legacy, make_assessment(legacy, answers={"R006_ITEM_SHARP": "NOT_APPLICABLE"}))
    assert d.recommendation is R.HUMAN_REVIEW and "G003" in d.uncertain_gates


# ------------------------------------------------------------------------------------------- conditional applicability
ANIMALS = dict(f_activity_type="ANIMAL_CARE", f_volunteers="NO", f_beneficiaries="NO", f_animals="YES", f_item_given="NO", f_handover="NO",
               f_volunteer_giver="NO", f_banner="NO")
EDU = dict(f_activity_type="EDUCATION_SUPPORT", f_item_given="NO", f_handover="NO", f_volunteer_giver="NO", f_banner="NO")


def test_an_education_photo_with_no_handover(legacy, proposed):
    a = lambda p: make_assessment(p, facts=EDU, answers={"R015_BEN_RECEIVING": "NO"})  # noqa: E731
    leg = decide(legacy, a(legacy))
    assert leg.recommendation is R.REJECT and "G012" in leg.failed_gates  # legacy keeps the strict original gate
    pro = decide(proposed, a(proposed), verified_metadata())
    assert pro.recommendation is R.APPROVE and "G012" in pro.not_applicable_gates and "G003" in pro.not_applicable_gates


def test_a_distribution_activity_still_needs_the_handover_in_the_proposed_profile(proposed):
    d = decide(proposed, make_assessment(proposed, answers={"R015_BEN_RECEIVING": "NO"}), verified_metadata())
    assert d.recommendation is R.REJECT and "G012" in d.failed_gates


def test_animal_care_and_photos_without_beneficiaries(legacy, proposed):
    facts = dict(f_activity_type="ANIMAL_CARE", f_beneficiaries="NO", f_animals="YES", f_item_given="NO", f_handover="NO", f_banner="NO")
    for prof in (legacy, proposed):
        d = decide(prof, make_assessment(prof, facts=facts, answers={"R003_FACE": "NO", "R015_BEN_RECEIVING": "NO"}), verified_metadata())
        assert d.recommendation is R.APPROVE, (prof.name, d.decision_reasons)  # beneficiary gates do not apply to animals
        assert {"G004", "G012", "G015"} <= set(d.not_applicable_gates)
    # the original human-subject requirement is a configurable option: an empty scene is rejected under both profiles
    nobody = make_assessment(proposed, facts=dict(facts, f_volunteers="NO", f_volunteer_giver="NO"), answers={"R015_PERSON_PRESENT": "NO"})
    assert decide(proposed, nobody, verified_metadata()).recommendation is R.REJECT


def test_g13_the_giving_volunteer_must_wear_white(legacy):
    assert "G013" in decide(legacy, make_assessment(legacy, answers={"R002_GIVER_WHITE": "NO"})).failed_gates
    assert "G013" in decide(legacy, make_assessment(legacy, facts={"f_volunteer_giver": "NO"}, answers={"R002_GIVER_WHITE": "NO"})).not_applicable_gates
    # the full client dress code (R002_WHITE) is a separate, non-gate criterion: failing it alone does not reject
    assert decide(legacy, make_assessment(legacy, answers={"R002_WHITE": "NO"})).recommendation is R.APPROVE


def test_g14_subject_presence(legacy):
    d = decide(legacy, make_assessment(legacy, facts=dict(f_beneficiaries="NO", f_volunteers="NO", f_volunteer_giver="NO"), answers={"R015_PERSON_PRESENT": "NO"}))
    assert d.recommendation is R.REJECT and "G014" in d.failed_gates


# ------------------------------------------------------------------------------------------- G15 and the direct-gaze exception
def test_g15_every_direct_gaze_by_a_beneficiary_is_disapproved_for_now(legacy, proposed):
    gaze = {"R004_BEN_GAZE": "NO"}
    for prof in (legacy, proposed):
        assert decide(prof, make_assessment(prof, answers=gaze), verified_metadata()).recommendation is R.REJECT
        # no exception exists for now: not even supplied metadata can allow a beneficiary's direct gaze
        d = decide(prof, make_assessment(prof, answers=gaze), verified_metadata(direct_gaze_approved=True))
        assert d.recommendation is R.REJECT and d.failed_gates == ["G015"]


def test_a_volunteer_looking_at_the_camera_is_not_a_gate(legacy):
    assert "R004_VOL_GAZE" not in legacy.criteria  # removed in checklist5: only beneficiaries are gated


# ------------------------------------------------------------------------------------------- back views are disallowed (G016)
BACK = dict(f_orientation="BACK")


@pytest.mark.parametrize("crit", ["R008_BACK", "T_FRONT", "R015_BEN_FRONT"])
def test_g016_a_back_view_is_rejected_whichever_component_catches_it(proposed, crit):
    d = decide(proposed, make_assessment(proposed, facts=BACK, answers={crit: "NO"}), verified_metadata())
    assert d.recommendation is R.REJECT and "G016" in d.failed_gates
    u = decide(proposed, make_assessment(proposed, facts=BACK, answers={crit: "CANNOT_TELL"}), verified_metadata())
    assert u.recommendation is R.HUMAN_REVIEW and "G016" in u.uncertain_gates


def test_g016_there_is_no_logo_exception_any_more(proposed):
    assert not any(e["id"] == "EXC_BACK_VIEW" for e in proposed.exceptions)
    assert not {"T_BACK_EXC_LOGO", "T_BACK_EXCEPTION"} & set(proposed.criteria)  # the questions of the withdrawn exception are not even asked
    d = decide(proposed, make_assessment(proposed, facts=BACK, answers={"R008_BACK": "NO", "T_BACK_EXCEPTION": "YES"}), verified_metadata())
    assert d.recommendation is R.REJECT  # a logo on the back, a dress code and visible beneficiaries do not rescue it


def test_g016_front_and_side_views_pass_and_it_is_skipped_without_people(proposed, legacy):
    assert decide(proposed, make_assessment(proposed, facts={"f_orientation": "SIDE"}), verified_metadata()).recommendation is R.APPROVE
    d = decide(proposed, make_assessment(proposed, facts=ANIMALS), verified_metadata())
    assert d.recommendation is R.APPROVE and "G016" in d.not_applicable_gates
    # the legacy profile is unchanged: back-view questions are reported, not gates
    assert decide(legacy, make_assessment(legacy, facts=BACK, answers={"R008_BACK": "NO", "T_FRONT": "NO"})).recommendation is R.APPROVE
    assert "G016" not in {g.gate_id for g in legacy.gates}


# ------------------------------------------------------------------------------------------- banner
def test_banners_are_optional_but_must_be_visible_distinguishable_and_uncreased_when_present(legacy, proposed):
    no_banner = dict(f_banner="NO", f_banner_prominent="NO")
    for crit in ("R001_FLAT", "R001_VISIBLE", "R001_DISTINCT"):
        d = decide(proposed, make_assessment(proposed, facts=no_banner, answers={crit: "NO"}), verified_metadata())
        assert d.recommendation is R.APPROVE and {"G009", "G017", "G018"} <= set(d.not_applicable_gates), crit  # no banner: nothing to judge
    for gate, crit in (("G009", "R001_FLAT"), ("G017", "R001_VISIBLE"), ("G018", "R001_DISTINCT")):
        for prominent in ("YES", "NO"):  # a banner that is merely present is judged too (the prominence narrowing was withdrawn)
            d = decide(proposed, make_assessment(proposed, facts={"f_banner_prominent": prominent}, answers={crit: "NO"}), verified_metadata())
            assert d.recommendation is R.REJECT and gate in d.failed_gates, (gate, prominent)
        u = decide(proposed, make_assessment(proposed, answers={crit: "CANNOT_TELL"}), verified_metadata())
        assert u.recommendation is R.HUMAN_REVIEW and gate in u.uncertain_gates
    # legacy is unchanged: only the crease gate exists; 'visible' is a diagnostic and 'distinguishable' is not asked at all
    assert "G009" in decide(legacy, make_assessment(legacy, answers={"R001_FLAT": "NO"})).failed_gates
    assert decide(legacy, make_assessment(legacy, answers={"R001_VISIBLE": "NO"})).recommendation is R.APPROVE
    assert "R001_DISTINCT" not in legacy.criteria


def test_animals_receiving_care_count_as_a_subject(proposed):
    d = decide(proposed, make_assessment(proposed, facts=ANIMALS), verified_metadata())
    assert d.recommendation is R.APPROVE and next(g for g in d.gates if g.gate_id == "G014").status.value == "YES"
    assert "animal receiving care" in proposed.criteria["R015_PERSON_PRESENT"].description_operational
    # with neither a person nor an animal, the model answers NO and the photo is rejected
    nothing = make_assessment(proposed, facts=dict(ANIMALS, f_animals="NO"), answers={"R015_PERSON_PRESENT": "NO"})
    assert decide(proposed, nothing, verified_metadata()).recommendation is R.REJECT


# ------------------------------------------------------------------------------------------- G10 and G11: legacy vs proposed
def test_coverage_and_zoomed_out_are_hard_in_legacy_and_soft_in_proposed(legacy, proposed):
    ans = {"R006_COVERAGE": "NO", "R006_ZOOMED_OUT": "NO"}
    leg = decide(legacy, make_assessment(legacy, answers=ans))
    assert leg.recommendation is R.REJECT and set(leg.failed_gates) == {"G010", "G011"}
    pro = decide(proposed, make_assessment(proposed, answers=ans), verified_metadata())
    assert pro.recommendation is R.APPROVE and not pro.failed_gates
    assert len(pro.quality.caps_applied) == 1 and pro.quality.dimension_scores["activity_communication"] == 2  # one defect, one cap
    assert pro.quality.overall_score == 87.5


# ------------------------------------------------------------------------------------------- G1 split into components
def test_g1_parent_gate_passes_only_when_every_applicable_component_passes(proposed):
    comp = ["G001_ACTIVITY_VISIBLE", "G001_BENEFICIARIES_VISIBLE", "G001_INTERACTION_VISIBLE", "G001_NOTHING_HIDDEN"]
    assert decide(proposed, make_assessment(proposed), verified_metadata()).recommendation is R.APPROVE
    for c in comp:
        d = decide(proposed, make_assessment(proposed, answers={c: "NO"}), verified_metadata())
        assert d.recommendation is R.REJECT and "G001" in d.failed_gates, c
        u = decide(proposed, make_assessment(proposed, answers={c: "CANNOT_TELL"}), verified_metadata())
        assert u.recommendation is R.HUMAN_REVIEW, c
    # the old single question no longer decides the parent gate in the layered profile
    assert decide(proposed, make_assessment(proposed, answers={"R007_ANGLE": "NO"}), verified_metadata()).recommendation is R.APPROVE
    # components that do not apply are skipped
    d = decide(proposed, make_assessment(proposed, facts={"f_beneficiaries": "NO"}, answers={"G001_BENEFICIARIES_VISIBLE": "NO"}), verified_metadata())
    assert "G001" not in d.failed_gates


# ------------------------------------------------------------------------------------------- medical protective equipment
def test_medical_ppe_is_not_an_automatic_violation_but_is_not_waved_through(proposed):
    med = dict(f_activity_type="HEALTHCARE_ASSISTANCE", f_medical="YES", f_item_given="NO", f_handover="NO")
    ans = {"R003_CLEAR": "NO", "R003_OBSTRUCTION_PPE_ONLY": "YES", "R015_BEN_RECEIVING": "YES"}
    d = decide(proposed, make_assessment(proposed, facts=med, answers=ans), verified_metadata())
    assert d.recommendation is R.HUMAN_REVIEW and "G005" in d.uncertain_gates  # escalated to a human
    other = decide(proposed, make_assessment(proposed, facts=med, answers={**ans, "R003_OBSTRUCTION_PPE_ONLY": "NO"}), verified_metadata())
    assert other.recommendation is R.REJECT  # a hand over the face is still a violation
    not_medical = decide(proposed, make_assessment(proposed, facts={**med, "f_medical": "NO"}, answers=ans), verified_metadata())
    assert not_medical.recommendation is R.REJECT  # the exception exists only in a medical scene


def test_the_legacy_profile_never_applies_proposed_exceptions(legacy):
    assert legacy.exceptions == []
