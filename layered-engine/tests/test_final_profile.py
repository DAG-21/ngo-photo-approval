"""layered_final_v1: the owner's rule-by-rule review of 2026-11-05 applied on top of layered_tuned_v1. No model is called.

They pin: the earlier profiles are untouched (prompt hashes, gates, HUMAN_REVIEW behaviour); the final profile differs from layered_tuned_v1 in exactly the agreed places (rules removed,
reworded and added, the eleven gates that can reject and the two soft gates, the facts kept); every agreed wording; that the built prompt carries no statement that contradicts another
(no guidance lines, no "role only" sentence, no removed rule or fact); the AI always gives a verdict (no HUMAN_REVIEW); the soft gates' degree; and that a fact a profile does not
ask is not required while a fact it does ask still is."""

from __future__ import annotations

import json
import re

import pytest

from conftest import FOOD, PROPOSED, make_assessment, response_text, verified_metadata
from layered_engine.annotation import parse_response
from layered_engine.compact import build_compact_schema
from layered_engine.decision_engine import evaluate_final_decision
from layered_engine.prompt_builder import asked_fact_fields, build_prompt
from layered_engine.rulebook import Rulebook, validate_rulebook
from layered_engine.schema_export import build_response_schema, check_provider_safe
from layered_engine.schemas import Recommendation

FINAL, TUNED = "layered_final_v1", "layered_tuned_v1"

REMOVED = {
    "R002_NEAT", "R002_WHITE", "R003_CLEAR", "R003_FACE_SIZE", "R004_BEN_GAZE", "R005_CANDID", "R005_INTERACT", "R006_COVERAGE", "R006_ZOOMED_OUT", "R006_HANDOVER", "R006_IDENTIFY",
    "R006_VOLUNTEER_DOING", "R007_FACES_SEEN", "G001_ACTIVITY_VISIBLE", "G001_BENEFICIARIES_VISIBLE", "G001_INTERACTION_VISIBLE", "G001_NOTHING_HIDDEN", "R007_FOREGROUND", "R007_HEIGHT",
    "R007_INTERACTION_VIEW", "R007_LEVEL", "R007_ORIENTATION", "R007_PERSPECTIVE", "R007_UNOBSTRUCTED", "R007_VERTICALS", "T_FRONT", "R009_FACE_LIT", "R009_DIRECTION", "R009_LIT",
    "R009_OVEREXPOSED", "R009_PEOPLE_LIT", "R009_UNDEREXPOSED", "R010_GRAIN", "R010_SHAKE", "R010_SHARP", "R011_CLUTTER", "R011_UNRELATED", "R015_BEN_FRONT", "R015_BEN_INTERACT",
    "R015_BEN_POSE", "R015_BEN_PRESENT", "R015_BEN_PROMINENT", "NEW-008", "NEW-009", "NEW-010", "NEW-011", "NEW-012", "NEW-013", "NEW-015", "NEW-016", "NEW-017", "NEW-018", "NEW-019",
    "NEW-020", "NEW-021", "NEW-022", "P004", "F_ANIMALS", "F_BANNER_PROMINENT", "F_GROUP", "F_HANDOVER", "F_LOGO", "F_ORIENTATION", "F_VOLUNTEER_GIVER"}
REJECTING = {"G001": "R007_ANGLE", "G003": "R006_ITEM_SHARP", "G004": "R003_FACE", "G006": "R003_FACE_SHARP", "G009": "R001_FLAT", "G012": "R015_BEN_RECEIVING",
             "G013": "R002_GIVER_WHITE", "G014": "R015_PERSON_PRESENT", "G016": "R008_BACK", "G017": "R001_VISIBLE", "G019": "R001_POSITION"}
ASKED = ["R001_POSITION", "R001_VISIBLE", "R001_FLAT", "R002_GIVER_WHITE", "R003_FACE", "R003_FACE_SHARP", "R003_CROP", "R003_OBSTRUCTION_PPE_ONLY", "R004_GENUINE", "R006_ITEM_SHARP",
         "R006_ACTION", "R006_SIZE", "R007_ANGLE", "R007_CUTOFF", "R007_DISTORTION", "R007_PLACEMENT", "R008_BACK", "R010_TECHNICAL", "R011_BACKGROUND", "R012_DIGNITY", "R013_HUMANITY",
         "R015_PERSON_PRESENT", "R015_BEN_RECEIVING", "NEW-014"]
FACTS_KEPT = {"F_PHOTO", "F_SINGLE", "F_BANNER", "F_VOLUNTEERS", "F_BENEFICIARIES", "F_ITEM_GIVEN", "F_MEDICAL", "F_ACTIVITY_TYPE"}
WORDING = {
    "R002_GIVER_WHITE": "Is every person who is giving something to a beneficiary wearing a white top?",
    "R003_FACE": "Is some part of every person's face visible, even if turned partly away: not covered, hidden or in shadow, and, for the main people, not too small to read?",
    "R003_FACE_SHARP": "Is every person's face sharp and in focus (not blurred by motion or poor focus)?",
    "R004_GENUINE": "Do the expressions and interactions look genuine, in a candid moment rather than staged or posed?",
    "R006_SIZE": "Are the main people and the seva activity clearly visible: large enough in the frame and not blocked from view?",
    "R006_ITEM_SHARP": ("If an item is being handed over, is the item sharp and in focus and clear enough to recognise? A sack or similar packed goods count as items, "
                        "and their contents need not be visible."),
    "R006_ACTION": "Is a seva activity actually taking place (for example giving food, teaching, treating, caring for an animal), not just people standing or waiting?",
    "R007_ANGLE": "Is the activity captured from roughly the same level as the people (a slightly higher or lower angle is fine, a very high or very low angle is not), with a level frame, so that the activity is clearly visible and not obstructed?",
    "R007_CUTOFF": "Is everything important fully inside the frame: no key person, hands or objects cut off at the edges or hidden by the camera angle?",
    "R007_PLACEMENT": "Is the main subject sensibly placed in the frame (not squeezed into a corner or edge) and free from obstruction by people or objects in front of it?",
    "R008_BACK": "Is every person seen from the front or the side, with some part of the face visible (nobody shown fully from behind)?",
    "R011_BACKGROUND": "Is the background free from clutter, scattered materials and unrelated objects that draw attention away from the seva?",
    "R015_BEN_RECEIVING": "If an item is being given, is at least one beneficiary receiving it (or already holding it)?",
    "R010_TECHNICAL": ("Is the photo technically sound: sharp and in focus, free from camera shake, motion blur, heavy grain and pixelation, and well exposed and well lit (not too dark, "
                       "not too bright, no washed-out or blacked-out areas, no harsh backlight, heavy shadows or glare hiding detail)?"),
    "F_ITEM_GIVEN": ("Is an item being handed over in this photo? An item is any substance, such as food, a packet, a sack or similar packed goods, medicine or a book, that"
                     + " is being given by a person's hands and received by a beneficiary's hands, whatever the giver is wearing. Only a visible hand-to-hand handover counts: people who are only interacting, standing together or talking, paperwork, forms, pamphlets or brochures, registration or checking of documents, and things lying on a table are not an item being handed over."),
    "F_VOLUNTEERS": "Is at least one volunteer visible in this photo? A volunteer is a person wearing a white top, or a person giving something to a beneficiary.",
}


@pytest.fixture(scope="module")
def rb():
    return Rulebook()


@pytest.fixture(scope="module")
def final(rb):
    return rb.resolve(FINAL)


def decide(rb, profile, **kw):
    p = rb.resolve(profile)
    return evaluate_final_decision(make_assessment(p, **kw), p, verified_metadata())


# ------------------------------------------------------------------------------------------------------------------------------ the profile itself
def test_the_rulebook_validates_and_the_earlier_profiles_are_untouched(rb):
    assert validate_rulebook(rb).ok
    assert build_prompt(rb, rb.resolve(PROPOSED)).hash == "11427c79d96c569a"
    assert build_prompt(rb, rb.resolve("checklist5_legacy")).hash == "0ac8d8c4c4000ada"
    assert build_prompt(rb, rb.resolve(TUNED)).hash == "185375c69d8cc3bb"
    assert len(rb.resolve(PROPOSED).hard_gates()) == len(rb.resolve(TUNED).hard_gates()) == 16
    assert "G002" in {g.gate_id for g in rb.resolve(PROPOSED).gates} and "R004_BEN_GAZE" in rb.resolve(TUNED).criteria and "no_human_review" not in rb.resolve(TUNED).settings


def test_the_final_profile_differs_from_the_tuned_one_in_exactly_the_agreed_places(rb, final):
    tuned = rb.resolve(TUNED)
    assert set(tuned.criteria) - set(final.criteria) == REMOVED
    assert set(final.criteria) - set(tuned.criteria) == {"R010_TECHNICAL"}
    assert sorted(c.criterion_id for c in final.asked_criteria()) == sorted(ASKED)
    assert {cid for cid, c in final.criteria.items() if c.decision_effect == "FACT" and cid.startswith("F_") and cid in {k.upper() for k in asked_fact_fields(final)} | FACTS_KEPT} >= FACTS_KEPT
    hard = {g.gate_id: g.criterion_id for g in final.hard_gates()}
    assert hard == REJECTING and len(hard) == 11
    soft = {g.gate_id: g.criterion_id for g in final.gates if g.gate_id not in hard}
    assert soft == {"G020": "R006_SIZE", "G021": "R010_TECHNICAL"}
    assert next(g for g in final.gates if g.gate_id == "G001").components == [] and next(g for g in final.gates if g.gate_id == "G016").components == ["R008_BACK"]
    assert final.settings["no_human_review"] is True and final.settings["guidance"] == [] and [e["id"] for e in final.exceptions] == ["EXC_MEDICAL_PPE_FINAL"]
    assert final.rulebook_version == "1.2.0-final-rules" and final.settings["unknown_is_fail"] is True
    for cid, c in final.criteria.items():                           # every other rule is exactly as it was in the tuned profile
        if cid in WORDING or cid in ("R010_TECHNICAL", "R002_GIVER_WHITE"):
            continue
        t = tuned.criteria[cid]
        assert (c.description_operational, c.decision_effect, c.applicability) == (t.description_operational, t.decision_effect, t.applicability) or cid in ("F_BENEFICIARIES", "F_VOLUNTEERS"), cid


def test_every_agreed_wording_is_in_the_profile(final):
    for cid, text in WORDING.items():
        assert final.criteria[cid].description_operational == text, cid
    assert final.criteria["F_BENEFICIARIES"].description_operational.startswith("Is at least one beneficiary visible in this photo? A beneficiary is any person not wearing a white top")
    assert "A person giving something to a beneficiary is a volunteer, whatever they wear. An animal can also be a beneficiary." in final.criteria["F_BENEFICIARIES"].description_operational
    assert final.criteria["R006_SIZE"].decision_effect == "SOFT_SCORE" and final.criteria["R010_TECHNICAL"].decision_effect == "SOFT_SCORE"
    for cid, cond in (("R002_GIVER_WHITE", "item_given"), ("R015_BEN_RECEIVING", "item_given"), ("R003_FACE", "people_visible"), ("R003_FACE_SHARP", "people_visible"), ("R008_BACK", "people_visible")):
        assert final.criteria[cid].applicability == cond, cid
    assert "animal receiving care" in final.criteria["R015_PERSON_PRESENT"].description_operational     # inherited from layered_proposed_v1 two steps up


# ------------------------------------------------------------------------------------------------------------------------------ the prompt
def test_the_prompt_carries_the_agreed_client_lines_and_no_contradiction(rb, final):
    text = build_prompt(rb, final).system_text
    client = [ln for ln in text.splitlines() if re.match(r"- R\d{3} - ", ln)]
    assert [ln.split(" - ")[0][2:] for ln in client] == ["R001", "R002", "R003", "R004", "R006", "R007", "R008", "R009", "R010", "R011"]
    assert "R002 - Volunteer Dress Code: Volunteers giving something to a beneficiary should wear white tops." in text
    assert "R004 - Natural Expressions: Capture genuine expressions and interactions." in text
    assert "Show people from the front or the side. Nobody should be shown fully from behind: some part of the face should be visible." in text
    assert "R003 - Beneficiary Visibility: Ensure the beneficiary" in text and "R009 - Proper Lighting: Ensure subjects are well-lit" in text    # the unchanged ones are verbatim
    for gone in ("ROLE in the activity only", "GUIDE", "looking straight into the camera", "NOT a violation", "IS a violation", "FRONT or SIDE of the people", "neat, clean", "white-on-white", "jeans",
                 "f_orientation", "f_group", "f_logo", "f_handover", "f_volunteer_giver", "f_animals", "f_banner_prominent", "HUMAN_REVIEW", "G002", "G005", "G007", "G008", "G015", "G018"):
        assert gone not in text, gone
    for cid in REMOVED:
        assert f"`{cid}`" not in text, cid
    assert "only by the definitions given in section B" in text
    ids = re.findall(r"^- `([A-Z0-9_\-]+)` \[", text, flags=re.M)
    assert sorted(ids) == sorted(ASKED) and len(ids) == len(set(ids))
    assert text.count("MANDATORY GATE") == 11 and "G020" not in text and "G021" not in text          # the soft gates are not announced as mandatory
    assert "0 = unusable (almost black, blown out or too blurred to make out the activity)" in text and "0 = the main people and the activity cannot be made out at all" in text


def test_the_compact_prompt_and_both_answer_schemas_leave_out_the_dropped_facts(rb, final):
    full = build_response_schema(final, build_prompt(rb, final).criterion_ids)
    facts = full["properties"]["layer_1_scene_understanding"]["properties"]["scene_facts"]
    assert set(facts["properties"]) == {"f_photo", "f_single", "f_volunteers", "f_banner", "f_beneficiaries", "f_item_given", "f_medical", "f_activity_type"} == set(facts["required"])
    assert not check_provider_safe(full)
    compact = build_compact_schema(final, build_prompt(rb, final, response_format="compact").criterion_ids)
    assert set(compact["properties"]["scene_facts"]["properties"]) == set(facts["properties"]) and not check_provider_safe(compact)
    older = build_response_schema(rb.resolve(TUNED), build_prompt(rb, rb.resolve(TUNED)).criterion_ids)["properties"]["layer_1_scene_understanding"]["properties"]["scene_facts"]
    assert len(older["properties"]) == 15 and set(older["required"]) == set(older["properties"])      # every earlier profile still asks (and requires) all fifteen


def test_a_fact_the_profile_does_not_ask_is_not_required_but_one_it_asks_still_is(rb, final):
    tuned = rb.resolve(TUNED)
    raw = json.loads(response_text(make_assessment(final)))
    for k in ("f_banner_prominent", "f_animals", "f_handover", "f_volunteer_giver", "f_logo", "f_orientation", "f_group"):
        raw["layer_1_scene_understanding"]["scene_facts"].pop(k)
    text = json.dumps(raw)
    assert parse_response(text, "x", final).layer_1_scene_understanding.scene_facts.f_logo is None
    for strict in (tuned, None):                                    # the earlier profiles, and a caller that names no profile, still require every fact
        with pytest.raises(ValueError, match="scene_facts"):
            parse_response(text, "x", strict)
    raw["layer_1_scene_understanding"]["scene_facts"].pop("f_item_given")
    with pytest.raises(ValueError, match="f_item_given"):
        parse_response(json.dumps(raw), "x", final)


# ------------------------------------------------------------------------------------------------------------------------------ decisions
def test_a_compliant_photo_is_approved_and_each_of_the_eleven_gates_rejects(rb, final):
    ok = decide(rb, FINAL)
    assert ok.recommendation is Recommendation.APPROVE and not ok.failed_gates
    for gid, cid in REJECTING.items():
        out = decide(rb, FINAL, answers={cid: "NO"})
        assert out.recommendation is Recommendation.REJECT and gid in out.failed_gates, (gid, out.recommendation)


def test_only_a_giver_not_in_white_rejects_the_dress_code_and_only_when_something_is_given(rb):
    assert "G013" in decide(rb, FINAL, answers={"R002_GIVER_WHITE": "NO"}).failed_gates
    nobody_giving = decide(rb, FINAL, answers={"R002_GIVER_WHITE": "NO", "R006_ITEM_SHARP": "NO", "R015_BEN_RECEIVING": "NO"}, facts={"f_item_given": "NO"})
    assert nobody_giving.recommendation is not Recommendation.REJECT and {"G013", "G003", "G012"} <= set(nobody_giving.not_applicable_gates)


def test_the_ai_always_gives_a_verdict_where_the_older_profiles_asked_for_a_human(rb):
    unsure = {"R003_FACE": "CANNOT_TELL"}
    old = decide(rb, TUNED, answers=unsure)
    assert old.recommendation is Recommendation.HUMAN_REVIEW
    new = decide(rb, FINAL, answers=unsure)
    assert new.recommendation is Recommendation.REJECT and any("could not confirm" in r for r in new.decision_reasons)
    borderline = {"activity_communication": 3, "technical_quality": 3, "composition": 2, "naturalness_storytelling": 2, "brand_presentation": 2, "visual_polish": 2}   # about 61 of 100
    assert decide(rb, TUNED, dims=borderline).recommendation is Recommendation.HUMAN_REVIEW
    low = decide(rb, FINAL, dims=borderline)
    assert low.recommendation is Recommendation.RESELECT and any("below the cutoff" in r for r in low.decision_reasons)
    flag = [{"flag_type": "DEGRADING_REPRESENTATION", "evidence": "x", "confidence": "HIGH"}]
    assert decide(rb, TUNED, flags=flag).recommendation is Recommendation.HUMAN_REVIEW
    assert decide(rb, FINAL, flags=flag).recommendation is Recommendation.REJECT
    for profile in (FINAL,):
        for kw in ({"answers": unsure}, {"dims": borderline}, {"flags": flag}, {}):
            assert decide(rb, profile, **kw).recommendation is not Recommendation.HUMAN_REVIEW


def test_the_two_soft_gates_lower_a_score_and_the_grade_gives_the_degree(rb):
    for cid, dim in (("R010_TECHNICAL", "technical_quality"), ("R006_SIZE", "activity_communication")):
        clean = decide(rb, FINAL)
        mild = decide(rb, FINAL, answers={cid: "NO"}, dims={dim: 2})              # a defect the model grades as acceptable: the score drops, the photograph is not rejected
        assert mild.recommendation is Recommendation.APPROVE and not mild.failed_gates and mild.quality.overall_score < clean.quality.overall_score
        capped = decide(rb, FINAL, answers={cid: "NO"}, dims={dim: 4})            # a NO caps the dimension once, whatever grade the model gave
        assert capped.quality.dimension_scores[dim] == 2 and any(cid in c for c in capped.quality.caps_applied)
        poor = decide(rb, FINAL, answers={cid: "NO"}, dims={dim: 1})
        assert poor.recommendation is Recommendation.RESELECT
        stop = decide(rb, FINAL, answers={cid: "NO"}, dims={dim: 0})
        assert stop.recommendation is Recommendation.RESELECT and any("unusable" in r.lower() for r in stop.decision_reasons)


def test_a_face_covered_only_by_protective_equipment_in_a_medical_scene_is_not_waved_through(rb):
    out = decide(rb, FINAL, answers={"R003_FACE": "NO", "R003_OBSTRUCTION_PPE_ONLY": "YES"}, facts={"f_medical": "YES"})
    assert out.recommendation is Recommendation.REJECT and any("could not confirm" in r for r in out.decision_reasons) and "G004" in out.uncertain_gates
    assert "G004" in decide(rb, FINAL, answers={"R003_FACE": "NO", "R003_OBSTRUCTION_PPE_ONLY": "YES"}).failed_gates          # not a medical scene: no exception


def test_the_dropped_facts_play_no_part_in_the_decision(rb):
    a = decide(rb, FINAL)
    b = decide(rb, FINAL, facts={"f_orientation": "BACK", "f_group": "CROWD", "f_logo": "NO", "f_animals": "YES", "f_handover": "NO", "f_banner_prominent": "NO", "f_volunteer_giver": "NO"})
    assert a.recommendation is b.recommendation is Recommendation.APPROVE and a.failed_gates == b.failed_gates == []
