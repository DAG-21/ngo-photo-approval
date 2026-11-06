"""layered_tuned_v1: layered_proposed_v1 with the banner rules tuned by the project owner (2026-11-05). No model is called.

They pin: the original profile is untouched; the tuned profile differs from it in exactly the intended places and nowhere else; the banner rules are asked in the agreed order with the
agreed wording; the new in-frame rule rejects a photograph whatever else is true, and only when a banner is present; the old profiles' prompts are unchanged; and `inherits` really
inherits (items active in the parent are active in the child)."""

from __future__ import annotations

import pytest

from conftest import PROPOSED, make_assessment, verified_metadata
from layered_engine.decision_engine import evaluate_final_decision
from layered_engine.prompt_builder import build_prompt
from layered_engine.rulebook import Rulebook, validate_rulebook
from layered_engine.schemas import Recommendation

TUNED = "layered_tuned_v1"
POSITION = "Is the entire banner, judged by its assumed boundary even where people or objects hide part of it, within the frame of the photo?"
VISIBLE = "If a banner is present, is at least 30% of it visible and readable? Blocking by people and minor creases are fine."
FLAT = "Is the banner free from crumpling so severe that its words cannot be read? Ordinary folds, creases and soft ripples are fine."
CHANGED = {"R001_POSITION", "R001_VISIBLE", "R001_FLAT"}


@pytest.fixture(scope="module")
def rb():
    return Rulebook()


def decide(rb, profile, **kw):
    p = rb.resolve(profile)
    return evaluate_final_decision(make_assessment(p, **kw), p, verified_metadata())


def test_the_rulebook_with_the_new_profile_validates_and_the_originals_are_untouched(rb):
    assert validate_rulebook(rb).ok
    assert build_prompt(rb, rb.resolve(PROPOSED)).hash == "11427c79d96c569a"
    assert build_prompt(rb, rb.resolve("checklist5_legacy")).hash == "0ac8d8c4c4000ada"


def test_the_tuned_profile_differs_from_the_original_in_exactly_the_banner_group(rb):
    old, new = rb.resolve(PROPOSED), rb.resolve(TUNED)
    assert set(old.criteria) - set(new.criteria) == {"R001_DISTINCT", "R001_STRAIGHT"} and not set(new.criteria) - set(old.criteria)
    assert {g.gate_id for g in old.gates} - {g.gate_id for g in new.gates} == {"G018"} and {g.gate_id for g in new.gates} - {g.gate_id for g in old.gates} == {"G019"}
    assert len(old.hard_gates()) == len(new.hard_gates()) == 16
    for cid, c in new.criteria.items():                                 # every other rule is exactly as it was
        if cid not in CHANGED:
            assert c.model_dump() == old.criteria[cid].model_dump(), cid
    for g in new.gates:
        if g.gate_id != "G019":
            og = next(x for x in old.gates if x.gate_id == g.gate_id)
            assert (g.criterion_id, g.components, g.forced_hard) == (og.criterion_id, og.components, og.forced_hard), g.gate_id
    assert [e["id"] for e in new.exceptions] == [e["id"] for e in old.exceptions] and new.taxonomy == old.taxonomy and new.quality == old.quality
    assert new.rulebook_version == "1.1.0-tuned-banner" and new.settings["unknown_is_fail"] is True


def test_the_banner_rules_are_asked_in_the_agreed_order_with_the_agreed_wording(rb):
    new = rb.resolve(TUNED)
    banner = [c for c in sorted(new.asked_criteria(), key=lambda c: (c.priority, c.criterion_id)) if c.parent_rule_id == "R001"]
    assert [c.criterion_id for c in banner] == ["R001_POSITION", "R001_VISIBLE", "R001_FLAT"]
    assert [c.description_operational for c in banner] == [POSITION, VISIBLE, FLAT]
    text = build_prompt(rb, new).system_text
    lines = [ln for ln in text.splitlines() if ln.startswith("- `R001_")]
    assert [ln.split("`")[1] for ln in lines] == ["R001_POSITION", "R001_VISIBLE", "R001_FLAT"]
    assert "MANDATORY GATE G019" in lines[0] and "MANDATORY GATE G017" in lines[1] and "MANDATORY GATE G009" in lines[2] and "(NOT_APPLICABLE only if no banner is present)" in lines[0]
    assert "R001_DISTINCT" not in text and "R001_STRAIGHT" not in text and "distinguishable from its surroundings" not in text
    assert "its whole boundary must be inside the frame of the photograph and at least 30% of it must be visible and readable" in text
    assert "clearly visible and clearly distinguishable" not in text      # no guidance line contradicts the new rules


def test_a_banner_that_is_not_entirely_in_the_frame_rejects_the_photograph_whatever_else_is_true(rb):
    ok = decide(rb, TUNED)
    assert ok.recommendation is not Recommendation.REJECT and "G019" not in ok.failed_gates
    out = decide(rb, TUNED, answers={"R001_POSITION": "NO"})
    assert out.recommendation is Recommendation.REJECT and out.failed_gates == ["G019"]
    assert decide(rb, TUNED, answers={"R001_POSITION": "NO"}, dims={d: 4 for d in rb.resolve(TUNED).quality["dimensions"]}).recommendation is Recommendation.REJECT


def test_the_in_frame_rule_applies_only_when_a_banner_is_present_and_a_doubtful_answer_goes_to_a_person(rb):
    no_banner = decide(rb, TUNED, answers={"R001_POSITION": "NO"}, facts={"f_banner": "NO", "f_banner_prominent": "NO"})
    assert "G019" in no_banner.not_applicable_gates and no_banner.recommendation is not Recommendation.REJECT      # no banner is never a failure
    unsure = decide(rb, TUNED, answers={"R001_POSITION": "CANNOT_TELL"})
    assert "G019" in unsure.uncertain_gates and unsure.recommendation is not Recommendation.APPROVE


def test_too_little_visible_banner_rejects_and_blocking_alone_does_not(rb):
    assert "G017" in decide(rb, TUNED, answers={"R001_VISIBLE": "NO"}).failed_gates
    assert decide(rb, TUNED).recommendation is not Recommendation.REJECT
    assert "G009" in decide(rb, TUNED, answers={"R001_FLAT": "NO"}).failed_gates


def test_the_original_profile_still_behaves_as_before_for_the_banner(rb):
    assert {"G017", "G018"} <= set(decide(rb, PROPOSED, answers={"R001_VISIBLE": "NO", "R001_DISTINCT": "NO"}).failed_gates)
    assert "G019" not in decide(rb, PROPOSED, answers={"R001_POSITION": "NO"}).failed_gates


def test_a_child_profile_inherits_what_is_active_in_its_parent_and_an_empty_list_removes_it(rb):
    assert "G016" in {g.gate_id for g in rb.resolve(TUNED).gates}      # active_in lists only the parent and v2: inherited, not lost
    v2 = rb.resolve("layered_proposed_v2")
    assert {"G016", "G017", "G018"} <= {g.gate_id for g in v2.gates} and "POST_ACTIVITY" in {a["id"] for a in v2.taxonomy} and "POST_ACTIVITY" not in {a["id"] for a in rb.resolve(TUNED).taxonomy}
    assert "R001_DISTINCT" not in rb.resolve(TUNED).criteria


def test_a_criterion_active_only_in_the_parent_is_active_in_the_child_but_not_in_an_unrelated_profile(tmp_path):
    import shutil

    import yaml

    from layered_engine.rulebook import RULEBOOK_DIR

    shutil.copytree(RULEBOOK_DIR, tmp_path / "rb")
    f = tmp_path / "rb" / "rulebook_v1.yaml"
    book = yaml.safe_load(f.read_text(encoding="utf-8"))
    next(c for c in book["criteria"] if c["criterion_id"] == "R002_NEAT")["active_in"] = [PROPOSED]       # the parent only
    f.write_text(yaml.safe_dump(book, sort_keys=False, allow_unicode=True), encoding="utf-8")
    rb2 = Rulebook(tmp_path / "rb")
    assert "R002_NEAT" in rb2.resolve(PROPOSED).criteria and "R002_NEAT" in rb2.resolve(TUNED).criteria and "R002_NEAT" not in rb2.resolve("checklist5_legacy").criteria
