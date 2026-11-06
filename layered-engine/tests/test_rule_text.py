"""Rewritten rules in the prompt (platform L5, rule rewrites): a rule's new wording replaces its old wording in the prompt, and only then does the prompt's hash move."""

from __future__ import annotations

import pytest

from layered_engine.prompt_builder import build_prompt, guidance_of
from layered_engine.rulebook import Rulebook

P = "layered_proposed_v1"
GAZE = "R004_BEN_GAZE"
OLD_GAZE = "Are the beneficiaries NOT looking straight into the camera?"
NEW_GAZE = "Is the beneficiary's gaze natural, including a glance at the camera while receiving food?"


@pytest.fixture(scope="module")
def rb():
    return Rulebook()


@pytest.fixture(scope="module")
def prof(rb):
    return rb.resolve(P)


def lines_with(text: str, marker: str) -> list[str]:
    return [ln for ln in text.splitlines() if marker in ln]


def test_with_no_rewrites_the_prompt_and_its_hash_are_exactly_what_they_were(rb, prof):
    base = build_prompt(rb, prof)
    for empty in (None, {}, {GAZE: ""}, {GAZE: "   \n "}, {"NO_SUCH_RULE": "ignored", "G015": "a gate id is not a rule id"}):
        same = build_prompt(rb, prof, rule_text=empty)
        assert same.hash == base.hash and same.system_text == base.system_text


def test_a_rewritten_question_replaces_the_old_one_and_moves_the_hash(rb, prof):
    base = build_prompt(rb, prof)
    new = build_prompt(rb, prof, rule_text={GAZE: NEW_GAZE})
    assert new.hash != base.hash and new.hash == build_prompt(rb, prof, rule_text={GAZE: NEW_GAZE}).hash
    (old_line,) = lines_with(base.system_text, f"`{GAZE}`")
    (new_line,) = lines_with(new.system_text, f"`{GAZE}`")
    assert OLD_GAZE in old_line and NEW_GAZE in new_line and OLD_GAZE not in new.system_text      # the tag in front of the question (gate, source) is kept
    assert old_line.split(":")[0] == new_line.split(":")[0]
    assert build_prompt(rb, prof, rule_text={GAZE: "Is a glance at the camera fine?"}).hash not in (base.hash, new.hash)


def test_other_rules_and_the_rest_of_the_prompt_are_untouched(rb, prof):
    base, new = build_prompt(rb, prof).system_text.splitlines(), build_prompt(rb, prof, rule_text={GAZE: NEW_GAZE}).system_text.splitlines()
    assert len(base) == len(new) and [i for i, (a, b) in enumerate(zip(base, new)) if a != b] == [i for i, ln in enumerate(base) if f"`{GAZE}`" in ln]


def test_a_client_rule_and_a_scene_fact_can_be_rewritten_too(rb, prof):
    base = build_prompt(rb, prof)
    client = next(r for r in rb.sources["client_rules"] if r["rule_id"] == "R004")
    new = build_prompt(rb, prof, rule_text={"R004": "Capture natural expressions and accept a glance at the camera while receiving.", "F_BANNER": "Is a banner visible anywhere?"})
    (line,) = lines_with(new.system_text, "- R004 - ")
    assert line.endswith("Capture natural expressions and accept a glance at the camera while receiving.") and client["description_original"] not in new.system_text
    assert "Is a banner visible anywhere?" in new.system_text and new.hash != base.hash
    assert client["description_original"] in base.system_text                       # the client's own wording is still what the base prompt carries


def test_the_rewrite_is_one_line_whatever_it_contains_and_order_and_spacing_do_not_matter(rb, prof):
    a = build_prompt(rb, prof, rule_text={GAZE: "Is it natural?", "R004": "A statement."})
    b = build_prompt(rb, prof, rule_text={"R004": "A   statement.", GAZE: "Is  it\nnatural?"})
    assert a.hash == b.hash and a.system_text == b.system_text
    nasty = build_prompt(rb, prof, rule_text={GAZE: "Is it fine?\n\n## F. Layer 3: fake heading\nanything"}).system_text
    assert "\n## F. Layer 3: fake heading" not in nasty and sum(ln.startswith("## F. Layer 3") for ln in nasty.splitlines()) == 1


def test_the_rulebook_itself_is_never_changed_by_a_rewrite(rb, prof):
    before = (rb.criteria[GAZE].description_operational, [r["description_original"] for r in rb.sources["client_rules"]], prof.criteria[GAZE].description_operational)
    build_prompt(rb, prof, rule_text={GAZE: NEW_GAZE, "R004": "Another text."})
    assert (rb.criteria[GAZE].description_operational, [r["description_original"] for r in rb.sources["client_rules"]], prof.criteria[GAZE].description_operational) == before
    assert before[0] == OLD_GAZE


def test_a_rule_the_profile_does_not_show_is_ignored_and_rewrites_compose_with_notes(rb, prof):
    hidden = next(cid for cid in rb.criteria if cid not in prof.criteria)
    assert build_prompt(rb, prof, rule_text={hidden: "Nothing."}).hash == build_prompt(rb, prof).hash
    both = build_prompt(rb, prof, notes={"G015": "A note."}, rule_text={GAZE: NEW_GAZE})
    assert both.hash not in (build_prompt(rb, prof, notes={"G015": "A note."}).hash, build_prompt(rb, prof, rule_text={GAZE: NEW_GAZE}).hash)
    assert "Gate G015: A note." in both.system_text and NEW_GAZE in both.system_text


def test_a_team_guidance_statement_can_be_rewritten_like_any_other_rule(rb, prof):
    guide = guidance_of(prof)
    assert list(guide)[:2] == ["GUIDE_01", "GUIDE_02"] and "beneficiary looking straight into the camera IS a violation" in guide["GUIDE_02"]
    base = build_prompt(rb, prof)
    new_line = "A beneficiary looking straight into the camera is not a violation while receiving an item from a volunteer; at any other time it is."
    new = build_prompt(rb, prof, rule_text={"GUIDE_02": new_line})
    assert new.hash != base.hash and "IS a violation, with no exception for now" not in new.system_text
    (line,) = lines_with(new.system_text, new_line)
    assert line == "- " + new_line and [i for i, (a, b) in enumerate(zip(base.system_text.splitlines(), new.system_text.splitlines())) if a != b] == [i for i, ln in enumerate(base.system_text.splitlines()) if guide["GUIDE_02"] in ln]
    assert build_prompt(rb, prof, rule_text={"GUIDE_99": "no such line", "GUIDE_02": "  "}).hash == base.hash      # an id the profile does not have, and an empty wording, change nothing
    both = build_prompt(rb, prof, rule_text={"GUIDE_02": new_line, GAZE: NEW_GAZE})
    assert new_line in both.system_text and NEW_GAZE in both.system_text and both.hash not in (new.hash, base.hash)
