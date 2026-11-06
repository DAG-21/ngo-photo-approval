"""The AI always gives a verdict on the review screens: Approve or Disapprove, with its reasons. It never says that it needs a person's decision (a person reviews every photograph anyway).
Text-level checks of the shipped screen code (no browser): the wording is gone, anything that is not an approval is shown as Disapprove, and a low quality score is explained in plain words."""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "src" / "ngo_agent" / "api" / "static"
APP = (STATIC / "app.js").read_text(encoding="utf-8")


def test_no_screen_says_the_ai_needs_a_person():
    for f in STATIC.glob("*.js"):
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"needs? (a )?(person|human)|person's decision|human review", text, re.I), f.name


def test_anything_that_is_not_an_approval_is_shown_as_disapprove():
    assert 'const aiRejects = (rec) => !!rec && !aiApproves(rec);' in APP
    assert 'recRejects = aiRejects(rec)' in APP and 'recApproves ? "Approve" : "Disapprove"' in APP
    assert 'rec === "REJECT"' not in APP                                   # no place decides "disapprove" only for the REJECT recommendation any more
    assert 'RESELECT: "bad"' in APP and 'HUMAN_REVIEW: "bad"' in APP


def test_a_low_quality_score_and_an_unconfirmed_check_are_explained_in_plain_words():
    assert "The overall photo quality is too low" in APP and "Could not confirm, so it is not approved" in APP and "Photo quality: " in APP


def test_the_screen_lists_the_low_quality_aspects_with_the_ai_s_own_reason():
    assert "graded ${dims[k].score} of 4" in APP and "Composition and framing" in APP
    assert "(why ? `. ${why}` : \".\")" in APP and "const why = text(dims[k].explanation)" in APP      # the AI's own sentence is part of the line
    assert "is too low: ${Math.round(Number(m[1]))} out of 100, and the cutoff is ${cut ? cut[1] : \"70\"}" in APP and "(?:cutoff of|threshold) ([0-9.]+)" in APP


def test_the_api_sends_the_quality_grades_with_the_decision():
    src = (Path(__file__).resolve().parents[1] / "src" / "ngo_agent" / "api" / "app.py").read_text(encoding="utf-8")
    assert re.search(r'for k in \("recommendation", "decision_reasons", "human_review_questions", "failed_gates", "quality",', src)


def test_a_reselect_counts_as_a_disapproval_when_the_reviewer_decides():
    from ngo_agent.review_service import _agreement

    assert _agreement("RESELECT", "APPROVE") == "DISAGREE" and _agreement("RESELECT", "REJECT") == "AGREE"
    assert _agreement("REJECT", "APPROVE") == "DISAGREE" and _agreement("APPROVE", "APPROVE") == "AGREE"
    assert _agreement("HUMAN_REVIEW", "APPROVE") == "AI_DEFERRED" and _agreement(None, "APPROVE") == "AI_DEFERRED"      # an older profile's undecided outcome still defers


def test_the_screen_lists_every_check_with_the_ai_s_reason_under_the_verdict():
    assert "function allChecks(run)" in APP and "list, allChecks(run));" in APP                      # shown as part of the AI verdict box
    assert '((run.layer_2 || {}).criteria || [])' in APP and "plainText(a.evidence)" in APP and "and the AI's reason for each" in APP
    names = APP[APP.index("const CHECK_NAMES"):APP.index("function allChecks")]
    asked = ["R001_POSITION", "R001_VISIBLE", "R001_FLAT", "R002_GIVER_WHITE", "R003_FACE", "R003_FACE_SHARP", "R003_CROP", "R003_OBSTRUCTION_PPE_ONLY", "R004_GENUINE", "R006_ITEM_SHARP",
             "R006_ACTION", "R006_SIZE", "R007_ANGLE", "R007_CUTOFF", "R007_DISTORTION", "R007_PLACEMENT", "R008_BACK", "R010_TECHNICAL", "R011_BACKGROUND", "R012_DIGNITY", "R013_HUMANITY",
             "R015_PERSON_PRESENT", "R015_BEN_RECEIVING", "NEW-014"]
    assert all(c in names for c in asked)                                                          # every one of the 24 has a plain name, none is shown as a code


def test_a_poor_part_with_a_good_total_is_explained_as_such_and_not_as_a_low_score():
    assert "One part of the photo quality is poor: " in APP and "but a poor part disapproves the photo on its own" in APP
    assert "aboveCutoff && poor" in APP and "/POOR:([a-z_]+)/" in APP                      # only when the total reaches the cutoff; a total under it still says the score is too low
    assert "The overall photo quality is too low: ${Math.round(Number(m[1]))} out of 100" in APP
