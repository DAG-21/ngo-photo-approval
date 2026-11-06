"""L3, gap analysis: reason quality, redaction, the offline baseline analyzer, running it after a decision without ever touching the decision, and keeping
generic reasons out of the pool of examples."""

from __future__ import annotations

import pytest
from sqlalchemy import select, text

from ngo_agent import gap_service as gs
from ngo_agent.gap_service import AnalyzerResult, BaselineGapAnalyzer
from ngo_agent.models import EvaluationRun, GapAnalysis, HumanReview, LearningExample
from ngo_agent.reason_quality import is_low_information, redact
from test_examples import BLUE, RED, decide, enable, recorder, solid, upload


# ------------------------------------------------------------------------------------------------------------------------- reason quality
@pytest.mark.parametrize("reason", ["looks good to me", "ok", "fine photo", "Good!", "", "   ", "yes, approved", "not good, rejected", "very nice picture"])
def test_generic_reasons_are_low_information(reason):
    assert is_low_information(reason) is True


@pytest.mark.parametrize("reason", ["Faces are clear and the activity is easy to understand.", "faces hidden behind volunteers", "The banner is cropped and the logo cannot be read.",
                                    "Children are looking at the camera, not at the volunteer."])
def test_specific_reasons_are_not_low_information(reason):
    assert is_low_information(reason) is False


def test_contact_details_are_removed_before_text_could_leave_the_platform():
    t = redact("Ask Mr Rao: rao.k@ngo-example.org or +91 98765 43210, see https://example.org/photo?id=7 and www.example.com/x. Room 12.")
    assert "[email]" in t and "[phone]" in t and t.count("[link]") == 2 and "rao.k@" not in t and "98765" not in t and "example.org" not in t and "Room 12" in t
    assert redact(None) == "" and redact("nothing to remove here") == "nothing to remove here"


# ------------------------------------------------------------------------------------------------------------------------------ analyzer
def payload(agreement, human_reason, gates=(), positives=(), concerns=(), reasons=(), corrections=()):
    return {"ai": {"recommendation": "REJECT", "reasons": list(reasons), "positives": list(positives), "concerns": list(concerns),
                   "failed_or_uncertain_gates": [{"gate_id": g, "criterion_id": c, "status": "NO", "evidence": e} for g, c, e in gates]},
            "human": {"decision": "APPROVE", "agreement": agreement, "reason": human_reason, "corrections": list(corrections)}}


def test_a_point_the_ai_already_made_is_not_a_gap():
    out = BaselineGapAnalyzer().analyze(payload("AGREE", "The volunteer is handing food to the beneficiaries.", positives=["Volunteer clearly handing food to the beneficiaries"]))
    assert out.gaps == [] and "nothing" in out.summary


def test_a_point_the_ai_never_mentioned_is_recorded_when_the_person_agrees():
    out = BaselineGapAnalyzer().analyze(payload("AGREE", "The volunteer is handing food to the beneficiaries. The banner shows the campaign slogan and the sponsor.",
                                                positives=["Volunteer clearly handing food to the beneficiaries"]))
    assert [g["kind"] for g in out.gaps] == ["MISSED_POINT"] and "campaign slogan" in out.gaps[0]["summary"] and out.gaps[0]["related_gates"] == []


def test_disagreeing_about_a_gate_the_ai_failed_records_the_contested_gate():
    out = BaselineGapAnalyzer().analyze(payload("DISAGREE", "The children's faces are visible enough; the child turned away is only in the background.",
                                                gates=[("G004", "R003_FACE", "Not every beneficiary face is visible; the child in the foreground has turned away.")]))
    assert len(out.gaps) == 2 and all((g["kind"], g["related_gates"]) == ("CONTESTED_GATE", ["G004"]) for g in out.gaps)   # both sentences are about that gate
    assert "2 contested" in out.summary


def test_when_the_ai_found_no_problem_a_rejection_reason_is_a_missed_point():
    out = BaselineGapAnalyzer().analyze(payload("DISAGREE", "The photograph shows a vehicle number plate in full view.", reasons=["Clear faces and a visible handover"]))
    assert [g["kind"] for g in out.gaps] == ["MISSED_POINT"]


def test_corrections_comments_count_too_and_repeats_and_fragments_are_ignored():
    out = BaselineGapAnalyzer().analyze(payload("AGREE", "Yes. Yes. Good.", corrections=[{"criterion_id": "G004", "comment": "Sponsor logo printed on the vest"},
                                                                                     {"criterion_id": "G005", "comment": "sponsor logo printed on the vest"}]))
    assert len(out.gaps) == 1 and "logo" in out.gaps[0]["summary"]                     # the short fragments are ignored and the same sentence is recorded once


# --------------------------------------------------------------------------------------------------------------------------- after a decision
def reason_row(P, review_id):
    with P.factory() as s:
        return s.scalars(select(GapAnalysis).where(GapAnalysis.review_id == review_id)).all()


def test_the_analysis_is_stored_after_a_decision_shown_to_the_reviewer_and_listed_for_the_administrator(PG):
    PG.use(__import__("ngo_agent.evaluators", fromlist=["ScriptedEvaluator"]).ScriptedEvaluator(answers={"R003_FACE": "NO"}, rulebook=PG.rb))
    pid, eid = upload(PG, solid(RED[0]))
    h = PG.login("rev_a")
    rid = PG.client.post(f"/evaluations/{eid}/review", headers=h).json()["review_id"]
    why = "The children's faces are visible enough, and the child turned away is only in the background."
    done = PG.client.post(f"/feedback/{rid}/submit", json={"decision": "APPROVE", "comment": why}, headers=h)
    assert done.status_code == 200
    rows = reason_row(PG, rid)
    assert len(rows) == 1 and rows[0].analyzer == "baseline-overlap-1" and rows[0].status == "DONE" and rows[0].agreement == "DISAGREE" and rows[0].low_information is False
    got = PG.client.get(f"/feedback/{rid}", headers=h).json()
    assert got["gap_analysis"][0]["gaps"][0]["kind"] == "CONTESTED_GATE" and got["low_information_reason"] is False
    listed = PG.client.get(f"/organisations/{PG.orgA_id}/gap-analyses", headers=PG.login("admin_a")).json()["gap_analyses"]
    assert [g["review_id"] for g in listed] == [rid]
    assert PG.client.get(f"/organisations/{PG.orgA_id}/gap-analyses", headers=h).status_code == 403                      # administrators only
    assert PG.client.get(f"/organisations/{PG.orgA_id}/gap-analyses", headers=PG.login("admin_b")).status_code == 404     # and never another organisation's
    assert gs.analyze_review(PG.factory, rid, BaselineGapAnalyzer()) == "EXISTS" and len(reason_row(PG, rid)) == 1       # running it again adds nothing
    with PG.engine.begin() as c:
        with pytest.raises(Exception, match="append-only"):
            c.execute(text("UPDATE gap_analyses SET summary='x'"))
    with PG.engine.begin() as c:
        with pytest.raises(Exception, match="append-only"):
            c.execute(text("DELETE FROM gap_analyses"))


def test_a_failing_analyzer_never_touches_the_decision(tmp_path, rb):
    from conftest import Platform

    class Boom:
        name = "boom-1"

        def analyze(self, payload):  # noqa: ANN001
            raise RuntimeError("the analyzer fell over: secret detail")

    (tmp_path / "x").mkdir()
    q = Platform(tmp_path / "x", rb, gap_analyzer=Boom())
    try:
        pid, eid = decide(q, solid(RED[0]))
        with q.factory() as s:
            rev = s.scalars(select(HumanReview)).one()
            assert rev.review_status == "FEEDBACK_SUBMITTED" and rev.human_final_decision == "APPROVE"             # the decision stands
            row = s.scalars(select(GapAnalysis)).one()
            assert row.status == "FAILED" and row.error == "RuntimeError" and "secret" not in (row.error or "") and row.gaps == []   # only the error type is kept
    finally:
        q.close()


def test_the_analyzer_receives_text_only_with_contact_details_removed(PG):
    seen = {}

    class Spy:
        name = "spy-1"

        def analyze(self, payload):  # noqa: ANN001
            seen.update(payload)
            return AnalyzerResult()

    pid, eid = decide(PG, solid(RED[0]), reason="Written by Mr Rao, rao.k@ngo-example.org, +91 98765 43210: the faces are clear and the handover is visible.")
    with PG.factory() as s:
        rid = s.scalars(select(HumanReview)).one().review_id
    assert gs.analyze_review(PG.factory, rid, Spy()) == "DONE"
    text_in = repr(seen)
    assert "[email]" in text_in and "[phone]" in text_in and "rao.k@" not in text_in and "98765" not in text_in
    assert not any(isinstance(v, (bytes, bytearray)) for v in text_in) and set(seen) == {"ai", "human"}     # no image bytes anywhere: two plain text structures


# ------------------------------------------------------------------------------------------------------------ generic reasons and the examples
def test_a_generic_reason_is_kept_but_never_shown_to_the_model_as_an_example(PE):
    good_pid, _ = decide(PE, solid(RED[0]), reason="Faces are clear and the handover is easy to see.")
    low_pid, _ = decide(PE, solid(RED[1], 161), reason="looks good to me")
    with PE.factory() as s:
        flags = {e.photo_id: e.low_information for e in s.scalars(select(LearningExample))}
    assert flags == {good_pid: False, low_pid: True}                                                          # the generic one is still stored as feedback
    rid = None
    with PE.factory() as s:
        rid = s.scalars(select(HumanReview).where(HumanReview.photo_id == low_pid)).one().review_id
    assert PE.client.get(f"/feedback/{rid}", headers=PE.login("rev_a")).json()["low_information_reason"] is True
    enable(PE)
    rec = recorder(PE)
    upload(PE, solid(RED[2], 162))
    assert {e["photo_id"] for e in rec.seen_examples[-1]} == {good_pid}                                       # only the specific reason is shown


def test_a_decision_with_an_informative_reason_is_not_marked(PG):
    pid, eid = decide(PG, solid(BLUE[0]), reason="The scene is unrelated to the NGO's activities and shows no beneficiaries.")
    with PG.factory() as s:
        assert s.scalars(select(LearningExample)).one().low_information is False
        run = s.get(EvaluationRun, eid)
        assert run.processing_status == "AI_EVALUATED"
