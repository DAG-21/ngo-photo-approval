"""L4, the learning loop: every few decided photographs it drafts a change to the settings, tests it on the stored staff decisions, applies it if it passes, watches
the photographs that follow, and undoes it if it does worse. Each safety rule has a test. No model is called anywhere here: the answers are scripted and the loop replays them."""

from __future__ import annotations

import itertools
import random

import pytest
from sqlalchemy import select, text

from ngo_agent import loop_service as loop
from ngo_agent.evaluators import ScriptedEvaluator
from ngo_agent.models import ChangeEvent, ChangeRecord, ChangeTest, EvaluationRun, ImprovementCycle, Organisation, RulebookVersion
from test_examples import decide, solid, upload

COUNTER = itertools.count(1)
GOOD_APPROVE = "The faces are clear, the handover is visible and the banner is readable."
GOOD_REJECT = "The faces are hidden behind the volunteers and the banner is cropped."
G004_ONLY = {"R003_FACE": "NO"}                            # fails gate G004 and nothing else
G004_AND_G005 = {"R003_FACE": "NO", "R003_CLEAR": "NO"}    # fails G004 and G005: still rejected if G004 is softened
SOFT_G010 = {"R006_COVERAGE": "NO"}                        # G010 is a soft gate: it only lowers the quality score
BORDERLINE = {"activity_communication": 2, "technical_quality": 2, "composition": 3, "naturalness_storytelling": 2, "brand_presentation": 3, "visual_polish": 2}   # score 57.5


class World:
    """Builds decided photographs of one organisation with chosen AI answers and staff decisions, and drives the loop."""

    def __init__(self, P, monkeypatch):
        self.P, self.held = P, set()
        monkeypatch.setattr(loop, "is_held_out", lambda pid, pct: pid in self.held)     # the held-out photographs are chosen by the test, not by a hash

    def photo(self, human="APPROVE", answers=None, dims=None, hold=False, who="rev_a", reason=None, org="A"):
        self.P.use(ScriptedEvaluator(answers=answers or {}, dims=dims or {}, rulebook=self.P.rb))
        pid, eid = decide(self.P, solid((20 + next(COUNTER) % 200, 90, 60), 100 + next(COUNTER)), human, reason or (GOOD_APPROVE if human == "APPROVE" else GOOD_REJECT), who=who)
        if hold:
            self.held.add(pid)
        return pid, eid

    def many(self, n, **kw):
        return [self.photo(**kw) for _ in range(n)]

    def settings(self, **kw):
        base = {"enabled": True, "confirm_automatic_changes": True, "every_n": 1000, "min_rows": 6, "min_pattern": 3, "min_rejected_for_loosening": 1, "min_gain": 1,
                "monitor_min": 2, "monitor_window": 4, "error_margin": 99}
        r = self.P.client.put(f"/organisations/{self.P.orgA_id}/learning-loop", json={**base, **kw}, headers=self.P.login("admin_a"))
        assert r.status_code == 200, r.text
        return r.json()

    def run(self):
        r = self.P.client.post(f"/organisations/{self.P.orgA_id}/learning-loop/run", headers=self.P.login("admin_a"))
        assert r.status_code == 200, r.text
        return r.json()["status"]

    def changelog(self, status=None):
        url = f"/organisations/{self.P.orgA_id}/changelog" + (f"?status={status}" if status else "")
        return self.P.client.get(url, headers=self.P.login("admin_a")).json()["changes"]

    def active(self):
        with self.P.factory() as s:
            org = s.get(Organisation, self.P.orgA_id)
            v = s.get(RulebookVersion, org.active_rulebook_version_id)
            return v.version_id, v.version_number, dict(v.configuration), org.active_prompt_version_id

    def rows(self, model):
        with self.P.factory() as s:
            return s.scalars(select(model)).all()


@pytest.fixture()
def W(P, monkeypatch):
    return World(P, monkeypatch)


def soften_pattern(W):
    """Staff approve photographs that gate G004 alone rejected; photographs failing G004 and G005 stay rejected."""
    W.many(2, answers=G004_ONLY, human="APPROVE", hold=True)            # held out
    W.many(3, answers=G004_ONLY, human="APPROVE")                       # the pattern the loop sees (3 photographs, all approved)
    W.many(3, human="APPROVE")                                          # photographs the AI approves and staff approve
    W.many(2, answers=G004_AND_G005, human="REJECT", hold=True)         # staff rejections that must stay rejected (held out)
    W.many(2, answers=G004_AND_G005, human="REJECT")


# ------------------------------------------------------------------------------------------------------------------------------- the switch
def test_the_loop_is_off_by_default_and_needs_an_administrator_and_a_confirmation(W):
    P, a, ha = W.P, W.P.orgA_id, W.P.login("admin_a")
    off = P.client.get(f"/organisations/{a}/learning-loop", headers=ha).json()
    assert off["enabled"] is False and off["every_n"] == 5 and off["decided_photographs"] == 0 and off["recent_cycles"] == []
    assert P.client.get(f"/organisations/{a}/learning-loop", headers=P.login("rev_a")).status_code == 403
    assert P.client.put(f"/organisations/{a}/learning-loop", json={"enabled": True, "confirm_automatic_changes": True}, headers=P.login("rev_a")).status_code == 403
    refused = P.client.put(f"/organisations/{a}/learning-loop", json={"enabled": True}, headers=ha)
    assert refused.status_code == 422 and "automatically" in refused.json()["detail"] and "tell the NGO" in refused.json()["detail"]
    for bad in ({"every_n": 0}, {"pattern_share": 2}, {"holdout_pct": 5}, {"min_gain": 0}, {"monitor_min": 8, "monitor_window": 3}, {"every_n": 2.5}):
        assert P.client.put(f"/organisations/{a}/learning-loop", json={"confirm_automatic_changes": True, **bad}, headers=ha).status_code == 422, bad
    assert P.client.post(f"/organisations/{a}/learning-loop/run", headers=ha).json()["status"] == "DISABLED"
    assert W.settings()["enabled"] is True
    assert P.client.put(f"/organisations/{a}/learning-loop", json={"paused": True}, headers=ha).json()["paused"] is True
    assert P.client.post(f"/organisations/{a}/learning-loop/run", headers=ha).json()["status"] == "PAUSED"
    assert P.client.get(f"/organisations/{P.orgB_id}/learning-loop", headers=ha).status_code == 404                           # another organisation's loop is not even visible
    assert P.client.get(f"/organisations/{P.orgB_id}/learning-loop", headers=P.login("admin_b")).json()["enabled"] is False  # and was not touched


# ------------------------------------------------------------------------------------------------------------------------------ nothing to do
def test_when_staff_and_the_ai_agree_the_loop_changes_nothing(W):
    W.many(4, human="APPROVE")
    W.many(4, answers=G004_AND_G005, human="REJECT")
    W.settings()
    vid, number, _, _ = W.active()
    assert W.run() == "NOTHING_TO_CHANGE"
    assert W.rows(ChangeRecord) == [] and W.active()[:2] == (vid, number)


def test_too_few_decisions_and_a_paused_loop_do_nothing(W):
    W.many(3, human="APPROVE")
    W.settings(min_rows=10)
    assert W.run() == "NOT_ENOUGH_DATA"
    W.settings(paused=True)
    before = len(W.rows(ImprovementCycle))
    assert W.run() == "PAUSED" and len(W.rows(ImprovementCycle)) == before


# ---------------------------------------------------------------------------------------------------------------------- loosening, applied
def test_a_gate_that_alone_rejected_photographs_staff_approved_is_softened_once_it_is_proved_safe(W, rb):
    soften_pattern(W)
    W.settings()
    vid, number, cfg, prompt = W.active()
    assert rb.resolve("layered_proposed_v1").is_hard(next(g for g in rb.resolve("layered_proposed_v1").gates if g.gate_id == "G004"))
    assert W.run() == "APPLIED"
    log = W.changelog()
    assert [(c["label"], c["kind"], c["target"], c["status"], c["loosening"]) for c in log] == [("L4-0001", "GATE_SOFTEN", "G004", "APPLIED", True)]
    assert "no longer rejects a photograph by itself" in log[0]["description"] and log[0]["latest_test"]["passed"] is True
    assert log[0]["latest_test"]["false_approvals"] == [0, 0] and log[0]["latest_test"]["false_rejections"][1] < log[0]["latest_test"]["false_rejections"][0]
    nvid, nnumber, ncfg, nprompt = W.active()
    assert nvid != vid and nnumber == f"{number}+L4-0001" and nprompt == prompt                                # a new rulebook version; the prompt version is the same
    assert ncfg["overlay"][0]["kind"] == "GATE_SOFTEN" and ncfg["auto_approval"]["by"] == "automatic guardrails" and ncfg["overlay_digest"]
    with W.P.factory() as s:
        v = s.get(RulebookVersion, nvid)
        assert v.approval_status == "APPROVED" and v.approved_by is None and v.created_by is None and v.effective_at is not None
    # the next photograph is decided with the new settings, and the model is asked exactly what it was asked before
    _, eid = W.photo(human="APPROVE", answers=G004_ONLY)
    with W.P.factory() as s:
        run = s.get(EvaluationRun, eid)
        assert run.rulebook_version_id == nvid and run.ai_recommendation != "REJECT" and run.decision_output["versions"]["overlay"] == ["L4-0001"]
        assert not any("differs from the registered" in w for w in run.warnings or [])                          # the prompt hash did not move
    assert W.rows(ImprovementCycle)[-1].status == "APPLIED"


def test_loosening_without_enough_staff_rejections_to_show_it_is_safe_is_held_for_developers(W):
    soften_pattern(W)
    W.settings(min_rejected_for_loosening=20)
    vid, number, _, _ = W.active()
    assert W.run() == "NO_PASSING_CHANGE"
    held = W.changelog("HELD")
    assert len(held) == 1 and held[0]["target"] == "G004" and held[0]["status"] == "HELD" and "fewer than the 20" in held[0]["history"][-1]["reason"]
    assert W.active()[:2] == (vid, number)                                                                     # nothing was applied
    W.run()
    assert len(W.rows(ChangeRecord)) == 1 and len(W.rows(ChangeTest)) == 2                                     # tested again, not drafted again


def test_a_change_that_would_approve_a_photograph_staff_rejected_is_blocked(W, rb):
    W.many(3, answers=G004_ONLY, human="APPROVE", hold=True)
    W.photo(answers=G004_ONLY, human="REJECT", hold=True)                                                      # staff rejected a photograph that only G004 failed
    W.many(2, human="APPROVE")
    cfg = {**loop.DEFAULTS, "min_rejected_for_loosening": 0, "min_gain": 1}
    with W.P.factory() as s:
        rows = loop.build_dataset(s, W.P.orgA_id, "layered_proposed_v1", 40)
    base = loop.replay(rows, rb, "layered_proposed_v1", [], [])
    out = loop.evaluate_candidate(rows, base, rb, "layered_proposed_v1", [], [], {"change_id": "X", "kind": "GATE_SOFTEN", "target": "G004", "set": {}}, True, cfg)
    assert out.status == "BLOCKED" and "would approve 1 photograph" in out.reason and out.guardrails["no_new_false_approval"] is False and len(out.flips["new_false_approvals"]) == 1


def test_a_change_without_a_measurable_gain_is_blocked(W, rb):
    W.many(4, human="APPROVE")
    W.many(2, answers=G004_AND_G005, human="REJECT", hold=True)
    cfg = {**loop.DEFAULTS, "min_rejected_for_loosening": 0}
    with W.P.factory() as s:
        rows = loop.build_dataset(s, W.P.orgA_id, "layered_proposed_v1", 40)
    base = loop.replay(rows, rb, "layered_proposed_v1", [], [])
    out = loop.evaluate_candidate(rows, base, rb, "layered_proposed_v1", [], [], {"change_id": "X", "kind": "QUALITY_THRESHOLDS", "set": {"review": 60}}, True, cfg)
    assert out.status == "BLOCKED" and "no measurable gain" in out.reason


# --------------------------------------------------------------------------------------------------------------------- tightening, thresholds
def test_a_soft_gate_that_staff_keep_rejecting_is_hardened_without_the_loosening_minimum(W, rb):
    prof = rb.resolve("layered_proposed_v1")
    assert next(g for g in prof.gates if g.gate_id == "G010").criterion_id == "R006_COVERAGE" and not prof.is_hard(next(g for g in prof.gates if g.gate_id == "G010"))
    W.many(2, answers=SOFT_G010, human="REJECT", hold=True)
    W.many(3, answers=SOFT_G010, human="REJECT")
    W.many(4, human="APPROVE")
    W.settings(min_rejected_for_loosening=1000)                                  # irrelevant: this change only makes the system stricter
    assert W.run() == "APPLIED"
    log = W.changelog()
    assert [(c["kind"], c["target"], c["loosening"]) for c in log] == [("GATE_HARDEN", "G010", False)]
    _, eid = W.photo(human="REJECT", answers=SOFT_G010)
    with W.P.factory() as s:
        run = s.get(EvaluationRun, eid)
        assert run.ai_recommendation == "REJECT" and "G010" in run.decision_output["failed_gates"]


def test_thresholds_are_lowered_when_staff_approve_what_the_quality_score_sent_to_review(W):
    W.many(2, dims=BORDERLINE, human="APPROVE", hold=True)
    W.many(4, dims=BORDERLINE, human="APPROVE")
    W.many(2, answers=G004_AND_G005, human="REJECT", hold=True)
    W.many(2, answers=G004_AND_G005, human="REJECT")
    W.settings()
    assert W.run() == "APPLIED"
    log = W.changelog()
    assert len(log) == 1 and log[0]["kind"] == "QUALITY_THRESHOLDS" and log[0]["loosening"] is True and "below 55" in log[0]["description"]
    _, eid = W.photo(human="APPROVE", dims=BORDERLINE)
    with W.P.factory() as s:
        assert s.get(EvaluationRun, eid).ai_recommendation in ("APPROVE", "APPROVED_PENDING_PUBLICATION_CHECKS")


# --------------------------------------------------------------------------------------------------------------------------- planted errors
def test_planted_errors_in_the_feedback_do_not_change_the_settings(W):
    rnd = random.Random(7)
    patterns = [{}, G004_ONLY, G004_AND_G005, SOFT_G010, {"dims": BORDERLINE}]
    for _ in range(18):
        p = rnd.choice(patterns)
        kw = {"dims": p["dims"]} if "dims" in p else {"answers": p}
        W.photo(human=rnd.choice(["APPROVE", "REJECT"]), hold=rnd.random() < 0.4, **kw)       # staff decisions unrelated to anything the AI saw
    W.settings(min_rejected_for_loosening=20, min_gain=1)                                          # the shipped evidence rules
    vid, number, _, _ = W.active()
    for _ in range(3):
        W.run()
    assert W.active()[:2] == (vid, number) and not [e for e in W.rows(ChangeEvent) if e.status == "APPLIED"]


# ------------------------------------------------------------------------------------------------------------------------------ rollback
def applied_world(W):
    soften_pattern(W)
    W.settings()
    before = W.active()
    assert W.run() == "APPLIED"
    return before


def test_a_change_that_makes_things_worse_is_rolled_back_by_itself_and_never_drafted_again(W):
    before = applied_world(W)
    W.many(3, answers=G004_ONLY, human="REJECT")          # after the change the AI approves these; staff reject them: new false approvals
    assert W.run() == "ROLLED_BACK"
    assert W.active()[0] == before[0]                       # the previous version is active again
    log = W.changelog()
    assert log[0]["status"] == "ROLLED_BACK" and "false approval" in log[0]["history"][-1]["reason"] and log[0]["history"][-1]["by_person"] is False
    assert W.run() in ("NOTHING_TO_CHANGE", "NO_PASSING_CHANGE") and len(W.rows(ChangeRecord)) == 1                 # it is not drafted again
    assert W.active()[0] == before[0]


def test_a_change_that_holds_up_is_confirmed_after_the_monitoring_window(W):
    applied_world(W)
    W.many(4, answers=G004_ONLY, human="APPROVE")           # staff keep approving what the softened gate lets through
    W.run()
    assert W.changelog()[0]["status"] == "MONITORED_OK"
    # a confirmed change is still in force, so a person can still undo it
    cid, before = W.changelog()[0]["change_id"], W.active()[0]
    ok = W.P.client.post(f"/organisations/{W.P.orgA_id}/changes/{cid}/rollback", json={"reason": "the NGO changed its mind"}, headers=W.P.login("admin_a"))
    assert ok.status_code == 200 and W.active()[0] != before and W.changelog()[0]["status"] == "ROLLED_BACK"


def test_a_person_can_roll_back_only_the_latest_applied_change(W):
    before = applied_world(W)
    P, a, ha = W.P, W.P.orgA_id, W.P.login("admin_a")
    cid = W.changelog()[0]["change_id"]
    assert P.client.post(f"/organisations/{a}/changes/{cid}/rollback", json={"reason": "  "}, headers=ha).status_code == 422
    assert P.client.post(f"/organisations/{a}/changes/{cid}/rollback", json={"reason": "the NGO objected"}, headers=P.login("rev_a")).status_code == 403
    assert P.client.post(f"/organisations/{a}/changes/{cid}/rollback", json={"reason": "the NGO objected"}, headers=P.login("admin_b")).status_code == 404
    ok = P.client.post(f"/organisations/{a}/changes/{cid}/rollback", json={"reason": "the NGO objected"}, headers=ha)
    assert ok.status_code == 200 and W.active()[0] == before[0]
    last = W.changelog()[0]["history"][-1]
    assert last["status"] == "ROLLED_BACK" and last["by_person"] is True and last["reason"] == "the NGO objected"
    assert P.client.post(f"/organisations/{a}/changes/{cid}/rollback", json={"reason": "again"}, headers=ha).status_code == 409
    # the pattern is still there and the change would still pass every test, yet a change a person undid is never applied again
    assert W.run() != "APPLIED" and W.active()[0] == before[0] and len(W.rows(ChangeRecord)) == 1


def test_only_one_change_is_applied_per_cycle_so_a_bad_one_can_be_traced(W):
    soften_pattern(W)                                          # a gate pattern (loosening)
    W.many(2, dims=BORDERLINE, human="APPROVE", hold=True)     # and a quality-threshold pattern (loosening)
    W.many(4, dims=BORDERLINE, human="APPROVE")
    W.settings()
    versions_before = len(W.rows(RulebookVersion))
    assert W.run() == "APPLIED"
    statuses = sorted(c["status"] for c in W.changelog())
    assert statuses == ["APPLIED", "PASSED_NOT_APPLIED"] and len(W.rows(RulebookVersion)) == versions_before + 1       # exactly one version was made
    assert W.run() == "APPLIED"                                # the other is tested again in the next cycle, on top of the first
    assert sorted(c["status"] for c in W.changelog()) == ["APPLIED", "APPLIED"] and len(W.rows(RulebookVersion)) == versions_before + 2
    assert W.active()[1].endswith("+L4-0002")


# ------------------------------------------------------------------------------------------------------------------ trigger and robustness
def test_a_cycle_runs_after_every_five_decided_photographs_and_not_while_paused(W):
    W.settings(every_n=5, min_rows=1000)                      # cycles will run but have too little data: only the trigger is being tested
    W.many(4, human="APPROVE")
    assert len(W.rows(ImprovementCycle)) == 0
    W.photo(human="APPROVE")
    assert [c.ordinal for c in W.rows(ImprovementCycle)] == [1]
    W.many(4, human="APPROVE")
    assert len(W.rows(ImprovementCycle)) == 1
    W.photo(human="APPROVE")
    assert [c.ordinal for c in W.rows(ImprovementCycle)] == [1, 2] and W.rows(ImprovementCycle)[0].decided_count == 5
    W.P.client.put(f"/organisations/{W.P.orgA_id}/learning-loop", json={"paused": True}, headers=W.P.login("admin_a"))
    W.many(5, human="APPROVE")
    assert len(W.rows(ImprovementCycle)) == 2                  # paused: nothing ran


def test_a_failing_cycle_is_recorded_and_never_disturbs_a_decision(W, monkeypatch):
    W.settings(every_n=1)
    monkeypatch.setattr(loop, "build_dataset", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    pid, eid = W.photo(human="APPROVE")                         # the decision still goes through
    with W.P.factory() as s:
        from ngo_agent.models import HumanReview
        assert s.scalars(select(HumanReview)).one().review_status == "FEEDBACK_SUBMITTED"
    cycle = W.rows(ImprovementCycle)[0]
    assert cycle.status == "FAILED" and cycle.summary["error"] == "RuntimeError"


def test_one_organisation_never_affects_anothers_settings(W):
    soften_pattern(W)
    W.settings()
    assert W.run() == "APPLIED"
    with W.P.factory() as s:
        assert [c for c in s.scalars(select(ChangeRecord)) if c.organisation_id != W.P.orgA_id] == []
        orgb = s.get(Organisation, W.P.orgB_id)
        assert loop.settings(orgb)["enabled"] is False and s.get(RulebookVersion, orgb.active_rulebook_version_id).configuration.get("overlay") is None
    assert W.P.client.get(f"/organisations/{W.P.orgA_id}/changelog", headers=W.P.login("admin_b")).status_code == 404


def test_the_loops_own_records_cannot_be_edited_or_deleted(W):
    soften_pattern(W)
    W.settings()
    W.run()
    for table in ("change_records", "change_events", "change_tests"):
        for sql in (f"UPDATE {table} SET organisation_id = organisation_id", f"DELETE FROM {table}"):
            with W.P.engine.begin() as c:
                with pytest.raises(Exception, match="append-only"):
                    c.execute(text(sql))
    with W.P.engine.begin() as c:
        with pytest.raises(Exception, match="can never be deleted"):
            c.execute(text("DELETE FROM improvement_cycles"))
    with W.P.engine.begin() as c:
        c.execute(text("UPDATE improvement_cycles SET status = status"))                                        # the status of a cycle is allowed to move
