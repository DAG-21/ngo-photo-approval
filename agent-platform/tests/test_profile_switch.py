"""Moving an organisation to another policy profile (profile_switch.py): new versions, the old ones kept, what was learned carried over where it still applies. FAKE evaluator: no model."""

from __future__ import annotations

import pytest
from layered_engine.prompt_builder import build_prompt
from sqlalchemy import select

from ngo_agent import profile_switch as ps
from ngo_agent import rule_service as rs
from ngo_agent.errors import Conflict, Invalid
from ngo_agent.evaluators import ScriptedEvaluator
from ngo_agent.models import Organisation, PromptVersion, RulebookVersion, User
from ngo_agent.rule_drafter import rule_catalogue
from test_examples import RED, run_of, solid, upload

OLD, NEW = "layered_proposed_v1", "layered_tuned_v1"
GOOD = "Is the beneficiary's gaze natural, including a glance at the camera while receiving food?"
GAZE = "R004_BEN_GAZE"


def state(P):
    with P.factory() as s:
        org = s.get(Organisation, P.orgA_id)
        return s.get(RulebookVersion, org.active_rulebook_version_id), s.get(PromptVersion, org.active_prompt_version_id)


def move(P, rb, profile=NEW):
    with P.factory() as s:
        org, admin = s.get(Organisation, P.orgA_id), s.get(User, P.ids["admin_a"])
        out = ps.adopt_profile(s, org, profile, rb, admin, "owner decision")
        s.commit()
        return out[0].version_number, out[1].version_number, out[2]


def test_the_organisation_moves_to_new_approved_versions_and_the_old_ones_are_kept(P, rb):
    rv0, pv0 = state(P)
    assert rv0.configuration["profile"] == OLD
    rv_no, pv_no, dropped = move(P, rb)
    rv, pv = state(P)
    assert rv.configuration["profile"] == NEW and rv.configuration["rulebook_version"] == "1.1.0-tuned-banner" and rv.approval_status == "APPROVED" and rv.configuration["approval"]["reason"] == "owner decision"
    assert pv.approval_status == "APPROVED" and pv.configuration["prompt_hash"] == build_prompt(rb, rb.resolve(NEW)).hash != rv0.configuration.get("prompt_hash") and dropped == []
    with P.factory() as s:
        numbers = {v.version_number for v in s.scalars(select(RulebookVersion).where(RulebookVersion.organisation_id == P.orgA_id))}
        assert rv0.version_number in numbers and rv_no in numbers                                              # the earlier version is still there, ready to be activated again


def test_a_new_evaluation_uses_the_new_profile_and_records_the_new_versions(P, rb):
    ev = ScriptedEvaluator(answers={"R001_POSITION": "NO"}, rulebook=rb)
    P.use(ev)
    move(P, rb)
    rv, pv = state(P)
    _, eid = upload(P, solid(RED[0], 400))
    run = run_of(P, eid)
    assert run.rulebook_version_id == rv.version_id and run.prompt_version_id == pv.version_id and run.evaluation_profile == NEW and run.ai_recommendation == "REJECT"
    assert [g["gate_id"] for g in run.decision_output["gates"] if g["status"] == "NO"] == ["G019"] and not any("differs" in w for w in run.warnings)


def test_what_was_learned_is_kept_where_it_still_applies_and_withdrawn_where_the_rule_is_gone(P, rb):
    with P.factory() as s:
        org, admin = s.get(Organisation, P.orgA_id), s.get(User, P.ids["admin_a"])
        rs.manual_edit(s, org, admin, GAZE, GOOD, "kept", "LOOSEN", rb, rule_catalogue(rb, OLD))
        rs.manual_edit(s, org, admin, "R001_STRAIGHT", "Is the banner level with the floor?", "this rule is removed by the new profile", "NEUTRAL", rb, rule_catalogue(rb, OLD))
        s.commit()
    rv_no, pv_no, dropped = move(P, rb)
    _, pv = state(P)
    assert dropped == ["R001_STRAIGHT"] and pv.configuration["rule_text"] == {GAZE: GOOD}
    with P.factory() as s:
        org = s.get(Organisation, P.orgA_id)
        assert set(rs.active_edits(s, org.organisation_id)) == {GAZE}
        gone = [x for x in rs.overview(s, org, False)["drafts"] if x["target_id"] == "R001_STRAIGHT"][0]
        assert gone["status"] == "WITHDRAWN" and "does not exist in the profile layered_tuned_v1" in gone["history"][-1]["reason"] and gone["history"][-1]["payload"]["by"] == "a profile change"
    assert build_prompt(rb, rb.resolve(NEW), rule_text={GAZE: GOOD}).hash == pv.configuration["prompt_hash"]


def test_notes_survive_the_move(P, rb):
    from ngo_agent import note_service as ns

    with P.factory() as s:
        org = s.get(Organisation, P.orgA_id)
        ns._set_notes(s, org, {"G015": "A note about the gaze."}, rb, "test")
        s.commit()
    move(P, rb)
    assert state(P)[1].configuration["notes"] == {"G015": "A note about the gaze."}


def test_the_move_is_refused_for_an_unknown_profile_the_same_profile_or_settings_the_new_profile_cannot_apply(P, rb):
    with P.factory() as s:
        org, admin = s.get(Organisation, P.orgA_id), s.get(User, P.ids["admin_a"])
        with pytest.raises(Invalid, match="unknown profile"):
            ps.adopt_profile(s, org, "nope", rb, admin, "x")
        with pytest.raises(Conflict, match="already uses"):
            ps.adopt_profile(s, org, OLD, rb, admin, "x")
        cur = s.get(RulebookVersion, org.active_rulebook_version_id)
        cur.configuration = {**cur.configuration, "overlay": [{"change_id": "c1", "kind": "GATE_SOFTEN", "target": "G018", "set": {}}]}   # a learned change to a gate the new profile does not have
        s.flush()
        with pytest.raises(Conflict, match="cannot be applied"):
            ps.adopt_profile(s, org, NEW, rb, admin, "x")
        s.rollback()
    assert state(P)[0].configuration["profile"] == OLD                                                         # nothing was changed
