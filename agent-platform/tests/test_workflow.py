"""Phase 2A behaviours, through the HTTP API. The fourteen listed behaviours and scenarios A to G are marked."""

from __future__ import annotations

import pytest
from sqlalchemy import select, text

from ngo_agent.evaluation_service import verify_integrity
from ngo_agent.evaluators import ScriptedEvaluator
from ngo_agent.models import CriterionCorrection, EvaluationRun, FeedbackEvent, LearningExample, LifecycleEvent, Photo
from conftest import jpeg_bytes

REJECT_ANSWERS = {"R003_FACE": "NO"}          # a hard-gate failure: G004 -> the engine rejects
WHY = "Faces are clear and the activity is easy to understand."   # every decision needs a written reason


def evaluation_of(P, photo_resp):
    return photo_resp.json()["evaluation_id"]


def open_review(P, who, ev_id):
    r = P.client.post(f"/evaluations/{ev_id}/review", headers=P.login(who))
    assert r.status_code in (200, 201), r.text
    return r.json()["review_id"]


# ------------------------------------------------------------------------------------------------------------------ 1-2 upload and evaluation
def test_upload_creates_photo_and_one_immutable_evaluation_run(P):
    r = P.upload()
    assert r.status_code == 201 and r.json()["created"] is True
    ev = P.client.get(f"/evaluations/{evaluation_of(P, r)}", headers=P.login("rev_a")).json()
    assert ev["processing_status"] == "AI_EVALUATED" and ev["ai_recommendation"] == "APPROVED_PENDING_PUBLICATION_CHECKS"
    assert ev["layer_1"] and ev["layer_2"]["gates"] and ev["layer_3"]["dimensions"] and ev["layer_4"] is not None
    assert ev["rulebook_version"].endswith("layered_proposed_v1") and ev["prompt_version"].startswith("1.0:11427c79d96c569a")
    assert ev["integrity_ok"] is True and ev["content_hash"]
    photo = P.client.get(f"/photos/{r.json()['photo']['photo_id']}", headers=P.login("rev_a")).json()
    assert photo["photo"]["photo_status"] == "PENDING_REVIEW"
    stages = [e["stage"] for e in photo["lifecycle"]]
    assert stages == ["PHOTO_UPLOADED", "AI_EVALUATION_STARTED", "AI_EVALUATION_COMPLETED", "AI_RECOMMENDATION_GENERATED", "WAITING_FOR_HUMAN_REVIEW"]


def test_identical_upload_is_stored_once_and_not_re_evaluated(P):
    a, b = P.upload(seed=5), P.upload(seed=5)
    assert a.json()["created"] and not b.json()["created"] and b.json()["duplicate"] and b.status_code == 200
    assert a.json()["photo"]["photo_id"] == b.json()["photo"]["photo_id"] and b.json()["evaluation_id"] is None
    with P.factory() as s:
        assert len(s.scalars(select(EvaluationRun)).all()) == 1


def test_bad_uploads_are_refused(P):
    h = P.login("rev_a")
    assert P.client.post("/photos/upload", files={"file": ("x.jpg", b"not an image", "image/jpeg")}, headers=h).status_code == 422
    assert P.client.post("/photos/upload", files={"file": ("x.jpg", b"", "image/jpeg")}, headers=h).status_code == 422
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "BMP")
    assert P.client.post("/photos/upload", files={"file": ("x.bmp", buf.getvalue(), "image/bmp")}, headers=h).status_code == 422
    assert P.client.post("/photos/upload", files={"file": ("x.jpg", jpeg_bytes(1), "image/jpeg")}).status_code == 401


def test_evaluation_failure_is_recorded_not_approved_and_retry_makes_a_new_run(P):
    P.use(ScriptedEvaluator(fail_with="the model response did not validate: Bearer abc.def.ghi AIza" + "x" * 30, rulebook=P.rb))
    r = P.upload()
    h = P.login("rev_a")
    ev = P.client.get(f"/evaluations/{evaluation_of(P, r)}", headers=h).json()
    assert ev["processing_status"] == "FAILED" and ev["ai_recommendation"] is None
    assert "abc.def" not in ev["error"] and "AIza" not in ev["error"]
    pid = r.json()["photo"]["photo_id"]
    assert P.client.get(f"/photos/{pid}", headers=h).json()["photo"]["photo_status"] == "FAILED"
    assert P.client.post(f"/evaluations/{ev['evaluation_id']}/review", headers=h).status_code == 409   # a failed run cannot be reviewed
    P.use(ScriptedEvaluator(rulebook=P.rb))                                                              # the retry works
    again = P.client.post(f"/photos/{pid}/evaluations", json={}, headers=h)
    assert again.status_code == 202
    photo = P.client.get(f"/photos/{pid}", headers=h).json()
    assert [e["processing_status"] for e in photo["evaluations"]] == ["AI_EVALUATED", "FAILED"] and photo["photo"]["photo_status"] == "PENDING_REVIEW"


# ------------------------------------------------------------------------------------------------------------------------ immutability
def test_completed_evaluation_cannot_be_changed_or_deleted_even_by_raw_sql(P):
    r = P.upload()
    eid = evaluation_of(P, r)
    with P.engine.begin() as c:
        with pytest.raises(Exception, match="immutable"):
            c.execute(text("UPDATE evaluation_runs SET ai_recommendation='APPROVE' WHERE evaluation_id=:i"), {"i": eid})
    with P.engine.begin() as c:
        with pytest.raises(Exception, match="never be deleted"):
            c.execute(text("DELETE FROM evaluation_runs WHERE evaluation_id=:i"), {"i": eid})
    with P.factory() as s:
        run = s.get(EvaluationRun, eid)
        run.quality_score = 1.0
        with pytest.raises(Exception, match="immutable"):
            s.commit()


def test_tampering_with_triggers_disabled_is_still_detected(P):
    from ngo_agent.triggers import install_triggers, remove_triggers
    eid = evaluation_of(P, P.upload())
    with P.factory() as s:
        assert verify_integrity(s.get(EvaluationRun, eid), P.storage)
    remove_triggers(P.engine)
    with P.engine.begin() as c:
        tamper = ("jsonb_set(decision_output,'{recommendation}','\"APPROVE\"')" if P.engine.dialect.name == "postgresql"
                  else "json_set(decision_output,'$.recommendation','APPROVE')")
        c.execute(text(f"UPDATE evaluation_runs SET decision_output={tamper} WHERE evaluation_id=:i"), {"i": eid})
    install_triggers(P.engine)
    with P.factory() as s:
        assert verify_integrity(s.get(EvaluationRun, eid), P.storage) is False
    raw = P.storage.root / next(iter((P.storage.root / P.orgA_id / "raw").iterdir())).relative_to(P.storage.root)
    assert raw.exists()


def test_events_and_photo_identity_are_append_only(P):
    pid = P.upload().json()["photo"]["photo_id"]
    with P.engine.begin() as c:
        with pytest.raises(Exception, match="append-only"):
            c.execute(text("UPDATE lifecycle_events SET stage='X'"))
    with P.engine.begin() as c:
        with pytest.raises(Exception, match="append-only"):
            c.execute(text("DELETE FROM lifecycle_events"))
    with P.engine.begin() as c:
        with pytest.raises(Exception, match="stored identity"):
            c.execute(text("UPDATE photos SET file_hash='0' WHERE photo_id=:p"), {"p": pid})
        with pytest.raises(Exception):
            c.execute(text("DELETE FROM photos WHERE photo_id=:p"), {"p": pid})


def test_database_refuses_an_evaluation_that_mixes_organisations(P):
    pid = P.upload("rev_a").json()["photo"]["photo_id"]
    with P.factory() as s:
        run = s.scalar(select(EvaluationRun))
        bad = EvaluationRun(photo_id=pid, organisation_id=P.orgB_id, prompt_version_id=run.prompt_version_id, rulebook_version_id=run.rulebook_version_id)
        s.add(bad)
        with pytest.raises(Exception):
            s.commit()


# ---------------------------------------------------------------------------------------------------------------- scenario A: human agrees
def test_scenario_a_human_accepts_ai(P):
    r = P.upload()
    eid = evaluation_of(P, r)
    h = P.login("rev_a")
    before = P.client.get(f"/evaluations/{eid}", headers=h).json()
    rid = open_review(P, "rev_a", eid)
    ok = P.client.post(f"/feedback/{rid}/submit", json={"decision": "APPROVE", "comment": WHY}, headers={**h, "Idempotency-Key": "k1"})
    assert ok.status_code == 200 and ok.json()["review"]["overall_agreement_with_ai"] == "AGREE" and ok.json()["replayed"] is False
    assert ok.json()["review"]["human_final_decision"] == "APPROVE" and ok.json()["review"]["corrections"] == []
    after = P.client.get(f"/evaluations/{eid}", headers=h).json()
    assert after["content_hash"] == before["content_hash"] and after["decision"] == before["decision"] and after["ai_recommendation"] == before["ai_recommendation"]
    photo = P.client.get(f"/photos/{r.json()['photo']['photo_id']}", headers=h).json()
    assert photo["photo"]["photo_status"] == "APPROVED"
    assert [e["stage"] for e in photo["lifecycle"]][-5:] == ["WAITING_FOR_HUMAN_REVIEW", "HUMAN_REVIEW_IN_PROGRESS", "HUMAN_ACCEPTS_AI", "FEEDBACK_RECORDED", "LEARNING_EXAMPLE_CREATED"]
    types = [e["event_type"] for e in P.client.get(f"/feedback/{rid}", headers=h).json()["events"]]
    assert types == ["REVIEW_OPENED", "AI_ACCEPTED", "LEARNING_EXAMPLE_CREATED", "FEEDBACK_SUBMITTED"]


# ---------------------------------------------------------------------------------------------------------------- scenario B: human disregards the AI
def test_scenario_b_disregarding_the_ai_needs_a_written_reason(P):
    P.use(ScriptedEvaluator(answers=REJECT_ANSWERS, rulebook=P.rb))
    r = P.upload()
    eid, h = evaluation_of(P, r), P.login("rev_a")
    assert P.client.get(f"/evaluations/{eid}", headers=h).json()["ai_recommendation"] == "REJECT"
    rid = open_review(P, "rev_a", eid)
    for body in ({}, {"comment": "   "}, {"comment": "no"}, {"reason_categories": ["RULE_APPLICABILITY_ERROR"]}):   # a ticked category is not a written reason
        refused = P.client.post(f"/feedback/{rid}/submit", json={"decision": "APPROVE", **body}, headers=h)
        assert refused.status_code == 422 and "reason box" in refused.json()["detail"], body
    assert P.client.get(f"/feedback/{rid}", headers=h).json()["review_status"] == "REVIEW_IN_PROGRESS"   # a refusal writes nothing
    why = "The person is in the background and is not a featured beneficiary."
    ok = P.client.post(f"/feedback/{rid}/submit", json={"decision": "APPROVE", "comment": why}, headers=h).json()["review"]
    assert ok["overall_agreement_with_ai"] == "DISAGREE" and ok["human_final_decision"] == "APPROVE" and ok["reviewer_comments"] == why
    final = [c for c in ok["corrections"] if c["criterion_type"] == "FINAL_DECISION"]
    assert len(final) == 1 and final[0]["ai_predicted_value"] == "REJECT" and final[0]["human_corrected_value"] == "APPROVE" and final[0]["reviewer_comment"] == why
    assert next(e for e in ok["events"] if e["event_type"] == "AI_OVERRIDDEN")["payload"]["reason"] == why
    ev = P.client.get(f"/evaluations/{eid}", headers=h).json()
    assert ev["ai_recommendation"] == "REJECT"                           # the AI's own record still says REJECT
    assert ev["my_review"]["human_final_decision"] == "APPROVE"          # three separate fields, never conflated
    assert P.client.get(f"/photos/{r.json()['photo']['photo_id']}", headers=h).json()["photo"]["photo_status"] == "APPROVED"


def test_every_decision_needs_a_written_reason_whether_or_not_the_person_agrees_with_the_ai(P):
    # the AI approves; the person disagrees: a reason is needed
    eid, h = evaluation_of(P, P.upload()), P.login("rev_a")
    assert P.client.get(f"/evaluations/{eid}", headers=h).json()["ai_recommendation"] in ("APPROVE", "APPROVED_PENDING_PUBLICATION_CHECKS")
    rid = open_review(P, "rev_a", eid)
    refused = P.client.post(f"/feedback/{rid}/submit", json={"decision": "REJECT"}, headers=h)
    assert refused.status_code == 422 and "disregarding" in refused.json()["detail"]
    ok = P.client.post(f"/feedback/{rid}/submit", json={"decision": "REJECT", "comment": "the banner is unreadable and the framing is poor"}, headers=h)
    assert ok.status_code == 200 and ok.json()["review"]["overall_agreement_with_ai"] == "DISAGREE"
    # the person agrees with the AI: a reason is still needed, because the AI may not have captured every point
    eid2 = evaluation_of(P, P.upload(seed=5))
    rid2 = open_review(P, "rev_a", eid2)
    for body in ({}, {"comment": "  "}, {"comment": "ok"}, {"reason_categories": ["OTHER"]}):
        refused = P.client.post(f"/feedback/{rid2}/submit", json={"decision": "APPROVE", **body}, headers=h)
        assert refused.status_code == 422 and "you agree" in refused.json()["detail"] and "may not have captured" in refused.json()["detail"], body
    assert P.client.get(f"/feedback/{rid2}", headers=h).json()["review_status"] == "REVIEW_IN_PROGRESS"   # a refusal writes nothing
    done = P.client.post(f"/feedback/{rid2}/submit", json={"decision": "APPROVE", "comment": WHY}, headers=h).json()["review"]
    assert done["overall_agreement_with_ai"] == "AGREE" and done["reviewer_comments"] == WHY
    assert next(e for e in done["events"] if e["event_type"] == "AI_ACCEPTED")["payload"]["reason"] == WHY   # the agreeing reason is on record too
    with P.factory() as s:   # and it travels with the learning example, so the agent can learn which points matter
        assert any(r["criterion_id"] == "final_decision" and r["comment"] == WHY for ex in s.scalars(select(LearningExample)).all() for r in ex.verified_reasons)


def test_when_the_ai_asks_for_a_person_to_decide_a_reason_is_needed_too(P):
    r, eid, h, rid = _blocked(P)          # a publication blocker makes the engine recommend HUMAN_REVIEW
    refused = P.client.post(f"/feedback/{rid}/submit", json={"decision": "REJECT"}, headers=h)
    assert refused.status_code == 422 and "reason box" in refused.json()["detail"]
    ok = P.client.post(f"/feedback/{rid}/submit", json={"decision": "REJECT", "comment": "The number plate is legible, so it cannot be used."}, headers=h)
    assert ok.status_code == 200 and ok.json()["review"]["overall_agreement_with_ai"] == "AI_DEFERRED"


# ------------------------------------------------------------------------------------------------------------ scenario C: correct one gate
def test_scenario_c_correct_an_individual_gate_without_touching_the_original(P):
    P.use(ScriptedEvaluator(answers=REJECT_ANSWERS, rulebook=P.rb))
    eid, h = evaluation_of(P, P.upload()), P.login("rev_a")
    before = P.client.get(f"/evaluations/{eid}", headers=h).json()
    gate = next(g for g in before["layer_2"]["gates"] if g["gate_id"] == "G004")
    assert gate["status"] == "NO"
    open_review(P, "rev_a", eid)
    body = {"corrections": [{"criterion_type": "MANDATORY_GATE", "criterion_id": "G004", "human_corrected_value": "YES", "correction_reason": ["RULE_APPLICABILITY_ERROR", "SUBJECT_IDENTIFICATION_ERROR"],
                             "reviewer_comment": "The person is in the background and is not a featured beneficiary.", "ai_predicted_value": "YES"}]}   # a client-supplied AI value is ignored
    r = P.client.patch(f"/evaluations/{eid}/corrections", json=body, headers=h)
    assert r.status_code == 200
    c = r.json()["corrections"][0]
    assert c["ai_predicted_value"] == "NO" and c["human_corrected_value"] == "YES" and c["revision"] == 1 and "validation_status" not in c
    assert c["correction_reason"] == ["RULE_APPLICABILITY_ERROR", "SUBJECT_IDENTIFICATION_ERROR"]
    after = P.client.get(f"/evaluations/{eid}", headers=h).json()
    assert after["content_hash"] == before["content_hash"] and after["layer_2"] == before["layer_2"]
    # a second correction of the same gate is a new revision that supersedes the first; both stay
    r2 = P.client.patch(f"/evaluations/{eid}/corrections", json={"corrections": [{"criterion_type": "MANDATORY_GATE", "criterion_id": "G004", "human_corrected_value": "CANNOT_TELL",
                                                                                 "correction_reason": ["MISSING_CONTEXT"]}]}, headers=h).json()["corrections"][0]
    assert r2["revision"] == 2 and r2["supersedes_correction_id"] == c["correction_id"] and r2["ai_predicted_value"] == "NO"
    with P.factory() as s:
        assert len(s.scalars(select(CriterionCorrection)).all()) == 2


def test_invalid_corrections_are_refused_and_a_batch_is_all_or_nothing(P):
    eid, h = evaluation_of(P, P.upload()), P.login("rev_a")
    assert P.client.patch(f"/evaluations/{eid}/corrections", json={"corrections": [{"criterion_type": "MANDATORY_GATE", "criterion_id": "G004", "human_corrected_value": "NO",
                                                                                   "correction_reason": ["OTHER"]}]}, headers=h).status_code == 409   # no open review yet
    open_review(P, "rev_a", eid)
    good = {"criterion_type": "MANDATORY_GATE", "criterion_id": "G004", "human_corrected_value": "NO", "correction_reason": ["VISUAL_PERCEPTION_ERROR"]}
    cases = [({**good, "criterion_id": "G999"}, "does not exist"), ({**good, "human_corrected_value": "YES"}, "same as the AI"), ({**good, "human_corrected_value": "MAYBE"}, "must be one of"),
             ({**good, "correction_reason": []}, "reason"), ({**good, "correction_reason": ["BECAUSE"]}, "unknown correction reason"),
             ({**good, "criterion_type": "WHATEVER"}, "unknown criterion_type"), ({**good, "criterion_type": "FINAL_DECISION"}, "submitted"),
             ({"criterion_type": "QUALITY_DIMENSION", "criterion_id": "composition", "human_corrected_value": 9, "correction_reason": ["OTHER"]}, "0 to 4"),
             ({"criterion_type": "ACTIVITY_TYPE", "criterion_id": "activity_type", "human_corrected_value": "MOON_LANDING", "correction_reason": ["ACTIVITY_CLASSIFICATION_ERROR"]}, "activity type")]
    for item, needle in cases:
        r = P.client.patch(f"/evaluations/{eid}/corrections", json={"corrections": [item]}, headers=h)
        assert r.status_code == 422 and needle in r.json()["detail"], (item, r.text)
    r = P.client.patch(f"/evaluations/{eid}/corrections", json={"corrections": [{**good, "human_corrected_value": "NO"}, {**good, "criterion_id": "G999"}]}, headers=h)
    assert r.status_code == 422
    with P.factory() as s:
        assert s.scalars(select(CriterionCorrection)).all() == []     # nothing from the failed batch was kept


# --------------------------------------------------------------------------------------------------------------- scenario D: tags and activity
def test_scenario_d_human_adds_a_missed_tag_removes_a_wrong_one_and_fixes_the_activity(P):
    P.use(ScriptedEvaluator(positives=["warm interaction"], concerns=["blurred banner"], rulebook=P.rb))
    eid, h = evaluation_of(P, P.upload()), P.login("rev_a")
    open_review(P, "rev_a", eid)
    items = [{"criterion_type": "TAG_ADDED", "criterion_id": "cluttered background", "human_corrected_value": "PRESENT", "correction_reason": ["VISUAL_PERCEPTION_ERROR"]},
             {"criterion_type": "TAG_REMOVED", "criterion_id": "blurred banner", "human_corrected_value": "ABSENT", "correction_reason": ["VISUAL_PERCEPTION_ERROR"]},
             {"criterion_type": "ACTIVITY_TYPE", "criterion_id": "activity_type", "human_corrected_value": "EDUCATION_SUPPORT", "correction_reason": ["ACTIVITY_CLASSIFICATION_ERROR"]}]
    got = P.client.patch(f"/evaluations/{eid}/corrections", json={"corrections": items}, headers=h).json()["corrections"]
    assert got[0]["ai_predicted_value"] is None and got[1]["ai_predicted_value"] == "PRESENT" and got[2]["ai_predicted_value"] == "FOOD_DISTRIBUTION"
    assert P.client.patch(f"/evaluations/{eid}/corrections", json={"corrections": [{"criterion_type": "TAG_ADDED", "criterion_id": "warm interaction", "human_corrected_value": "PRESENT",
                                                                                   "correction_reason": ["OTHER"]}]}, headers=h).status_code == 422   # the AI already reported it
    with P.factory() as s:
        types = [e.event_type for e in s.scalars(select(FeedbackEvent).order_by(FeedbackEvent.created_at))]
    assert {"TAG_ADDED", "TAG_REMOVED", "ACTIVITY_CHANGED"} <= set(types)


# ---------------------------------------------------------------------------------------------------------------- idempotent, retry-safe submit
def test_submit_is_safe_to_retry(P):
    eid, h = evaluation_of(P, P.upload()), P.login("rev_a")
    rid = open_review(P, "rev_a", eid)
    first = P.client.post(f"/feedback/{rid}/submit", json={"decision": "APPROVE", "comment": WHY}, headers={**h, "Idempotency-Key": "abc"})
    again = P.client.post(f"/feedback/{rid}/submit", json={"decision": "APPROVE", "comment": WHY}, headers={**h, "Idempotency-Key": "abc"})
    assert first.json()["replayed"] is False and again.status_code == 200 and again.json()["replayed"] is True
    assert P.client.post(f"/feedback/{rid}/submit", json={"decision": "REJECT", "reason_categories": ["OTHER"]}, headers=h).status_code == 409
    with P.factory() as s:
        assert len(s.scalars(select(FeedbackEvent).where(FeedbackEvent.event_type == "FEEDBACK_SUBMITTED")).all()) == 1   # the retry wrote nothing twice
    assert P.client.post(f"/feedback/{rid}/submit", json={"decision": "APPROVE"}, headers=P.login("rev2_a")).status_code == 403   # not the reviewer's own review


def test_open_review_is_resumed_not_duplicated_and_a_second_reviewer_can_review(P):
    eid = evaluation_of(P, P.upload())
    a = P.client.post(f"/evaluations/{eid}/review", headers=P.login("rev_a"))
    b = P.client.post(f"/evaluations/{eid}/review", headers=P.login("rev_a"))
    c = P.client.post(f"/evaluations/{eid}/review", headers=P.login("rev2_a"))
    assert a.status_code == 201 and b.status_code == 200 and a.json()["review_id"] == b.json()["review_id"] and c.json()["review_id"] != a.json()["review_id"]


# ---------------------------------------------------------------------------------------------------------------- scenario E: the submission is final
def _overridden(P, reasons=None, seed=1):
    """The AI rejects; the person corrects one gate and disregards the verdict with a written reason."""
    P.use(ScriptedEvaluator(answers=REJECT_ANSWERS, rulebook=P.rb))
    r = P.upload(seed=seed)
    eid, h = evaluation_of(P, r), P.login("rev_a")
    rid = open_review(P, "rev_a", eid)
    P.client.patch(f"/evaluations/{eid}/corrections", json={"corrections": [{"criterion_type": "MANDATORY_GATE", "criterion_id": "G004", "human_corrected_value": "YES",
                                                                             "correction_reason": reasons or ["RULE_APPLICABILITY_ERROR"], "reviewer_comment": "background"}]}, headers=h)
    done = P.client.post(f"/feedback/{rid}/submit", json={"decision": "APPROVE", "comment": "The person is in the background, not a featured beneficiary.",
                                                          "reason_categories": reasons or []}, headers=h)
    assert done.status_code == 200, done.text
    return eid, rid, h


def test_scenario_e_a_submitted_review_is_final_and_becomes_an_inactive_learning_example(P):
    eid, rid, h = _overridden(P)
    review = P.client.get(f"/feedback/{rid}", headers=h).json()
    assert review["review_status"] == "FEEDBACK_SUBMITTED" and "LEARNING_EXAMPLE_CREATED" in [e["event_type"] for e in review["events"]]
    submitted = next(e for e in review["events"] if e["event_type"] == "FEEDBACK_SUBMITTED")["payload"]
    assert submitted["learning_example_id"] and submitted["not_a_learning_example"] is None
    listed = P.client.get(f"/organisations/{P.orgA_id}/learning-examples", headers=P.login("admin_a")).json()["learning_examples"]
    assert len(listed) == 1 and listed[0]["verified_decision"] == "APPROVE" and listed[0]["active_for_retrieval"] is False and listed[0]["human_review_id"] == rid
    with P.factory() as s:
        row = s.scalars(select(LearningExample)).one()
        assert row.verified_gate_results["G004"] == "YES"                              # the person's correction, applied
        assert any(r["criterion_id"] == "G004" for r in row.verified_reasons)
        assert any(r["criterion_id"] == "final_decision" and "background" in r["comment"] for r in row.verified_reasons)   # the written reason travels with the example
    # nobody validates anything any more, and the decision cannot be reopened or changed
    assert P.client.post("/feedback/corrections/anything/validate", json={"status": "VALIDATED"}, headers=P.login("admin_a")).status_code in (404, 405)
    assert P.client.post(f"/feedback/{rid}/approve-for-learning", headers=P.login("admin_a")).status_code in (404, 405)
    assert P.client.post(f"/feedback/{rid}/submit", json={"decision": "REJECT", "comment": "I changed my mind about this one"}, headers=h).status_code == 409
    assert P.client.get(f"/evaluations/{eid}", headers=h).json()["integrity_ok"] is True        # the AI evaluation is untouched by all of it


@pytest.mark.parametrize("reason", ["POLICY_CHANGE", "HUMAN_REVIEW_ERROR"])
def test_a_policy_change_or_a_self_declared_error_never_becomes_a_learning_example(P, reason):
    eid, rid, h = _overridden(P, reasons=[reason])
    review = P.client.get(f"/feedback/{rid}", headers=h).json()
    assert review["human_final_decision"] == "APPROVE"                                          # the decision itself stands
    submitted = next(e for e in review["events"] if e["event_type"] == "FEEDBACK_SUBMITTED")["payload"]
    assert submitted["learning_example_id"] is None and reason in submitted["not_a_learning_example"]
    assert "LEARNING_EXAMPLE_CREATED" not in [e["event_type"] for e in review["events"]]
    assert P.client.get(f"/organisations/{P.orgA_id}/learning-examples", headers=P.login("admin_a")).json()["learning_examples"] == []


def test_corrections_and_reviews_are_fully_frozen_and_the_validation_columns_are_gone(P):
    from sqlalchemy import inspect
    eid, rid, h = _overridden(P)
    cid = next(c["correction_id"] for c in P.client.get(f"/feedback/{rid}", headers=h).json()["corrections"] if c["criterion_id"] == "G004")
    for sql in ("UPDATE criterion_corrections SET human_corrected_value='\"NO\"' WHERE correction_id=:c", "UPDATE criterion_corrections SET reviewer_comment='x' WHERE correction_id=:c"):
        with P.engine.begin() as c:
            with pytest.raises(Exception, match="cannot be edited"):
                c.execute(text(sql), {"c": cid})
    with P.engine.begin() as c:
        with pytest.raises(Exception, match="never be deleted"):
            c.execute(text("DELETE FROM criterion_corrections"))
    with P.engine.begin() as c:
        with pytest.raises(Exception, match="cannot be edited"):
            c.execute(text("UPDATE human_reviews SET human_final_decision='REJECT'"))
    columns = {c["name"] for c in inspect(P.engine).get_columns("criterion_corrections")}
    assert not {"validation_status", "validated_by", "validated_at", "validation_comment"} & columns


# ---------------------------------------------------------------------------------------------------------------- scenario F: re-evaluation
def test_scenario_f_re_evaluation_creates_a_new_run_and_leaves_the_old_one_alone(P):
    r = P.upload()
    pid, eid, h = r.json()["photo"]["photo_id"], evaluation_of(P, r), P.login("rev_a")
    hist_before = P.client.get(f"/evaluations/{eid}", headers=h).json()
    # a new, approved rulebook version
    ha = P.login("admin_a")
    vers = P.client.get(f"/organisations/{P.orgA_id}/versions", headers=ha).json()
    v1 = next(v for v in vers["rulebook_versions"] if v["configuration"]["profile"] == "layered_proposed_v1")
    new = P.client.post(f"/organisations/{P.orgA_id}/versions", json={"kind": "rulebook", "version_number": "1.1.0:v1-recheck", "configuration": {**v1["configuration"]},
                                                                      "change_description": "same rules, new registration"}, headers=ha)
    assert new.status_code == 201
    nid = new.json()["version_id"]
    assert P.client.post(f"/photos/{pid}/evaluations", json={"rulebook_version_id": nid}, headers=h).status_code == 403          # a reviewer cannot choose a version
    assert P.client.post(f"/photos/{pid}/evaluations", json={"rulebook_version_id": nid}, headers=ha).status_code == 403         # an unapproved version is refused
    assert P.client.post(f"/organisations/{P.orgA_id}/versions/rulebook/{nid}/approve", headers=ha).status_code == 403           # the creator cannot approve it
    P.client.post(f"/organisations/{P.orgA_id}/users", json={"email": "admin2-a@example.org", "password": "another long password", "role": "ORG_ADMIN"}, headers=ha)
    h2 = {"Authorization": "Bearer " + P.client.post("/auth/login", json={"email": "admin2-a@example.org", "password": "another long password"}).json()["access_token"]}
    assert P.client.post(f"/organisations/{P.orgA_id}/versions/rulebook/{nid}/approve", headers=h2).status_code == 200
    r2 = P.client.post(f"/photos/{pid}/evaluations", json={"rulebook_version_id": nid}, headers=ha)
    assert r2.status_code == 202
    photo = P.client.get(f"/photos/{pid}", headers=h).json()
    assert len(photo["evaluations"]) == 2 and {e["rulebook_version_id"] for e in photo["evaluations"]} == {v1["version_id"], nid}
    assert P.client.get(f"/evaluations/{eid}", headers=h).json()["content_hash"] == hist_before["content_hash"]            # the earlier evaluation is untouched
    assert any(e["stage"] == "RE_EVALUATION_REQUESTED" for e in photo["lifecycle"])
    hist = P.client.get(f"/organisations/{P.orgA_id}/evaluation-history?photo_id={pid}", headers=h).json()["evaluations"]
    assert len(hist) == 2


def test_re_evaluation_after_a_human_decision_does_not_reopen_the_photo(P):
    r = P.upload()
    pid, eid, h = r.json()["photo"]["photo_id"], evaluation_of(P, r), P.login("rev_a")
    rid = open_review(P, "rev_a", eid)
    P.client.post(f"/feedback/{rid}/submit", json={"decision": "APPROVE", "comment": WHY}, headers=h)
    assert P.client.post(f"/photos/{pid}/evaluations", json={}, headers=h).status_code == 202
    assert P.client.get(f"/photos/{pid}", headers=h).json()["photo"]["photo_status"] == "APPROVED"      # the person's decision stands


def test_draft_v2_cannot_be_used_unless_an_administrator_asks_for_an_experiment(P):
    pid, h, ha = P.upload().json()["photo"]["photo_id"], P.login("rev_a"), P.login("admin_a")
    vers = P.client.get(f"/organisations/{P.orgA_id}/versions", headers=ha).json()
    v2r = next(v for v in vers["rulebook_versions"] if v["configuration"]["profile"] == "layered_proposed_v2")
    v2p = next(v for v in vers["prompt_versions"] if v["configuration"]["profile"] == "layered_proposed_v2")
    assert v2r["approval_status"] == "DRAFT"
    assert P.client.post(f"/organisations/{P.orgA_id}/versions/rulebook/{v2r['version_id']}/activate", headers=ha).status_code == 403
    assert P.client.post(f"/photos/{pid}/evaluations", json={"rulebook_version_id": v2r["version_id"], "prompt_version_id": v2p["version_id"]}, headers=ha).status_code == 403
    ok = P.client.post(f"/photos/{pid}/evaluations", json={"rulebook_version_id": v2r["version_id"], "prompt_version_id": v2p["version_id"], "allow_unapproved": True}, headers=ha)
    assert ok.status_code == 202
    ev = P.client.get(f"/evaluations/{ok.json()['evaluation_id']}", headers=h).json()
    assert ev["evaluation_profile"] == "layered_proposed_v2" and any("EXPERIMENTAL" in w for w in ev["warnings"])
    assert ev["decision"]["policy_validation_required"] is True
    assert P.client.get(f"/organisations/{P.orgA_id}/versions", headers=ha).json()["active_rulebook_version_id"] != v2r["version_id"]   # still V1


# ---------------------------------------------------------------------------------------------------------------- scenario G: two organisations
def test_scenario_g_two_organisations_are_fully_isolated(P):
    ra, rb_ = P.upload("rev_a", seed=11), P.upload("rev_b", seed=12)
    pa, pb = ra.json()["photo"]["photo_id"], rb_.json()["photo"]["photo_id"]
    ea, eb = evaluation_of(P, ra), evaluation_of(P, rb_)
    ha, hb = P.login("rev_a"), P.login("rev_b")
    rid_a = open_review(P, "rev_a", ea)
    P.client.post(f"/feedback/{rid_a}/submit", json={"decision": "APPROVE", "comment": WHY}, headers=ha)
    # B sees none of A's records, and cannot learn that they exist: always 404, never 403
    for url in (f"/photos/{pa}", f"/photos/{pa}/image", f"/evaluations/{ea}", f"/feedback/{rid_a}", f"/organisations/{P.orgA_id}/feedback", f"/organisations/{P.orgA_id}/evaluation-history",
                f"/organisations/{P.orgA_id}/versions"):
        assert P.client.get(url, headers=hb).status_code == 404, url
    assert P.client.post(f"/evaluations/{ea}/review", headers=hb).status_code == 404
    assert P.client.patch(f"/evaluations/{ea}/corrections", json={"corrections": []}, headers=hb).status_code == 404
    assert P.client.post(f"/photos/{pa}/evaluations", json={}, headers=hb).status_code == 404
    assert P.client.post(f"/feedback/{rid_a}/submit", json={"decision": "REJECT", "reason_categories": ["OTHER"]}, headers=hb).status_code == 404
    # lists contain only the caller's own records
    assert [p["photo_id"] for p in P.client.get("/photos", headers=hb).json()["photos"]] == [pb]
    assert P.client.get(f"/organisations/{P.orgB_id}/feedback", headers=hb).json()["reviews"] == []
    assert len(P.client.get(f"/organisations/{P.orgA_id}/feedback", headers=ha).json()["reviews"]) == 1
    # each organisation has its own versions: B cannot evaluate with A's version
    va = P.client.get(f"/organisations/{P.orgA_id}/versions", headers=P.login("admin_a")).json()["rulebook_versions"][0]["version_id"]
    assert P.client.post(f"/photos/{pb}/evaluations", json={"rulebook_version_id": va}, headers=P.login("admin_b")).status_code == 404
    # identical bytes uploaded by both organisations are two photos and two stored objects
    x, y = P.upload("rev_a", seed=99), P.upload("rev_b", seed=99)
    assert x.json()["photo"]["photo_id"] != y.json()["photo"]["photo_id"] and x.json()["photo"]["image_width"] == y.json()["photo"]["image_width"]
    with P.factory() as s:
        paths = {p.image_storage_path for p in s.scalars(select(Photo).where(Photo.file_hash == x.json()["photo"]["file_hash"]))}
    assert len(paths) == 2 and all(p.startswith((P.orgA_id, P.orgB_id)) for p in paths)
    # feedback in A never appears in B's learning examples or history
    assert P.client.get(f"/organisations/{P.orgB_id}/evaluation-history", headers=hb).json()["evaluations"] and all(e["photo_id"] in (pb, y.json()["photo"]["photo_id"])
                                                                                                                     for e in P.client.get(f"/organisations/{P.orgB_id}/evaluation-history", headers=hb).json()["evaluations"])
    with P.factory() as s:
        assert s.scalars(select(LifecycleEvent).where(LifecycleEvent.organisation_id == P.orgB_id, LifecycleEvent.photo_id == pa)).all() == []


# ------------------------------------------------------------------------------------------------------------------- no validator, server-side AI value
def test_there_is_no_validator_role_and_the_reviewer_alone_is_the_final_word(P):
    r = P.client.post(f"/organisations/{P.orgA_id}/users", json={"email": "v@example.org", "password": "a long password", "role": "VALIDATOR"}, headers=P.login("admin_a"))
    assert r.status_code == 422
    eid, rid, h = _overridden(P)
    assert P.client.get(f"/feedback/{rid}", headers=h).json()["human_final_decision"] == "APPROVE"   # no second person was needed
    assert P.client.get(f"/photos/{P.client.get(f'/evaluations/{eid}', headers=h).json()['photo_id']}", headers=h).json()["photo"]["photo_status"] == "APPROVED"


def test_the_ai_value_is_always_read_from_the_stored_evaluation_even_if_a_caller_supplies_one(P):
    from ngo_agent.models import User
    from ngo_agent.review_service import add_corrections, open_review as svc_open
    P.use(ScriptedEvaluator(answers=REJECT_ANSWERS, rulebook=P.rb))
    eid = evaluation_of(P, P.upload())
    with P.factory() as s:
        user = s.scalar(select(User).where(User.email == "rev_a@example.org"))
        svc_open(s, user, eid)
        out = add_corrections(s, user, eid, [{"criterion_type": "MANDATORY_GATE", "criterion_id": "G004", "human_corrected_value": "YES", "correction_reason": ["OTHER"],
                                              "ai_predicted_value": "YES"}])
        assert out[0].ai_predicted_value == "NO"


# --------------------------------------------------------------------------------------------------------------------- robustness
def test_a_run_stuck_in_processing_is_closed_as_failed_never_approved(P):
    from ngo_agent.evaluation_service import fail_stale_runs, request_evaluation
    from ngo_agent.models import Photo, User
    pid = P.upload().json()["photo"]["photo_id"]
    with P.factory() as s:
        user = s.scalar(select(User).where(User.email == "rev_a@example.org"))
        run = request_evaluation(s, user, s.get(Photo, pid))      # created, but the worker "crashed" before running it
        s.commit()
        eid = run.evaluation_id
    assert fail_stale_runs(P.factory, older_than_s=3600) == 0       # too recent: left alone
    assert fail_stale_runs(P.factory, older_than_s=-1) == 1
    ev = P.client.get(f"/evaluations/{eid}", headers=P.login("rev_a")).json()
    assert ev["processing_status"] == "FAILED" and ev["ai_recommendation"] is None and "did not finish" in ev["error"]


def test_heic_photographs_are_accepted(P):
    import io
    from PIL import Image
    import pillow_heif
    pillow_heif.register_heif_opener()
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 10, 90)).save(buf, "HEIF")
    r = P.client.post("/photos/upload", files={"file": ("IMG.jpg", buf.getvalue(), "image/jpeg")}, headers=P.login("rev_a"))   # a HEIC saved with a .jpg name
    assert r.status_code == 201 and r.json()["photo"]["file_format"] == "HEIF"
    assert P.client.get(f"/photos/{r.json()['photo']['photo_id']}/image", headers=P.login("rev_a")).headers["content-type"] == "image/heic"


# ------------------------------------------------------------------------------------------- approving over a publication blocker
PLATE = [{"flag_type": "CONFIDENTIAL_INFORMATION", "evidence": "vehicle number plate is legible", "confidence": "HIGH"}]
GOOD = "the plate will be blurred before publishing"


def _blocked(P, flags=PLATE):
    P.use(ScriptedEvaluator(flags=flags, rulebook=P.rb))
    r = P.upload()
    eid, h = evaluation_of(P, r), P.login("rev_a")
    return r, eid, h, open_review(P, "rev_a", eid)


def _submit(P, rid, h, **body):
    return P.client.post(f"/feedback/{rid}/submit", json={"decision": "APPROVE", "comment": WHY, "reason_categories": ["RULE_APPLICABILITY_ERROR"], **body}, headers=h)


def test_approving_over_a_publication_blocker_needs_a_written_statement_and_records_it(P):
    r, eid, h, rid = _blocked(P)
    ev = P.client.get(f"/evaluations/{eid}", headers=h).json()
    assert ev["publication_blockers"] and "CONFIDENTIAL_INFORMATION" in ev["publication_blockers"][0]
    for body in ({}, {"publication_blocker_statement": None}, {"publication_blocker_statement": "   "}, {"publication_blocker_statement": "ok"}):
        refused = _submit(P, rid, h, **body)
        assert refused.status_code == 422 and "publication blocker" in refused.json()["detail"], body
    # nothing was written by the refusals: the review is still open, the photograph is not approved, no event says so
    mine = P.client.get(f"/feedback/{rid}", headers=h).json()
    assert mine["review_status"] == "REVIEW_IN_PROGRESS" and mine["human_final_decision"] is None
    assert not {"FEEDBACK_SUBMITTED", "BLOCKER_ACKNOWLEDGED"} & {e["event_type"] for e in mine["events"]}
    assert P.client.get(f"/photos/{r.json()['photo']['photo_id']}", headers=h).json()["photo"]["photo_status"] != "APPROVED"
    ok = _submit(P, rid, h, publication_blocker_statement=GOOD)
    assert ok.status_code == 200 and ok.json()["review"]["human_final_decision"] == "APPROVE"
    ack = [e for e in ok.json()["review"]["events"] if e["event_type"] == "BLOCKER_ACKNOWLEDGED"]
    assert len(ack) == 1 and ack[0]["payload"]["statement"] == GOOD and ack[0]["payload"]["blockers"] == ev["publication_blockers"]
    assert ack[0]["actor_id"] == P.ids["rev_a"]
    submitted = next(e for e in ok.json()["review"]["events"] if e["event_type"] == "FEEDBACK_SUBMITTED")["payload"]
    assert submitted["learning_example_id"] is None and "publication blocker" in submitted["not_a_learning_example"]   # decided on context outside the photograph
    after = P.client.get(f"/evaluations/{eid}", headers=h).json()
    assert after["content_hash"] == ev["content_hash"] and after["publication_blockers"] == ev["publication_blockers"]   # the AI's record is untouched


def test_a_rejection_or_a_photograph_without_blockers_never_needs_the_statement(P):
    r, eid, h, rid = _blocked(P)
    rejected = P.client.post(f"/feedback/{rid}/submit", json={"decision": "REJECT", "comment": "A different photograph would serve the story better.", "reason_categories": ["OTHER"]}, headers=h)
    assert rejected.status_code == 200 and rejected.json()["review"]["human_final_decision"] == "REJECT"
    assert "BLOCKER_ACKNOWLEDGED" not in {e["event_type"] for e in rejected.json()["review"]["events"]}
    # no blockers: a statement the client supplies anyway is not recorded as an acknowledgement
    h2 = P.login("rev_a")
    P.use(ScriptedEvaluator(rulebook=P.rb))
    eid2 = evaluation_of(P, P.upload(seed=8))
    rid2 = open_review(P, "rev_a", eid2)
    assert P.client.get(f"/evaluations/{eid2}", headers=h2).json()["publication_blockers"] == []
    done = P.client.post(f"/feedback/{rid2}/submit", json={"decision": "APPROVE", "comment": WHY, "publication_blocker_statement": GOOD}, headers=h2)
    assert done.status_code == 200 and "BLOCKER_ACKNOWLEDGED" not in {e["event_type"] for e in done.json()["review"]["events"]}


def test_a_low_confidence_flag_is_not_a_blocker(P):
    low = [{"flag_type": "CONFIDENTIAL_INFORMATION", "evidence": "something that may be a plate", "confidence": "LOW"}]
    r, eid, h, rid = _blocked(P, flags=low)
    assert P.client.get(f"/evaluations/{eid}", headers=h).json()["publication_blockers"] == []
    done = P.client.post(f"/feedback/{rid}/submit", json={"decision": "APPROVE", "comment": WHY, "reason_categories": ["RULE_APPLICABILITY_ERROR"]}, headers=h)
    assert done.status_code == 200


def test_blocker_statement_is_part_of_the_idempotent_submission(P):
    r, eid, h, rid = _blocked(P)
    first = _submit(P, rid, {**h, "Idempotency-Key": "k"}, publication_blocker_statement=GOOD)
    again = _submit(P, rid, {**h, "Idempotency-Key": "k"}, publication_blocker_statement=GOOD)
    assert first.json()["replayed"] is False and again.json()["replayed"] is True
    other = _submit(P, rid, {**h, "Idempotency-Key": "k"}, publication_blocker_statement="a different but long enough explanation")
    assert other.status_code == 409
    with P.factory() as s:
        assert len(s.scalars(select(FeedbackEvent).where(FeedbackEvent.event_type == "BLOCKER_ACKNOWLEDGED")).all()) == 1


def test_with_a_blocker_the_engine_defers_to_a_person_and_the_approval_is_recorded_as_a_human_decision(P):
    """Under V1 a blocker makes the engine ask for a human (it never approves), so approving over it is always a human decision on an AI_DEFERRED review."""
    r, eid, h, rid = _blocked(P)
    assert P.client.get(f"/evaluations/{eid}", headers=h).json()["ai_recommendation"] == "HUMAN_REVIEW"
    ok = _submit(P, rid, h, publication_blocker_statement=GOOD).json()["review"]
    assert ok["overall_agreement_with_ai"] == "AI_DEFERRED" and ok["human_final_decision"] == "APPROVE"
