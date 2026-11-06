"""L2, the example store: embeddings, similar-photograph search inside one NGO, balanced selection, flags, the per-NGO switch, and what the model is shown."""

from __future__ import annotations

import io

import pytest
from PIL import Image
from sqlalchemy import select, text

from ngo_agent import example_service as xs
from ngo_agent.evaluation_service import _hash_content, canonical, verify_integrity
from ngo_agent.evaluators import ScriptedEvaluator
from ngo_agent.models import EvaluationRun, ExampleFlag, LearningExample, PhotoEmbedding

RED = [(200, 30, 30), (195, 40, 35), (205, 25, 40), (198, 35, 32)]      # approved group
BROWN = [(150, 90, 60), (152, 88, 62)]                                  # rejected group: similar to the reds, but less so
BLUE = [(30, 30, 200), (35, 32, 195)]                                   # unrelated
WHY = "Clear faces, a visible handover and a readable banner."


def solid(color, width=160):
    """A solid-colour JPEG; a different width gives different bytes for the same colour, so it is a different photograph."""
    buf = io.BytesIO()
    Image.new("RGB", (width, 120), color).save(buf, "JPEG")
    return buf.getvalue()


def upload(P, data, who="rev_a"):
    r = P.client.post("/photos/upload", files={"file": ("p.jpg", data, "image/jpeg")}, headers=P.login(who))
    assert r.status_code == 201, r.text
    return r.json()["photo"]["photo_id"], r.json()["evaluation_id"]


def decide(P, data, decision="APPROVE", reason=WHY, tags=(), categories=None, who="rev_a"):
    pid, eid = upload(P, data, who)
    h = P.login(who)
    rid = P.client.post(f"/evaluations/{eid}/review", headers=h).json()["review_id"]
    if tags:
        P.client.patch(f"/evaluations/{eid}/corrections", json={"corrections": [{"criterion_type": "TAG_ADDED", "criterion_id": t, "human_corrected_value": "PRESENT",
                                                                                 "correction_reason": ["VISUAL_PERCEPTION_ERROR"]} for t in tags]}, headers=h)
    body = {"decision": decision, "comment": reason, **({"reason_categories": categories} if categories else {})}
    done = P.client.post(f"/feedback/{rid}/submit", json=body, headers=h)
    assert done.status_code == 200, done.text
    return pid, eid


def enable(P, admin="admin_a", **extra):
    return P.client.put(f"/organisations/{P.ids and (P.orgA_id if admin.endswith('_a') else P.orgB_id)}/few-shot", json={"enabled": True, "confirm_cost_increase": True, **extra},
                        headers=P.login(admin))


def recorder(P):
    rec = ScriptedEvaluator(rulebook=P.rb)
    P.use(rec)
    return rec


def run_of(P, eid):
    with P.factory() as s:
        return s.get(EvaluationRun, eid)


# ------------------------------------------------------------------------------------------------------------------------------ embeddings
def test_a_photograph_is_embedded_at_upload_and_the_embedding_is_immutable(PE):
    pid, _ = upload(PE, solid(RED[0]))
    again = PE.client.post("/photos/upload", files={"file": ("p.jpg", solid(RED[0]), "image/jpeg")}, headers=PE.login("rev_a"))
    assert again.status_code == 200 and again.json()["duplicate"] is True       # the same bytes again: the same photograph, not a second embedding
    with PE.factory() as s:
        rows = s.scalars(select(PhotoEmbedding)).all()
        assert len(rows) == 1 and rows[0].photo_id == pid and rows[0].model == "thumbnail-8x8-v1" and rows[0].dim == len(rows[0].vector) == 192
    with PE.engine.begin() as c:
        with pytest.raises(Exception, match="cannot be edited"):
            c.execute(text("UPDATE photo_embeddings SET dim=1"))
    with PE.engine.begin() as c:
        with pytest.raises(Exception, match="cannot be deleted"):
            c.execute(text("DELETE FROM photo_embeddings"))


def other_platform(tmp_path, rb, name, embedder=None):
    """A second, separate platform (its own database and storage) inside one test."""
    from conftest import Platform

    (tmp_path / name).mkdir()
    return Platform(tmp_path / name, rb, embedder=embedder)


def test_without_an_embedder_nothing_is_embedded_and_an_embedding_failure_never_fails_the_upload(tmp_path, rb):
    p = other_platform(tmp_path, rb, "none")
    try:
        upload(p, solid(RED[0]))
        with p.factory() as s:
            assert s.scalars(select(PhotoEmbedding)).all() == []
    finally:
        p.close()

    class Boom:
        name = "boom"

        def embed(self, data):  # noqa: ANN001
            raise RuntimeError("model missing")

    q = other_platform(tmp_path, rb, "boom", Boom())
    try:
        pid, eid = upload(q, solid(RED[1]))
        assert q.client.get(f"/evaluations/{eid}", headers=q.login("rev_a")).json()["processing_status"] == "AI_EVALUATED"
    finally:
        q.close()


# --------------------------------------------------------------------------------------------------------------------------------- selection
def C(pid, sim, decision):
    return xs.Candidate(pid, sim, decision)


def test_selection_is_balanced_bounded_and_ordered_least_similar_first():
    pool = [C("a1", 0.99, "APPROVE"), C("a2", 0.98, "APPROVE"), C("a3", 0.97, "APPROVE"), C("r1", 0.60, "REJECT"), C("r2", 0.55, "REJECT")]
    got = xs.select_examples(pool, 3)
    assert [c.photo_id for c in got] == ["r1", "a2", "a1"]                        # the best of each class, then the next most similar; most similar last
    assert [c.photo_id for c in xs.select_examples(pool, 2)] == ["r1", "a1"]
    assert [c.photo_id for c in xs.select_examples(pool, 1)] == ["a1"]            # one slot: simply the most similar
    assert [c.photo_id for c in xs.select_examples(pool[:3], 3)] == ["a3", "a2", "a1"]   # only one class exists: top k of it
    assert [c.photo_id for c in xs.select_examples(pool, 3, min_similarity=0.7)] == ["a3", "a2", "a1"]   # a floor removes the rejected ones
    assert xs.select_examples([], 3) == [] and xs.select_examples(pool, 3, min_similarity=1.0) == []


# ------------------------------------------------------------------------------------------------------------------------- switch and settings
def test_the_feature_is_off_by_default_and_needs_an_administrator_a_confirmation_and_an_embedder(PE, tmp_path, rb):
    a, h = PE.orgA_id, PE.login("admin_a")
    off = PE.client.get(f"/organisations/{a}/few-shot", headers=h).json()
    assert off["enabled"] is False and off["k"] == 3 and off["embedder"] == "thumbnail-8x8-v1"
    assert PE.client.get(f"/organisations/{a}/few-shot", headers=PE.login("rev_a")).status_code == 403
    assert PE.client.put(f"/organisations/{a}/few-shot", json={"enabled": True, "confirm_cost_increase": True}, headers=PE.login("rev_a")).status_code == 403
    refused = PE.client.put(f"/organisations/{a}/few-shot", json={"enabled": True}, headers=h)
    assert refused.status_code == 422 and "past photographs" in refused.json()["detail"] and "46%" in refused.json()["detail"]
    for bad in ({"k": 0}, {"k": 9}, {"k": 2.5}, {"min_similarity": 1.5}, {"min_similarity": -0.1}):
        assert PE.client.put(f"/organisations/{a}/few-shot", json={"enabled": True, "confirm_cost_increase": True, **bad}, headers=h).status_code == 422, bad
    assert enable(PE, k=2, min_similarity=0.5).json() == {"enabled": True, "k": 2, "min_similarity": 0.5}
    assert PE.client.get(f"/organisations/{PE.orgB_id}/few-shot", headers=PE.login("admin_a")).status_code == 404         # another organisation's setting is not even visible
    assert PE.client.get(f"/organisations/{PE.orgB_id}/few-shot", headers=PE.login("admin_b")).json()["enabled"] is False  # and was not changed
    # a server with no embedder cannot turn it on
    P = other_platform(tmp_path, rb, "noembed")
    try:
        r = P.client.put(f"/organisations/{P.orgA_id}/few-shot", json={"enabled": True, "confirm_cost_increase": True}, headers=P.login("admin_a"))
        assert r.status_code == 409 and "embedder" in r.json()["detail"]
    finally:
        P.close()


def test_examples_are_usable_only_while_the_feature_is_on(PE):
    decide(PE, solid(RED[0]))
    with PE.factory() as s:
        assert [e.active_for_retrieval for e in s.scalars(select(LearningExample))] == [False]
    enable(PE)
    with PE.factory() as s:
        assert [e.active_for_retrieval for e in s.scalars(select(LearningExample))] == [True]
    decide(PE, solid(RED[1]))                                                           # made while on: usable at once
    with PE.factory() as s:
        assert [e.active_for_retrieval for e in s.scalars(select(LearningExample))] == [True, True]
    PE.client.put(f"/organisations/{PE.orgA_id}/few-shot", json={"enabled": False}, headers=PE.login("admin_a"))
    with PE.factory() as s:
        assert [e.active_for_retrieval for e in s.scalars(select(LearningExample))] == [False, False]


# ------------------------------------------------------------------------------------------------------------- what the model is shown
def test_nothing_is_shown_when_the_feature_is_off_or_the_pool_is_empty_and_the_run_is_then_exactly_what_it_was(PE):
    rec = recorder(PE)
    decide(PE, solid(RED[0]))
    _, eid = upload(PE, solid(RED[1]))                                                  # feature off: the pool exists but is not used
    assert rec.seen_examples[-1] is None and run_of(PE, eid).few_shot is None
    # on, but nothing is similar enough: a similarity floor no photograph can reach leaves the pool empty
    enable(PE)
    PE.client.put(f"/organisations/{PE.orgA_id}/few-shot", json={"enabled": True, "min_similarity": 0.999999}, headers=PE.login("admin_a"))
    _, eid2 = upload(PE, solid(BLUE[0]))
    assert rec.seen_examples[-1] is None and run_of(PE, eid2).few_shot is None
    run = run_of(PE, eid2)
    assert run.content_hash == _hash_content(run.validated_response, run.decision_output, run.raw_response_sha256, run.model_id, run.evaluation_profile, run.prompt_version_id,
                                             run.rulebook_version_id)                  # the hash formula of a run without examples is unchanged
    assert run.content_hash != _hash_content(run.validated_response, run.decision_output, run.raw_response_sha256, run.model_id, run.evaluation_profile, run.prompt_version_id,
                                             run.rulebook_version_id, {"examples": [1]})


def test_similar_decided_photographs_are_shown_with_images_tags_and_reasons_and_are_recorded(PE):
    for i, c in enumerate(RED[:3]):
        decide(PE, solid(c, 160 + i), "APPROVE", f"Warm interaction and clear faces in photograph {i}.", tags=["warm interaction"] if i == 0 else ())
    brown_pid, _ = decide(PE, solid(BROWN[0]), "REJECT", "The faces are hidden behind the volunteers.", categories=["OTHER"])
    blue_pid, _ = decide(PE, solid(BLUE[0]), "REJECT", "The scene is unrelated to the NGO's activities.")
    enable(PE)
    rec = recorder(PE)
    _, eid = upload(PE, solid(RED[3]))
    shown = rec.seen_examples[-1]
    assert shown is not None and len(shown) == 3
    assert {e["decision"] for e in shown} == {"APPROVE", "REJECT"}                      # balanced: the best approved and the best rejected are both there
    assert brown_pid in {e["photo_id"] for e in shown} and blue_pid not in {e["photo_id"] for e in shown}
    assert [e["similarity"] for e in shown] == sorted(e["similarity"] for e in shown)   # least similar first, most similar last
    assert all(isinstance(e["image_bytes"], bytes) and e["mime_type"] == "image/jpeg" for e in shown)
    rejected = next(e for e in shown if e["decision"] == "REJECT")
    assert rejected["reasons"] == "The faces are hidden behind the volunteers."
    run = run_of(PE, eid)
    rec_ex = run.few_shot["examples"]
    assert run.few_shot["embedding_model"] == "thumbnail-8x8-v1" and run.few_shot["k"] == 3 and len(run.few_shot["format_hash"]) == 16
    assert [e["photo_id"] for e in rec_ex] == [e["photo_id"] for e in shown] and all("image_bytes" not in e and e["learning_example_id"] for e in rec_ex)
    assert verify_integrity(run) is True                                                # the examples are part of the content hash
    assert PE.client.get(f"/evaluations/{eid}", headers=PE.login("rev_a")).json()["few_shot"]["examples"][0]["decision"] in ("APPROVE", "REJECT")   # shown on the review screen's data
    # tampering with the record of what was shown is detected
    from ngo_agent.triggers import install_triggers, remove_triggers
    remove_triggers(PE.engine)
    with PE.engine.begin() as c:
        c.execute(text("UPDATE evaluation_runs SET few_shot = NULL WHERE evaluation_id = :i"), {"i": eid})
    install_triggers(PE.engine)
    assert verify_integrity(run_of(PE, eid)) is False


def test_a_photograph_is_never_its_own_example_and_examples_never_cross_organisations(PE):
    pid, eid = decide(PE, solid(RED[0]))
    pid_b, _ = decide(PE, solid(RED[0]), who="rev_b")                                    # organisation B decides the very same bytes
    decide(PE, solid(RED[1], 161), who="rev_b")
    enable(PE)
    enable(PE, admin="admin_b")
    rec = recorder(PE)
    # re-evaluating a decided photograph: it is not shown to itself, and organisation B's photographs are never shown to A
    r = PE.client.post(f"/photos/{pid}/evaluations", json={}, headers=PE.login("rev_a"))
    assert r.status_code == 202
    assert rec.seen_examples[-1] is None                                                 # A has no other decided photograph, and B's are invisible to it
    _, eid_a = upload(PE, solid(RED[2], 162))
    assert {e["photo_id"] for e in rec.seen_examples[-1]} == {pid}
    _, eid_b = upload(PE, solid(RED[3], 163), who="rev_b")
    assert pid not in {e["photo_id"] for e in rec.seen_examples[-1]} and pid_b in {e["photo_id"] for e in rec.seen_examples[-1]}


def test_only_decided_learning_examples_that_are_not_flagged_can_be_shown(PE):
    ok_pid, _ = decide(PE, solid(RED[0]))
    policy_pid, _ = decide(PE, solid(RED[1], 161), "APPROVE", "The new school-uniform rule should apply here.", categories=["POLICY_CHANGE"])   # never becomes a learning example
    undecided_pid, _ = upload(PE, solid(RED[2], 162))                                    # uploaded and evaluated, but nobody has decided it
    flagged_pid, _ = decide(PE, solid(RED[3], 163))
    enable(PE)
    rec = recorder(PE)
    ha = PE.login("admin_a")
    assert PE.client.post(f"/photos/{flagged_pid}/example-flag", json={"flagged": True, "reason": "x"}, headers=ha).status_code == 422      # a reason is needed
    assert PE.client.post(f"/photos/{flagged_pid}/example-flag", json={"flagged": True, "reason": "The reason given was mistaken."}, headers=PE.login("rev_a")).status_code == 403
    assert PE.client.post(f"/photos/{flagged_pid}/example-flag", json={"flagged": True, "reason": "The reason given was mistaken."}, headers=PE.login("admin_b")).status_code == 404
    assert PE.client.post(f"/photos/{flagged_pid}/example-flag", json={"flagged": True, "reason": "The reason given was mistaken."}, headers=ha).status_code == 201
    upload(PE, solid(RED[0], 170))
    assert {e["photo_id"] for e in rec.seen_examples[-1]} == {ok_pid}                    # not the policy change, not the undecided one, not the flagged one
    assert PE.client.post(f"/photos/{flagged_pid}/example-flag", json={"flagged": False}, headers=ha).status_code == 201   # clearing the mark brings it back
    upload(PE, solid(RED[0], 171))
    assert {e["photo_id"] for e in rec.seen_examples[-1]} == {ok_pid, flagged_pid}
    with PE.factory() as s:
        assert [f.action for f in s.scalars(select(ExampleFlag).order_by(ExampleFlag.created_at))] == ["FLAG", "UNFLAG"]    # append-only history
    with PE.engine.begin() as c:
        with pytest.raises(Exception, match="append-only"):
            c.execute(text("UPDATE example_flags SET action='FLAG'"))
    assert policy_pid and undecided_pid


def test_an_example_switched_off_individually_is_not_shown_even_while_the_feature_is_on(PE):
    on_pid, _ = decide(PE, solid(RED[0]))
    off_pid, _ = decide(PE, solid(RED[1], 161))
    enable(PE)
    with PE.factory() as s:                                        # a developer deactivates one example directly
        s.scalars(select(LearningExample).where(LearningExample.photo_id == off_pid)).one().active_for_retrieval = False
        s.commit()
    rec = recorder(PE)
    upload(PE, solid(RED[2], 162))
    assert {e["photo_id"] for e in rec.seen_examples[-1]} == {on_pid}


def test_a_problem_finding_examples_never_stops_an_evaluation(PE, monkeypatch):
    decide(PE, solid(RED[0]))
    enable(PE)

    def boom(*a, **k):  # noqa: ANN002
        raise RuntimeError("search failed")

    monkeypatch.setattr(xs, "examples_for_evaluation", boom)
    _, eid = upload(PE, solid(RED[1], 161))
    run = run_of(PE, eid)
    assert run.processing_status == "AI_EVALUATED" and run.few_shot is None


def test_only_tags_a_person_added_are_shown_as_staff_tags(PE):
    """The AI's own observations that nobody removed are not staff tags."""
    P_ = PE
    P_.use(ScriptedEvaluator(positives=["warm interaction seen by the AI"], concerns=["blurred banner seen by the AI"], rulebook=P_.rb))
    pid, _ = decide(P_, solid(RED[0]), tags=["cluttered foreground"])
    enable(P_)
    new_pid, _ = upload(P_, solid(RED[1], 161))
    got = P_.client.get(f"/photos/{new_pid}/similar", headers=P_.login("rev_a")).json()["examples"]
    assert [e["photo_id"] for e in got] == [pid] and got[0]["tags"] == ["cluttered foreground"]


def test_the_preview_shows_what_would_be_used_without_the_images(PE):
    pid, _ = decide(PE, solid(RED[0]), tags=["warm interaction"])
    new_pid, _ = upload(PE, solid(RED[1], 161))
    h = PE.login("rev_a")
    assert PE.client.get(f"/photos/{new_pid}/similar", headers=h).json() == {"enabled": False, "examples": []}
    enable(PE)
    got = PE.client.get(f"/photos/{new_pid}/similar", headers=h).json()
    assert got["enabled"] is True and [e["photo_id"] for e in got["examples"]] == [pid] and got["examples"][0]["tags"] == ["warm interaction"]
    assert "image_bytes" not in got["examples"][0]
    assert PE.client.get(f"/photos/{new_pid}/similar", headers=PE.login("rev_b")).status_code == 404


def test_the_old_hash_formula_is_unchanged_for_runs_without_examples():
    legacy = __import__("hashlib").sha256(canonical({"validated": {"a": 1}, "decision": {"b": 2}, "raw": "r", "model": "m", "profile": "p", "prompt_version_id": "pv",
                                                     "rulebook_version_id": "rv"}).encode()).hexdigest()
    assert _hash_content({"a": 1}, {"b": 2}, "r", "m", "p", "pv", "rv") == legacy == _hash_content({"a": 1}, {"b": 2}, "r", "m", "p", "pv", "rv", None)
