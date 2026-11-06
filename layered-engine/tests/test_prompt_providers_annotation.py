from __future__ import annotations

import csv
import json

import pytest

from layered_engine.annotation import HISTORICAL, NEW_PHOTO, build_engine, evaluate_one, parse_response, plan_run, run_batch
from layered_engine.export import build_enriched
from layered_engine.ingestion import IngestionError, load_image
from layered_engine.prompt_builder import build_prompt
from layered_engine.providers import AuthorizationError, GeminiProvider, ProviderError, read_env_file, scrub
from layered_engine.schemas import Observation, Recommendation as R

from conftest import FakeProvider, make_assessment, response_text, verified_metadata, write_image


# ------------------------------------------------------------------------------------------------ prompt
def test_the_prompt_quotes_the_client_rules_verbatim_and_marks_every_gate(rb, legacy, proposed):
    for prof in (legacy, proposed):
        p = build_prompt(rb, prof)
        for r in rb.sources["client_rules"]:
            assert r["description_original"] in p.system_text
        for g in prof.hard_gates():
            assert f"MANDATORY GATE {g.gate_id}" in p.system_text
        assert "team rule, not in the client's document" in p.system_text  # internal rules are never presented as the client's
    assert "MANDATORY GATE G010" in build_prompt(rb, legacy).system_text
    assert "MANDATORY GATE G010" not in build_prompt(rb, proposed).system_text  # a soft criterion is not described as mandatory


def test_the_prompt_asks_the_profile_s_questions_only(rb, legacy, proposed):
    pl, pp = build_prompt(rb, legacy), build_prompt(rb, proposed)
    assert len(pl.criterion_ids) == 61 and set(pl.criterion_ids) - {"T_BACK_EXCEPTION"} < set(pp.criterion_ids)  # T_BACK_EXCEPTION is retired with the back-view exception
    assert "NEW-008" in pp.system_text and "NEW-008" not in pl.system_text
    assert pl.hash != pp.hash and pl.template_version == "1.0"


def test_the_prompt_has_no_room_for_a_historical_decision(rb, proposed):
    text = build_prompt(rb, proposed).system_text.lower()
    assert "must not be guessed" in text  # the only mention of history is the instruction not to guess it
    assert "historical decision" not in text and "historical label" not in text
    import inspect

    assert "label" not in inspect.signature(build_prompt).parameters  # the builder cannot even receive a label


def test_the_model_is_not_asked_for_a_decision_or_a_total(rb, proposed):
    p = build_prompt(rb, proposed)
    assert "Do not give a final decision, an overall score or a recommendation" in p.system_text
    e = build_engine({**__import__("layered_engine.providers", fromlist=["x"]).load_gemini_config()}, None, "layered_proposed_v1")
    assert "model_overall_score" not in json.dumps(e.schema) and "model_recommendation" not in json.dumps(e.schema)


# ------------------------------------------------------------------------------------------------ response parsing
def test_a_valid_response_parses_and_a_bad_one_is_an_error_not_an_evaluation(proposed):
    good = response_text(make_assessment(proposed))
    assert parse_response(good, "img_1").image_id == "img_1"
    for bad in ("not json", "[]", json.dumps({"layer_1_scene_understanding": {}}), good.replace('"YES"', '"MAYBE"', 1)):
        with pytest.raises(ValueError):
            parse_response(bad, "img_1")
    dup = json.loads(good)
    dup["layer_2_criteria"].append(dup["layer_2_criteria"][0])
    with pytest.raises(ValueError, match="more than once"):
        parse_response(json.dumps(dup), "img_1")
    noscore = json.loads(good)
    noscore["layer_3_quality_dimensions"]["composition"] = {"applicable": True}
    with pytest.raises(ValueError):
        parse_response(json.dumps(noscore), "img_1")


# ------------------------------------------------------------------------------------------------ providers fail closed
def test_the_provider_refuses_without_authorization_and_without_a_key(cfg):
    with pytest.raises(AuthorizationError, match="not authorized"):
        GeminiProvider(cfg, env={"GEMINI_API_KEY": "k"})
    with pytest.raises(AuthorizationError, match="GEMINI_API_KEY"):
        GeminiProvider(cfg, env={"NGO_ALLOW_CLOUD_INFERENCE": "true"})
    assert GeminiProvider(cfg, env={"NGO_ALLOW_CLOUD_INFERENCE": "true", "GEMINI_API_KEY": "k"}).model_id == "gemini-2.5-flash-lite"


def test_errors_never_contain_the_key():
    assert scrub("boom secret-123 boom", "secret-123") == "boom <key> boom"


def test_the_env_file_reader_returns_only_the_named_keys(tmp_path):
    f = tmp_path / ".env"
    f.write_text("GEMINI_API_KEY=abc\nOTHER=zzz\nNGO_ALLOW_CLOUD_INFERENCE=true\n#C=1\n", encoding="utf-8")
    assert read_env_file(f, ("GEMINI_API_KEY", "NGO_ALLOW_CLOUD_INFERENCE")) == {"GEMINI_API_KEY": "abc", "NGO_ALLOW_CLOUD_INFERENCE": "true"}


class _Resp:
    text = "{}"
    usage_metadata = None


class _Models:
    def __init__(self, behaviours):
        self.b = list(behaviours)
        self.calls = 0

    def generate_content(self, **kw):
        self.calls += 1
        x = self.b.pop(0)
        if isinstance(x, Exception):
            raise x
        return x


class _Err(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code} secret-key")
        self.code = code


class _Client:
    def __init__(self, behaviours):
        self.models = _Models(behaviours)


def test_transient_errors_are_retried_and_permanent_ones_are_not(cfg):
    env = {"NGO_ALLOW_CLOUD_INFERENCE": "true", "GEMINI_API_KEY": "secret-key"}
    sleeps = []
    c = _Client([_Err(503), _Err(429), _Resp()])
    p = GeminiProvider(cfg, client=c, sleep=sleeps.append, env=env)
    out = p.assess(b"x", "image/jpeg", "sys", "user", {})
    assert out.text == "{}" and c.models.calls == 3 and len(sleeps) == 2 and out.usage["attempts"] == 3
    c2 = _Client([_Err(403)])
    with pytest.raises(ProviderError, match="permission") as e:
        GeminiProvider(cfg, client=c2, sleep=sleeps.append, env=env).assess(b"x", "image/jpeg", "s", "u", {})
    assert c2.models.calls == 1 and "secret-key" not in str(e.value)
    c3 = _Client([_Err(500)] * 3)
    with pytest.raises(ProviderError):
        GeminiProvider(cfg, client=c3, sleep=lambda s: None, env=env).assess(b"x", "image/jpeg", "s", "u", {})
    assert c3.models.calls == 3  # exactly the configured number of attempts


def test_text_only_requests_carry_no_media_and_share_the_retries_and_the_key_scrubbing(cfg):
    env = {"NGO_ALLOW_CLOUD_INFERENCE": "true", "GEMINI_API_KEY": "secret-key"}
    sent = []

    class Recording(_Models):
        def generate_content(self, **kw):
            sent.append(kw)
            return super().generate_content(**kw)

    c = _Client([_Err(503), _Resp()])
    c.models = Recording([_Err(503), _Resp()])
    out = GeminiProvider(cfg, client=c, sleep=lambda s: None, env=env).complete_text("system words", "user words", {"type": "object"}, max_output_tokens=2048)
    assert out.text == "{}" and c.models.calls == 2 and out.usage["attempts"] == 2                      # the transient error was retried, as for photographs
    request = sent[-1]
    parts = request["contents"][0].parts
    assert len(parts) == 1 and parts[0].text == "user words" and parts[0].inline_data is None           # one text part, no image
    assert request["config"].system_instruction == "system words" and request["config"].max_output_tokens == 2048 and request["config"].media_resolution is None
    c2 = _Client([_Err(403)])
    with pytest.raises(ProviderError, match="permission") as e:
        GeminiProvider(cfg, client=c2, sleep=lambda s: None, env=env).complete_text("s", "u", {})
    assert c2.models.calls == 1 and "secret-key" not in str(e.value)
    c3 = _Client([_Err(500)] * 3)
    with pytest.raises(ProviderError) as e3:
        GeminiProvider(cfg, client=c3, sleep=lambda s: None, env=env).complete_text("s", "u", {})
    assert c3.models.calls == 3 and "secret-key" not in str(e3.value)                                   # the configured number of attempts, and the key never in the message


# ------------------------------------------------------------------------------------------------ ingestion
def test_the_original_image_bytes_are_used_and_oversize_files_are_refused(tmp_path):
    p = write_image(tmp_path / "a.jpg", (40, 30))
    img = load_image(p)
    assert img.data == p.read_bytes() and img.metadata["width"] == 40 and img.metadata["height"] == 30 and img.mime_type == "image/jpeg"
    with pytest.raises(IngestionError, match="not found"):
        load_image(tmp_path / "missing.jpg")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(IngestionError, match="unsupported"):
        load_image(tmp_path / "notes.txt")
    big = tmp_path / "big.jpg"
    big.write_bytes(b"0" * (19 * 1024 * 1024))
    with pytest.raises(IngestionError, match="not resized"):
        load_image(big)


# ------------------------------------------------------------------------------------------------ the two execution modes
@pytest.fixture
def engine(cfg):
    e = build_engine(cfg, FakeProvider(), "layered_proposed_v1")
    return e


def good_text(engine, **kw):
    return response_text(make_assessment(engine.profile, **kw))


def test_historical_mode_never_sees_a_label_and_decides_in_python(engine, tmp_path):
    engine.provider.texts = [good_text(engine)]
    row = evaluate_one(engine, "NGO_IMG_000001", write_image(tmp_path / "approved_or_not.jpg"), HISTORICAL)
    assert row["ok"] and row["decision"]["recommendation"] == "APPROVED_PENDING_PUBLICATION_CHECKS" and row["mode"] == HISTORICAL
    system, user = engine.provider.seen[0]
    assert "approved_or_not" not in system + user and "NGO_IMG_000001" not in system + user  # neither the file name nor the id reaches the model
    assert row["model_id"] == "fake-model" and row["prompt_hash"] == engine.prompt.hash and row["rulebook_version"] == "1.1.0"


def test_historical_mode_rejects_labels_and_publication_metadata(engine, tmp_path):
    img = write_image(tmp_path / "a.jpg")
    assert "labels" in evaluate_one(engine, "i", img, HISTORICAL, {"label": "approved"})["error"]
    assert "no publication metadata" in evaluate_one(engine, "i", img, HISTORICAL, {"consent_status": "VERIFIED"})["error"]
    assert engine.provider.calls == 0  # refused before anything was sent


def test_new_photo_mode_uses_verified_metadata(engine, tmp_path):
    engine.provider.texts = [good_text(engine)]
    row = evaluate_one(engine, "i", write_image(tmp_path / "a.jpg"), NEW_PHOTO, verified_metadata())
    assert row["decision"]["recommendation"] == R.APPROVE.value and row["metadata_used"]["width"] == 64


def test_a_failed_call_or_a_bad_response_is_recorded_as_a_failure(engine, tmp_path):
    img = write_image(tmp_path / "a.jpg")
    engine.provider.texts = [ProviderError("authentication or permission error (HTTP 403)"), "not json", good_text(engine, omit=("R003_FACE",))]
    r1, r2, r3 = (evaluate_one(engine, f"i{n}", img, NEW_PHOTO, {}, tmp_path / "raw") for n in range(3))
    assert not r1["ok"] and "403" in r1["error"] and "decision" not in r1
    assert not r2["ok"] and "not valid JSON" in r2["error"] and "decision" not in r2
    assert r3["ok"] and r3["decision"]["recommendation"] == "HUMAN_REVIEW"  # a missing gate is escalated, never passed
    assert (tmp_path / "raw" / f"i1__{engine.prompt.hash}.json").read_text(encoding="utf-8") == "not json"  # the raw response is preserved


def test_batch_runs_are_resumable_capped_and_idempotent(engine, tmp_path):
    entries = [{"image_id": f"i{n}", "path": write_image(tmp_path / f"{n}.jpg")} for n in range(4)]
    out = tmp_path / "run"
    engine.provider.texts = [good_text(engine), good_text(engine), "garbage", good_text(engine)]
    with pytest.raises(RuntimeError, match="max-calls"):
        run_batch(engine, entries, out, HISTORICAL, max_calls=2)
    assert engine.provider.calls == 0  # the cap is checked before the first call
    s = run_batch(engine, entries, out, HISTORICAL, max_calls=10, workers=1)
    assert s == {"ok": 3, "failed": 1, "skipped_done": 0}
    plan, todo = plan_run(engine, entries, out, HISTORICAL)
    assert plan.n_done == 3 and plan.n_todo == 1  # only the failure comes back
    engine.provider.texts = [good_text(engine)]
    run_batch(engine, entries, out, HISTORICAL, max_calls=10, workers=1)
    assert engine.provider.calls == 5 and plan_run(engine, entries, out, HISTORICAL)[0].n_todo == 0
    # a different prompt (profile) is never confused with finished work
    other = build_engine(engine.config, engine.provider, "checklist5_legacy")
    assert plan_run(other, entries, out, HISTORICAL)[0].n_todo == 4


def test_the_enriched_dataset_keeps_label_observations_and_human_findings_apart(engine, tmp_path):
    entries = [{"image_id": "i0", "path": write_image(tmp_path / "0.jpg")}, {"image_id": "i1", "path": write_image(tmp_path / "1.jpg")}]
    a = make_assessment(engine.profile).model_copy(update={"potential_rejection_factors": [Observation(text="banner looked odd", evidence="left edge")]})
    engine.provider.texts = [response_text(a), good_text(engine, answers={"R003_FACE": "NO"})]
    out = tmp_path / "run"
    run_batch(engine, entries, out, HISTORICAL, max_calls=5, workers=1)
    labels, human = tmp_path / "labels.csv", tmp_path / "human.csv"
    labels.write_text("image_id,label\ni0,approved\ni1,not_approved\n", encoding="utf-8")
    human.write_text("image_id,confirmed_tags,confirmed_rejection_reasons,reviewer_notes\ni1,FACE_HIDDEN|DARK,R003,checked by Asha\n", encoding="utf-8")
    summary = build_enriched(out / "results.jsonl", out / "enriched", labels, human)
    assert summary == {"photos": 2, "with_label": 2, "human_reviewed": 1}
    recs = {json.loads(l)["image_id"]: json.loads(l) for l in (out / "enriched.jsonl").read_text(encoding="utf-8").splitlines()}
    assert recs["i0"]["historical_decision"] == "approved" and recs["i1"]["historical_decision"] == "not_approved"
    assert recs["i0"]["model_observations"]["potential_rejection_factors"][0]["text"] == "banner looked odd"
    assert recs["i0"]["human_validation"]["confirmed_rejection_reasons"] == []  # a model's guess is never promoted to a confirmed reason
    assert recs["i1"]["human_validation"] == {"reviewed": True, "confirmed_tags": ["FACE_HIDDEN", "DARK"], "confirmed_rejection_reasons": ["R003"], "reviewer_notes": "checked by Asha"}
    rows = list(csv.DictReader((out / "enriched.csv").open(encoding="utf-8")))
    assert {r["image_id"]: r["historical_decision"] for r in rows} == {"i0": "approved", "i1": "not_approved"}
    assert rows[1]["failed_gates"] == "G004" and "gate_G004" in rows[1]


def test_the_prompt_states_the_ngo_decisions_and_no_withdrawn_exception(rb, legacy, proposed):
    p = build_prompt(rb, proposed).system_text
    assert "NO exception for a logo on the back" in p and "A volunteer looking straight into the camera is NOT a violation" in p
    assert "Banners are not expected in every photograph" in p and "animal receiving care counts" in p
    assert "MANDATORY GATE G016" in p and "MANDATORY GATE G017" in p and "MANDATORY GATE G018" in p
    assert "T_BACK_EXC" not in p and "EXC_BACK_VIEW" not in p and "EXC_DIRECT_GAZE" not in p  # dormant exceptions are not shown to the model
    assert "EXC_MEDICAL_PPE" in p
    legacy_prompt = build_prompt(rb, legacy).system_text
    assert "G016" not in legacy_prompt and "(none are active)" in legacy_prompt


def test_the_schema_has_no_list_size_limits_and_completeness_is_enforced_in_python(rb, proposed, cfg):
    import json as _json

    e = build_engine(cfg, None, "layered_proposed_v1")
    assert "minItems" not in _json.dumps(e.schema) and "maxItems" not in _json.dumps(e.schema)  # rejected by the live API (HTTP 400)
    d = __import__("layered_engine.decision_engine", fromlist=["x"]).evaluate_final_decision(
        make_assessment(proposed, omit=tuple(c for c in e.prompt.criterion_ids if c != "R003_FACE")), proposed, {})
    assert d.recommendation is not R.APPROVE and any("synthesized" in w for w in d.warnings)  # a short answer is never a pass
