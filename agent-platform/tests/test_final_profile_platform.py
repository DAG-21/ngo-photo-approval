"""layered_final_v1 on the platform side. A FAKE provider: no model is called.

They pin: the rule catalogue of the new profile (the client lines it hides are gone, the ones it rewords carry the owner's wording, there are no guidance lines); the evaluator
accepts an answer that leaves out the seven facts the profile no longer asks, in the short format and in the full one, while an older profile still fails on a missing fact; and
the AI always gives a verdict (a gate it cannot confirm is a rejection with the reason, never "human review")."""

from __future__ import annotations

import json

import pytest
from layered_engine.compact import split_ids
from layered_engine.prompt_builder import build_prompt
from layered_engine.providers import ProviderResult, load_gemini_config

from ngo_agent.evaluators import EvaluationFailure, LayeredEngineEvaluator, ScriptedEvaluator
from ngo_agent.rule_drafter import rule_catalogue

FINAL, TUNED = "layered_final_v1", "layered_tuned_v1"
APPROVED = ("APPROVE", "APPROVED_PENDING_PUBLICATION_CHECKS")      # without publication metadata an approval is "pending publication checks"
KEPT_FACTS = dict(f_photo="YES", f_single="YES", f_volunteers="YES", f_banner="YES", f_beneficiaries="YES", f_item_given="YES", f_medical="NO", f_activity_type="FOOD_DISTRIBUTION")


class FakeProvider:
    model_id = "gemini-2.5-flash-lite"

    def __init__(self, text):
        self.text, self.calls = text, []

    def assess(self, image_bytes, mime_type, system_text, user_text, schema, **kw):
        self.calls.append({"system": system_text, "schema": schema})
        return ProviderResult(text=self.text, usage={"prompt_tokens": 7000, "output_tokens": 500}, model_id=self.model_id)


def config(fmt):
    cfg = json.loads(json.dumps(load_gemini_config()))
    cfg["evaluation"]["response_format"] = fmt
    return cfg


def short_answer(rb, **checks):
    prof = rb.resolve(FINAL)
    ids = build_prompt(rb, prof, response_format="compact").criterion_ids
    explicit, _ = split_ids(prof, ids)
    return json.dumps({"activity_description": "Volunteers serve food.", "scene_facts": KEPT_FACTS, "checks": {**{i: "YES" for i in explicit}, **checks}, "other_criteria_not_yes": [],
                       "evidence": [{"criterion_id": k, "text": "could not tell from the photograph"} for k, v in checks.items() if v != "YES"],
                       "quality": {d: 4 for d in prof.quality["dimensions"]}, "visual_flags": []})


def test_the_catalogue_of_the_final_profile_hides_what_it_hides_and_shows_the_owners_wording(rb):
    cat = rule_catalogue(rb, FINAL)
    assert not any(k in cat for k in ("R005", "R012", "R013", "R014", "R015", "R016")) and not any(v.kind == "GUIDANCE" for v in cat.values())
    assert cat["R004"].base == "Capture natural expressions and interactions. A posed photograph, such as a group photograph of volunteers, is fine." and cat["R004"].client_wording.startswith("Capture genuine expressions and interactions. Beneficiaries")
    assert {k for k, v in cat.items() if v.kind == "CLIENT_RULE"} == {"R001", "R002", "R003", "R004", "R006", "R007", "R008", "R009", "R010", "R011"}
    assert "R004_BEN_GAZE" not in cat and "R010_TECHNICAL" in cat and "F_BENEFICIARIES" in cat and "F_LOGO" not in cat
    older = rule_catalogue(rb, TUNED)
    assert {"R005", "R016", "GUIDE_01", "GUIDE_04", "R004_BEN_GAZE", "F_LOGO"} <= set(older)                 # the earlier profile is unchanged


def test_the_short_answer_without_the_dropped_facts_is_accepted_and_decided(rb):
    fake = FakeProvider(short_answer(rb))
    out = LayeredEngineEvaluator(fake, config("compact"), rb).evaluate(b"img", "image/jpeg", {"profile": FINAL}, {"width": 1000, "height": 800})
    assert out.decision["recommendation"] in APPROVED and out.profile == FINAL
    assert "f_logo" not in fake.calls[0]["schema"]["properties"]["scene_facts"]["properties"] and "f_logo" not in fake.calls[0]["system"]
    tuned_text = short_answer(rb)                                                       # the same answer is NOT enough for an earlier profile: it asks all fifteen facts and the brand presentation grade
    with pytest.raises(EvaluationFailure, match="scene fact|schema|quality dimension"):
        LayeredEngineEvaluator(FakeProvider(tuned_text), config("compact"), rb).evaluate(b"img", "image/jpeg", {"profile": TUNED}, {"width": 1000, "height": 800})


def test_a_gate_the_model_cannot_confirm_is_a_rejection_with_the_reason_not_human_review(rb):
    out = LayeredEngineEvaluator(FakeProvider(short_answer(rb, R003_FACE="CANNOT_TELL")), config("compact"), rb).evaluate(b"img", "image/jpeg", {"profile": FINAL}, {})
    assert out.decision["recommendation"] == "REJECT" and out.decision["failed_gates"] == [] and out.decision["uncertain_gates"] == ["G004"]
    assert any("could not confirm" in r for r in out.decision["decision_reasons"])


def test_the_scripted_evaluator_and_the_gaze_no_longer_matter_for_the_final_profile(rb):
    ok = ScriptedEvaluator(rulebook=rb).evaluate(b"img", "image/jpeg", {"profile": FINAL}, {})
    assert ok.decision["recommendation"] in APPROVED
    bad = ScriptedEvaluator(answers={"R002_GIVER_WHITE": "NO"}, rulebook=rb).evaluate(b"img", "image/jpeg", {"profile": FINAL}, {})
    assert bad.decision["recommendation"] == "REJECT" and bad.decision["failed_gates"] == ["G013"]
    older = ScriptedEvaluator(answers={"R004_BEN_GAZE": "NO"}, rulebook=rb).evaluate(b"img", "image/jpeg", {"profile": TUNED}, {})
    assert older.decision["recommendation"] == "REJECT" and older.decision["failed_gates"] == ["G015"]       # the earlier profile still rejects a gaze


def test_the_full_format_answer_without_the_dropped_facts_is_accepted_for_the_final_profile_only(rb):
    raw = json.loads(ScriptedEvaluator(rulebook=rb).evaluate(b"img", "image/jpeg", {"profile": FINAL}, {}).raw_text)
    for k in ("f_banner_prominent", "f_animals", "f_handover", "f_volunteer_giver", "f_logo", "f_orientation", "f_group"):
        raw["layer_1_scene_understanding"]["scene_facts"].pop(k, None)
    for k in ("image_id", "model_recommendation", "model_overall_score", "schema_version"):
        raw.pop(k, None)
    text = json.dumps(raw)
    out = LayeredEngineEvaluator(FakeProvider(text), config("full"), rb).evaluate(b"img", "image/jpeg", {"profile": FINAL}, {})
    assert out.decision["recommendation"] in APPROVED
    with pytest.raises(EvaluationFailure, match="scene_facts"):
        LayeredEngineEvaluator(FakeProvider(text), config("full"), rb).evaluate(b"img", "image/jpeg", {"profile": TUNED}, {})


def test_the_haiku_provider_runs_the_whole_evaluation_with_a_fake_client_and_counts_the_cache(rb):
    from types import SimpleNamespace

    from layered_engine.providers import AnthropicProvider, load_vision_config

    cfg = load_vision_config("haiku")
    usage = SimpleNamespace(input_tokens=1500, cache_read_input_tokens=0, cache_creation_input_tokens=8000, output_tokens=2000)
    seen = []

    class Messages:
        def create(self, **kw):
            seen.append(kw)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=short_answer(rb))], stop_reason="end_turn", usage=usage)

    provider = AnthropicProvider(cfg, client=SimpleNamespace(messages=Messages()), env={"NGO_ALLOW_ANTHROPIC_INFERENCE": "true", "ANTHROPIC_API_KEY": "k"})
    out = LayeredEngineEvaluator(provider, cfg, rb, provider_name="anthropic-claude").evaluate(b"img", "image/jpeg", {"profile": FINAL}, {"width": 1000, "height": 800})
    assert out.decision["recommendation"] in APPROVED and out.model_provider == "anthropic-claude" and out.model_id == "claude-haiku-5-5"
    assert "How to respond (short format)" in seen[0]["system"][0]["text"] and seen[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert out.usage["cache_write_tokens"] == 8000 and out.cost_usd == pytest.approx((1500 * 0.10 + 8000 * 0.125 + 2000 * 0.50) / 1e6)
