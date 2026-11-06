"""The platform's evaluator with the short answer format switched on. A FAKE provider: no model is called.

They pin: with the setting off nothing changes (the full prompt, the full schema); with it on the provider is asked for the short answer, the answer is stored as the usual full
assessment and decided by the usual rules, and the cost counts thinking and cached input."""

from __future__ import annotations

import json

import pytest
from layered_engine.compact import split_ids
from layered_engine.prompt_builder import build_prompt
from layered_engine.providers import ProviderResult, load_gemini_config

from ngo_agent.evaluators import LayeredEngineEvaluator

P = "layered_proposed_v1"
GAZE = "R004_BEN_GAZE"
FACTS = dict(f_photo="YES", f_single="YES", f_volunteers="YES", f_banner="YES", f_banner_prominent="YES", f_beneficiaries="YES", f_animals="NO", f_item_given="YES",
             f_handover="YES", f_volunteer_giver="YES", f_logo="YES", f_medical="NO", f_orientation="FRONT", f_group="GROUP", f_activity_type="FOOD_DISTRIBUTION")


class FakeProvider:
    model_id = "gemini-3.5-flash-lite"

    def __init__(self, text, usage):
        self.text, self.usage, self.calls = text, usage, []

    def assess(self, image_bytes, mime_type, system_text, user_text, schema, **kw):
        self.calls.append({"system": system_text, "schema": schema})
        return ProviderResult(text=self.text, usage=dict(self.usage), model_id=self.model_id)


def config(fmt):
    cfg = json.loads(json.dumps(load_gemini_config()))
    cfg["model_id"] = "gemini-3.5-flash-lite"
    cfg["evaluation"]["response_format"] = fmt
    return cfg


def short_answer(rb, gaze="NO"):
    prof = rb.resolve(P)
    ids = build_prompt(rb, prof, response_format="compact").criterion_ids
    explicit, _ = split_ids(prof, ids)
    return json.dumps({"activity_description": "Volunteers serve food.", "scene_facts": FACTS, "checks": {**{i: "YES" for i in explicit}, GAZE: gaze}, "other_criteria_not_yes": [],
                       "evidence": [{"criterion_id": GAZE, "text": "The beneficiary looks into the lens."}] if gaze == "NO" else [],
                       "quality": {d: 4 for d in prof.quality["dimensions"]}, "visual_flags": []})


USAGE = {"prompt_tokens": 7000, "output_tokens": 500, "thinking_tokens": 100, "cached_tokens": 6000}


def test_with_the_short_format_on_the_provider_is_asked_for_it_and_the_usual_decision_comes_out(rb):
    fake = FakeProvider(short_answer(rb), USAGE)
    out = LayeredEngineEvaluator(fake, config("compact"), rb).evaluate(b"img", "image/jpeg", {"profile": P}, {"width": 1000, "height": 800})
    seen = fake.calls[0]
    assert "How to respond (short format)" in seen["system"] and "checks" in seen["schema"]["properties"] and "layer_2_criteria" not in seen["schema"]["properties"]
    assert out.decision["recommendation"] == "REJECT" and any(g["gate_id"] == "G015" and g["status"] == "NO" and "looks into the lens" in g["evidence"] for g in out.decision["gates"])
    assert len(out.validated["layer_2_criteria"]) == len(build_prompt(rb, rb.resolve(P)).criterion_ids)             # stored as the usual full assessment
    assert out.prompt_hash == build_prompt(rb, rb.resolve(P), response_format="compact").hash != build_prompt(rb, rb.resolve(P)).hash
    assert out.cost_usd == pytest.approx((1000 * 0.30 + 6000 * 0.03 + 600 * 2.50) / 1e6)                              # thinking billed as output, cached input at the cached rate


def test_with_the_setting_off_the_request_is_exactly_the_full_one(rb):
    full = json.dumps({"layer_1_scene_understanding": {"activity_description": "x", "scene_facts": FACTS}, "layer_2_criteria": [],
                       "layer_3_quality_dimensions": {d: {"applicable": True, "score": 4} for d in rb.resolve(P).quality["dimensions"]}})
    fake = FakeProvider(full, {"prompt_tokens": 7400, "output_tokens": 5500, "thinking_tokens": 1500})
    cfg = config("full")
    cfg["model_id"] = "gemini-3.8-flash"
    out = LayeredEngineEvaluator(fake, cfg, rb).evaluate(b"img", "image/jpeg", {"profile": P}, {"width": 1000, "height": 800})
    assert "layer_2_criteria" in fake.calls[0]["schema"]["properties"] and "short format" not in fake.calls[0]["system"]
    assert out.prompt_hash == build_prompt(rb, rb.resolve(P)).hash and out.cost_usd == pytest.approx((7400 * 0.75 + 7000 * 3.75) / 1e6)


def test_an_unusable_short_answer_fails_the_evaluation_instead_of_being_guessed(rb):
    from ngo_agent.evaluators import EvaluationFailure

    with pytest.raises(EvaluationFailure, match="did not validate"):
        LayeredEngineEvaluator(FakeProvider("not json", USAGE), config("compact"), rb).evaluate(b"img", "image/jpeg", {"profile": P}, {})
