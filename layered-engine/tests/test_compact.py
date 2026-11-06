"""The short answer format (compact.py), the real price accounting (costs.py) and the thinking setting of the provider. No model is called: providers are fakes.

They pin: the full prompt is untouched; what the short prompt asks; which criteria must be answered explicitly; that a missing or unreadable decisive answer is never a pass; that
the short answer becomes the same full assessment the rule engine always decided from; and that cost counts thinking as output and cached input at the cached rate."""

from __future__ import annotations

import json

import pytest

from layered_engine.annotation import parse_response
from layered_engine.compact import build_compact_schema, expand_compact, split_ids
from layered_engine.costs import cost_usd, prices
from layered_engine.decision_engine import evaluate_final_decision
from layered_engine.prompt_builder import build_prompt
from layered_engine.providers import GeminiProvider, load_gemini_config
from layered_engine.rulebook import Rulebook
from layered_engine.schema_export import check_provider_safe

P = "layered_proposed_v1"
FROZEN_HASH = "11427c79d96c569a"
GAZE = "R004_BEN_GAZE"
FACTS = dict(f_photo="YES", f_single="YES", f_volunteers="YES", f_banner="YES", f_banner_prominent="YES", f_beneficiaries="YES", f_animals="NO", f_item_given="YES",
             f_handover="YES", f_volunteer_giver="YES", f_logo="YES", f_medical="NO", f_orientation="FRONT", f_group="GROUP", f_activity_type="FOOD_DISTRIBUTION")


@pytest.fixture(scope="module")
def rb():
    return Rulebook()


@pytest.fixture(scope="module")
def prof(rb):
    return rb.resolve(P)


@pytest.fixture(scope="module")
def ids(rb, prof):
    return build_prompt(rb, prof, response_format="compact").criterion_ids


def answer(prof, ids, checks=None, others=None, evidence=None, quality=None, facts=None, flags=None, drop=()):
    explicit, _ = split_ids(prof, ids)
    body = {"activity_description": "Volunteers serve food.", "scene_facts": {**FACTS, **(facts or {})}, "checks": {i: "YES" for i in explicit},
            "other_criteria_not_yes": others or [], "evidence": evidence or [], "quality": quality or {d: 4 for d in prof.quality["dimensions"]}, "visual_flags": flags or []}
    body["checks"].update(checks or {})
    for k in drop:
        body["checks"].pop(k, None)
    return json.dumps(body)


# --------------------------------------------------------------------------------------------------------------------------------- the prompt
def test_the_full_prompt_and_its_hash_are_untouched(rb, prof):
    assert build_prompt(rb, prof).hash == build_prompt(rb, prof, response_format="full").hash == FROZEN_HASH


def test_the_short_prompt_asks_the_same_questions_for_a_short_answer(rb, prof):
    full, short = build_prompt(rb, prof), build_prompt(rb, prof, response_format="compact")
    assert short.criterion_ids == full.criterion_ids and short.hash != full.hash
    s = short.system_text
    assert "How to respond (short format)" in s and "## H. Additional observations" not in s and "Subjects: list each" not in s
    assert "Give evidence for every NO and CANNOT_TELL." in s and "Give only the score, a whole number" in s and "Mandatory gates must carry evidence." not in s
    assert "counts as YES unless you list it here" in s and len(s) < len(full.system_text)
    for line in full.system_text.splitlines():                          # every question is still asked, word for word
        if line.startswith("- `") and "]: " in line:
            assert line in s
    with pytest.raises(ValueError, match="unknown response format"):
        build_prompt(rb, prof, response_format="tiny")


# ---------------------------------------------------------------------------------------------------------------------------- what must be explicit
def test_every_criterion_that_can_change_a_decision_must_be_answered_explicitly(prof, ids):
    explicit, others = split_ids(prof, ids)
    assert set(explicit) | set(others) == set(ids) and not set(explicit) & set(others)
    assert explicit == [i for i in ids if i in set(explicit)] and others == [i for i in ids if i in set(others)]              # the order asked
    for g in prof.gates:
        assert g.criterion_id in explicit and all(c in explicit for c in g.components if c in ids)
    assert all(prof.criteria[i].decision_effect not in ("GATE", "SOFT_SCORE") for i in others)
    for cap in prof.quality.get("soft_caps", []):
        assert cap["criterion_id"] not in others
    for ex in prof.exceptions:
        assert not set(ex.get("requires_criteria_all_yes", [])) & set(others) and not set(ex.get("applies_to", [])) & set(others)
    assert GAZE in explicit and others                                  # there are criteria that only inform, and only they may default to YES


def test_each_reason_for_being_decisive_counts_on_its_own(prof, ids):
    import copy

    only_gates = copy.deepcopy(prof)                                    # gates alone: G016-G018 are criteria that only inform, made into gates
    for g in only_gates.gates:
        g.components = []
    only_gates.quality = {**only_gates.quality, "soft_caps": []}
    only_gates.exceptions = []
    explicit, _ = split_ids(only_gates, ids)
    assert all(g.criterion_id in explicit for g in only_gates.gates) and {"R008_BACK", "R001_VISIBLE", "R001_DISTINCT"} <= set(explicit)
    only_caps = copy.deepcopy(prof)                                     # soft caps alone
    only_caps.gates, only_caps.exceptions = [], []
    explicit, _ = split_ids(only_caps, ids)
    caps = [c["criterion_id"] for c in prof.quality.get("soft_caps", []) if c["criterion_id"] in ids]
    assert caps and all(c in explicit for c in caps)


def test_the_short_schema_requires_every_decisive_answer_and_is_safe_for_the_provider(prof, ids):
    explicit, others = split_ids(prof, ids)
    s = build_compact_schema(prof, ids)
    assert s["required"] == ["activity_description", "scene_facts", "checks", "other_criteria_not_yes", "evidence", "quality", "visual_flags"]
    assert s["properties"]["checks"]["required"] == explicit and set(s["properties"]["checks"]["properties"]) == set(explicit)
    assert s["properties"]["other_criteria_not_yes"]["items"]["properties"]["criterion_id"]["enum"] == others
    assert s["properties"]["other_criteria_not_yes"]["items"]["properties"]["status"]["enum"] == ["NO", "CANNOT_TELL", "NOT_APPLICABLE"]
    assert s["properties"]["quality"]["required"] == list(prof.quality["dimensions"]) and check_provider_safe(s) == []


# ------------------------------------------------------------------------------------------------------------------------------ reading the answer
def test_the_short_answer_becomes_the_full_assessment(prof, ids):
    _, others = split_ids(prof, ids)
    a = expand_compact(answer(prof, ids, checks={GAZE: "NO"}, others=[{"criterion_id": others[0], "status": "CANNOT_TELL"}],
                              evidence=[{"criterion_id": GAZE, "text": "  The beneficiary looks into the lens.  "}, {"criterion_id": GAZE, "text": "a second one is ignored"}],
                              quality={**{d: 3 for d in prof.quality["dimensions"]}, "brand_presentation": -1},
                              flags=[{"flag_type": list(prof.publication["visual_flags"])[0], "evidence": "x", "confidence": "HIGH"}, {"flag_type": "NOT_A_FLAG"}]),
                       "img-1", prof, ids)
    by = {c.criterion_id: c for c in a.layer_2_criteria}
    assert [c.criterion_id for c in a.layer_2_criteria] == ids and a.image_id == "img-1"
    assert by[GAZE].status.value == "NO" and by[GAZE].evidence == "The beneficiary looks into the lens." and by[others[0]].status.value == "CANNOT_TELL"
    assert all(by[i].status.value == "YES" and by[i].evidence == "" for i in others[1:])
    assert a.layer_3_quality_dimensions["brand_presentation"].applicable is False and a.layer_3_quality_dimensions["composition"].score == 3
    assert [f.flag_type for f in a.layer_4_visual_flags] == [list(prof.publication["visual_flags"])[0]] and a.layer_1_scene_understanding.scene_facts.f_activity_type == "FOOD_DISTRIBUTION"


def test_a_missing_or_unreadable_decisive_answer_is_never_a_pass(prof, ids):
    a = expand_compact(answer(prof, ids, checks={"R007_ANGLE": "MAYBE"}, drop=[GAZE]), "x", prof, ids)
    by = {c.criterion_id: c.status.value for c in a.layer_2_criteria}
    assert by[GAZE] == "CANNOT_TELL" and by["R007_ANGLE"] == "CANNOT_TELL"
    explicit, _ = split_ids(prof, ids)
    sneaky = expand_compact(answer(prof, ids, others=[{"criterion_id": GAZE, "status": "NO"}, {"criterion_id": "NOT_A_RULE", "status": "NO"}]), "x", prof, ids)
    assert {c.criterion_id: c.status.value for c in sneaky.layer_2_criteria}[GAZE] == "YES"   # a decisive criterion is read only from `checks`
    assert all({c.criterion_id: c.status.value for c in sneaky.layer_2_criteria}[i] == "YES" for i in explicit)


@pytest.mark.parametrize("bad, why", [
    ("not json", "not valid JSON"), ("[]", "not a JSON object"), ("```json\n[]\n```", "not a JSON object"),
    ({"quality": {"composition": 7}}, "whole number from -1 to 4"), ({"quality": {"composition": True}}, "whole number"), ({"quality": {"composition": 2.5}}, "whole number"),
    ({"quality": {}}, "whole number"), ({"facts": {"f_photo": "PERHAPS"}}, "does not match"), ({"facts": {"f_orientation": None}}, "does not match")])
def test_an_unusable_short_answer_is_an_error_not_a_guess(prof, ids, bad, why):
    if isinstance(bad, str):
        text = bad
    else:
        obj = json.loads(answer(prof, ids))
        if "quality" in bad:
            obj["quality"] = {**obj["quality"], **bad["quality"]} if bad["quality"] else {}
        if "facts" in bad:
            obj["scene_facts"] = {**obj["scene_facts"], **bad["facts"]}
        text = json.dumps(obj)
    with pytest.raises(ValueError, match=why):
        expand_compact(text, "x", prof, ids)


def test_a_fenced_short_answer_is_read(prof, ids):
    assert expand_compact("```json\n" + answer(prof, ids) + "\n```", "x", prof, ids).layer_2_criteria


def test_the_rule_engine_decides_the_short_answer_exactly_as_it_decides_a_full_one(prof, ids):
    failed = expand_compact(answer(prof, ids, checks={GAZE: "NO"}, evidence=[{"criterion_id": GAZE, "text": "Looks into the lens."}]), "x", prof, ids)
    d = evaluate_final_decision(failed, prof, {}, versions={"prompt_version": "t", "prompt_hash": "t", "model_id": "t"})
    assert d.recommendation.value == "REJECT" and any(g.gate_id == "G015" and g.status.value == "NO" for g in d.gates)
    passed = evaluate_final_decision(expand_compact(answer(prof, ids), "x", prof, ids), prof, {}, versions={"prompt_version": "t", "prompt_hash": "t", "model_id": "t"})
    assert passed.recommendation.value != "REJECT" and not [g for g in passed.gates if g.status.value == "NO"]
    as_full = parse_response(json.dumps(failed.model_dump(mode="json")), "x")    # the same answers, read through the full format's own parser, decide the same way
    again = evaluate_final_decision(as_full, prof, {}, versions={"prompt_version": "t", "prompt_hash": "t", "model_id": "t"})
    assert again.recommendation == d.recommendation and [(g.gate_id, g.status) for g in again.gates] == [(g.gate_id, g.status) for g in d.gates]


# --------------------------------------------------------------------------------------------------------------------------------------- cost
CFG = {"model_id": "gemini-3.8-flash", "cost": {"per_model": {"gemini-3.8-flash": {"input": 0.75, "output": 3.75, "cached_input": 0.075}, "lite": {"input": 0.10, "output": 0.40}},
                                                 "input_per_million_usd": 0.30, "output_per_million_usd": 2.50}}


def test_thinking_is_billed_as_output_and_cached_input_at_the_cached_rate():
    assert cost_usd({"prompt_tokens": 7400, "output_tokens": 5500, "thinking_tokens": 1500}, CFG) == pytest.approx((7400 * 0.75 + 7000 * 3.75) / 1e6)
    assert cost_usd({"prompt_tokens": 7400, "output_tokens": 500, "cached_tokens": 6000}, CFG) == pytest.approx((1400 * 0.75 + 6000 * 0.075 + 500 * 3.75) / 1e6)
    assert cost_usd({"prompt_tokens": 100, "output_tokens": 0, "cached_tokens": 999}, CFG) == pytest.approx(100 * 0.075 / 1e6)          # never more cached than sent
    assert cost_usd({"prompt_tokens": 1000, "output_tokens": 0, "cached_tokens": 1000}, CFG, "lite") == pytest.approx(1000 * 0.10 / 1e6)   # no cached rate known: full rate


def test_prices_come_per_model_and_fall_back_to_the_flat_figures():
    assert prices(CFG) == {"input": 0.75, "output": 3.75, "cached_input": 0.075} and prices(CFG, "unknown-model") == {"input": 0.30, "output": 2.50, "cached_input": None}
    assert prices({"cost": {}}) is None and cost_usd({"prompt_tokens": 1, "output_tokens": 1}, {}) is None and cost_usd({"output_tokens": 1}, CFG) is None
    real = load_gemini_config()
    # the live model is gemini-2.5-flash-lite since 2026-11-05 (short answer format, thinking off); the other models' prices stay in the table for a switch back
    assert prices(real) == {"input": 0.10, "output": 0.40, "cached_input": None} and real["model_id"] == "gemini-2.5-flash-lite"
    assert prices(real, "gemini-3.8-flash") == {"input": 0.75, "output": 3.75, "cached_input": 0.075} and prices(real, "gemini-3.5-flash-lite")["output"] == 2.50
    assert real["evaluation"]["response_format"] == "compact" and real["generation"]["thinking"] == {"budget": 0}


# ------------------------------------------------------------------------------------------------------------------------------------ thinking
class _Models:
    def __init__(self):
        self.configs = []

    def generate_content(self, **kw):
        self.configs.append(kw["config"])
        from types import SimpleNamespace

        return SimpleNamespace(text="{}", usage_metadata=SimpleNamespace(prompt_token_count=7400, candidates_token_count=500, thoughts_token_count=120, total_token_count=8020,
                                                                         cached_content_token_count=6000))


class _Client:
    def __init__(self):
        self.models = _Models()


@pytest.mark.parametrize("thinking, level, budget", [(None, None, None), ({"level": "minimal"}, "MINIMAL", None), ({"budget": 0}, None, 0)])
def test_the_thinking_setting_reaches_the_request_and_the_usage_reports_thinking_and_cached_tokens(thinking, level, budget):
    cfg = json.loads(json.dumps(load_gemini_config()))
    if thinking:
        cfg["generation"]["thinking"] = thinking
    else:
        cfg["generation"].pop("thinking", None)          # the real config now sets one; this case is "no setting at all"
    c = _Client()
    out = GeminiProvider(cfg, client=c, sleep=lambda s: None, env={"NGO_ALLOW_CLOUD_INFERENCE": "true", "GEMINI_API_KEY": "k"}).assess(b"x", "image/jpeg", "s", "u", {})
    tc = c.models.configs[0].thinking_config
    if thinking is None:
        assert tc is None
    else:
        assert (tc.thinking_level.value if tc.thinking_level else None, tc.thinking_budget) == (level, budget)
    assert out.usage["thinking_tokens"] == 120 and out.usage["cached_tokens"] == 6000 and out.usage["output_tokens"] == 500
