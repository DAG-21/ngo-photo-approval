"""Claude Haiku 5.5 as the vision model (AnthropicProvider). A FAKE client: no API call is made and the anthropic package is not needed.

They pin: the request (the rules marked for caching, the photograph as base64, structured output, thinking off, no sampling parameter, no beta header); the answer schema made
acceptable to structured outputs without changing the engine's own schema; the usage mapped to what the cost code expects (cache reads and writes included) and the real cost per
photograph; fail-closed authorisation (its own switch and key); retries, authentication errors, refusals, cut-off answers; that the key never appears in an error; that the
Gemini provider and its prices are untouched; and that the provider is chosen by configuration."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest

from conftest import make_assessment, response_text
from layered_engine.compact import build_compact_schema
from layered_engine.costs import cost_usd, prices, write_rate
from layered_engine.prompt_builder import build_prompt
from layered_engine.providers import (AnthropicProvider, AuthorizationError, GeminiProvider, ProviderError, anthropic_schema, load_vision_config, make_provider)
from layered_engine.schema_export import build_response_schema

PHOTO = bytes([255, 216]) + b"photo"      # JPEG-looking bytes that are not valid text
ENV = {"NGO_ALLOW_ANTHROPIC_INFERENCE": "true", "ANTHROPIC_API_KEY": "fake-anthropic-key-0123456789"}
SCHEMA = {"type": "object", "properties": {"a": {"type": "integer", "minimum": 0, "maximum": 4, "default": 1, "title": "A"}, "title": {"type": "string", "minLength": 1},
                                           "items": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {"x": {"type": "string", "enum": ["P", "Q"]}}, "required": ["x"]}}},
          "required": ["a"]}


class Err(Exception):
    def __init__(self, code, msg="boom"):
        super().__init__(msg)
        self.status_code = code


class FakeMessages:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, script):
        self.messages = FakeMessages(script)


def reply(text='{"ok": true}', stop="end_turn", inp=900, read=0, wrote=0, out=300):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=inp, cache_read_input_tokens=read, cache_creation_input_tokens=wrote, output_tokens=out))


@pytest.fixture(scope="module")
def cfg():
    return load_vision_config("haiku")


def provider(cfg, script, **kw):
    return AnthropicProvider(cfg, client=FakeClient(script), sleep=lambda s: None, env=ENV, **kw)


def objects(node):
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            yield node
        for v in node.values():
            yield from objects(v)
    elif isinstance(node, list):
        for v in node:
            yield from objects(v)


# ------------------------------------------------------------------------------------------------------------------------------ the request
def test_the_request_marks_the_rules_for_caching_sends_the_photograph_and_asks_for_structured_json(cfg):
    p = provider(cfg, [reply()])
    out = p.assess(PHOTO, "image/jpeg", "THE RULES", "Assess this.", SCHEMA)
    kw = p._client.messages.calls[0]
    assert kw["model"] == "claude-haiku-5-5" and kw["max_tokens"] == 8192
    assert kw["system"] == [{"type": "text", "text": "THE RULES", "cache_control": {"type": "ephemeral"}}]
    img, text = kw["messages"][0]["content"]
    assert img["type"] == "image" and img["source"]["type"] == "base64" and img["source"]["media_type"] == "image/jpeg" and img["source"]["data"] == base64.standard_b64encode(PHOTO).decode()
    assert text == {"type": "text", "text": "Assess this."} and kw["messages"][0]["role"] == "user"
    assert kw["thinking"] == {"type": "disabled"}
    assert kw["output_config"]["format"]["type"] == "json_schema" and "effort" not in kw["output_config"]
    for banned in ("temperature", "top_p", "top_k", "betas", "extra_headers", "tools", "tool_choice"):
        assert banned not in kw, banned
    assert out.text == '{"ok": true}' and out.model_id == "claude-haiku-5-5"


def test_the_caching_lifetime_effort_and_thinking_follow_the_configuration(cfg):
    import copy

    c = copy.deepcopy(cfg)
    c["cache"] = {"enabled": True, "ttl": "1h"}
    c["generation"].update(effort="low", thinking="adaptive")
    kw = provider(c, [reply()])._request("S", [], SCHEMA, 100)
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"} and kw["output_config"]["effort"] == "low" and kw["thinking"] == {"type": "adaptive"}
    c["cache"] = {"enabled": False}
    del c["generation"]["thinking"]
    kw = provider(c, [reply()])._request("S", [], SCHEMA, 100)
    assert kw["system"] == "S" and "thinking" not in kw


def test_similar_photographs_are_sent_as_examples_before_the_new_one(cfg):
    p = provider(cfg, [reply()])
    ex = [{"image_bytes": b"E1", "mime_type": "image/png", "decision": "APPROVE", "tags": ["x"], "reasons": "fine"}]
    p.assess(b"NEW", "image/jpeg", "S", "go", SCHEMA, examples=ex)
    blocks = p._client.messages.calls[0]["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["text", "image", "text", "text", "image", "text"] and blocks[1]["source"]["media_type"] == "image/png"
    assert "Past decision 1: staff approved" in blocks[2]["text"] and blocks[-1]["text"] == "go"


def test_the_answer_schema_is_made_acceptable_to_structured_outputs_without_changing_the_original(cfg):
    import copy

    before = copy.deepcopy(SCHEMA)
    out = anthropic_schema(SCHEMA)
    assert SCHEMA == before                                                       # the engine's own schema (and so the Gemini request) is untouched
    assert all(o["additionalProperties"] is False for o in objects(out))
    assert out["properties"]["a"] == {"type": "integer"} and "title" in out["properties"] and out["properties"]["title"] == {"type": "string"}   # a PROPERTY called title stays
    assert out["properties"]["items"] == {"type": "array", "items": {"type": "object", "properties": {"x": {"type": "string", "enum": ["P", "Q"]}}, "required": ["x"], "additionalProperties": False}}


def test_both_real_answer_schemas_of_the_final_profile_become_valid_for_claude(rb):
    final = rb.resolve("layered_final_v1")
    for fmt in ("full", "compact"):
        prompt = build_prompt(rb, final, response_format=fmt)
        schema = build_response_schema(final, prompt.criterion_ids) if fmt == "full" else build_compact_schema(final, prompt.criterion_ids)
        out = anthropic_schema(schema)
        text = json.dumps(out)
        assert all(o["additionalProperties"] is False for o in objects(out)), fmt
        for k in ("minimum", "maximum", "minItems", "maxItems", "minLength", "maxLength", '"default"', '"$ref"'):
            assert k not in text, (fmt, k)
        assert "scene_facts" in text and "f_logo" not in text and "f_beneficiaries" in text


# ------------------------------------------------------------------------------------------------------------------------------ the answer and its cost
def test_usage_counts_cache_reads_and_writes_and_the_cost_is_the_real_one(cfg):
    out = provider(cfg, [reply(inp=1500, read=0, wrote=8000, out=2000)]).assess(b"x", "image/jpeg", "S", "u", SCHEMA)
    assert out.usage["prompt_tokens"] == 9500 and out.usage["cache_write_tokens"] == 8000 and out.usage["cached_tokens"] == 0 and out.usage["output_tokens"] == 2000
    first = cost_usd(out.usage, cfg)
    assert first == pytest.approx((1500 * 0.10 + 8000 * 0.125 + 2000 * 0.50) / 1e6)             # the first photograph pays the cache write
    warm = provider(cfg, [reply(inp=1500, read=8000, wrote=0, out=2000)]).assess(b"x", "image/jpeg", "S", "u", SCHEMA)
    second = cost_usd(warm.usage, cfg)
    assert second == pytest.approx((1500 * 0.10 + 8000 * 0.01 + 2000 * 0.50) / 1e6) and second < first
    assert prices(cfg) == {"input": 0.10, "output": 0.50, "cached_input": 0.01} and write_rate(cfg) == 0.125
    assert "thinking_tokens" in warm.usage and warm.usage["total_tokens"] == 11500


def test_the_cost_code_is_unchanged_for_gemini_and_a_missing_write_price_falls_back_to_the_input_price(cfg):
    g = load_vision_config("gemini")
    usage = {"prompt_tokens": 7000, "output_tokens": 500, "thinking_tokens": 100, "cached_tokens": 6000}
    assert cost_usd(usage, g, "gemini-3.8-flash") == pytest.approx((1000 * 0.75 + 6000 * 0.075 + 600 * 3.75) / 1e6)
    assert write_rate(g, "gemini-2.5-flash-lite") == 0.10 and cost_usd({**usage, "cache_write_tokens": 500}, g, "gemini-2.5-flash-lite") == cost_usd(usage, g, "gemini-2.5-flash-lite")
    assert prices(g, "gemini-2.5-flash-lite") == {"input": 0.10, "output": 0.40, "cached_input": None}


# ------------------------------------------------------------------------------------------------------------------------------ failing closed and failing honestly
def test_it_fails_closed_without_its_own_switch_and_its_own_key(cfg):
    with pytest.raises(AuthorizationError, match="Anthropic is not authorized"):
        AnthropicProvider(cfg, env={"ANTHROPIC_API_KEY": "k"})
    with pytest.raises(AuthorizationError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider(cfg, env={"NGO_ALLOW_ANTHROPIC_INFERENCE": "true"})
    with pytest.raises(AuthorizationError, match="not authorized"):                      # the Google switch does not authorise Anthropic
        AnthropicProvider(cfg, env={"NGO_ALLOW_CLOUD_INFERENCE": "true", "ANTHROPIC_API_KEY": "k", "GEMINI_API_KEY": "g"})
    assert AnthropicProvider(cfg, env=ENV).model_id == "claude-haiku-5-5"


def test_transient_errors_are_retried_and_a_bad_key_is_not(cfg):
    p = provider(cfg, [Err(529), Err(429), reply(out=100)])
    out = p.assess(b"x", "image/jpeg", "S", "u", SCHEMA)
    assert out.usage["attempts"] == 3 and len(p._client.messages.calls) == 3
    p = provider(cfg, [Err(401, "invalid x-api-key fake-anthropic-key-0123456789")])
    with pytest.raises(ProviderError, match="authentication") as e:
        p.assess(b"x", "image/jpeg", "S", "u", SCHEMA)
    assert "fake-anthropic-key-0123456789" not in str(e.value) and len(p._client.messages.calls) == 1
    p = provider(cfg, [Err(500, "server said fake-anthropic-key-0123456789")] * 3)
    with pytest.raises(ProviderError) as e:
        p.assess(b"x", "image/jpeg", "S", "u", SCHEMA)
    assert "fake-anthropic-key-0123456789" not in str(e.value) and "<key>" in str(e.value) and len(p._client.messages.calls) == 3
    p = provider(cfg, [Err(400, "bad request")])
    with pytest.raises(ProviderError, match="HTTP 400"):
        p.assess(b"x", "image/jpeg", "S", "u", SCHEMA)
    assert len(p._client.messages.calls) == 1                                           # a rejected request is never retried


def test_a_refusal_and_a_cut_off_answer_are_errors_never_an_evaluation(cfg):
    with pytest.raises(ProviderError, match="declined"):
        provider(cfg, [reply(stop="refusal", text="")]).assess(b"x", "image/jpeg", "S", "u", SCHEMA)
    with pytest.raises(ProviderError, match="cut off"):
        provider(cfg, [reply(stop="max_tokens", text='{"a":')]).assess(b"x", "image/jpeg", "S", "u", SCHEMA)


def test_a_connection_error_is_retried_by_its_name(cfg):
    class APIConnectionError(Exception):
        pass

    p = provider(cfg, [APIConnectionError("net"), reply()])
    assert p.assess(b"x", "image/jpeg", "S", "u", SCHEMA).usage["attempts"] == 2


def test_the_text_only_request_has_no_photograph(cfg):
    p = provider(cfg, [reply('{"r": 1}')])
    out = p.complete_text("SYS", "reasons text", SCHEMA, max_output_tokens=500)
    kw = p._client.messages.calls[0]
    assert kw["max_tokens"] == 500 and kw["messages"][0]["content"] == [{"type": "text", "text": "reasons text"}] and out.text == '{"r": 1}'


# ------------------------------------------------------------------------------------------------------------------------------ choosing the provider
def test_the_provider_is_chosen_by_configuration_and_gemini_is_still_the_default(cfg):
    assert load_vision_config()["provider"] == "google" and load_vision_config(env={"NGO_VISION": "haiku"})["provider"] == "anthropic" and load_vision_config("haiku")["model_id"] == "claude-haiku-5-5"
    with pytest.raises(ValueError, match="unknown vision model"):
        load_vision_config("gpt")
    assert isinstance(make_provider(cfg, client=FakeClient([]), env=ENV), AnthropicProvider)
    g = load_vision_config("gemini")
    assert isinstance(make_provider(g, client=object(), env={"NGO_ALLOW_CLOUD_INFERENCE": "true", "GEMINI_API_KEY": "k"}), GeminiProvider)
    with pytest.raises(ValueError, match="unknown provider"):
        make_provider({"provider": "other"})


def test_the_haiku_configuration_matches_the_decisions_taken(cfg):
    assert cfg["provider"] == "anthropic" and cfg["generation"]["thinking"] == "disabled" and cfg["evaluation"]["response_format"] == "compact" and cfg["evaluation"]["profile"] == "layered_final_v1"
    assert cfg["auth"] == {"api_key_env": "ANTHROPIC_API_KEY", "authorization_env": "NGO_ALLOW_ANTHROPIC_INFERENCE"} and cfg["cache"]["enabled"] is True
    assert "temperature" not in cfg["generation"]                                          # Haiku 5.5 rejects sampling parameters
    assert load_vision_config("gemini")["model_id"] == "gemini-2.5-flash-lite"              # the Gemini setting of earlier today is untouched


def test_the_request_fits_the_real_sdk_when_it_is_installed(cfg):
    """Guards against drift between our request and the installed anthropic package (its method signature and typed dictionaries). Nothing is sent: only types are inspected."""
    import inspect

    anthropic = pytest.importorskip("anthropic")
    import anthropic.types as t

    kw = provider(cfg, [reply()])._request("RULES", [{"type": "text", "text": "x"}], SCHEMA, 100)
    sig = set(inspect.signature(anthropic.Anthropic(api_key="not-a-real-key", max_retries=0).messages.create).parameters)
    assert set(kw) <= sig, set(kw) - sig
    assert set(kw["output_config"]) <= set(t.OutputConfigParam.__annotations__) and set(kw["output_config"]["format"]) <= set(t.JSONOutputFormatParam.__annotations__)
    assert set(kw["system"][0]["cache_control"]) <= set(t.CacheControlEphemeralParam.__annotations__) and set(kw["thinking"]) <= set(t.ThinkingConfigDisabledParam.__annotations__)
    assert "status_code" in inspect.getsource(anthropic.APIStatusError.__init__)       # the attribute our retry logic reads
