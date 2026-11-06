"""The Groq text model for the gap analyzer. Everything here goes through a FAKE HTTP transport: no request is ever sent to Groq (a guard makes any real network
call fail the test), so these tests pin the request, the authorisation, the retries and the error handling without any key or any spend."""

from __future__ import annotations

import json

import httpx
import pytest
from layered_engine.providers import AuthorizationError, ProviderError

from ngo_agent.groq_text_model import (AUTH_ENV, DEFAULT_MODEL, KEY_ENV, URL, GroqTextModel, Rejected, strictify)
from ngo_agent.llm_gap_analyzer import SCHEMA, SYSTEM, LlmGapAnalyzer, ModelOutputError, make_gap_analyzer, make_text_model
from test_llm_gaps import CASE, point

KEY = "gsk-secret-key"
ENV = {AUTH_ENV: "true", KEY_ENV: KEY}


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    def refuse(self, request):  # noqa: ANN001
        raise AssertionError("a real network request was attempted")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)


def server(*responses):
    """A fake Groq: answers with the scripted responses in turn (the last repeats) and records every request."""
    calls, queue = [], list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"url": str(request.url), "headers": dict(request.headers), "body": json.loads(request.content)})
        r = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(r, Exception):
            raise r
        status, body, headers = r
        return httpx.Response(status, json=body, headers=headers) if isinstance(body, (dict, list)) else httpx.Response(status, text=body, headers=headers)

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def ok(content, finish="stop", prompt=100, out=20, model=DEFAULT_MODEL):
    return 200, {"choices": [{"message": {"content": content}, "finish_reason": finish}], "usage": {"prompt_tokens": prompt, "completion_tokens": out}, "model": model}, {}


GOOD = json.dumps({"points": []})


def model(client, sleeps=None, env=ENV, **kw):
    return GroqTextModel(env, client=client, sleep=(sleeps.append if sleeps is not None else (lambda s: None)), **kw)


def test_the_guard_really_blocks_a_real_request():
    with pytest.raises(AssertionError, match="real network request"):
        httpx.Client().get("http://groq.invalid/")


# ----------------------------------------------------------------------------------------------------------------------------- authorisation
def test_groq_has_its_own_switch_and_key_and_never_borrows_the_gemini_switch():
    for env, wanted in [({}, AUTH_ENV), ({"NGO_ALLOW_CLOUD_INFERENCE": "true", KEY_ENV: KEY}, AUTH_ENV), ({AUTH_ENV: "false", KEY_ENV: KEY}, AUTH_ENV), ({AUTH_ENV: "true"}, KEY_ENV)]:
        with pytest.raises(AuthorizationError, match=wanted) as e:
            GroqTextModel(env)
        assert KEY not in str(e.value)
    env = dict(ENV)
    assert GroqTextModel(env).model_id == DEFAULT_MODEL and env == ENV                                      # the mapping is only read, never changed
    assert GroqTextModel({**ENV, "GROQ_MODEL": "meta/some-model"}).model_id == "meta/some-model"
    assert GroqTextModel({**ENV, "GROQ_MODEL": "meta/some-model"}, model="explicit/one").model_id == "explicit/one"


# ----------------------------------------------------------------------------------------------------------------------------------- request
def test_a_gpt_oss_model_is_asked_with_a_strict_schema_and_the_answer_is_priced_at_zero():
    client, calls = server(ok(GOOD, prompt=1200, out=300))
    ans = model(client).complete("system words", "user words", SCHEMA)
    c = calls[0]
    assert c["url"] == URL and c["headers"]["authorization"] == f"Bearer {KEY}"
    b = c["body"]
    assert b["model"] == DEFAULT_MODEL and [m["role"] for m in b["messages"]] == ["system", "user"] and b["messages"][0]["content"] == "system words" and b["messages"][1]["content"] == "user words"
    assert b["response_format"] == {"type": "json_schema", "json_schema": {"name": "gap_analysis", "strict": True, "schema": strictify(SCHEMA)}}
    assert b["temperature"] == 0.1 and b["max_completion_tokens"] == 4096
    assert (ans.text, ans.model_id, ans.cost_usd) == (GOOD, DEFAULT_MODEL, 0.0)
    assert ans.usage == {"prompt_tokens": 1200, "output_tokens": 300, "requests": 1, "attempts": 1}


def test_the_strict_form_of_a_schema_closes_every_object_requires_every_property_and_drops_length_keywords():
    s = strictify(SCHEMA)
    item = s["properties"]["points"]["items"]
    assert s["additionalProperties"] is False and s["required"] == ["points"]
    assert item["additionalProperties"] is False and item["required"] == list(SCHEMA["properties"]["points"]["items"]["properties"])      # rule_note is now required too
    assert "maxItems" not in json.dumps(s) and "maxItems" in json.dumps(SCHEMA) and "rule_note" not in SCHEMA["properties"]["points"]["items"]["required"]   # the original is untouched


def test_any_other_model_gets_plain_json_mode_with_the_schema_written_into_the_instruction():
    client, calls = server(ok(GOOD))
    model(client, model="llama-3.3-70b-versatile").complete(SYSTEM, "u", SCHEMA)
    b = calls[0]["body"]
    assert b["response_format"] == {"type": "json_object"} and b["model"] == "llama-3.3-70b-versatile"
    assert b["messages"][0]["content"].startswith(SYSTEM) and "one JSON object" in b["messages"][0]["content"] and json.dumps(SCHEMA) in b["messages"][0]["content"]


# ---------------------------------------------------------------------------------------------------------------------- an unusable answer
@pytest.mark.parametrize("first", [ok("Sorry, I cannot help with that."), ok(""), ok('{"points": [', finish="length"), ok(GOOD, finish="length"), ok("[1, 2]"), (200, {"choices": []}, {})])
def test_an_answer_that_is_not_a_json_object_is_asked_for_once_more(first):
    client, calls = server(first, ok(GOOD))
    ans = model(client).complete("s", "u", SCHEMA)
    assert ans.text == GOOD and len(calls) == 2 and ans.usage["attempts"] == 2 and ans.usage["requests"] == 2


def test_after_the_one_retry_the_unusable_text_is_returned_and_the_analyzer_refuses_it(rb):
    client, calls = server(ok("still not json"))
    m = model(client)
    ans = m.complete("s", "u", SCHEMA)
    assert ans.text == "still not json" and len(calls) == 2                                                  # exactly one retry, never a third request
    with pytest.raises(ModelOutputError):
        LlmGapAnalyzer(model(server(ok("still not json"))[0]), rb).analyze(CASE)
    assert ans.usage["prompt_tokens"] == 200 and ans.usage["output_tokens"] == 40                            # both requests are counted


# ------------------------------------------------------------------------------------------------------------------------------------ errors
def test_transient_errors_are_retried_with_backoff_and_retry_after_is_honoured(monkeypatch):
    monkeypatch.setattr("ngo_agent.groq_text_model.random.random", lambda: 0.5)                              # no jitter, so the waits are exact
    sleeps = []
    client, calls = server((429, {"error": "rate limit"}, {"retry-after": "3"}), (503, {"error": "busy"}, {}), ok(GOOD))
    ans = model(client, sleeps).complete("s", "u", SCHEMA)
    assert ans.text == GOOD and len(calls) == 3 and len(sleeps) == 2 and ans.usage["requests"] == 3 and ans.usage["attempts"] == 1
    assert sleeps == [3.0, 4.0]                                                                              # Retry-After first, then the doubled backoff


def test_a_very_long_retry_after_is_capped(monkeypatch):
    monkeypatch.setattr("ngo_agent.groq_text_model.random.random", lambda: 0.5)
    sleeps = []
    client, _ = server((429, {"error": "rate limit"}, {"retry-after": "120"}), ok(GOOD))
    model(client, sleeps).complete("s", "u", SCHEMA)
    assert sleeps == [30.0]


def test_permanent_errors_are_not_retried_and_the_key_never_appears_in_a_message():
    client, calls = server((401, {"error": "bad key gsk-secret-key"}, {}))
    with pytest.raises(ProviderError, match="authentication") as e:
        model(client).complete("s", "u", SCHEMA)
    assert len(calls) == 1 and KEY not in str(e.value)
    client, calls = server((500, {"error": "boom"}, {}))
    with pytest.raises(ProviderError) as e:
        model(client).complete("s", "u", SCHEMA)
    assert len(calls) == 3 and KEY not in str(e.value)                                                       # the configured number of attempts, then it gives up
    client, calls = server((413, "request too large gsk-secret-key", {}))
    with pytest.raises(Rejected) as e:
        model(client).complete("s", "u", SCHEMA)
    assert e.value.status == 413 and len(calls) == 1 and KEY not in str(e.value) and "<key>" in str(e.value)  # an echoed key is scrubbed


def test_a_network_failure_is_retried_then_reported_by_type_only():
    sleeps = []
    client, calls = server(httpx.ConnectError("could not connect, key gsk-secret-key"))
    with pytest.raises(ProviderError, match="ConnectError") as e:
        model(client, sleeps).complete("s", "u", SCHEMA)
    assert len(calls) == 3 and len(sleeps) == 2 and KEY not in str(e.value)


def test_a_response_that_is_not_in_the_expected_shape_is_an_error():
    client, _ = server((200, "<html>not json</html>", {}))
    with pytest.raises(ProviderError, match="expected shape"):
        model(client).complete("s", "u", SCHEMA)


# -------------------------------------------------------------------------------------------------------------------- strict schema refused
def test_a_refused_strict_schema_falls_back_once_to_plain_json_and_remembers_it():
    client, calls = server((400, {"error": {"message": "schema not supported"}}, {}), ok(GOOD), ok(GOOD))
    m = model(client)
    assert m.complete("s", "u", SCHEMA).text == GOOD
    assert [c["body"]["response_format"]["type"] for c in calls] == ["json_schema", "json_object"]
    assert m.complete("s", "u", SCHEMA).text == GOOD
    assert [c["body"]["response_format"]["type"] for c in calls] == ["json_schema", "json_object", "json_object"]      # the next request goes straight to plain JSON


def test_a_400_in_plain_json_mode_is_not_retried():
    client, calls = server((400, {"error": "bad request"}, {}))
    with pytest.raises(Rejected) as e:
        model(client, model="llama-3.3-70b-versatile").complete("s", "u", SCHEMA)
    assert e.value.status == 400 and len(calls) == 1


# ------------------------------------------------------------------------------------------------------------------------------ cost, wiring
def test_prices_can_be_given_when_the_account_is_upgraded():
    client, _ = server(ok(GOOD, prompt=1_000_000, out=500_000))
    ans = model(client, input_per_million_usd=1.0, output_per_million_usd=2.0).complete("s", "u", SCHEMA)
    assert ans.cost_usd == 2.0


def test_the_analyzer_works_end_to_end_through_the_groq_model(rb):
    client, calls = server(ok(json.dumps({"points": [point("The reviewer accepts the gaze because the child is receiving food.", "CONTRADICTS_AI", ["G015"], "Accepted when receiving.")]}), model="openai/gpt-oss-120b"))
    out = LlmGapAnalyzer(model(client), rb).analyze(CASE)
    assert out.gaps[0]["kind"] == "CONTESTED_GATE" and out.gaps[0]["related_gates"] == ["G015"] and out.model_id == "openai/gpt-oss-120b" and out.cost_usd == 0.0
    assert "receiving food" in calls[0]["body"]["messages"][1]["content"] and "<reviewer_text>" in calls[0]["body"]["messages"][1]["content"]     # the same text-only request as for Gemini


def test_the_model_provider_is_chosen_by_name_and_fails_closed_without_its_own_authorisation(rb):
    assert isinstance(make_text_model("groq", env=ENV), GroqTextModel)
    an = make_gap_analyzer("llm", llm="groq", env=ENV, rulebook=rb, max_calls=5)
    assert isinstance(an, LlmGapAnalyzer) and isinstance(an.model, GroqTextModel) and an.max_calls == 5
    with pytest.raises(AuthorizationError):
        make_gap_analyzer("llm", llm="groq", env={}, rulebook=rb)
    with pytest.raises(ValueError, match="needs the Gemini provider"):
        make_gap_analyzer("llm", llm="gemini", rulebook=rb)
    with pytest.raises(ValueError, match="unknown model provider"):
        make_gap_analyzer("llm", llm="nope", rulebook=rb)
