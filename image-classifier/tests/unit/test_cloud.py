from __future__ import annotations

import json

import pytest

from ngo_photo_approval.config import CloudConfig, RetryConfig
from ngo_photo_approval.exceptions import (
    AuthorizationError,
    ConfigError,
    ModelUnavailableError,
    ProviderAuthError,
    ProviderError,
    ResponseValidationError,
    TransientProviderError,
)
from ngo_photo_approval.models.baseline_analyzer import RuleAnalyzer, build_prompt, estimate_cost
from ngo_photo_approval.models.cloud_client import (
    GeminiBackend,
    RateLimiter,
    call_with_retry,
    map_exception,
    require_inference_authorization,
)
from ngo_photo_approval.models.response_parser import (
    build_response_schema,
    parse_decision_label,
    parse_visual_analysis,
)
from ngo_photo_approval.models.tuned_model import TunedDecisionClassifier, resolve_tuned_model_id
from ngo_photo_approval.schemas import UsageInfo

from ..conftest import FakeBackend, checks_json, make_image

ALL_PASS = {"R001": "PASS", "R002": "PASS", "R003": "PASS", "R004": "PASS"}


class SdkError(Exception):
    """Stand-in for google.genai.errors.APIError (has an int ``code``)."""

    def __init__(self, code: int) -> None:
        super().__init__(f"http {code}")
        self.code = code


# ------------------------------------------------------------------ response parsing


def test_valid_structured_response_parses(rule_set):
    a = parse_visual_analysis(checks_json(ALL_PASS), rule_set.active(), image_id="i1")
    assert a.image_id == "i1" and len(a.checks) == 4 and a.checks[0].status.value == "PASS"


def test_code_fenced_and_prose_wrapped_json_tolerated(rule_set):
    body = checks_json(ALL_PASS)
    assert len(parse_visual_analysis(f"```json\n{body}\n```", rule_set.active()).checks) == 4
    assert len(parse_visual_analysis(f"Here you go: {body} Thanks.", rule_set.active()).checks) == 4


@pytest.mark.parametrize("text", ["", "   ", "not json", "[1,2,3]", '{"checks": "nope"}', '{"foo": 1}'])
def test_malformed_responses_raise(rule_set, text):
    with pytest.raises(ResponseValidationError):
        parse_visual_analysis(text, rule_set.active())


def test_incomplete_duplicate_unknown_and_bad_status_raise(rule_set):
    rules = rule_set.active()
    with pytest.raises(ResponseValidationError, match="missing"):
        parse_visual_analysis(checks_json({"R001": "PASS"}), rules)
    dup = checks_json(ALL_PASS, extra=[{"rule_id": "R001", "status": "PASS", "evidence": ""}])
    with pytest.raises(ResponseValidationError, match="duplicated"):
        parse_visual_analysis(dup, rules)
    unk = checks_json(ALL_PASS, extra=[{"rule_id": "R777", "status": "PASS", "evidence": ""}])
    with pytest.raises(ResponseValidationError, match="unknown"):
        parse_visual_analysis(unk, rules)
    with pytest.raises(ResponseValidationError):
        parse_visual_analysis(checks_json({**ALL_PASS, "R001": "APPROVED"}), rules)


def test_self_reported_confidence_is_recorded_separately(rule_set):
    payload = json.loads(checks_json(ALL_PASS))
    payload["self_reported_confidence"] = 0.97
    a = parse_visual_analysis(json.dumps(payload), rule_set.active())
    assert a.self_reported_confidence == 0.97
    with pytest.raises(ResponseValidationError):
        payload["self_reported_confidence"] = 5
        parse_visual_analysis(json.dumps(payload), rule_set.active())


def test_response_schema_constrains_ids_and_statuses(rule_set):
    s = build_response_schema(rule_set.active())
    item = s["properties"]["checks"]["items"]["properties"]
    assert item["rule_id"]["enum"] == ["R001", "R002", "R003", "R004"]
    assert item["status"]["enum"] == ["PASS", "FAIL", "UNCERTAIN"]


def test_decision_label_parser():
    assert parse_decision_label('{"decision": "approved"}') == "approved"
    for bad in ('{"decision": "maybe"}', "{}", "approved"):
        with pytest.raises(ResponseValidationError):
            parse_decision_label(bad)


def test_prompt_lists_every_rule_and_forbids_identification(rule_set):
    prompt = build_prompt(rule_set.active())
    assert all(r in prompt for r in ("R001", "R002", "R003", "R004"))
    from ngo_photo_approval.models.baseline_analyzer import SYSTEM_INSTRUCTION

    assert "facial recognition" in SYSTEM_INSTRUCTION and "Never guess a PASS" in SYSTEM_INSTRUCTION


# ------------------------------------------------------------------ error mapping and retry


@pytest.mark.parametrize(("exc", "cls"), [
    (SdkError(429), TransientProviderError), (SdkError(503), TransientProviderError),
    (SdkError(500), TransientProviderError), (TimeoutError(), TransientProviderError),
    (SdkError(401), ProviderAuthError), (SdkError(403), ProviderAuthError),
    (SdkError(404), ModelUnavailableError), (SdkError(400), ProviderError),
    (RuntimeError("boom"), ProviderError),
])
def test_exception_mapping(exc, cls):
    mapped = map_exception(exc)
    assert type(mapped) is cls
    assert mapped.retryable == (cls is TransientProviderError)


def test_retry_recovers_from_transient_errors():
    calls = []
    sleeps = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise SdkError(429)
        return "ok"

    assert call_with_retry(fn, RetryConfig(max_attempts=4, initial_backoff_s=1.0), sleep=sleeps.append,
                           jitter=lambda: 0.0) == "ok"
    assert len(calls) == 3 and sleeps == [0.5, 1.0]  # exponential backoff with jitter floor


def test_retry_gives_up_after_max_attempts():
    calls = []

    def fn():
        calls.append(1)
        raise SdkError(503)

    with pytest.raises(TransientProviderError):
        call_with_retry(fn, RetryConfig(max_attempts=3, initial_backoff_s=0), sleep=lambda s: None)
    assert len(calls) == 3


@pytest.mark.parametrize("code", [401, 403, 404, 400])
def test_non_transient_errors_are_not_retried(code):
    calls = []

    def fn():
        calls.append(1)
        raise SdkError(code)

    with pytest.raises(ProviderError):
        call_with_retry(fn, RetryConfig(max_attempts=5), sleep=lambda s: None)
    assert len(calls) == 1


def test_rate_limiter_spaces_calls():
    t = [0.0]
    slept = []

    def sleep(d):
        slept.append(round(d, 3))
        t[0] += d

    lim = RateLimiter(2.0, clock=lambda: t[0], sleep=sleep)
    for _ in range(3):
        lim.wait()
    assert slept == [0.5, 0.5]
    assert RateLimiter(0).wait() is None


# ------------------------------------------------------------------ authorization


def test_cloud_inference_requires_explicit_authorization():
    with pytest.raises(AuthorizationError):
        require_inference_authorization(CloudConfig())
    with pytest.raises(AuthorizationError):
        GeminiBackend(CloudConfig(project_id="p"))
    with pytest.raises(ConfigError):
        GeminiBackend(CloudConfig(allow_cloud_inference=True))
    GeminiBackend(CloudConfig(allow_cloud_inference=True, project_id="p"))  # constructs, sends nothing


def test_api_key_mode_requires_the_key_and_not_a_project(monkeypatch):
    cloud = CloudConfig(allow_cloud_inference=True, auth_mode="api_key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        GeminiBackend(cloud)
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(ConfigError):
        GeminiBackend(cloud)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-not-a-real-key")
    GeminiBackend(cloud)  # no project needed in api_key mode
    with pytest.raises(AuthorizationError):  # the transfer gate still applies in api_key mode
        GeminiBackend(CloudConfig(auth_mode="api_key"))


def test_api_key_is_used_only_in_api_key_mode(monkeypatch):
    seen = {}

    class FakeGenai:
        @staticmethod
        def Client(**kw):  # noqa: N802 - mimics genai.Client
            seen.update(kw)
            return object()

    import google

    monkeypatch.setattr(google, "genai", FakeGenai, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-not-a-real-key")
    GeminiBackend(CloudConfig(allow_cloud_inference=True, auth_mode="api_key"))._get_client()
    assert seen == {"api_key": "dummy-not-a-real-key"}
    seen.clear()
    GeminiBackend(CloudConfig(allow_cloud_inference=True, project_id="p", location="us"))._get_client()
    assert seen == {"vertexai": True, "project": "p", "location": "us"}  # a stray GEMINI_API_KEY is ignored


def test_auth_mode_env_override_is_validated(monkeypatch):
    from ngo_photo_approval.config import load_project_config

    monkeypatch.setenv("NGO_AUTH_MODE", "api_key")
    assert load_project_config().cloud.auth_mode == "api_key"
    monkeypatch.setenv("NGO_AUTH_MODE", "password")
    with pytest.raises(ConfigError):
        load_project_config()


def test_conditional_rules_reach_the_prompt():
    from ngo_photo_approval.models.baseline_analyzer import PROMPT_VERSION, SYSTEM_INSTRUCTION
    from ngo_photo_approval.rules.loader import load_rule_set

    prompt = build_prompt(load_rule_set().active())
    assert "applies only when: a banner is present in the photo" in prompt
    assert 'client wording (authoritative, apply exactly): "If a banner is present, ensure it is clearly visible' in prompt
    assert "N/A:" in SYSTEM_INSTRUCTION and PROMPT_VERSION == "rules-v4"
    assert prompt.count("rule_id:") == 16


# ------------------------------------------------------------------ analyzer


@pytest.fixture
def image(tmp_path):
    return make_image(tmp_path / "x.jpg")


def analyzer(rule_set, model_cfg, script):
    backend = FakeBackend(script)
    return RuleAnalyzer(backend, rule_set, model_cfg, "gemini-test"), backend


def test_analyzer_success(rule_set, model_cfg, image):
    a, backend = analyzer(rule_set, model_cfg, [checks_json(ALL_PASS)])
    r = a.analyze(image, "i1")
    assert r.ok and r.analysis and len(r.analysis.checks) == 4 and r.model_id == "gemini-test"
    assert r.usage.total_tokens == 120 and r.rule_set_version == "test-1.0"
    req = backend.calls[0]
    assert req.mime_type == "image/jpeg" and req.image_bytes == image.read_bytes()  # no needless re-encoding


def test_analyzer_malformed_response_is_a_failure_not_an_approval(rule_set, model_cfg, image):
    for bad in ("garbage", checks_json({"R001": "PASS"})):
        a, _ = analyzer(rule_set, model_cfg, [bad])
        r = a.analyze(image, "i1")
        assert not r.ok and r.analysis is None and r.error_type == "invalid_response"


def test_analyzer_provider_failures_are_explicit(rule_set, model_cfg, image):
    for exc, retryable in [(SdkError(503), True), (SdkError(401), False), (SdkError(404), False), (TimeoutError(), True)]:
        a, backend = analyzer(rule_set, model_cfg, [exc])
        r = a.analyze(image, "i1")
        assert not r.ok and r.error_type == "provider_error" and r.retryable is retryable
        assert len(backend.calls) == (model_cfg.retry.max_attempts if retryable else 1)


def test_analyzer_retries_then_succeeds(rule_set, model_cfg, image):
    a, backend = analyzer(rule_set, model_cfg, [SdkError(429), SdkError(503), checks_json(ALL_PASS)])
    assert a.analyze(image, "i1").ok and len(backend.calls) == 3


def test_analyzer_missing_or_corrupt_image_never_calls_provider(rule_set, model_cfg, tmp_path):
    a, backend = analyzer(rule_set, model_cfg, [checks_json(ALL_PASS)])
    assert a.analyze(tmp_path / "ghost.jpg", "i1").error_type == "invalid_image"
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"xx")
    assert a.analyze(bad, "i2").error_type == "invalid_image"
    assert backend.calls == []


def test_authorization_error_is_not_swallowed(rule_set, model_cfg, image):
    a, _ = analyzer(rule_set, model_cfg, [AuthorizationError("no")])
    with pytest.raises(AuthorizationError):
        a.analyze(image, "i1")


def test_cost_estimate_unknown_until_priced(model_cfg):
    u = UsageInfo(prompt_tokens=1_000_000, output_tokens=100_000)
    assert estimate_cost(u, model_cfg.pricing) is None
    model_cfg.pricing.input_per_million_tokens_usd = 1.0
    model_cfg.pricing.output_per_million_tokens_usd = 10.0
    assert estimate_cost(u, model_cfg.pricing) == 2.0
    assert estimate_cost(None, model_cfg.pricing) is None


# ------------------------------------------------------------------ tuned model wrappers


def test_tuned_model_never_guessed(model_cfg):
    with pytest.raises(ConfigError):
        resolve_tuned_model_id(model_cfg)
    model_cfg.tuned_model.endpoint = "projects/p/locations/us/endpoints/1"
    assert resolve_tuned_model_id(model_cfg).endswith("/1")


def test_tuned_decision_classifier(model_cfg, image):
    ok = TunedDecisionClassifier(FakeBackend(['{"decision": "approved"}']), model_cfg, "tuned")
    r = ok.classify(image, "i1")
    assert r.ok and r.decision == "approved"
    bad = TunedDecisionClassifier(FakeBackend(["approved!"]), model_cfg, "tuned").classify(image, "i2")
    assert not bad.ok and bad.error_type == "invalid_response" and bad.decision is None
    down = TunedDecisionClassifier(FakeBackend([SdkError(404)]), model_cfg, "tuned").classify(image, "i3")
    assert not down.ok and down.error_type == "provider_error"


def test_client_is_created_once_even_with_concurrent_first_calls(monkeypatch):
    """Regression: parallel batch workers raced to build the SDK client and the first calls failed."""
    import threading
    import time

    created = []

    class FakeGenai:
        @staticmethod
        def Client(**kw):  # noqa: N802 - mimics genai.Client
            time.sleep(0.05)  # widen the race window
            obj = object()
            created.append(obj)
            return obj

    import google

    monkeypatch.setattr(google, "genai", FakeGenai, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-not-a-real-key")
    backend = GeminiBackend(CloudConfig(allow_cloud_inference=True, auth_mode="api_key"))
    got: list = []
    threads = [threading.Thread(target=lambda: got.append(backend._get_client())) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(created) == 1 and len(got) == 8 and all(g is created[0] for g in got)


def test_unexpected_errors_keep_a_scrubbed_message(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key-123")
    err = map_exception(RuntimeError("client closed while using super-secret-key-123"))
    assert "client closed" in str(err) and "super-secret-key-123" not in str(err) and "<key>" in str(err)


def test_camera_from_the_front_guidance_reaches_the_prompt_for_the_right_rules_only():
    from ngo_photo_approval.rules.loader import load_rule_set

    rs = load_rule_set()
    with_guidance = {r.rule_id for r in rs.rules if r.additional_guidance}
    assert with_guidance == {"R003", "R007", "R008", "R015", "R016"}
    blocks = build_prompt(rs.active()).split("\n- rule_id: ")[1:]
    for block in blocks:
        rid = block.split("\n", 1)[0]
        has = "additional guidance from the NGO team (not part of the client wording" in block
        assert has == (rid in with_guidance), rid
        if has:
            assert "taken from the front of the people, not from behind them" in block
            assert "client wording (authoritative, apply exactly)" in block  # the SOP text is still sent verbatim


def test_template_placeholders_in_env_count_as_not_set(monkeypatch):
    from ngo_photo_approval.config import load_project_config

    monkeypatch.setenv("NGO_GCS_BUCKET", "REPLACE_WITH_YOUR_BUCKET_NAME")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "  ")
    cfg = load_project_config()
    assert cfg.cloud.gcs_bucket is None and cfg.cloud.project_id is None  # preflight will therefore flag them
    monkeypatch.setenv("NGO_GCS_BUCKET", "my-real-bucket")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
    cfg = load_project_config()
    assert cfg.cloud.gcs_bucket == "my-real-bucket" and cfg.cloud.project_id == "my-project"
