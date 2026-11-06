from __future__ import annotations

import pytest

from judge_trial.backend import GeminiJudgeBackend
from judge_trial.prompt import build_prompt

from conftest import good_answer, make_image


class FakeResponse:
    text = good_answer()
    usage_metadata = type("U", (), {"prompt_token_count": 5000, "cached_content_token_count": 4800,
                                    "candidates_token_count": 200, "thoughts_token_count": 0, "total_token_count": 5200})()


class FakeClient:
    def __init__(self, cache_fails=False):
        self.created, self.deleted, self.generated = [], [], []
        self.cache_fails = cache_fails
        outer = self

        class Caches:
            def create(self, model, config):
                if outer.cache_fails:
                    raise RuntimeError("cache too small")
                outer.created.append((model, config))
                return type("C", (), {"name": "cachedContents/abc"})()

            def delete(self, name):
                outer.deleted.append(name)

        class Models:
            def generate_content(self, model, contents, config):
                outer.generated.append((model, contents, config))
                return FakeResponse()

        self.caches, self.models = Caches(), Models()


@pytest.fixture
def authorized(monkeypatch):
    monkeypatch.setattr("ngo_photo_approval.models.cloud_client.require_inference_authorization", lambda cloud: None)


def examples(cfg, n=8):
    ex = []
    for i in range(n):
        make_image(cfg, f"ex_{i}")
        ex.append({"image_id": f"ex_{i}", "label": "approved" if i % 2 else "not_approved", "reason": None})
    return ex


def test_without_authorization_no_backend_exists(cfg, rule_set, monkeypatch):
    monkeypatch.setenv("NGO_ALLOW_CLOUD_INFERENCE", "false")
    with pytest.raises(Exception, match="not authorized"):
        GeminiJudgeBackend(cfg, client=FakeClient())


def test_examples_are_cached_once_and_photos_use_the_cache(cfg, rule_set, authorized):
    client = FakeClient()
    p = build_prompt(cfg, "D", rule_set, examples(cfg))
    b = GeminiJudgeBackend(cfg, client=client)
    b.prepare(p)
    assert b.cache_active and len(client.created) == 1
    cached = client.created[0][1]
    assert cached.system_instruction == p.system_text and len(cached.contents) == 2  # examples turn + acknowledgement
    res = b.judge(b"\xff\xd8fake", "image/jpeg")
    _, contents, config = client.generated[0]
    assert config.cached_content == "cachedContents/abc" and config.system_instruction is None
    assert len(contents) == 1  # only the photo travels with the request
    assert res.usage["cached_tokens"] == 4800 and res.usage["prompt_tokens"] == 5000
    b.close()
    assert client.deleted == ["cachedContents/abc"] and not b.cache_active
    b.close()
    assert client.deleted == ["cachedContents/abc"]  # closing twice deletes once


def test_auto_mode_falls_back_to_sending_everything_inline(cfg, rule_set, authorized):
    client = FakeClient(cache_fails=True)
    p = build_prompt(cfg, "D", rule_set, examples(cfg))
    b = GeminiJudgeBackend(cfg, client=client)
    b.prepare(p)
    assert not b.cache_active
    b.judge(b"\xff\xd8fake", "image/jpeg")
    _, contents, config = client.generated[0]
    assert config.system_instruction == p.system_text and config.cached_content is None
    assert len(contents) == 3  # example turn + acknowledgement + the photo


def test_cache_mode_on_does_not_hide_a_failure(cfg, rule_set, authorized):
    cfg.cache.mode = "on"
    b = GeminiJudgeBackend(cfg, client=FakeClient(cache_fails=True))
    with pytest.raises(RuntimeError, match="too small"):
        b.prepare(build_prompt(cfg, "D", rule_set, examples(cfg)))


def test_short_prompts_and_mode_off_skip_caching(cfg, rule_set, authorized):
    client = FakeClient()
    b = GeminiJudgeBackend(cfg, client=client)
    b.prepare(build_prompt(cfg, "A", rule_set, []))
    assert not b.cache_active and not client.created
    cfg.cache.mode = "off"
    b2 = GeminiJudgeBackend(cfg, client=client)
    b2.prepare(build_prompt(cfg, "D", rule_set, examples(cfg)))
    assert not b2.cache_active and not client.created
