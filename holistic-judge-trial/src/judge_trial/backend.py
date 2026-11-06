"""Gemini access for the trial, with context caching.

* Nothing is sent anywhere unless ``--execute`` is passed to the run script AND the main project's own authorization
  (``NGO_ALLOW_CLOUD_INFERENCE=true`` plus credentials) is in place; construction fails closed otherwise.
* The fixed part of the prompt (instructions + worked examples) is sent once into a cache when that is allowed and
  worthwhile; each photo is then judged against the cache by name. If caching is not possible (``auto`` mode) the same
  content is sent inline with every photo instead.
* Image bytes, prompts and keys are never logged.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .backend_cost import estimate_prefix_tokens
from .config import TrialConfig, ensure_main_importable
from .prompt import ACK_TEXT, BuiltPrompt

logger = logging.getLogger(__name__)


@dataclass
class BackendResult:
    text: str
    usage: dict[str, Any] = field(default_factory=dict)


class JudgeBackend(Protocol):
    cache_active: bool

    def prepare(self, prompt: BuiltPrompt) -> None: ...
    def judge(self, image_bytes: bytes, mime_type: str) -> BackendResult: ...
    def close(self) -> None: ...


class GeminiJudgeBackend:
    """google-genai implementation (Vertex project/ADC, or a Gemini API key), with optional explicit caching."""

    def __init__(self, cfg: TrialConfig, client: Any | None = None, sleep=time.sleep) -> None:
        ensure_main_importable(cfg)
        from ngo_photo_approval.config import load_model_config, load_project_config
        from ngo_photo_approval.models.cloud_client import RateLimiter, require_inference_authorization

        self._cfg = cfg
        self._project = load_project_config()
        require_inference_authorization(self._project.cloud)  # fails closed: no authorization, no client
        self._retry = load_model_config().retry
        self._limiter = RateLimiter(self._retry.rate_limit_rps)
        self._client = client
        self._sleep = sleep
        self._prompt: BuiltPrompt | None = None
        self._cache_name: str | None = None
        self.cache_active = False
        self._prefix_contents: list[Any] = []

    # -- client ---------------------------------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai

            cloud = self._project.cloud
            if cloud.auth_mode == "api_key":
                self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])  # never logged
            else:
                self._client = genai.Client(vertexai=True, project=cloud.project_id, location=cloud.location)
        return self._client

    # -- fixed prefix ---------------------------------------------------------------------------------------

    def _build_prefix(self, prompt: BuiltPrompt) -> list[Any]:
        from google.genai import types

        if not prompt.examples:
            return []
        parts: list[Any] = [types.Part(text=prompt.examples_intro)]
        for n, e in enumerate(prompt.examples, 1):
            parts += [types.Part(text=f"Example {n}:"),
                      types.Part.from_bytes(data=e.path.read_bytes(), mime_type="image/jpeg"),
                      types.Part(text=e.decision_text)]
        return [types.Content(role="user", parts=parts), types.Content(role="model", parts=[types.Part(text=ACK_TEXT)])]

    def prepare(self, prompt: BuiltPrompt) -> None:
        from google.genai import types

        self._prompt = prompt
        self._prefix_contents = self._build_prefix(prompt)
        mode = self._cfg.cache.mode
        worthwhile = estimate_prefix_tokens(prompt, self._cfg.cost.tokens_per_image) >= self._cfg.cache.min_prefix_tokens
        if mode == "off" or (mode == "auto" and not worthwhile):
            logger.info("context caching not used (mode=%s, worthwhile=%s)", mode, worthwhile)
            return
        try:
            cache = self._get_client().caches.create(
                model=self._cfg.model.id,
                config=types.CreateCachedContentConfig(
                    display_name=f"judge-trial-{prompt.variant}-{prompt.hash}",
                    system_instruction=prompt.system_text,
                    contents=self._prefix_contents or None,
                    ttl=f"{self._cfg.cache.ttl_seconds}s",
                ),
            )
        except Exception as exc:  # noqa: BLE001
            if mode == "on":
                raise
            logger.warning("could not create the context cache (%s); sending the prompt with every photo instead",
                           type(exc).__name__)
            return
        self._cache_name = cache.name
        self.cache_active = True

    # -- per photo ------------------------------------------------------------------------------------------

    def judge(self, image_bytes: bytes, mime_type: str) -> BackendResult:
        from google.genai import types
        from ngo_photo_approval.models.cloud_client import call_with_retry, map_exception

        assert self._prompt is not None, "prepare() must be called first"
        g = self._cfg.generation
        turn = types.Content(role="user", parts=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                                                 types.Part(text=self._prompt.user_text)])
        kwargs: dict[str, Any] = dict(
            temperature=g.temperature, max_output_tokens=g.max_output_tokens,
            response_mime_type="application/json", response_json_schema=self._prompt.schema,
            media_resolution=getattr(types.MediaResolution, g.media_resolution),
            http_options=types.HttpOptions(timeout=int(g.request_timeout_s * 1000)),
        )
        if self.cache_active:
            kwargs["cached_content"] = self._cache_name  # the instructions live in the cache
            contents = [turn]
        else:
            kwargs["system_instruction"] = self._prompt.system_text
            contents = [*self._prefix_contents, turn]
        config = types.GenerateContentConfig(**kwargs)

        def once():
            self._limiter.wait()
            started = time.monotonic()
            try:
                resp = self._get_client().models.generate_content(model=self._cfg.model.id, contents=contents, config=config)
            except Exception as exc:  # noqa: BLE001
                raise map_exception(exc) from exc
            return resp, time.monotonic() - started

        resp, latency = call_with_retry(once, self._retry, sleep=self._sleep)
        um = getattr(resp, "usage_metadata", None)
        usage = {
            "prompt_tokens": getattr(um, "prompt_token_count", None),
            "cached_tokens": getattr(um, "cached_content_token_count", None),
            "output_tokens": getattr(um, "candidates_token_count", None),
            "thinking_tokens": getattr(um, "thoughts_token_count", None),
            "total_tokens": getattr(um, "total_token_count", None),
            "latency_s": round(latency, 3),
        }
        text = resp.text if getattr(resp, "text", None) is not None else ""
        return BackendResult(text=text, usage=usage)

    def close(self) -> None:
        if self._cache_name:
            name, self._cache_name, self.cache_active = self._cache_name, None, False
            try:
                self._get_client().caches.delete(name=name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not delete cache %s (%s); it expires on its own after the TTL", name, type(exc).__name__)
