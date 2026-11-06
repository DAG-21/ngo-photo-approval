"""Vision providers behind one small interface, so the rule engine never depends on Gemini.

GeminiProvider and AnthropicProvider (Claude Haiku 5.5) are the implementations. Each fails closed: no client is created unless its own authorization switch and its own
API key are both present (they are different companies, so each has its own switch). The key is read from the environment, never logged, and scrubbed from error messages.
The photograph is sent as the original bytes (not resized or recompressed). `make_provider(cfg)` picks the one the configuration names (`provider: google` or `anthropic`).
"""

from __future__ import annotations

import base64
import copy
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import yaml

from .rulebook import CONFIG_DIR

TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}


class ProviderError(RuntimeError):
    """A failed call. Never converted into a successful evaluation."""


class AuthorizationError(ProviderError):
    pass


@dataclass
class ProviderResult:
    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    model_id: str = ""


class VisionProvider(Protocol):
    model_id: str

    def assess(self, image_bytes: bytes, mime_type: str, system_text: str, user_text: str, schema: dict[str, Any], examples: Any = None) -> ProviderResult: ...


def load_gemini_config(path: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((path or CONFIG_DIR / "gemini_config.yaml").read_text(encoding="utf-8"))


VISION_CONFIGS = {"gemini": "gemini_config.yaml", "haiku": "anthropic_config.yaml"}


def load_vision_config(which: Optional[str] = None, env: Optional[dict[str, str]] = None) -> dict[str, Any]:
    """The configuration of the vision model in use: `which` ("gemini" or "haiku"), else the NGO_VISION environment variable, else gemini (the default, as before)."""
    env = os.environ if env is None else env
    name = str(which or env.get("NGO_VISION") or "gemini").lower()
    if name not in VISION_CONFIGS:
        raise ValueError(f"unknown vision model {name!r}; choose from {sorted(VISION_CONFIGS)}")
    return load_gemini_config(CONFIG_DIR / VISION_CONFIGS[name])


def read_env_file(path: Path, wanted: tuple[str, ...]) -> dict[str, str]:
    """Reads only the named keys from a .env file (never prints them)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() in wanted:
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def scrub(text: str, secret: str) -> str:
    return text.replace(secret, "<key>") if secret else text


class GeminiProvider:
    def __init__(self, cfg: dict[str, Any], client: Any = None, sleep: Callable[[float], None] = time.sleep, env: Optional[dict[str, str]] = None) -> None:
        env = os.environ if env is None else env
        auth = cfg["auth"]
        if str(env.get(auth["authorization_env"], "")).lower() != "true":
            raise AuthorizationError(f"cloud inference is not authorized: set {auth['authorization_env']}=true once the NGO's data-transfer authorization is in place")
        self._key = env.get(auth["api_key_env"], "")
        if not self._key and client is None:
            raise AuthorizationError(f"{auth['api_key_env']} is not set (put it in the environment or .env, never in code)")
        self.cfg = cfg
        self.model_id: str = cfg["model_id"]
        self._client = client
        self._sleep = sleep
        self._client_lock = threading.Lock()  # concurrent first calls must not each build (and then discard and close) a client

    def _get_client(self) -> Any:
        with self._client_lock:
            if self._client is None:
                from google import genai

                self._client = genai.Client(api_key=self._key)
            return self._client

    def assess(self, image_bytes: bytes, mime_type: str, system_text: str, user_text: str, schema: dict[str, Any], examples: Any = None) -> ProviderResult:
        """`examples` (optional): similar past staff decisions, each {image_bytes, mime_type, decision, tags, reasons}; see few_shot.py. None or empty: the
        request is exactly what it has always been."""
        from google.genai import types

        from .few_shot import build_parts
        g = self.cfg["generation"]
        config = types.GenerateContentConfig(
            system_instruction=system_text, temperature=g["temperature"], max_output_tokens=g["max_output_tokens"],
            response_mime_type=g["response_mime_type"], response_json_schema=schema,
            media_resolution=getattr(types.MediaResolution, g["media_resolution"]),
            http_options=types.HttpOptions(timeout=int(g["request_timeout_s"] * 1000)), **self._thinking(types))
        parts = [types.Part.from_bytes(data=p[0], mime_type=p[1]) if kind == "image" else types.Part(text=p) for kind, p in build_parts(examples, image_bytes, mime_type, user_text)]
        contents = [types.Content(role="user", parts=parts)]
        return self._generate(contents, config)

    def _thinking(self, types: Any) -> dict[str, Any]:
        """generation.thinking: {level: MINIMAL|LOW|MEDIUM|HIGH} (Gemini 3.5 and later) or {budget: tokens, 0 = off} (2.5 models). Absent: the model's own default, as before.
        Thinking is billed as output, so a lower setting is cheaper; whether it costs accuracy has to be measured."""
        th = self.cfg["generation"].get("thinking") or {}
        if th.get("level"):
            return {"thinking_config": types.ThinkingConfig(thinking_level=getattr(types.ThinkingLevel, str(th["level"]).upper()))}
        if th.get("budget") is not None:
            return {"thinking_config": types.ThinkingConfig(thinking_budget=int(th["budget"]))}
        return {}

    def complete_text(self, system_text: str, user_text: str, schema: dict[str, Any], max_output_tokens: Optional[int] = None) -> ProviderResult:
        """A text-only request with a JSON answer (no photograph): the same authorisation, retries and key scrubbing as `assess`. Used for analysing written
        reasons, never for judging a photograph. `max_output_tokens` overrides the configured limit, which is sized for a photograph's long answer."""
        from google.genai import types

        g = self.cfg["generation"]
        config = types.GenerateContentConfig(
            system_instruction=system_text, temperature=g["temperature"], max_output_tokens=max_output_tokens or g["max_output_tokens"],
            response_mime_type=g["response_mime_type"], response_json_schema=schema, http_options=types.HttpOptions(timeout=int(g["request_timeout_s"] * 1000)))
        return self._generate([types.Content(role="user", parts=[types.Part(text=user_text)])], config)

    def _generate(self, contents: Any, config: Any) -> ProviderResult:
        attempts = int(self.cfg["processing"]["retry_attempts"])
        delay = float(self.cfg["processing"]["initial_backoff_s"])
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            started = time.monotonic()
            try:
                resp = self._get_client().models.generate_content(model=self.model_id, contents=contents, config=config)
                um = getattr(resp, "usage_metadata", None)
                usage = {"prompt_tokens": getattr(um, "prompt_token_count", None), "output_tokens": getattr(um, "candidates_token_count", None),
                         "thinking_tokens": getattr(um, "thoughts_token_count", None), "total_tokens": getattr(um, "total_token_count", None),
                         "cached_tokens": getattr(um, "cached_content_token_count", None),
                         "latency_s": round(time.monotonic() - started, 3), "attempts": attempt}
                return ProviderResult(text=resp.text if getattr(resp, "text", None) is not None else "", usage=usage, model_id=self.model_id)
            except Exception as exc:  # noqa: BLE001
                code = getattr(exc, "code", None)
                last = exc
                if code in (401, 403):
                    raise ProviderError(f"authentication or permission error (HTTP {code})") from None
                if code not in TRANSIENT_CODES or attempt == attempts:
                    detail = scrub(str(exc).replace("\n", " ")[:200], self._key)
                    raise ProviderError(f"{type(exc).__name__} (HTTP {code}): {detail}") from None
                self._sleep(min(delay, float(self.cfg["processing"]["max_backoff_s"])) * (0.75 + random.random() / 2))
                delay *= 2
        raise ProviderError(f"gave up after {attempts} attempts: {type(last).__name__}")


# ------------------------------------------------------------------------------------------------------------------------------ Claude (Anthropic)
# Structured outputs accept a subset of JSON Schema: every object must say additionalProperties: false, and numeric / string / array-size constraints are not allowed.
_ANTHROPIC_UNSUPPORTED = frozenset({"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength", "minItems", "maxItems", "uniqueItems", "default", "title"})


def anthropic_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """The answer schema of the engine, made acceptable to Claude's structured outputs: additionalProperties false on every object, unsupported constraints dropped. A copy: the
    engine's own schema (and so the Gemini request) is never changed. Property NAMES are never mistaken for keywords."""
    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(x) for x in node]
        if not isinstance(node, dict):
            return node
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k in _ANTHROPIC_UNSUPPORTED:
                continue
            out[k] = {name: walk(sub) for name, sub in v.items()} if k == "properties" and isinstance(v, dict) else walk(v)
        if out.get("type") == "object" or "properties" in out:
            out["additionalProperties"] = False
        return out

    return walk(copy.deepcopy(schema))


class AnthropicProvider:
    """Claude Haiku 5.5 behind the same small interface as GeminiProvider. Its own switch (NGO_ALLOW_ANTHROPIC_INFERENCE) and key (ANTHROPIC_API_KEY), because the photographs
    then go to Anthropic. The long, identical instructions are marked for prompt caching (cheaper from the second photograph if it comes within the cache lifetime); thinking is
    switched off by configuration (it is billed as output); no sampling parameter is sent (Haiku 5.5 rejects them)."""

    def __init__(self, cfg: dict[str, Any], client: Any = None, sleep: Callable[[float], None] = time.sleep, env: Optional[dict[str, str]] = None) -> None:
        env = os.environ if env is None else env
        auth = cfg["auth"]
        if str(env.get(auth["authorization_env"], "")).lower() != "true":
            raise AuthorizationError(f"cloud inference to Anthropic is not authorized: set {auth['authorization_env']}=true once the NGO's data-transfer authorization covers Anthropic")
        self._key = env.get(auth["api_key_env"], "")
        if not self._key and client is None:
            raise AuthorizationError(f"{auth['api_key_env']} is not set (put it in the environment or .env, never in code)")
        self.cfg = cfg
        self.model_id: str = cfg["model_id"]
        self._client = client
        self._sleep = sleep
        self._client_lock = threading.Lock()

    def _get_client(self) -> Any:
        with self._client_lock:
            if self._client is None:
                import anthropic   # the official SDK; installed on demand (pip install anthropic), not needed for the other provider

                # our own retry loop counts attempts and backs off, so the SDK's automatic retries are off
                self._client = anthropic.Anthropic(api_key=self._key, timeout=float(self.cfg["generation"]["request_timeout_s"]), max_retries=0)
            return self._client

    @staticmethod
    def _block(kind: str, payload: Any) -> dict[str, Any]:
        if kind == "image":
            return {"type": "image", "source": {"type": "base64", "media_type": payload[1], "data": base64.standard_b64encode(payload[0]).decode("ascii")}}
        return {"type": "text", "text": payload}

    def assess(self, image_bytes: bytes, mime_type: str, system_text: str, user_text: str, schema: dict[str, Any], examples: Any = None) -> ProviderResult:
        """`examples` (optional): similar past staff decisions, exactly as for Gemini (few_shot.build_parts); None or empty: the photograph, then the user text."""
        from .few_shot import build_parts

        content = [self._block(kind, p) for kind, p in build_parts(examples, image_bytes, mime_type, user_text)]
        return self._create(system_text, content, schema, int(self.cfg["generation"]["max_output_tokens"]))

    def complete_text(self, system_text: str, user_text: str, schema: dict[str, Any], max_output_tokens: Optional[int] = None) -> ProviderResult:
        """A text-only request with a JSON answer (no photograph): the same authorisation, retries and key scrubbing as `assess`."""
        return self._create(system_text, [{"type": "text", "text": user_text}], schema, int(max_output_tokens or self.cfg["generation"]["max_output_tokens"]))

    def _request(self, system_text: str, content: list[dict[str, Any]], schema: dict[str, Any], max_tokens: int) -> dict[str, Any]:
        g, cache = self.cfg["generation"], self.cfg.get("cache") or {}
        system: Any = system_text
        if cache.get("enabled"):
            marker: dict[str, Any] = {"type": "ephemeral"}
            if str(cache.get("ttl", "5m")) == "1h":
                marker["ttl"] = "1h"
            system = [{"type": "text", "text": system_text, "cache_control": marker}]   # the rules and the answer instructions: identical for every photograph of a prompt version
        out_cfg: dict[str, Any] = {"format": {"type": "json_schema", "schema": anthropic_schema(schema)}}
        if g.get("effort"):
            out_cfg["effort"] = g["effort"]
        kwargs: dict[str, Any] = {"model": self.model_id, "max_tokens": max_tokens, "system": system, "messages": [{"role": "user", "content": content}], "output_config": out_cfg}
        if g.get("thinking"):
            kwargs["thinking"] = {"type": str(g["thinking"])}   # "disabled" (the default here: thinking is billed as output) or "adaptive"
        return kwargs

    def _create(self, system_text: str, content: list[dict[str, Any]], schema: dict[str, Any], max_tokens: int) -> ProviderResult:
        kwargs = self._request(system_text, content, schema, max_tokens)
        attempts = int(self.cfg["processing"]["retry_attempts"])
        delay = float(self.cfg["processing"]["initial_backoff_s"])
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            started = time.monotonic()
            try:
                resp = self._get_client().messages.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                code = getattr(exc, "status_code", None)
                last = exc
                if code in (401, 403):
                    raise ProviderError(f"authentication or permission error (HTTP {code})") from None
                transient = code in (TRANSIENT_CODES | {529}) or type(exc).__name__ in ("APIConnectionError", "APITimeoutError")
                if not transient or attempt == attempts:
                    detail = scrub(str(exc).replace("\n", " ")[:200], self._key)
                    raise ProviderError(f"{type(exc).__name__} (HTTP {code}): {detail}") from None
                self._sleep(min(delay, float(self.cfg["processing"]["max_backoff_s"])) * (0.75 + random.random() / 2))
                delay *= 2
                continue
            stop = getattr(resp, "stop_reason", None)
            if stop == "refusal":
                raise ProviderError("the model declined to answer (stop_reason refusal)")
            if stop == "max_tokens":
                raise ProviderError("the answer was cut off: it needed more than max_output_tokens")
            text = "".join(b.text for b in (getattr(resp, "content", None) or []) if getattr(b, "type", None) == "text")
            u = getattr(resp, "usage", None)
            fresh, read, wrote = (int(getattr(u, k, 0) or 0) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
            out = int(getattr(u, "output_tokens", 0) or 0)
            usage = {"prompt_tokens": fresh + read + wrote, "output_tokens": out, "thinking_tokens": None, "total_tokens": fresh + read + wrote + out,   # thinking, if any, is inside output_tokens
                     "cached_tokens": read, "cache_write_tokens": wrote, "latency_s": round(time.monotonic() - started, 3), "attempts": attempt}
            return ProviderResult(text=text, usage=usage, model_id=self.model_id)
        raise ProviderError(f"gave up after {attempts} attempts: {type(last).__name__}")


def make_provider(cfg: dict[str, Any], client: Any = None, env: Optional[dict[str, str]] = None, sleep: Callable[[float], None] = time.sleep) -> Any:
    """The provider the configuration names: `provider: google` (Gemini) or `provider: anthropic` (Claude). Fails closed exactly like each provider does."""
    kind = str(cfg.get("provider", "google")).lower()
    if kind == "anthropic":
        return AnthropicProvider(cfg, client=client, env=env, sleep=sleep)
    if kind == "google":
        return GeminiProvider(cfg, client=client, env=env, sleep=sleep)
    raise ValueError(f"unknown provider {kind!r} (expected google or anthropic)")
