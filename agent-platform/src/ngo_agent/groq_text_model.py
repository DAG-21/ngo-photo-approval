"""A text-only model on Groq, for the model-based gap analyzer (`llm_gap_analyzer.TextModel`). It uses Groq's OpenAI-style chat API over HTTPS and nothing else.

* Its own authorisation: it refuses to start unless `NGO_ALLOW_GROQ_INFERENCE=true` and `GROQ_API_KEY` are both set. Groq is a different company from Google, so the NGO's
  approval for Gemini does not cover it, and the Gemini switch (`NGO_ALLOW_CLOUD_INFERENCE`) is NOT reused. The key is read into memory from the environment mapping you
  pass (or from `.env` by the caller) and is never printed, logged or put in an error message.
* Only the gap analyzer's text goes through it: the AI's findings and the reviewer's written reasons, contact details already removed, never a photograph.
* JSON answers: models listed in `STRICT_MODELS` are asked with a strict JSON schema (constrained decoding). If Groq refuses that schema (HTTP 400) the request is repeated
  once without it, in plain JSON mode with the schema written into the instruction, and that is remembered. Any other model goes straight to plain JSON mode. Either way the
  analyzer still checks the answer itself. If the answer is not a JSON object (or was cut off) it is asked ONCE more; after that the unusable text is returned and the
  analysis is stored as failed.
* Transient errors (rate limit 429, server errors, network failures) are retried with backoff, honouring `Retry-After`. 401 and 403 are not retried.
* Cost is recorded as zero unless prices are given, because the free tier costs nothing. If the account is upgraded, pass the real per-million prices.

Not verified against the live service: nothing here has been sent to Groq (the tests use a fake transport), so the default model id and the strict-schema behaviour are
as the public documentation describes them and have not been confirmed with an API key.
"""

from __future__ import annotations

import copy
import json
import random
import time
from typing import Any, Callable, Mapping, Optional

import httpx
from layered_engine.providers import AuthorizationError, ProviderError, scrub

from .llm_gap_analyzer import TextAnswer

URL = "https://api.groq.com/openai/v1/chat/completions"
AUTH_ENV = "NGO_ALLOW_GROQ_INFERENCE"
KEY_ENV = "GROQ_API_KEY"
MODEL_ENV = "GROQ_MODEL"
ENV_NAMES = (AUTH_ENV, KEY_ENV, MODEL_ENV)           # the only names read from a .env file
DEFAULT_MODEL = "openai/gpt-oss-120b"
STRICT_MODELS = frozenset({"openai/gpt-oss-20b", "openai/gpt-oss-120b"})   # per Groq's documentation, the models with constrained (strict) JSON-schema output
TRANSIENT = frozenset({429, 500, 502, 503, 504})
ANSWER_TOKENS = 4096
MAX_WAIT_S = 30.0


class Rejected(ProviderError):
    """Groq answered with an HTTP error that retrying will not fix; `status` is the code."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status


def strictify(schema: dict[str, Any]) -> dict[str, Any]:
    """The schema in the form strict mode accepts: every object closed (`additionalProperties: false`) with all its properties required, and the length keywords
    (`maxItems`, `minItems`) removed. The analyzer enforces the caps itself, so nothing is lost."""
    s = copy.deepcopy(schema)

    def walk(n: Any) -> None:
        if isinstance(n, dict):
            n.pop("maxItems", None)
            n.pop("minItems", None)
            if n.get("type") == "object" and isinstance(n.get("properties"), dict):
                n["additionalProperties"] = False
                n["required"] = list(n["properties"])
            for v in list(n.values()):
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(s)
    return s


def _is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


class GroqTextModel:
    def __init__(self, env: Mapping[str, str], client: Optional[httpx.Client] = None, sleep: Callable[[float], None] = time.sleep, model: Optional[str] = None,
                 input_per_million_usd: float = 0.0, output_per_million_usd: float = 0.0, max_attempts: int = 3, timeout_s: float = 60.0) -> None:
        if str(env.get(AUTH_ENV, "")).lower() != "true":
            raise AuthorizationError(f"Groq inference is not authorized: set {AUTH_ENV}=true once the NGO has authorised sending reviewers' written reasons to Groq")
        self._key = env.get(KEY_ENV, "")
        if not self._key:
            raise AuthorizationError(f"{KEY_ENV} is not set (put it in the environment or .env, never in code)")
        self.model_id = model or env.get(MODEL_ENV) or DEFAULT_MODEL
        self._client = client or httpx.Client(timeout=timeout_s)
        self._sleep, self._prices, self.max_attempts = sleep, (input_per_million_usd, output_per_million_usd), max_attempts
        self._strict = self.model_id in STRICT_MODELS

    # ----------------------------------------------------------------------------------------------------------------------------- one request
    def _payload(self, system_text: str, user_text: str, schema: dict[str, Any], strict: bool) -> dict[str, Any]:
        if strict:
            response_format: dict[str, Any] = {"type": "json_schema", "json_schema": {"name": "gap_analysis", "strict": True, "schema": strictify(schema)}}
            system = system_text
        else:
            response_format = {"type": "json_object"}
            system = system_text + "\n\nAnswer with one JSON object and nothing else, matching this JSON schema:\n" + json.dumps(schema)
        return {"model": self.model_id, "temperature": 0.1, "max_completion_tokens": ANSWER_TOKENS, "response_format": response_format,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_text}]}

    def _post(self, payload: dict[str, Any], usage: dict[str, Any]) -> httpx.Response:
        delay = 2.0
        for attempt in range(1, self.max_attempts + 1):
            usage["requests"] += 1
            try:
                r = self._client.post(URL, json=payload, headers={"Authorization": f"Bearer {self._key}"})
            except httpx.HTTPError as exc:
                if attempt == self.max_attempts:
                    raise ProviderError(f"{type(exc).__name__} after {attempt} attempts") from None
                self._sleep(min(delay, MAX_WAIT_S) * (0.75 + random.random() / 2))
                delay *= 2
                continue
            if r.status_code == 200:
                return r
            if r.status_code in (401, 403):
                raise ProviderError(f"authentication or permission error (HTTP {r.status_code})")
            if r.status_code in TRANSIENT and attempt < self.max_attempts:
                try:
                    wait = float(r.headers.get("retry-after", ""))
                except ValueError:
                    wait = delay
                self._sleep(min(wait, MAX_WAIT_S) * (0.75 + random.random() / 2))
                delay *= 2
                continue
            raise Rejected(r.status_code, scrub(r.text.replace("\n", " ")[:200], self._key))
        raise ProviderError("gave up")   # not reached: the loop returns or raises

    def _ask(self, system_text: str, user_text: str, schema: dict[str, Any], usage: dict[str, Any]) -> tuple[str, str, str]:
        try:
            r = self._post(self._payload(system_text, user_text, schema, self._strict), usage)
        except Rejected as exc:
            if not (self._strict and exc.status == 400):
                raise
            self._strict = False                                                    # Groq refused the strict schema: ask again in plain JSON mode, and remember
            r = self._post(self._payload(system_text, user_text, schema, False), usage)
        try:
            data = r.json()
            choice = (data.get("choices") or [{}])[0]
            u = data.get("usage") or {}
            text = (choice.get("message") or {}).get("content") or ""
        except (ValueError, AttributeError, IndexError, TypeError):
            raise ProviderError("the response was not in the expected shape") from None
        usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        usage["output_tokens"] += int(u.get("completion_tokens") or 0)
        return str(text), str(choice.get("finish_reason") or ""), str(data.get("model") or self.model_id)

    # ---------------------------------------------------------------------------------------------------------------------------- the interface
    def complete(self, system_text: str, user_text: str, schema: dict[str, Any]) -> TextAnswer:
        usage: dict[str, Any] = {"prompt_tokens": 0, "output_tokens": 0, "requests": 0, "attempts": 0}
        text, model = "", self.model_id
        for attempt in (1, 2):                                                     # one more ask when the answer is not a JSON object or was cut off
            usage["attempts"] = attempt
            text, finish, model = self._ask(system_text, user_text, schema, usage)
            if finish != "length" and _is_json_object(text):
                break
        cost = round(usage["prompt_tokens"] / 1e6 * self._prices[0] + usage["output_tokens"] / 1e6 * self._prices[1], 6)
        return TextAnswer(text=text, usage=usage, model_id=model, cost_usd=cost)
