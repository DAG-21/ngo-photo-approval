"""
Gemini Vision API Client with:
- Environment variable API key loading (`GEMINI_API_KEY`)
- Label-blind payload sanitization (strips original filenames and paths before sending bytes)
- Low-variance generation config (`temperature=0.1`, `response_mime_type="application/json"`)
- Exponential backoff retry logic
- Raw response archiving before parsing (`outputs/raw_gemini_responses/`)
- Pydantic schema validation and token/latency tracking
"""

import io
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar
import yaml
from PIL import Image, ImageOps
from pydantic import BaseModel

from src.env import load_env_file

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class GeminiConfigError(RuntimeError):
    """Missing API key or SDK: retrying cannot fix it, so callers fail immediately."""


class GeminiVisionClient:
    def __init__(
        self,
        config_path: Path,
        raw_output_dir: Path,
        mock_responder: Optional[Callable[[str, Optional[Path]], Dict[str, Any]]] = None,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        load_env_file()  # picks up GEMINI_API_KEY from the project's .env without overriding the real environment
        self.api_key_env = self.config.get("api_key_env_var", "GEMINI_API_KEY")
        self.api_key = os.environ.get(self.api_key_env)
        self.model_name = self.config.get("model_name", "gemini-2.5-flash")
        self.gen_cfg = self.config.get("generation_config", {})
        self.retry_cfg = self.config.get("concurrency_and_retry", {})
        self.raw_output_dir = raw_output_dir
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        self.mock_responder = mock_responder
        self.usage_ledger: List[Dict[str, Any]] = []

    @staticmethod
    def load_sanitized_image_bytes(image_path: Path) -> bytes:
        """
        Reads image pixels and re-encodes in memory without filename or file path metadata
        so that Gemini cannot infer historical approval status from file paths.
        EXIF orientation is applied first because the re-encoded JPEG carries no EXIF, which
        would otherwise leave phone photos rotated sideways.
        """
        with Image.open(image_path) as img:
            rgb = ImageOps.exif_transpose(img).convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=92)
            return buf.getvalue()

    def _call_gemini_sdk(
        self,
        prompt_text: str,
        image_paths: Optional[List[Path]] = None,
    ) -> Tuple[str, Dict[str, int]]:
        """Invokes Google GenAI SDK (`google-genai` or `google-generativeai`)."""
        if self.mock_responder is not None:
            primary_img = image_paths[0] if image_paths else None
            mock_dict = self.mock_responder(prompt_text, primary_img)
            raw_json = json.dumps(mock_dict, indent=2)
            return raw_json, {"prompt_tokens": 850, "completion_tokens": 420, "total_tokens": 1270}

        if not self.api_key:
            raise GeminiConfigError(
                f"Missing API key in environment variable '{self.api_key_env}'. "
                "Set GEMINI_API_KEY in your environment or .env file."
            )

        # Support official google-genai SDK
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            contents: List[Any] = []
            if image_paths:
                for p in image_paths:
                    img_bytes = self.load_sanitized_image_bytes(p)
                    contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
            contents.append(prompt_text)

            cfg_kwargs: Dict[str, Any] = dict(
                temperature=self.gen_cfg.get("temperature", 0.1),
                top_p=self.gen_cfg.get("top_p", 0.95),
                top_k=self.gen_cfg.get("top_k", 40),
                max_output_tokens=self.gen_cfg.get("max_output_tokens", 16384),
                response_mime_type="application/json",
            )
            # Gemini 2.5 "thinking" tokens count against max_output_tokens, so an unbounded thinking
            # budget can starve the JSON answer and truncate it mid-string.
            if self.gen_cfg.get("thinking_budget") is not None:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=self.gen_cfg["thinking_budget"])

            resp = client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            meta = resp.usage_metadata
            finish = resp.candidates[0].finish_reason if resp.candidates else None
            usage = {
                "prompt_tokens": getattr(meta, "prompt_token_count", 0) or 0 if meta else 0,
                "completion_tokens": getattr(meta, "candidates_token_count", 0) or 0 if meta else 0,
                "thinking_tokens": getattr(meta, "thoughts_token_count", 0) or 0 if meta else 0,
                "total_tokens": getattr(meta, "total_token_count", 0) or 0 if meta else 0,
                "finish_reason": getattr(finish, "name", str(finish)),
            }
            return resp.text, usage
        except ImportError as exc:
            raise GeminiConfigError("Install `google-genai` package to call Gemini API in production.") from exc

    def generate_validated_json(
        self,
        record_id: str,
        prompt_text: str,
        prompt_version: str,
        schema_cls: Type[T],
        image_paths: Optional[List[Path]] = None,
    ) -> Tuple[T, Dict[str, Any]]:
        """
        Executes Gemini request with exponential backoff, saves raw response before parsing,
        and validates output against the provided Pydantic schema.
        """
        max_retries = self.retry_cfg.get("max_retries", 4)
        backoff = self.retry_cfg.get("initial_backoff_seconds", 2.0)
        multiplier = self.retry_cfg.get("backoff_multiplier", 2.0)
        max_backoff = self.retry_cfg.get("max_backoff_seconds", 30.0)

        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            start_ts = time.perf_counter()
            try:
                raw_text, token_usage = self._call_gemini_sdk(prompt_text, image_paths=image_paths)
                latency_ms = round((time.perf_counter() - start_ts) * 1000, 2)

                # Always archive raw response BEFORE parsing
                envelope = {
                    "record_id": record_id,
                    "model_version": self.model_name,
                    "prompt_version": prompt_version,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "attempt": attempt,
                    "latency_ms": latency_ms,
                    "token_usage": token_usage,
                    "raw_response": raw_text,
                }
                raw_file = self.raw_output_dir / f"{record_id}_raw.json"
                raw_file.write_text(json.dumps(envelope, indent=2), encoding="utf-8")

                parsed_dict = json.loads(raw_text)
                validated_obj = schema_cls.parse_dict(parsed_dict)

                self.usage_ledger.append({
                    "record_id": record_id,
                    "status": "SUCCESS",
                    "attempt": attempt,
                    "latency_ms": latency_ms,
                    **token_usage,
                })
                return validated_obj, envelope

            except GeminiConfigError:
                raise
            except Exception as err:
                if "API_KEY_INVALID" in str(err) or "API key not valid" in str(err):
                    raise GeminiConfigError(
                        f"Gemini rejected the API key in '{self.api_key_env}' (API_KEY_INVALID)."
                    ) from err
                last_err = err
                logger.warning(
                    "Gemini evaluation attempt %d/%d failed for %s: %s",
                    attempt,
                    max_retries,
                    record_id,
                    err,
                )
                if attempt < max_retries:
                    time.sleep(min(backoff, max_backoff))
                    backoff *= multiplier

        self.usage_ledger.append({
            "record_id": record_id,
            "status": "FAILED",
            "error": str(last_err),
        })
        raise RuntimeError(f"Failed to annotate/evaluate {record_id} after {max_retries} attempts: {last_err}")
