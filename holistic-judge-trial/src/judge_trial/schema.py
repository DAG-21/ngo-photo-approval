"""The structured answer the judge must return, and a strict parser for it."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


class ResponseError(ValueError):
    """The model's answer is not valid against the schema (counted as a failed call, never guessed around)."""


def response_schema(rule_ids: list[str], summary_reason: bool = False) -> dict[str, Any]:
    """``summary_reason`` adds one overall plain-language reason (prompt v2 onward), placed before the score."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "observations": {"type": "string"},
            "reasons": {
                "type": "array",
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "rule_id": {"type": "string", "enum": [*rule_ids, "OTHER"]},
                        "effect": {"type": "string", "enum": ["supports", "against"]},
                        "note": {"type": "string"},
                    },
                    "required": ["rule_id", "effect", "note"],
                    "propertyOrdering": ["rule_id", "effect", "note"],
                },
            },
            "score": {"type": "integer", "minimum": 0, "maximum": 100},
            "decision": {"type": "string", "enum": ["APPROVED", "DISAPPROVED"]},
            "cannot_judge": {"type": "boolean"},
        },
        "required": ["observations", "reasons", "score", "decision", "cannot_judge"],
        "propertyOrdering": ["observations", "reasons", "score", "decision", "cannot_judge"],
    }
    if summary_reason:
        schema["properties"]["reason"] = {"type": "string"}
        schema["required"].insert(2, "reason")
        schema["propertyOrdering"].insert(2, "reason")
    return schema


@dataclass
class JudgeAnswer:
    observations: str
    reasons: list[dict[str, str]] = field(default_factory=list)
    score: int = 0
    decision: str = "DISAPPROVED"
    cannot_judge: bool = False
    reason: str | None = None  # overall explanation; only prompts with summary_reason ask for it


def parse_answer(text: str, rule_ids: list[str], summary_reason: bool = False) -> JudgeAnswer:
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ResponseError(f"not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ResponseError("answer is not a JSON object")
    required = ("observations", "reasons", "score", "decision", "cannot_judge") + (("reason",) if summary_reason else ())
    missing = [k for k in required if k not in raw]
    if missing:
        raise ResponseError(f"missing fields: {missing}")
    score = raw["score"]
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 100:
        raise ResponseError(f"score must be an integer 0-100, got {score!r}")
    if raw["decision"] not in ("APPROVED", "DISAPPROVED"):
        raise ResponseError(f"bad decision {raw['decision']!r}")
    if not isinstance(raw["cannot_judge"], bool):
        raise ResponseError("cannot_judge must be a boolean")
    reasons = raw["reasons"]
    if not isinstance(reasons, list):
        raise ResponseError("reasons must be a list")
    allowed = {*rule_ids, "OTHER"}
    clean: list[dict[str, str]] = []
    for item in reasons:
        if not isinstance(item, dict) or item.get("rule_id") not in allowed or item.get("effect") not in ("supports", "against"):
            raise ResponseError(f"bad reason item: {item!r}")
        clean.append({"rule_id": item["rule_id"], "effect": item["effect"], "note": str(item.get("note", ""))})
    reason = None
    if summary_reason:
        if not isinstance(raw["reason"], str) or not raw["reason"].strip():
            raise ResponseError("reason must be a non-empty string")
        reason = raw["reason"].strip()
    return JudgeAnswer(str(raw["observations"]), clean, score, raw["decision"], raw["cannot_judge"], reason)
