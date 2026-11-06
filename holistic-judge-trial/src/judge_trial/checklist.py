"""Checklist mode: itemised yes/no questions per rule instead of one holistic judgement.

The model answers every question with YES / NO / CANNOT_TELL / NOT_APPLICABLE (YES = the requirement is met) and never
gives a score or decision. Everything graded or combined (a rule's partial result, the overall prediction) is computed
here, in code, from those answers, so it can be inspected and re-weighted without calling the model again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from .config import PROMPTS_DIR
from .schema import ResponseError

ANSWERS = ("YES", "NO", "CANNOT_TELL", "NOT_APPLICABLE")


def load_checks(version: str) -> dict[str, Any]:
    spec = yaml.safe_load((PROMPTS_DIR / version / "checks.yaml").read_text(encoding="utf-8"))
    ids = [c["id"] for c in spec["checks"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate check ids in checks.yaml")
    return spec


def check_ids(spec: dict[str, Any]) -> list[str]:
    return [c["id"] for c in spec["checks"]]


def checks_block(spec: dict[str, Any], rules) -> str:
    """The question list, grouped by rule, each group headed by the client's own wording. Fails closed if a rule that
    has questions is not an enabled rule (the rules changed and the checklist was not updated)."""
    by_id = {r.rule_id: r for r in rules}
    groups: dict[str, list[dict]] = {}
    for c in spec["checks"]:
        groups.setdefault(c["rule"], []).append(c)
    unknown = sorted(g for g in groups if g != "FACT" and g not in by_id)
    if unknown:
        raise ValueError(f"checks.yaml has questions for rules that are not enabled: {unknown}. Update the checklist version.")
    uncovered = sorted(set(by_id) - set(groups) - set(spec.get("covers", {})))
    if uncovered:
        raise ValueError(f"rules with no questions and no 'covers' entry: {uncovered}. A rule must never be silently left out.")
    lines = ["## The questions (an itemised version of the rules above; answer every one)", ""]
    for rule_id, items in groups.items():
        if rule_id == "FACT":
            lines.append("### Facts about the photograph (not quality checks)")
        else:
            r = by_id[rule_id]
            lines.append(f"### {rule_id} - {r.source_ref}: \"{r.original_text}\"")
        for c in items:
            note = f"  [NOT_APPLICABLE if {c['na']}]" if c.get("na") else ""
            tag = "  (team guidance, not in the client's document)" if c["source"] == "team" else (
                "  (added from the NGO taggers' notes, not in the client's rules)" if c["source"] == "tagger" else "")
            lines.append(f"- {c['id']}: {c['q']}{note}{tag}")
        lines.append("")
    if spec.get("covers"):
        lines.append("Rules " + ", ".join(spec["covers"]) + " restate other rules and have no questions of their own: "
                     + "; ".join(f"{k} is covered by {', '.join(v)}" for k, v in spec["covers"].items()) + ".")
    return "\n".join(lines).rstrip()


def checklist_schema(ids: list[str]) -> dict[str, Any]:
    entry = {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": list(ANSWERS)}, "evidence": {"type": "string"}},
        "required": ["answer", "evidence"],
        "propertyOrdering": ["answer", "evidence"],
    }
    return {
        "type": "object",
        "properties": {
            "scene": {"type": "string"},
            "checks": {"type": "object", "properties": {i: entry for i in ids}, "required": list(ids),
                       "propertyOrdering": list(ids)},
            "cannot_judge": {"type": "boolean"},
        },
        "required": ["scene", "checks", "cannot_judge"],
        "propertyOrdering": ["scene", "checks", "cannot_judge"],
    }


@dataclass
class ChecklistAnswer:
    scene: str
    checks: dict[str, dict[str, str]] = field(default_factory=dict)  # id -> {"answer": ..., "evidence": ...}
    cannot_judge: bool = False


def parse_checklist(text: str, ids: list[str]) -> ChecklistAnswer:
    import json

    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ResponseError(f"not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ResponseError("answer is not a JSON object")
    missing = [k for k in ("scene", "checks", "cannot_judge") if k not in raw]
    if missing:
        raise ResponseError(f"missing fields: {missing}")
    if not isinstance(raw["cannot_judge"], bool):
        raise ResponseError("cannot_judge must be a boolean")
    got = raw["checks"]
    if not isinstance(got, dict):
        raise ResponseError("checks must be an object")
    absent = [i for i in ids if i not in got]
    extra = [k for k in got if k not in ids]
    if absent or extra:
        raise ResponseError(f"checks do not match the checklist (missing {absent[:5]}, unknown {extra[:5]})")
    clean: dict[str, dict[str, str]] = {}
    for i in ids:
        item = got[i]
        if not isinstance(item, dict) or item.get("answer") not in ANSWERS:
            raise ResponseError(f"bad answer for {i}: {item!r}")
        clean[i] = {"answer": item["answer"], "evidence": str(item.get("evidence", "")).strip()}
    return ChecklistAnswer(str(raw["scene"]), clean, raw["cannot_judge"])


def rule_summary(answers: dict[str, dict[str, str]], spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per rule: how many of its applicable checks were met (a partial score), and a PASS/FAIL/UNCERTAIN/NOT_APPLICABLE label.

    Facts (rule FACT) are not graded. A rule FAILs if any of its applicable checks is NO; it is UNCERTAIN if none is NO but
    some could not be told; NOT_APPLICABLE if every check is NOT_APPLICABLE.
    """
    out: dict[str, dict[str, Any]] = {}
    for c in spec["checks"]:
        if c["rule"] == "FACT":
            continue
        a = answers[c["id"]]["answer"]
        s = out.setdefault(c["rule"], {"yes": 0, "no": 0, "unknown": 0, "na": 0})
        s["yes" if a == "YES" else "no" if a == "NO" else "unknown" if a == "CANNOT_TELL" else "na"] += 1
    for s in out.values():
        graded = s["yes"] + s["no"]
        s["fraction_met"] = s["yes"] / graded if graded else None
        s["status"] = ("FAIL" if s["no"] else "UNCERTAIN" if s["unknown"] else "PASS" if s["yes"] else "NOT_APPLICABLE")
    return out
