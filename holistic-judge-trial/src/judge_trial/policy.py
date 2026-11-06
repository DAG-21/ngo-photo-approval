"""The decision policy for checklist runs: which answers force a disapproval.

The model never decides. It answers the checklist questions; this module applies the team's gates deterministically, so an instruction
such as "a creased banner means disapproved" is followed every time and can be tested, instead of depending on how the model
weighs a prompt. Every gate carries the instruction it came from.
"""

from __future__ import annotations

from typing import Any

import yaml

from .config import PROMPTS_DIR


class PolicyError(ValueError):
    pass


def load_policy(version: str, spec: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = yaml.safe_load((PROMPTS_DIR / version / "policy.yaml").read_text(encoding="utf-8"))
    gates = policy["gates"]
    ids = [g["id"] for g in gates]
    if len(set(ids)) != len(ids):
        raise PolicyError("a check is listed twice in policy.yaml")
    if spec is not None:
        known = {c["id"] for c in spec["checks"]}
        unknown = sorted(set(ids) - known)
        if unknown:
            raise PolicyError(f"policy.yaml gates that are not in the checklist (renamed or removed?): {unknown}")
    for g in gates:
        if not g.get("instruction") or not g.get("why"):
            raise PolicyError(f"gate {g['id']} needs the instruction it came from and a reason")
    return policy


def decide(answers: dict[str, dict[str, str]], policy: dict[str, Any], cannot_judge: bool = False) -> dict[str, Any]:
    """APPROVED unless a gate fails. A gate fails on NO, and on CANNOT_TELL when ``unknown_is_fail`` is set. NOT_APPLICABLE never fails.

    ``cannot_judge`` (the whole photo is unreadable) gives NEEDS_REVIEW instead of a guess.
    """
    strict = bool(policy.get("unknown_is_fail", True))
    triggered = []
    for g in policy["gates"]:
        a = answers.get(g["id"])
        if a is None:
            raise PolicyError(f"no answer for gate {g['id']}")
        if a["answer"] == "NO" or (strict and a["answer"] == "CANNOT_TELL"):
            triggered.append({"id": g["id"], "instruction": g["instruction"], "why": g["why"], "answer": a["answer"],
                              "evidence": a.get("evidence", "")})
    if cannot_judge:
        return {"decision": "NEEDS_REVIEW", "triggered": triggered}
    return {"decision": "DISAPPROVED" if triggered else "APPROVED", "triggered": triggered}


def render(result: dict[str, Any]) -> str:
    return "\n".join(f"[{t['instruction']}] {t['id']} ({t['answer']}): {t['why']}" + (f" - {t['evidence']}" if t["evidence"] else "")
                     for t in result["triggered"])
