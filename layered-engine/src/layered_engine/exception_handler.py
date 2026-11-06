"""Exceptions and precedence.

An exception may only change a criterion listed in its `applies_to`, and only when that criterion's status is NO. It never
touches any other criterion and can never turn an UNKNOWN into a pass. Exceptions run in the order they are listed.
"""

from __future__ import annotations

from typing import Any, Mapping

from .schemas import CriterionResult, Status

_EFFECT = {"SET_YES": Status.YES, "SET_NOT_APPLICABLE": Status.NOT_APPLICABLE, "SET_CANNOT_TELL": Status.CANNOT_TELL}


def _conditions_met(exc: Mapping[str, Any], results: Mapping[str, CriterionResult], facts: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    for cid in exc.get("requires_criteria_all_yes", []):
        r = results.get(cid)
        if r is None or r.status is not Status.YES:
            return False
    for key, want in (exc.get("requires_facts") or {}).items():
        if facts.get(key) != want:
            return False
    for key, want in (exc.get("requires_metadata") or {}).items():
        if metadata.get(key) != want:
            return False
    return True


def apply_exceptions(results: dict[str, CriterionResult], exceptions: list[dict[str, Any]], facts: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[str]:
    """Mutates `results` and returns the ids of the exceptions that changed something."""
    # conditions are checked against the statuses BEFORE any exception ran, so one exception cannot enable another
    snapshot = {k: v.model_copy() for k, v in results.items()}
    fired: list[str] = []
    for exc in exceptions:
        if exc["effect"] not in _EFFECT:
            raise ValueError(f"exception {exc['id']}: unknown effect {exc['effect']!r}")
        if not _conditions_met(exc, snapshot, facts, metadata):
            continue
        for cid in exc["applies_to"]:
            r = results.get(cid)
            if r is not None and r.status is Status.NO:
                r.status = _EFFECT[exc["effect"]]
                r.notes.append(f"exception:{exc['id']}")
                if exc["id"] not in fired:
                    fired.append(exc["id"])
    return fired
