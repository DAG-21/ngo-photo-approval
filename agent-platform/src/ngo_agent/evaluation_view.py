"""What a reviewer can correct in a stored evaluation, and what the AI said for each of those items.

The server reads the AI's value from the immutable EvaluationRun; a client can never supply it. Pure functions over the stored JSON.
"""

from __future__ import annotations

from typing import Any, Optional

from .enums import CriterionType as CT

GATE_VALUES = {"YES", "NO", "CANNOT_TELL", "NOT_APPLICABLE"}
TRI_VALUES = {"YES", "NO", "UNKNOWN"}
FINAL_VALUES = {"APPROVE", "REJECT"}


class CorrectionError(ValueError):
    pass


def _tags(validated: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for o in validated.get("positive_attributes") or []:
        out[o["text"]] = "POSITIVE"
    for o in validated.get("potential_rejection_factors") or []:
        out[o["text"]] = "CONCERN"
    return out


def addressable_items(run: Any) -> dict[tuple[str, str], Any]:
    """{(criterion_type, criterion_id): the AI's value} for everything a reviewer may correct."""
    v = run.validated_response or {}
    d = run.decision_output or {}
    items: dict[tuple[str, str], Any] = {}
    facts = (v.get("layer_1_scene_understanding") or {}).get("scene_facts") or {}
    for k, val in facts.items():
        if k == "f_activity_type":
            items[(CT.ACTIVITY_TYPE.value, "activity_type")] = val
        else:
            items[(CT.SCENE_FACT.value, k)] = val
    for g in d.get("gates") or []:
        items[(CT.MANDATORY_GATE.value, g["gate_id"])] = g["status"]
    for a in v.get("layer_2_criteria") or []:
        items[(CT.CRITERION.value, a["criterion_id"])] = a["status"]
    for name, dim in (v.get("layer_3_quality_dimensions") or {}).items():
        items[(CT.QUALITY_DIMENSION.value, name)] = dim.get("score") if dim.get("applicable", True) else None
    for flag in v.get("layer_4_visual_flags") or []:
        items[(CT.PUBLICATION.value, flag["flag_type"])] = "PRESENT"
    for check in (d.get("publication") or {}).get("checks") or []:
        cid = check.get("check_id") or check.get("id")
        if cid:
            items[(CT.PUBLICATION.value, cid)] = check.get("status")
    for text, kind in _tags(v).items():
        items[(CT.TAG_REMOVED.value, text)] = "PRESENT"
        items[(CT.EDITORIAL_OBSERVATION.value, text)] = kind
    items[(CT.FINAL_DECISION.value, "final_decision")] = run.ai_recommendation
    return items


def publication_blockers(run: Any) -> list[str]:
    """The publication blockers the stored evaluation lists (a visual flag the engine treats as blocking, or a metadata check that is BLOCKED)."""
    pub = (run.decision_output or {}).get("publication") or {}
    return [str(b) for b in pub.get("blockers") or []]


def resolve_ai_value(run: Any, criterion_type: str, criterion_id: str) -> Any:
    """The AI's value for an item. TAG_ADDED has no AI value (the AI missed it); everything else must exist in the stored run."""
    if criterion_type == CT.TAG_ADDED.value:
        if (CT.TAG_REMOVED.value, criterion_id) in addressable_items(run):
            raise CorrectionError(f"the AI already reported the tag {criterion_id!r}")
        return None
    items = addressable_items(run)
    key = (criterion_type, criterion_id)
    if key not in items:
        raise CorrectionError(f"{criterion_type} {criterion_id!r} does not exist in this evaluation")
    return items[key]


def validate_corrected_value(criterion_type: str, value: Any, activity_ids: Optional[set[str]] = None) -> Any:
    t = criterion_type
    if t in (CT.MANDATORY_GATE.value, CT.CRITERION.value):
        if value not in GATE_VALUES:
            raise CorrectionError(f"a gate or criterion value must be one of {sorted(GATE_VALUES)}")
    elif t == CT.SCENE_FACT.value:
        if value not in TRI_VALUES and not (isinstance(value, str) and value.isupper()):
            raise CorrectionError("a scene fact value must be YES, NO, UNKNOWN or an upper-case enumeration value")
    elif t == CT.ACTIVITY_TYPE.value:
        if activity_ids is not None and value not in activity_ids:
            raise CorrectionError(f"the activity type must be one of {sorted(activity_ids)}")
    elif t == CT.QUALITY_DIMENSION.value:
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 4):
            raise CorrectionError("a quality score must be an integer from 0 to 4 (or null for not applicable)")
    elif t == CT.TAG_ADDED.value:
        if value != "PRESENT":
            raise CorrectionError("an added tag has the value PRESENT")
    elif t == CT.TAG_REMOVED.value:
        if value != "ABSENT":
            raise CorrectionError("a removed tag has the value ABSENT")
    elif t == CT.FINAL_DECISION.value:
        if value not in FINAL_VALUES:
            raise CorrectionError("the final decision must be APPROVE or REJECT")
    elif t in (CT.EDITORIAL_OBSERVATION.value, CT.PUBLICATION.value):
        if value is None or (isinstance(value, str) and not value.strip()):
            raise CorrectionError("an observation or publication correction needs a value")
    else:
        raise CorrectionError(f"unknown criterion type {t!r}")
    return value


EVENT_FOR_TYPE = {
    CT.TAG_ADDED.value: "TAG_ADDED", CT.TAG_REMOVED.value: "TAG_REMOVED", CT.ACTIVITY_TYPE.value: "ACTIVITY_CHANGED",
    CT.EDITORIAL_OBSERVATION.value: "OBSERVATION_ADDED", CT.PUBLICATION.value: "CONTEXT_SUPPLIED",
}


def effective_value(corrections: list[Any], criterion_type: str, criterion_id: str) -> Optional[Any]:
    """The latest non-superseded correction value for an item, or None when it was never corrected."""
    mine = [c for c in corrections if c.criterion_type == criterion_type and c.criterion_id == criterion_id]
    return max(mine, key=lambda c: c.revision).human_corrected_value if mine else None
