"""Which criteria apply to THIS photograph. Decided in Python from the Layer-1 facts and the metadata, never by the model."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .expressions import evaluate
from .rulebook import Criterion, ResolvedProfile
from .schemas import ModelAssessment


def facts_dict(assessment: ModelAssessment) -> dict[str, Any]:
    f = assessment.layer_1_scene_understanding.scene_facts
    return {k: (v.value if hasattr(v, "value") else v) for k, v in f.model_dump().items()}


def applicability_of(criterion: Criterion, profile: ResolvedProfile, facts: Mapping[str, Any], metadata: Mapping[str, Any]) -> Optional[bool]:
    """True / False / None (unknown: treated as applicable)."""
    return evaluate({"ref": criterion.applicability}, facts, metadata, profile.conditions)


def validate_activity(assessment: ModelAssessment, profile: ResolvedProfile) -> Optional[str]:
    """The activity type must be one of the taxonomy values (the taxonomy is configuration, not code)."""
    value = assessment.layer_1_scene_understanding.scene_facts.f_activity_type
    return None if value in profile.activity_ids else f"activity type {value!r} is not in the activity taxonomy"
