"""Response JSON schema for the provider, generated from the Pydantic models and patched from the configuration.

The activity list, the criterion ids, the quality dimensions and the visual-flag names all come from the rulebook, so the schema
changes when the configuration changes. $refs are inlined and titles/defaults dropped because provider schema support is a subset
of JSON Schema.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .prompt_builder import asked_fact_fields
from .rulebook import CONFIG_DIR, ResolvedProfile
from .schemas import (OPTIONAL_FACTS, CriterionAnswer, DimensionAssessment, GazeTag, ModelAssessment, SceneFacts, SceneUnderstanding)

# fields the model must not fill in (we do not trust them); they stay in the Pydantic model only for audit use
OMITTED = ("model_recommendation", "model_overall_score", "schema_version", "image_id")  # image_id is attached by us, never trusted from the model


def simplify(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            if "anyOf" in node:
                options = [o for o in node["anyOf"] if o.get("type") != "null"]
                if len(options) == 1:
                    merged = {k: v for k, v in node.items() if k != "anyOf"}
                    merged.update(walk(options[0]))
                    return walk(merged)
            return {k: walk(v) for k, v in node.items() if k not in ("title", "default", "$defs", "additionalProperties")}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk({k: v for k, v in schema.items() if k != "$defs"} | {"$defs": defs})


def build_response_schema(profile: ResolvedProfile, criterion_ids: list[str]) -> dict[str, Any]:
    raw = ModelAssessment.model_json_schema()
    raw["$defs"] = raw.get("$defs", {})
    s = simplify(raw)
    props = s["properties"]
    for k in OMITTED:
        props.pop(k, None)
    s["required"] = [r for r in s.get("required", []) if r not in OMITTED]
    # activity types from the taxonomy
    facts_schema = props["layer_1_scene_understanding"]["properties"]["scene_facts"]
    facts_schema["properties"]["f_activity_type"] = {"type": "string", "enum": profile.activity_ids}
    # the facts a profile may stop asking are optional in the model class: here they are required when the profile asks them (so the schema of every earlier profile is exactly
    # what it was) and absent from the schema when it does not
    asked = asked_fact_fields(profile)
    for name in OPTIONAL_FACTS:
        if name not in asked:
            facts_schema["properties"].pop(name, None)
    facts_schema["required"] = [n for n in SceneFacts.model_fields if n in facts_schema["properties"] and n != "f_gaze_tag"]
    if "F_GAZE_TAG" in profile.criteria:  # v2 only: the v1 response schema stays exactly as it was
        facts_schema["properties"]["f_gaze_tag"] = {"type": "string", "enum": [t.value for t in GazeTag]}
        facts_schema["required"] = list(facts_schema.get("required", [])) + ["f_gaze_tag"]
    else:
        facts_schema["properties"].pop("f_gaze_tag", None)
    # criterion ids the model must answer
    items = props["layer_2_criteria"]["items"]
    items["properties"]["criterion_id"] = {"type": "string", "enum": list(criterion_ids)}
    items["properties"].pop("source", None)
    items["required"] = [r for r in items.get("required", []) if r != "source"] or ["criterion_id", "status"]
    # NOTE: no minItems/maxItems. A live call showed gemini-3.8-flash rejects the schema (HTTP 400) when the 83-item list carries them.
    # Completeness is enforced in Python instead: any criterion missing from the answer becomes CANNOT_TELL ("synthesized").
    # one property per configured dimension
    dim_schema = simplify(DimensionAssessment.model_json_schema())
    dim_names = list(profile.quality["dimensions"])
    props["layer_3_quality_dimensions"] = {"type": "object", "properties": {n: copy.deepcopy(dim_schema) for n in dim_names}, "required": dim_names}
    # visual flag names from the publication config
    flag_item = props["layer_4_visual_flags"]["items"]
    flag_item["properties"]["flag_type"] = {"type": "string", "enum": list(profile.publication["visual_flags"])}
    return s


def check_provider_safe(schema: Any, path: str = "") -> list[str]:
    """Problems that would make a provider reject the schema: leftover $ref, anyOf, title or default."""
    bad: list[str] = []
    if isinstance(schema, dict):
        for k, v in schema.items():
            if k in ("$ref", "anyOf", "title", "default", "$defs", "additionalProperties", "minItems", "maxItems"):
                bad.append(f"{path}/{k}")
            bad += check_provider_safe(v, f"{path}/{k}")
    elif isinstance(schema, list):
        for i, v in enumerate(schema):
            bad += check_provider_safe(v, f"{path}[{i}]")
    return bad


def export_schemas(profile: ResolvedProfile, criterion_ids: list[str], out_dir: Path | None = None) -> list[Path]:
    """Writes one schema file per layer plus the combined response schema into config/schemas/."""
    out_dir = out_dir or CONFIG_DIR / "schemas"
    out_dir.mkdir(parents=True, exist_ok=True)
    full = build_response_schema(profile, criterion_ids)
    p = full["properties"]
    files = {
        "scene_understanding.schema.json": p["layer_1_scene_understanding"],
        "gate_assessment.schema.json": p["layer_2_criteria"],
        "quality_assessment.schema.json": p["layer_3_quality_dimensions"],
        "publication_readiness.schema.json": p["layer_4_visual_flags"],
        "final_assessment.schema.json": full,
    }
    paths = []
    for name, schema in files.items():
        path = out_dir / name
        path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        paths.append(path)
    return paths
