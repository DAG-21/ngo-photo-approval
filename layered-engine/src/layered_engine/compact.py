"""The short answer format: the same questions, a much shorter answer, to cut the cost of a photograph.

The full format asks the model to write evidence for every one of the ~82 criteria, an explanation and lists of observations for every quality dimension, and lists of positive
and negative points: about 5,500 output tokens (plus its thinking), which is about 80% of the cost. The short format keeps everything the rule engine decides from and drops the
prose nobody reads:

* `checks`: an explicit status for every criterion that can change a decision (every gate and gate component, every soft-score criterion, every criterion a soft cap or an
  active exception depends on). These are required by the schema, and Python treats a missing or unreadable one as CANNOT_TELL, never as a pass.
* `other_criteria_not_yes`: the remaining (diagnostic) criteria are YES unless listed. This is the one place where an omission counts as YES; it is limited to criteria that
  cannot reject a photograph.
* `evidence`: one short phrase for every NO and CANNOT_TELL only (what the reviewer is shown).
* `quality`: one whole number per dimension (-1 when it does not apply), no prose.
* `scene_facts`, `activity_description`, `visual_flags`: as before.

`expand_compact` turns the short answer back into the full `ModelAssessment`, so the rule engine, the stored record and every screen work unchanged.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from .prompt_builder import asked_fact_fields
from .rulebook import ResolvedProfile
from .schema_export import build_response_schema
from .schemas import (Confidence, CriterionAnswer, DimensionAssessment, ModelAssessment, SceneFacts, SceneUnderstanding, Status, VisualFlag, missing_facts)

NOT_YES = ("NO", "CANNOT_TELL", "NOT_APPLICABLE")
MAX_EVIDENCE = 300


def split_ids(profile: ResolvedProfile, criterion_ids: list[str]) -> tuple[list[str], list[str]]:
    """(criteria that need an explicit answer, criteria that are YES unless listed), both in the order asked."""
    if profile.settings.get("check_reasons"):   # a profile that wants a reason for every answer: nothing is "YES unless listed", every criterion is answered explicitly
        return list(criterion_ids), []
    decisive: set[str] = set()
    for g in profile.gates:
        decisive.add(g.criterion_id)
        decisive.update(g.components)
    for cid in criterion_ids:
        c = profile.criteria.get(cid)
        if c is not None and c.decision_effect in ("GATE", "SOFT_SCORE"):
            decisive.add(cid)
    decisive.update(str(cap.get("criterion_id")) for cap in profile.quality.get("soft_caps", []) or [])
    for ex in profile.exceptions:
        decisive.update(ex.get("requires_criteria_all_yes", []) or [])
        decisive.update(ex.get("applies_to", []) or [])
    explicit = [c for c in criterion_ids if c in decisive]
    return explicit, [c for c in criterion_ids if c not in decisive]


def build_compact_schema(profile: ResolvedProfile, criterion_ids: list[str]) -> dict[str, Any]:
    full = build_response_schema(profile, criterion_ids)["properties"]
    explicit, others = split_ids(profile, criterion_ids)
    status = {"type": "string", "enum": [s.value for s in Status]}
    dims = list(profile.quality["dimensions"])
    props: dict[str, Any] = {
        "activity_description": {"type": "string"},
        "scene_facts": full["layer_1_scene_understanding"]["properties"]["scene_facts"],
        "checks": {"type": "object", "properties": {i: dict(status) for i in explicit}, "required": list(explicit)},
        "evidence": {"type": "array", "items": {"type": "object", "properties": {"criterion_id": {"type": "string", "enum": list(criterion_ids)}, "text": {"type": "string"}},
                                                 "required": ["criterion_id", "text"]}},
        "quality": {"type": "object", "properties": {d: {"type": "integer"} for d in dims}, "required": dims},
        "visual_flags": full["layer_4_visual_flags"],
    }
    required = ["activity_description", "scene_facts", "checks", "evidence", "quality", "visual_flags"]
    if profile.settings.get("quality_notes"):   # a short reason for every dimension graded 2 or lower (layered_final_v1): so a re-select for poor quality comes with the reasons behind it
        props["quality_notes"] = {"type": "object", "properties": {d: {"type": "string"} for d in dims}}
        required.insert(required.index("quality") + 1, "quality_notes")
    if others:
        props["other_criteria_not_yes"] = {"type": "array", "items": {"type": "object", "properties": {"criterion_id": {"type": "string", "enum": list(others)},
                                                                                                    "status": {"type": "string", "enum": list(NOT_YES)}},
                                                                     "required": ["criterion_id", "status"]}}
        required.insert(3, "other_criteria_not_yes")
    return {"type": "object", "properties": props, "required": required}


def expand_compact(text: str, image_id: str, profile: ResolvedProfile, criterion_ids: list[str]) -> ModelAssessment:
    """The short answer as a full `ModelAssessment`. Raises ValueError when it is not usable (the evaluation then fails and can be tried again; it is never guessed)."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        obj = json.loads(t)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("the response is not a JSON object")
    explicit, others = split_ids(profile, criterion_ids)
    explicit_set, others_set = set(explicit), set(others)
    checks = obj.get("checks") if isinstance(obj.get("checks"), dict) else {}
    statuses: dict[str, str] = {}
    for cid in criterion_ids:
        if cid in explicit_set:
            s = checks.get(cid)
            statuses[cid] = s if s in {x.value for x in Status} else "CANNOT_TELL"   # a missing or unreadable decisive answer is never a pass
        else:
            statuses[cid] = "YES"
    for o in obj.get("other_criteria_not_yes") or []:
        if isinstance(o, dict) and o.get("criterion_id") in others_set and o.get("status") in NOT_YES:
            statuses[o["criterion_id"]] = o["status"]
    evidence: dict[str, str] = {}
    for e in obj.get("evidence") or []:
        if isinstance(e, dict) and e.get("criterion_id") in statuses and str(e.get("text") or "").strip():
            evidence.setdefault(e["criterion_id"], " ".join(str(e["text"]).split())[:MAX_EVIDENCE])
    answers = [CriterionAnswer(criterion_id=c, status=Status(statuses[c]), evidence=evidence.get(c, ""), confidence=Confidence.MEDIUM) for c in criterion_ids]
    q = obj.get("quality") if isinstance(obj.get("quality"), dict) else {}
    notes = obj.get("quality_notes") if isinstance(obj.get("quality_notes"), dict) else {}   # optional: only a profile with `quality_notes` asks for them
    dims: dict[str, DimensionAssessment] = {}
    for d in profile.quality["dimensions"]:
        v = q.get(d)
        if isinstance(v, bool) or not isinstance(v, int) or not -1 <= v <= 4:
            raise ValueError(f"quality dimension {d!r} needs a whole number from -1 to 4, got {v!r}")
        note = " ".join(str(notes.get(d) or "").split())[:MAX_EVIDENCE] if isinstance(notes.get(d), str) else ""
        dims[d] = DimensionAssessment(applicable=False) if v == -1 else DimensionAssessment(applicable=True, score=v, explanation=note)
    known_flags = set(profile.publication["visual_flags"])
    flags = []
    for f in obj.get("visual_flags") or []:
        if isinstance(f, dict) and f.get("flag_type") in known_flags:
            conf = f.get("confidence") if f.get("confidence") in {c.value for c in Confidence} else Confidence.MEDIUM.value
            flags.append(VisualFlag(flag_type=f["flag_type"], evidence=str(f.get("evidence") or "")[:MAX_EVIDENCE], confidence=Confidence(conf)))
    try:
        facts = SceneFacts(**(obj.get("scene_facts") or {}))
        lacking = missing_facts(facts, asked_fact_fields(profile))
        if lacking:
            raise ValueError(f"the short answer does not match the schema: the scene fact {lacking[0]} is missing")
        scene = SceneUnderstanding(activity_description=str(obj.get("activity_description") or "")[:MAX_EVIDENCE], scene_facts=facts)
        return ModelAssessment(image_id=image_id, layer_1_scene_understanding=scene, layer_2_criteria=answers, layer_3_quality_dimensions=dims, layer_4_visual_flags=flags)
    except (ValidationError, TypeError) as exc:
        raise ValueError(f"the short answer does not match the schema: {str(exc).splitlines()[0]}") from exc
