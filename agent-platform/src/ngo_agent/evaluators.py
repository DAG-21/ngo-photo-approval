"""Evaluators: anything that turns image bytes into the layered engine's assessment and decision.

`LayeredEngineEvaluator` is the real one (Gemini through layered_engine; fails closed without authorisation, exactly like the engine).
`ScriptedEvaluator` produces a deterministic compliant assessment with chosen deviations; it makes no network call and is used by tests and
the demo. The architecture stays model-independent: anything with `evaluate(...)` can be plugged in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from layered_engine.annotation import Engine, parse_response
from layered_engine.compact import build_compact_schema, expand_compact
from layered_engine.decision_engine import evaluate_final_decision
from layered_engine.prompt_builder import build_prompt
from layered_engine.rulebook import Rulebook
from layered_engine.schema_export import build_response_schema
from layered_engine.schemas import CriterionAnswer, DimensionAssessment, ModelAssessment, SceneFacts, SceneUnderstanding, Status

DIMENSIONS = ("activity_communication", "technical_quality", "composition", "naturalness_storytelling", "brand_presentation", "visual_polish")


class EvaluationFailure(RuntimeError):
    """The evaluator could not produce a valid assessment. Recorded as a FAILED run; never turned into an approval."""


@dataclass
class EvaluatorOutput:
    raw_text: str
    validated: dict[str, Any]
    decision: dict[str, Any]
    model_provider: str
    model_id: str
    profile: str
    prompt_hash: str
    rulebook_version: str
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: Optional[float] = None
    warnings: list[str] = field(default_factory=list)


class Evaluator(Protocol):
    def evaluate(self, image_bytes: bytes, mime_type: str, rulebook_config: dict[str, Any], metadata: dict[str, Any]) -> EvaluatorOutput: ...
    # An evaluator may also accept `examples=` (similar past staff decisions, see layered_engine.few_shot). The service passes it only when there are
    # examples, so an evaluator written before L2 keeps working unchanged.


def _cost(usage: dict[str, Any], cfg: dict[str, Any]) -> Optional[float]:
    from layered_engine.costs import cost_usd   # thinking billed as output, cached input at the cached rate, prices per model

    return cost_usd(usage, cfg)


class LayeredEngineEvaluator:
    """Real evaluation with the layered engine and a vision provider (Gemini). `rulebook_config` = {"profile": ..., "without": [...]}."""

    def __init__(self, provider: Any, engine_config: dict[str, Any], rulebook: Optional[Rulebook] = None, provider_name: str = "google-gemini") -> None:
        self.provider, self.cfg, self.rb, self.provider_name = provider, engine_config, rulebook or Rulebook(), provider_name

    def evaluate(self, image_bytes: bytes, mime_type: str, rulebook_config: dict[str, Any], metadata: dict[str, Any], examples: Optional[list[dict[str, Any]]] = None) -> EvaluatorOutput:
        without = rulebook_config.get("without", [])
        profile = self.rb.resolve(rulebook_config["profile"], without=without, overlay=rulebook_config.get("overlay"))   # decisions use the stored overlay of learned settings
        fmt = (self.cfg.get("evaluation") or {}).get("response_format", "full")                                         # "compact": the short answer format (compact.py)
        prompt = build_prompt(self.rb, self.rb.resolve(rulebook_config["profile"], without=without),                      # the prompt never does: its hash must not move
                              notes=rulebook_config.get("prompt_notes"),                                                   # except by the reviewers' notes (L5), when there are any,
                              rule_text=rulebook_config.get("rule_text"), response_format=fmt)                             # by rules rewritten from their feedback, and by the format
        schema = build_compact_schema(profile, prompt.criterion_ids) if fmt == "compact" else build_response_schema(profile, prompt.criterion_ids)
        engine = Engine(self.rb, profile, prompt, schema, self.provider, self.cfg)
        extra = {"examples": examples} if examples else {}   # no examples: the call is exactly the one that was always made
        res = engine.provider.assess(image_bytes, mime_type, prompt.system_text, prompt.user_text, engine.schema, **extra)
        try:
            assessment = expand_compact(res.text, "pending", profile, prompt.criterion_ids) if fmt == "compact" else parse_response(res.text, "pending", profile)
        except ValueError as exc:
            raise EvaluationFailure(f"the model response did not validate: {exc}") from exc
        meta = {k: v for k, v in metadata.items() if k in ("width", "height")}
        decision = evaluate_final_decision(assessment, profile, meta, versions={"prompt_version": prompt.template_version, "prompt_hash": prompt.hash, "model_id": engine.model_id})
        return EvaluatorOutput(raw_text=res.text, validated=assessment.model_dump(mode="json"), decision=decision.model_dump(mode="json"), model_provider=self.provider_name,
                               model_id=engine.model_id, profile=profile.name, prompt_hash=prompt.hash, rulebook_version=profile.rulebook_version, usage=dict(res.usage),
                               cost_usd=_cost(res.usage, self.cfg))


class ScriptedEvaluator:
    """No model: a compliant food-distribution assessment with chosen deviations (`answers`, `facts`, `dims`), decided by the real rule engine."""

    FOOD = dict(f_photo="YES", f_single="YES", f_volunteers="YES", f_banner="YES", f_banner_prominent="YES", f_beneficiaries="YES", f_animals="NO", f_item_given="YES",
                f_handover="YES", f_volunteer_giver="YES", f_logo="YES", f_medical="NO", f_orientation="FRONT", f_group="GROUP", f_activity_type="FOOD_DISTRIBUTION")

    def __init__(self, answers: Optional[dict[str, str]] = None, facts: Optional[dict[str, str]] = None, dims: Optional[dict[str, Any]] = None,
                 positives: Optional[list[str]] = None, concerns: Optional[list[str]] = None, fail_with: Optional[str] = None, rulebook: Optional[Rulebook] = None,
                 flags: Optional[list[dict[str, str]]] = None) -> None:
        self.answers, self.facts, self.dims, self.positives, self.concerns, self.fail_with = answers or {}, facts or {}, dims or {}, positives or [], concerns or [], fail_with
        self.flags = flags or []   # visual flags, e.g. {"flag_type": "CONFIDENTIAL_INFORMATION", "evidence": "number plate legible", "confidence": "HIGH"}
        self.seen_examples: list[Optional[list[dict[str, Any]]]] = []   # what each call was given, for tests
        self.rb = rulebook or Rulebook()
        self.calls = 0

    def evaluate(self, image_bytes: bytes, mime_type: str, rulebook_config: dict[str, Any], metadata: dict[str, Any], examples: Optional[list[dict[str, Any]]] = None) -> EvaluatorOutput:
        self.calls += 1
        self.seen_examples.append(examples)
        if self.fail_with:
            raise EvaluationFailure(self.fail_with)
        without = rulebook_config.get("without", [])
        profile = self.rb.resolve(rulebook_config["profile"], without=without, overlay=rulebook_config.get("overlay"))
        self.seen_notes = rulebook_config.get("prompt_notes")   # for tests: the reviewers' notes this call was given
        self.seen_rule_text = rulebook_config.get("rule_text")   # likewise the rewritten rules
        prompt = build_prompt(self.rb, self.rb.resolve(rulebook_config["profile"], without=without), notes=self.seen_notes,
                              rule_text=self.seen_rule_text)   # the prompt is built from the profile WITHOUT the overlay
        crit = [CriterionAnswer(criterion_id=c.criterion_id, status=Status(self.answers.get(c.criterion_id, "YES")), evidence="seen in the photograph") for c in profile.asked_criteria()]
        dims = {d: DimensionAssessment(applicable=True, score=self.dims.get(d, 4), explanation="scripted") for d in DIMENSIONS}
        facts = {**self.FOOD, **self.facts}
        if "F_GAZE_TAG" in profile.criteria:
            facts.setdefault("f_gaze_tag", "NATURAL_CANDID_GAZE")
        from layered_engine.schemas import Observation, VisualFlag

        a = ModelAssessment(image_id="scripted", layer_1_scene_understanding=SceneUnderstanding(activity_description="volunteers serving food", scene_facts=SceneFacts(**facts)),
                            layer_2_criteria=crit, layer_3_quality_dimensions=dims, layer_4_visual_flags=[VisualFlag(**f) for f in self.flags], positive_attributes=[Observation(text=t, evidence="scripted") for t in self.positives],
                            potential_rejection_factors=[Observation(text=t, evidence="scripted") for t in self.concerns])
        decision = evaluate_final_decision(a, profile, {k: v for k, v in metadata.items() if k in ("width", "height")},
                                           versions={"prompt_version": prompt.template_version, "prompt_hash": prompt.hash, "model_id": "scripted"})
        return EvaluatorOutput(raw_text=a.model_dump_json(), validated=a.model_dump(mode="json"), decision=decision.model_dump(mode="json"), model_provider="scripted",
                               model_id="scripted", profile=profile.name, prompt_hash=prompt.hash, rulebook_version=profile.rulebook_version,
                               usage={"prompt_tokens": 0, "output_tokens": 0, "latency_s": 0.0}, cost_usd=0.0)
