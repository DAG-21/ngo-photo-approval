"""
Phase 5: Final Production Photo Evaluation & Decision-Making Layer (Prompt 4)
Enforces the 5-Priority Hierarchy:
  Priority 1: Mandatory NGO hard-gate rules -> Any FAIL forces `REJECT`; any UNCERTAIN/NOT_VISUALLY_ASSESSABLE forces `HUMAN_REVIEW`.
  Priority 2: Safety, privacy & safeguarding -> Never assumes visual consent; routes unverified minor/beneficiary consent to `HUMAN_REVIEW`.
  Priority 3: Technical & visual quality -> Enforces minimum resolution and sharpness standards.
  Priority 4: Editorial & storytelling quality -> Evaluates narrative clarity.
  Priority 5: Brand suitability -> Enforces beneficiary dignity and non-exploitative framing.
Never averages dimension scores to override a hard-gate rule or unresolved safeguarding check.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from src.gemini.client import GeminiVisionClient
from src.retrieval.vector_index import HistoricalFewShotRetriever
from src.schemas import ProductionEvaluationOutput


class PhotoApprovalDecisionEngine:
    def __init__(
        self,
        client: GeminiVisionClient,
        rules_path: Path,
        eval_config_path: Path,
        prompt4_template_path: Path,
        decisions_output_dir: Path,
        retriever: Optional[HistoricalFewShotRetriever] = None,
    ):
        self.client = client
        self.rules = json.loads(rules_path.read_text(encoding="utf-8"))
        self.rules_by_id = {r["rule_id"]: r for r in self.rules}
        self.eval_config = yaml.safe_load(eval_config_path.read_text(encoding="utf-8"))
        self.prompt4_template = prompt4_template_path.read_text(encoding="utf-8")
        self.decisions_output_dir = decisions_output_dir
        self.decisions_output_dir.mkdir(parents=True, exist_ok=True)
        self.retriever = retriever

    def build_production_prompt(
        self,
        new_image_id: str,
        few_shot_examples: List[Dict[str, Any]],
        external_metadata: Optional[Dict[str, Any]] = None,
        include_enhanced_rubric: bool = True,
    ) -> str:
        rules_json = json.dumps(self.rules, indent=2)
        rubric_data = (
            self.eval_config.get("enhanced_evaluation_rubric", {})
            if include_enhanced_rubric
            else {"note": "Baseline mode: evaluate using original 16 rules only."}
        )
        rubric_json = json.dumps(rubric_data, indent=2)
        refs_json = json.dumps(few_shot_examples, indent=2) if few_shot_examples else "[]"
        ext_json = json.dumps(external_metadata or {}, indent=2)

        return (
            self.prompt4_template
            .replace("{NGO_16_RULES}", rules_json)
            .replace("{ENHANCED_EVALUATION_RUBRIC}", rubric_json)
            .replace("{FEW_SHOT_REFERENCES}", refs_json)
            .replace("{NEW_IMAGE_ID}", new_image_id)
            .replace("{EXTERNAL_METADATA}", ext_json)
        )

    def apply_deterministic_guardrails(
        self,
        model_output: ProductionEvaluationOutput,
        external_metadata: Optional[Dict[str, Any]] = None,
    ) -> ProductionEvaluationOutput:
        """
        Applies deterministic post-processing guardrails over Gemini's Prompt 4 output
        to guarantee zero false positives on mandatory hard-gate and consent rules.
        """
        external_metadata = external_metadata or {}
        policy = self.eval_config.get("decision_policy", {})
        hard_gate_ids = set(policy.get("hard_gate_rules", []))
        external_ver_ids = set(policy.get("external_verification_rules", []))

        failed_hard_gates = []
        uncertain_mandatory = []

        for r in model_output.rule_assessments:
            rule_def = self.rules_by_id.get(r.rule_id, {})
            is_mandatory = rule_def.get("severity") == "MANDATORY"

            # Check if external metadata explicitly authorizes a camera-facing shot for RULE_04
            if r.rule_id == "RULE_04" and external_metadata.get("camera_gaze_required") is True:
                r.status = "PASS"
                r.decision_effect = "NO_BLOCK"

            if r.rule_id in hard_gate_ids and r.status == "FAIL":
                failed_hard_gates.append(r.rule_id)
            elif is_mandatory and r.status in ("UNCERTAIN", "NOT_VISUALLY_ASSESSABLE"):
                uncertain_mandatory.append(r.rule_id)
            elif r.rule_id in external_ver_ids and r.status == "FAIL":
                failed_hard_gates.append(r.rule_id)

        # Priority 1: Conclusively violated mandatory hard-gate rule -> REJECT
        if failed_hard_gates:
            model_output.recommendation = "REJECT"
            model_output.human_review_required = False
            reason_msg = f"Mandatory hard-gate rule violation(s): {', '.join(failed_hard_gates)}"
            if reason_msg not in model_output.primary_reasons:
                model_output.primary_reasons.insert(0, reason_msg)
            return model_output

        # Priority 2: Unverified mandatory rule or safety/consent uncertainty -> HUMAN_REVIEW
        if (
            uncertain_mandatory
            or model_output.safety_assessment.status in ("CONCERN", "UNCERTAIN", "NOT_ASSESSABLE")
            or len(model_output.safety_assessment.external_checks_required) > 0
        ):
            if model_output.recommendation == "APPROVE":
                model_output.recommendation = "HUMAN_REVIEW"
            model_output.human_review_required = True
            if uncertain_mandatory:
                q = f"Verify external compliance/consent for mandatory rule(s): {', '.join(uncertain_mandatory)}"
                if q not in model_output.human_review_questions:
                    model_output.human_review_questions.append(q)
            return model_output

        # Priority 3-5: Minimum editorial/brand/technical thresholds for auto-approval
        min_tech = policy.get("min_technical_score_for_auto_approve", 7)
        min_comp = policy.get("min_composition_score_for_auto_approve", 7)
        min_edit = policy.get("min_editorial_score_for_auto_approve", 7)
        min_brand = policy.get("min_brand_score_for_auto_approve", 8)

        if model_output.recommendation == "APPROVE":
            if (
                model_output.technical_assessment.score < min_tech
                or model_output.composition_assessment.score < min_comp
                or model_output.editorial_assessment.score < min_edit
                or model_output.brand_assessment.score < min_brand
            ):
                model_output.recommendation = "HUMAN_REVIEW"
                model_output.human_review_required = True
                model_output.human_review_questions.append(
                    "One or more dimension scores fell below the strict auto-approval threshold; editorial review required."
                )

        return model_output

    def evaluate_photograph(
        self,
        new_image_id: str,
        image_path: Path,
        external_metadata: Optional[Dict[str, Any]] = None,
        mode: str = "EXPERIMENT_D",
        fixed_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> ProductionEvaluationOutput:
        """
        Evaluates a photograph under one of the 4 experimental configurations:
        - `EXPERIMENT_A`: Original 16 rules only, zero few-shot examples
        - `EXPERIMENT_B`: Enhanced evaluation rubric, zero few-shot examples
        - `EXPERIMENT_C`: Enhanced evaluation rubric + fixed representative few-shot examples
        - `EXPERIMENT_D`: Enhanced evaluation rubric + dynamically retrieved few-shot examples (Production default)
        """
        include_enhanced = mode in ("EXPERIMENT_B", "EXPERIMENT_C", "EXPERIMENT_D")
        few_shot_refs: List[Dict[str, Any]] = []

        if mode == "EXPERIMENT_C" and fixed_examples:
            few_shot_refs = fixed_examples
        elif mode == "EXPERIMENT_D" and self.retriever is not None:
            event_id = (external_metadata or {}).get("event_id")
            few_shot_refs = self.retriever.retrieve_examples(
                target_image_path=image_path,
                target_image_id=new_image_id,
                target_event_id=event_id,
            )

        prompt_text = self.build_production_prompt(
            new_image_id=new_image_id,
            few_shot_examples=few_shot_refs,
            external_metadata=external_metadata,
            include_enhanced_rubric=include_enhanced,
        )

        # Attach target image + retrieved reference images
        image_paths = [image_path]
        for ref in few_shot_refs:
            ref_path = Path(ref["filepath"])
            if ref_path.exists():
                image_paths.append(ref_path)

        raw_eval, _ = self.client.generate_validated_json(
            record_id=f"{new_image_id}_{mode}",
            prompt_text=prompt_text,
            prompt_version="2.0.0",
            schema_cls=ProductionEvaluationOutput,
            image_paths=image_paths,
        )

        final_eval = self.apply_deterministic_guardrails(raw_eval, external_metadata=external_metadata)
        out_file = self.decisions_output_dir / f"{new_image_id}_{mode}_decision.json"
        out_file.write_text(json.dumps(final_eval.to_dict(), indent=2), encoding="utf-8")
        return final_eval
