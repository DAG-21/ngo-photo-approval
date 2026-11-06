"""
Pydantic data validation models enforcing strict schemas across all phases:
- Phase 0: NGO Rules & Image Metadata
- Phase 1: Prompt 1 Individual Image Annotation
- Phase 2: Prompt 2 Taxonomy Normalization
- Phase 4: Enriched Dataset Record (with immutable historical label & separate human validation)
- Phase 5: Prompt 4 Production Evaluation Output
"""

from typing import Annotated, Dict, List, Literal, Optional
from pydantic import BaseModel, BeforeValidator, Field, model_validator

class CompatModel(BaseModel):
    def to_dict(self):
        return self.model_dump() if hasattr(self, "model_dump") else self.dict()

    @classmethod
    def parse_dict(cls, data):
        return cls.model_validate(data) if hasattr(cls, "model_validate") else cls.parse_obj(data)



# --- Phase 0: Rules & Metadata ---
class NGORule(CompatModel):
    rule_id: str
    rule_name: str
    description: str
    severity: Literal["MANDATORY", "ADVISORY"]
    decision_effect: Literal["HARD_GATE", "HUMAN_REVIEW_IF_UNVERIFIED", "EDITORIAL_WEIGHT", "NO_BLOCK"]
    visually_assessable: bool
    additional_information_required: bool


# --- Phase 1: Prompt 1 Individual Historical Annotation ---
def _normalize_level(value):
    """LLMs often answer MODERATE for a HIGH/MEDIUM/LOW scale; map the obvious alias, reject the rest."""
    if isinstance(value, str):
        value = value.strip().upper()
        return {"MODERATE": "MEDIUM"}.get(value, value)
    return value


def _stringify_items(value):
    """Flatten list items the model returned as objects (e.g. {"item": ..., "severity": ...}) into strings."""
    if not isinstance(value, list):
        return value
    out = []
    for item in value:
        if isinstance(item, dict):
            out.append("; ".join(f"{k}: {v}" for k, v in item.items()))
        else:
            out.append(item)
    return out


StrList = Annotated[List[str], BeforeValidator(_stringify_items)]

RuleStatus = Literal["PASS", "FAIL", "UNCERTAIN", "NOT_APPLICABLE", "NOT_VISUALLY_ASSESSABLE"]
ConfidenceLevel = Annotated[Literal["HIGH", "MEDIUM", "LOW"], BeforeValidator(_normalize_level)]
SeverityLevel = Annotated[Literal["HIGH", "MEDIUM", "LOW"], BeforeValidator(_normalize_level)]
GeneralAssessment = Literal["GOOD", "ACCEPTABLE", "POOR", "UNCERTAIN"]
SafetyAssessmentStatus = Literal["CLEAR", "CONCERN", "UNCERTAIN", "NOT_ASSESSABLE"]
PublicationRecommendation = Literal["SUITABLE", "NOT_SUITABLE", "NEEDS_HUMAN_REVIEW", "INSUFFICIENT_INFORMATION"]
FactorStrength = Literal["STRONG_CONCERN", "MODERATE_CONCERN", "MINOR_CONCERN"]
EvidenceType = Literal[
    "DIRECTLY_OBSERVABLE",
    "CONTEXTUAL_INTERPRETATION",
    "SUBJECTIVE_EDITORIAL",
    "EXTERNAL_VERIFICATION_REQUIRED",
]


class RuleAssessmentItem(CompatModel):
    rule_id: str
    status: RuleStatus
    evidence: str
    confidence: ConfidenceLevel
    external_verification_required: bool = False


class StandardDimensionAssessment(CompatModel):
    assessment: GeneralAssessment
    issues: StrList = Field(default_factory=list)
    strengths: StrList = Field(default_factory=list)
    confidence: ConfidenceLevel


class SafetyDimensionAssessment(CompatModel):
    assessment: SafetyAssessmentStatus
    issues: StrList = Field(default_factory=list)
    externally_verifiable_items: StrList = Field(default_factory=list)
    confidence: ConfidenceLevel


class ContextualDimensionAssessment(CompatModel):
    assessment: GeneralAssessment
    issues: StrList = Field(default_factory=list)
    confidence: ConfidenceLevel


class DimensionAssessments(CompatModel):
    technical_quality: StandardDimensionAssessment
    composition: StandardDimensionAssessment
    editorial_storytelling: StandardDimensionAssessment
    brand_suitability: StandardDimensionAssessment
    compliance_safety: SafetyDimensionAssessment
    contextual_suitability: ContextualDimensionAssessment


class IdentifiedTag(CompatModel):
    tag_id: str
    category: str
    severity: SeverityLevel
    evidence: str
    evidence_type: EvidenceType
    confidence: ConfidenceLevel


class PotentialRejectionFactor(CompatModel):
    factor: str
    reason: str
    strength: FactorStrength
    evidence: str


class AdditionalFactorProposal(CompatModel):
    short_name: str
    description: str
    suitability_impact_reason: str
    visible_evidence: str
    is_recurring: bool = False


class VisualPublicationAssessment(CompatModel):
    recommendation: PublicationRecommendation
    summary: str


class HistoricalAnnotationRecord(CompatModel):
    """Schema for Prompt 1 output."""
    image_id: str
    annotation_version: str = "1.0"
    rule_assessments: List[RuleAssessmentItem]
    dimension_assessments: DimensionAssessments
    identified_tags: List[IdentifiedTag] = Field(default_factory=list)
    potential_rejection_factors: List[PotentialRejectionFactor] = Field(default_factory=list)
    additional_factor_proposals: List[AdditionalFactorProposal] = Field(default_factory=list)
    positive_attributes: StrList = Field(default_factory=list)
    external_verification_required: StrList = Field(default_factory=list)
    visual_publication_assessment: VisualPublicationAssessment
    uncertainties: StrList = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_all_16_rules_present(self):
        expected = {f"RULE_{i:02d}" for i in range(1, 17)}
        present = {r.rule_id for r in self.rule_assessments}
        missing = expected - present
        if missing:
            raise ValueError(f"Missing mandatory rule assessments: {sorted(missing)}")
        return self


# --- Phase 2: Taxonomy Normalization ---
class CanonicalTagDefinition(CompatModel):
    canonical_tag_id: str
    display_name: str
    parent_category: str
    definition: str
    examples: List[str] = Field(default_factory=list)
    severity_guidance: str
    visually_assessable: bool
    human_validation_recommended: bool
    merged_source_tags: List[str] = Field(default_factory=list)
    tag_type: Literal["EXPLICIT_POLICY_CONCERN", "EDITORIAL_CONSIDERATION", "OBSERVATIONAL_CHARACTERISTIC"]


class NormalizedTaxonomyOutput(CompatModel):
    taxonomy_version: str = "1.0"
    canonical_tags: List[CanonicalTagDefinition]
    raw_to_canonical_mapping: Dict[str, str]


# --- Phase 4: Enriched Dataset Schema ---
class ModelAssessmentSummary(CompatModel):
    recommendation: PublicationRecommendation
    annotation_version: str
    model_version: str


class HumanValidationRecord(CompatModel):
    reviewed: bool = False
    confirmed_tags: List[str] = Field(default_factory=list)
    confirmed_rejection_reasons: List[str] = Field(default_factory=list)
    reviewer_notes: Optional[str] = None


class EnrichedImageMetadata(CompatModel):
    event_id: Optional[str] = None
    capture_date: Optional[str] = None
    image_width: int
    image_height: int


class EnrichedDatasetRecord(CompatModel):
    """Schema for Phase 4 enriched dataset record."""
    image_id: str
    historical_decision: Literal["APPROVED", "REJECTED"]
    rule_assessments: Dict[str, RuleStatus]
    technical_tags: List[str] = Field(default_factory=list)
    composition_tags: List[str] = Field(default_factory=list)
    editorial_tags: List[str] = Field(default_factory=list)
    brand_tags: List[str] = Field(default_factory=list)
    safety_tags: List[str] = Field(default_factory=list)
    context_tags: List[str] = Field(default_factory=list)
    potential_rejection_factors: List[str] = Field(default_factory=list)
    model_assessment: ModelAssessmentSummary
    human_validation: HumanValidationRecord = Field(default_factory=HumanValidationRecord)
    metadata: EnrichedImageMetadata


# --- Phase 5: Prompt 4 Production Photo Evaluation ---
ProductionRecommendation = Literal["APPROVE", "REJECT", "HUMAN_REVIEW"]


class ProductionRuleAssessment(CompatModel):
    rule_id: str
    status: RuleStatus
    evidence: str
    decision_effect: Literal["HARD_GATE", "HUMAN_REVIEW_IF_UNVERIFIED", "EDITORIAL_WEIGHT", "NO_BLOCK"]


class ScoredDimensionAssessment(CompatModel):
    score: int = Field(ge=1, le=10)
    identified_issues: StrList = Field(default_factory=list)
    strengths: StrList = Field(default_factory=list)


class ProductionSafetyAssessment(CompatModel):
    status: SafetyAssessmentStatus
    identified_flags: List[str] = Field(default_factory=list)
    external_checks_required: List[str] = Field(default_factory=list)


class HistoricalExampleComparison(CompatModel):
    reference_image_id: str
    historical_decision: Literal["APPROVED", "REJECTED"]
    relevance: str
    comparison: str


class ProductionEvaluationOutput(CompatModel):
    """Schema for Prompt 4 final production output."""
    image_id: str
    recommendation: ProductionRecommendation
    decision_summary: str
    rule_assessments: List[ProductionRuleAssessment]
    technical_assessment: ScoredDimensionAssessment
    composition_assessment: ScoredDimensionAssessment
    editorial_assessment: ScoredDimensionAssessment
    brand_assessment: ScoredDimensionAssessment
    safety_assessment: ProductionSafetyAssessment
    identified_tags: List[str] = Field(default_factory=list)
    primary_reasons: List[str] = Field(default_factory=list)
    historical_examples_considered: List[HistoricalExampleComparison] = Field(default_factory=list)
    human_review_required: bool
    human_review_questions: List[str] = Field(default_factory=list)
    confidence_level: ConfidenceLevel

    @model_validator(mode="after")
    def validate_all_16_rules_present(self):
        expected = {f"RULE_{i:02d}" for i in range(1, 17)}
        present = {r.rule_id for r in self.rule_assessments}
        missing = expected - present
        if missing:
            raise ValueError(f"Missing mandatory rule assessments in production output: {sorted(missing)}")
        return self
