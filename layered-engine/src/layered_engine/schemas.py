"""Typed models: what the vision model returns (provider-neutral), the metadata we supply, and the decision we compute.

Nothing in here knows about Gemini. A future computer-vision component can add CriterionAnswer rows with source="cv" and the
rest of the engine does not change.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"


class Status(str, Enum):
    YES = "YES"
    NO = "NO"
    CANNOT_TELL = "CANNOT_TELL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Tri(str, Enum):
    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


CONFIDENCE_RANK = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}


class Orientation(str, Enum):
    FRONT = "FRONT"
    SIDE = "SIDE"
    BACK = "BACK"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class GazeTag(str, Enum):
    """Editorial tag (v2): how the beneficiaries' gaze reads. Never a rejection by itself in the v2 profile."""

    DIRECT_CAMERA_GAZE = "DIRECT_CAMERA_GAZE"
    NATURAL_CANDID_GAZE = "NATURAL_CANDID_GAZE"
    INTENTIONAL_PORTRAIT = "INTENTIONAL_PORTRAIT"
    UNCERTAIN_GAZE = "UNCERTAIN_GAZE"


class GroupType(str, Enum):
    SINGLE = "SINGLE"
    GROUP = "GROUP"
    CROWD = "CROWD"
    UNKNOWN = "UNKNOWN"


class SubjectCategory(str, Enum):
    VOLUNTEER = "VOLUNTEER"
    BENEFICIARY = "BENEFICIARY"
    ANIMAL = "ANIMAL"
    OTHER = "OTHER"


class SubjectRole(str, Enum):
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    INCIDENTAL = "INCIDENTAL"


class Recommendation(str, Enum):
    APPROVE = "APPROVE"
    APPROVED_PENDING_PUBLICATION_CHECKS = "APPROVED_PENDING_PUBLICATION_CHECKS"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    RESELECT = "RESELECT"
    REJECT = "REJECT"


# --------------------------------------------------------------------------------------------- Layer 1
class Subject(BaseModel):
    ref: str
    category: SubjectCategory
    role: SubjectRole
    description: str = ""


OPTIONAL_FACTS = ("f_banner_prominent", "f_animals", "f_handover", "f_volunteer_giver", "f_logo", "f_orientation", "f_group")
"""Facts that a profile may stop asking (layered_final_v1 does, 2026-11-05). The model class lets them be absent, but a fact the profile DOES ask is still required: the check is
`missing_facts`, and it runs wherever an answer is parsed (annotation.parse_response, compact.expand_compact). With no profile every one of them is required, as it always was."""


class SceneFacts(BaseModel):
    """Layer-1 facts. All fields are required: an omitted fact is an error, never a silent default."""
    # (the sentence above is part of the exported JSON schema, which is pinned byte for byte, so it stays. In fact a profile may stop asking the seven OPTIONAL_FACTS: a fact
    # the profile asks is still required, see `missing_facts`.)

    f_photo: Tri
    f_single: Tri
    f_volunteers: Tri
    f_banner: Tri
    f_banner_prominent: Optional[Tri] = None
    f_beneficiaries: Tri
    f_animals: Optional[Tri] = None
    f_item_given: Tri
    f_handover: Optional[Tri] = None
    f_volunteer_giver: Optional[Tri] = None
    f_logo: Optional[Tri] = None
    f_medical: Tri
    f_orientation: Optional[Orientation] = None
    f_group: Optional[GroupType] = None
    f_activity_type: str
    f_gaze_tag: Optional[GazeTag] = None  # asked only by profiles that have the F_GAZE_TAG fact (v2); absent from v1 responses


def missing_facts(facts: "SceneFacts", asked: "set[str] | None" = None) -> list[str]:
    """The optional facts that were asked (`asked` = their field names; None = all of them) but are absent from the answer. A profile that does not ask a fact never fails for it."""
    names = OPTIONAL_FACTS if asked is None else [f for f in OPTIONAL_FACTS if f in asked]
    return [f for f in names if getattr(facts, f) is None]


class SceneUnderstanding(BaseModel):
    activity_description: str
    scene_facts: SceneFacts
    subjects: list[Subject] = Field(default_factory=list)
    candidate_applicable_rule_groups: list[str] = Field(default_factory=list)  # advisory only; Python decides applicability
    uncertainties: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------------- Layers 2 and 3
class CriterionAnswer(BaseModel):
    criterion_id: str
    status: Status
    evidence: str = ""
    confidence: Confidence = Confidence.MEDIUM
    source: str = "gemini"  # a computer-vision component would write "cv"


class DimensionAssessment(BaseModel):
    applicable: bool = True
    score: Optional[int] = Field(default=None, ge=0, le=4)
    explanation: str = ""
    positive_observations: list[str] = Field(default_factory=list)
    negative_observations: list[str] = Field(default_factory=list)
    evidence: str = ""
    confidence: Confidence = Confidence.MEDIUM

    @model_validator(mode="after")
    def _score_needed_when_applicable(self):
        if self.applicable and self.score is None:
            raise ValueError("an applicable dimension needs a score 0-4")
        return self


# --------------------------------------------------------------------------------------------- Layer 4 (visual flags only)
class VisualFlag(BaseModel):
    flag_type: str
    evidence: str = ""
    confidence: Confidence = Confidence.MEDIUM


class Observation(BaseModel):
    text: str
    evidence: str = ""


class FactorProposal(BaseModel):
    name: str
    description: str = ""
    why_it_matters: str = ""
    evidence: str = ""


class ModelAssessment(BaseModel):
    """The strict structure the model must return. Aggregates it reports (overall status, final score) are NOT trusted."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    schema_version: str = SCHEMA_VERSION
    image_id: str
    layer_1_scene_understanding: SceneUnderstanding
    layer_2_criteria: list[CriterionAnswer]
    layer_3_quality_dimensions: dict[str, DimensionAssessment]
    layer_4_visual_flags: list[VisualFlag] = Field(default_factory=list)
    positive_attributes: list[Observation] = Field(default_factory=list)
    potential_rejection_factors: list[Observation] = Field(default_factory=list)
    new_factor_proposals: list[FactorProposal] = Field(default_factory=list)
    model_recommendation: Optional[str] = None  # recorded for audit; never used
    model_overall_score: Optional[float] = None  # recorded for audit; never used

    @field_validator("layer_2_criteria")
    @classmethod
    def _unique_ids(cls, v):
        ids = [a.criterion_id for a in v]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise ValueError(f"criterion answered more than once: {sorted(dup)}")
        return v


# --------------------------------------------------------------------------------------------- metadata (never from the model)
class Metadata(BaseModel):
    """Verified information that cannot be seen in a photograph. Missing values are PENDING, never VERIFIED."""

    model_config = ConfigDict(extra="allow")

    width: Optional[int] = None
    height: Optional[int] = None
    platform: Optional[str] = None
    consent_status: Optional[str] = None
    rights_status: Optional[str] = None
    child_safeguarding_status: Optional[str] = None
    caption_status: Optional[str] = None
    event_details_status: Optional[str] = None
    credits_rights_status: Optional[str] = None
    event_claims_asserted: bool = False
    direct_gaze_approved: bool = False

    FORBIDDEN_KEYS: ClassVar[tuple] = ("label", "historical_label", "historical_decision", "approved", "rejected")


# --------------------------------------------------------------------------------------------- computed results
class CriterionResult(BaseModel):
    criterion_id: str
    applicable: Optional[bool]  # None = the facts needed to decide are UNKNOWN (treated as applicable)
    model_status: Optional[Status] = None
    status: Status  # the effective status after applicability, exceptions and safety rules
    evidence: str = ""
    confidence: Confidence = Confidence.MEDIUM
    notes: list[str] = Field(default_factory=list)  # synthesized / na_conflict / exception:<id> / advisory_ignored


class GateResult(BaseModel):
    gate_id: str
    criterion_id: str
    source: str
    parent_rule_id: Optional[str]
    hard: bool  # False when the active profile treats the gate as soft
    status: Status
    applicable: Optional[bool]
    evidence: str = ""
    confidence: Confidence = Confidence.MEDIUM
    failure_reason: Optional[str] = None
    components: list[CriterionResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class QualityResult(BaseModel):
    computed: bool
    overall_score: Optional[float] = None
    band: Optional[str] = None
    dimension_scores: dict[str, Optional[float]] = Field(default_factory=dict)
    caps_applied: list[str] = Field(default_factory=list)
    applicable_weight: float = 0.0
    incomplete: bool = False
    severe_weakness: Optional[str] = None  # "UNUSABLE:<dim>" or "POOR:<dim>"
    warnings: list[str] = Field(default_factory=list)
    calibrated: bool = False


class PublicationCheck(BaseModel):
    check_id: str
    label: str
    status: str
    detail: str = ""


class PublicationResult(BaseModel):
    status: str  # READY / PENDING_VERIFICATION / BLOCKED / NOT_EVALUATED
    checks: list[PublicationCheck] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    pending: list[str] = Field(default_factory=list)
    review_notes: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    recommendation: Recommendation
    photo_approval_status: str  # APPROVED / REJECTED / REVIEW_REQUIRED / RESELECT
    publication_readiness_status: str
    decision_reasons: list[str] = Field(default_factory=list)
    human_review_questions: list[str] = Field(default_factory=list)
    gates: list[GateResult] = Field(default_factory=list)
    failed_gates: list[str] = Field(default_factory=list)
    uncertain_gates: list[str] = Field(default_factory=list)
    not_applicable_gates: list[str] = Field(default_factory=list)
    quality: QualityResult = Field(default_factory=lambda: QualityResult(computed=False))
    publication: PublicationResult = Field(default_factory=lambda: PublicationResult(status="NOT_EVALUATED"))
    warnings: list[str] = Field(default_factory=list)
    policy_validation_required: bool = False
    versions: dict[str, Any] = Field(default_factory=dict)
