"""Pydantic schemas shared across the project (rules, model output, prediction response)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------- enums


class Label(StrEnum):
    APPROVED = "approved"
    NOT_APPROVED = "not_approved"
    PENDING = "pending"


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"


class Decision(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REVIEW = "REVIEW"


class RequirementType(StrEnum):
    MANDATORY = "mandatory"
    SCORED = "scored"


class CheckMethod(StrEnum):
    VLM = "vlm"
    PYTHON = "python"
    OCR = "ocr"
    DETECTOR = "detector"


class UncertainPolicy(StrEnum):
    REJECT = "reject"
    REVIEW = "review"
    SCORE_ONLY = "score_only"  # only valid for scored (non-mandatory) rules


class ProbabilityType(StrEnum):
    CALIBRATED = "calibrated"
    UNCALIBRATED_CLASSIFIER = "uncalibrated_classifier"
    HEURISTIC = "heuristic"
    NONE = "none"


class ReasonCode(StrEnum):
    MANDATORY_RULE_FAILED = "MANDATORY_RULE_FAILED"
    MANDATORY_RULE_UNCERTAIN = "MANDATORY_RULE_UNCERTAIN"
    RULE_RESULT_MISSING = "RULE_RESULT_MISSING"
    RULE_RESULT_DUPLICATED = "RULE_RESULT_DUPLICATED"
    RULE_RESULT_UNKNOWN = "RULE_RESULT_UNKNOWN"
    RESULT_INVALID = "RESULT_INVALID"
    SCORED_RULE_FAILED = "SCORED_RULE_FAILED"
    SCORED_RULE_UNCERTAIN = "SCORED_RULE_UNCERTAIN"
    BELOW_THRESHOLD = "BELOW_THRESHOLD"
    PROBABILITY_UNAVAILABLE = "PROBABILITY_UNAVAILABLE"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    INVALID_IMAGE = "INVALID_IMAGE"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    APPROVED = "APPROVED"


# --------------------------------------------------------------------------- rules


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(pattern=r"^R\d{3,}$")
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    original_text: str = ""
    # Identifier of the item in the client's rules document (section title / "Final QC Test Q<n>").
    source_ref: str = ""
    description: str
    visual_evidence: str = ""
    # Condition under which the rule applies at all ("a banner is visible"). When it is not met the
    # analyzer returns PASS with evidence starting "N/A:". Empty = always applies.
    applies_when: str = ""
    # Extra instruction from the NGO team that is NOT in the client's rules document (e.g. a clarification found
    # while testing). It is sent to the model separately from the client wording and clearly labelled as such;
    # `original_text` stays verbatim. Empty = none.
    additional_guidance: str = ""
    requirement_type: RequirementType
    mandatory_status_confirmed: bool = False
    check_method: CheckMethod = CheckMethod.VLM
    enabled: bool = True
    uncertain_policy: UncertainPolicy = UncertainPolicy.REJECT
    applies_to_programs: list[str] = Field(default_factory=list)
    confirmed: bool = False
    needs_manual_annotation: bool = True
    limitations: str = ""
    notes: str = ""

    @model_validator(mode="after")
    def _mandatory_cannot_be_score_only(self) -> Rule:
        if (
            self.requirement_type == RequirementType.MANDATORY
            and self.uncertain_policy == UncertainPolicy.SCORE_ONLY
        ):
            raise ValueError(
                f"{self.rule_id}: a mandatory rule cannot use uncertain_policy=score_only "
                "(an uncertain result must never count as a pass)."
            )
        return self


class RuleSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_set_version: str
    rule_set_confirmed: bool = False
    rules: list[Rule]

    @model_validator(mode="after")
    def _unique_ids(self) -> RuleSet:
        ids = [r.rule_id for r in self.rules]
        names = [r.name for r in self.rules]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate rule_id in rule set: {sorted({i for i in ids if ids.count(i) > 1})}")
        if len(set(names)) != len(names):
            raise ValueError("duplicate rule name in rule set")
        return self

    def active(self, program: str | None = None) -> list[Rule]:
        """Enabled rules that apply to ``program``.

        A rule with no program list applies everywhere. If the program is unknown (``None``),
        program-restricted rules are kept: evaluating more rules is the stricter, safer choice.
        """
        out = []
        for r in self.rules:
            if not r.enabled:
                continue
            if r.applies_to_programs and program is not None and program not in r.applies_to_programs:
                continue
            out.append(r)
        return out

    def by_id(self, rule_id: str) -> Rule:
        for r in self.rules:
            if r.rule_id == rule_id:
                return r
        raise KeyError(rule_id)


# --------------------------------------------------------------------------- model output


class RuleCheck(BaseModel):
    """One rule result as returned by a visual analyzer."""

    model_config = ConfigDict(extra="ignore")

    rule_id: str
    status: CheckStatus
    evidence: str = ""

    @field_validator("status", mode="before")
    @classmethod
    def _norm_status(cls, v: object) -> object:
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence(cls, v: object) -> object:
        return "" if v is None else v


class VisualAnalysis(BaseModel):
    """Structured output of a visual analyzer for one image."""

    model_config = ConfigDict(extra="ignore")

    image_id: str | None = None
    checks: list[RuleCheck]
    # Self-reported confidence is recorded but is NEVER a calibrated probability.
    self_reported_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class UsageInfo(BaseModel):
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_s: float | None = None
    estimated_cost_usd: float | None = None


# --------------------------------------------------------------------------- policy output


class RuleIssue(BaseModel):
    rule_id: str | None
    reason_code: ReasonCode
    detail: str = ""


class PolicyResult(BaseModel):
    """Outcome of the deterministic rule engine (independent of the probability layer)."""

    rule_set_version: str
    checks: list[RuleCheck]
    hard_rule_failures: list[str] = Field(default_factory=list)
    uncertain_rules: list[str] = Field(default_factory=list)
    missing_rules: list[str] = Field(default_factory=list)
    unmet_scored_rules: list[str] = Field(default_factory=list)
    issues: list[RuleIssue] = Field(default_factory=list)
    # "reject" / "review" / "pass": whether rules alone allow proceeding to the threshold step.
    rule_outcome: Literal["reject", "review", "pass"]
    reason_codes: list[ReasonCode] = Field(default_factory=list)


# --------------------------------------------------------------------------- API/service response


class CheckOut(BaseModel):
    rule_id: str
    status: CheckStatus
    evidence: str = ""


class PredictionResponse(BaseModel):
    """Public response contract of the prediction service / API."""

    model_config = ConfigDict(protected_namespaces=())

    image_id: str
    decision: Decision
    approval_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    approval_percentage: float | None = Field(default=None, ge=0.0, le=100.0)
    probability_type: ProbabilityType = ProbabilityType.NONE
    threshold: float = Field(ge=0.0, le=1.0)
    hard_rule_failures: list[str] = Field(default_factory=list)
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    checks: list[CheckOut] = Field(default_factory=list)
    rule_set_version: str
    rule_set_confirmed: bool = False
    model_id: str | None = None
    model_version: str | None = None
    scoring_model_version: str | None = None
    raw_classifier_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    # Fraction of active rules passed, 0-100. A heuristic summary, NOT a probability.
    heuristic_score: float | None = Field(default=None, ge=0.0, le=100.0)
    self_reported_confidence_uncalibrated: float | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None

    @model_validator(mode="after")
    def _consistent_percentage(self) -> PredictionResponse:
        if self.approval_probability is None:
            if self.approval_percentage is not None:
                raise ValueError("approval_percentage set without approval_probability")
        else:
            expected = round(self.approval_probability * 100.0, 4)
            if self.approval_percentage is None or abs(self.approval_percentage - expected) > 1e-3:
                self.approval_percentage = expected
        return self
