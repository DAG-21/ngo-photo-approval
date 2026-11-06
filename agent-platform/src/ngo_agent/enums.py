"""Controlled vocabularies. Stored as plain strings in the database (portable) and validated here."""

from __future__ import annotations

from enum import Enum


class _S(str, Enum):
    def __str__(self) -> str:  # noqa: D105
        return self.value


class PhotoStatus(_S):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    AI_EVALUATED = "AI_EVALUATED"
    PENDING_REVIEW = "PENDING_REVIEW"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    REVIEW_IN_PROGRESS = "REVIEW_IN_PROGRESS"
    APPROVED = "APPROVED"            # a person's final decision: approve (nothing is published automatically)
    REJECTED = "REJECTED"            # a person's final decision: reject
    FAILED = "FAILED"


class ProcessingStatus(_S):
    PROCESSING = "PROCESSING"
    AI_EVALUATED = "AI_EVALUATED"
    FAILED = "FAILED"


class ReviewStatus(_S):
    REVIEW_IN_PROGRESS = "REVIEW_IN_PROGRESS"
    FEEDBACK_SUBMITTED = "FEEDBACK_SUBMITTED"


class AiRecommendation(_S):
    APPROVE = "APPROVE"
    APPROVED_PENDING_PUBLICATION_CHECKS = "APPROVED_PENDING_PUBLICATION_CHECKS"
    REJECT = "REJECT"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    RESELECT = "RESELECT"


AI_APPROVING = {AiRecommendation.APPROVE.value, AiRecommendation.APPROVED_PENDING_PUBLICATION_CHECKS.value}
AI_DEFERRING = {AiRecommendation.HUMAN_REVIEW.value}   # only the older profiles still produce this; a re-select (poor quality) is a disapproval, so a reviewer who approves it DISAGREES with the AI and the loop can learn from it


class HumanDecision(_S):
    APPROVE = "APPROVE"
    REJECT = "REJECT"


class Agreement(_S):
    AGREE = "AGREE"
    DISAGREE = "DISAGREE"
    AI_DEFERRED = "AI_DEFERRED"      # the engine itself asked for a person, so there is no AI verdict to agree with


class Role(_S):
    REVIEWER = "REVIEWER"
    ORG_ADMIN = "ORG_ADMIN"


ROLE_RANK = {Role.REVIEWER.value: 1, Role.ORG_ADMIN.value: 2}


class CriterionType(_S):
    SCENE_FACT = "SCENE_FACT"
    ACTIVITY_TYPE = "ACTIVITY_TYPE"
    MANDATORY_GATE = "MANDATORY_GATE"
    CRITERION = "CRITERION"
    QUALITY_DIMENSION = "QUALITY_DIMENSION"
    TAG_ADDED = "TAG_ADDED"
    TAG_REMOVED = "TAG_REMOVED"
    EDITORIAL_OBSERVATION = "EDITORIAL_OBSERVATION"
    PUBLICATION = "PUBLICATION"
    FINAL_DECISION = "FINAL_DECISION"


class CorrectionReason(_S):
    VISUAL_PERCEPTION_ERROR = "VISUAL_PERCEPTION_ERROR"
    SUBJECT_IDENTIFICATION_ERROR = "SUBJECT_IDENTIFICATION_ERROR"
    ACTIVITY_CLASSIFICATION_ERROR = "ACTIVITY_CLASSIFICATION_ERROR"
    RULE_APPLICABILITY_ERROR = "RULE_APPLICABILITY_ERROR"
    RULE_INTERPRETATION_ERROR = "RULE_INTERPRETATION_ERROR"
    EXCEPTION_NOT_RECOGNISED = "EXCEPTION_NOT_RECOGNISED"
    EDITORIAL_DISAGREEMENT = "EDITORIAL_DISAGREEMENT"
    MISSING_CONTEXT = "MISSING_CONTEXT"
    POLICY_CHANGE = "POLICY_CHANGE"
    HISTORICAL_DECISION_DISAGREEMENT = "HISTORICAL_DECISION_DISAGREEMENT"
    HUMAN_REVIEW_ERROR = "HUMAN_REVIEW_ERROR"
    OTHER = "OTHER"


# reasons that never produce a learning example: a policy change belongs to the rulebook approval process,
# and a self-declared reviewer error is, by its own account, not trustworthy
NOT_FOR_LEARNING = {CorrectionReason.POLICY_CHANGE.value, CorrectionReason.HUMAN_REVIEW_ERROR.value}


class VersionApproval(_S):
    DRAFT = "DRAFT"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    RETIRED = "RETIRED"


class FeedbackEventType(_S):
    REVIEW_OPENED = "REVIEW_OPENED"
    AI_ACCEPTED = "AI_ACCEPTED"
    AI_OVERRIDDEN = "AI_OVERRIDDEN"
    CRITERION_CORRECTED = "CRITERION_CORRECTED"
    TAG_ADDED = "TAG_ADDED"
    TAG_REMOVED = "TAG_REMOVED"
    ACTIVITY_CHANGED = "ACTIVITY_CHANGED"
    OBSERVATION_ADDED = "OBSERVATION_ADDED"
    CONTEXT_SUPPLIED = "CONTEXT_SUPPLIED"
    POLICY_AMBIGUITY_FLAGGED = "POLICY_AMBIGUITY_FLAGGED"
    FEEDBACK_SUBMITTED = "FEEDBACK_SUBMITTED"
    BLOCKER_ACKNOWLEDGED = "BLOCKER_ACKNOWLEDGED"   # a person approved a photograph although its evaluation lists publication blockers
    LEARNING_EXAMPLE_CREATED = "LEARNING_EXAMPLE_CREATED"


# lifecycle stages (system side), in the order of the specification
LIFECYCLE_STAGES = ["PHOTO_UPLOADED", "AI_EVALUATION_STARTED", "AI_EVALUATION_COMPLETED", "AI_RECOMMENDATION_GENERATED", "WAITING_FOR_HUMAN_REVIEW",
                    "HUMAN_REVIEW_IN_PROGRESS", "HUMAN_ACCEPTS_AI", "HUMAN_OVERRIDES_AI", "FEEDBACK_RECORDED",
                    "LEARNING_EXAMPLE_CREATED", "AI_EVALUATION_FAILED", "RE_EVALUATION_REQUESTED"]
