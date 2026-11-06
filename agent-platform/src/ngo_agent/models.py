"""ORM models for Phase 2A. Design and rationale: docs/PHASE2A_DESIGN.md."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, ForeignKeyConstraint, Index, Integer, String, Text, UniqueConstraint, false, text)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

Json = JSON().with_variant(JSONB(), "postgresql")


def new_id() -> str:
    return str(uuid.uuid4())


def now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


def _id() -> Mapped[str]:
    return mapped_column(String(36), primary_key=True, default=new_id)


def _ts(**kw: Any) -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), default=now, **kw)


class Organisation(Base):
    __tablename__ = "organisations"
    organisation_id: Mapped[str] = _id()
    name: Mapped[str] = mapped_column(String(200), unique=True)
    configuration: Mapped[dict] = mapped_column(Json, default=dict)
    active_rulebook_version_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)  # set by activation; not an FK, to avoid a cycle
    active_prompt_version_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = _ts()
    updated_at: Mapped[datetime] = _ts()


class User(Base):
    __tablename__ = "users"
    user_id: Mapped[str] = _id()
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    display_name: Mapped[str] = mapped_column(String(200), default="")
    password_hash: Mapped[str] = mapped_column(String(300))
    role: Mapped[str] = mapped_column(String(20))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = _ts()


class Photo(Base):
    __tablename__ = "photos"
    __table_args__ = (UniqueConstraint("organisation_id", "file_hash", name="uq_photo_org_hash"), UniqueConstraint("photo_id", "organisation_id", name="uq_photo_org"))
    photo_id: Mapped[str] = _id()
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    image_storage_path: Mapped[str] = mapped_column(String(500))
    original_filename: Mapped[str] = mapped_column(String(300))
    file_hash: Mapped[str] = mapped_column(String(64))
    uploaded_by: Mapped[str] = mapped_column(ForeignKey("users.user_id"))
    uploaded_at: Mapped[datetime] = _ts()
    event_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    capture_date: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    file_format: Mapped[str] = mapped_column(String(20))
    image_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    image_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    photo_status: Mapped[str] = mapped_column(String(30), default="UPLOADED", index=True)
    photo_metadata: Mapped[dict] = mapped_column("metadata", Json, default=dict)


class _Version:
    version_id: Mapped[str] = _id()
    version_number: Mapped[str] = mapped_column(String(80))
    configuration: Mapped[dict] = mapped_column(Json, default=dict)
    change_description: Mapped[str] = mapped_column(Text, default="")
    approval_status: Mapped[str] = mapped_column(String(20), default="DRAFT")
    created_at: Mapped[datetime] = _ts()
    effective_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class RulebookVersion(_Version, Base):
    __tablename__ = "rulebook_versions"
    __table_args__ = (UniqueConstraint("organisation_id", "version_number", name="uq_rulebook_org_number"),)
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    created_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.user_id"), nullable=True)
    approved_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.user_id"), nullable=True)


class PromptVersion(_Version, Base):
    __tablename__ = "prompt_versions"
    __table_args__ = (UniqueConstraint("organisation_id", "version_number", name="uq_prompt_org_number"),)
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    created_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.user_id"), nullable=True)
    approved_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.user_id"), nullable=True)


class EvaluationRun(Base):
    """One AI evaluation. Written once: created PROCESSING, filled exactly once, then frozen by a database trigger. Never deleted."""

    __tablename__ = "evaluation_runs"
    __table_args__ = (ForeignKeyConstraint(["photo_id", "organisation_id"], ["photos.photo_id", "photos.organisation_id"]),
                      UniqueConstraint("evaluation_id", "organisation_id", name="uq_eval_org"))
    evaluation_id: Mapped[str] = _id()
    photo_id: Mapped[str] = mapped_column(String(36), index=True)
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    model_provider: Mapped[str] = mapped_column(String(60), default="")
    model_id: Mapped[str] = mapped_column(String(100), default="")
    prompt_version_id: Mapped[str] = mapped_column(ForeignKey("prompt_versions.version_id"))
    rulebook_version_id: Mapped[str] = mapped_column(ForeignKey("rulebook_versions.version_id"))
    evaluation_profile: Mapped[str] = mapped_column(String(100), default="")
    raw_response_location: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    raw_response_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    validated_response: Mapped[Optional[dict]] = mapped_column(Json, nullable=True)
    layer_1_output: Mapped[Optional[dict]] = mapped_column(Json, nullable=True)
    layer_2_output: Mapped[Optional[dict]] = mapped_column(Json, nullable=True)
    layer_3_output: Mapped[Optional[dict]] = mapped_column(Json, nullable=True)
    layer_4_output: Mapped[Optional[dict]] = mapped_column(Json, nullable=True)
    decision_output: Mapped[Optional[dict]] = mapped_column(Json, nullable=True)
    ai_recommendation: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    processing_status: Mapped[str] = mapped_column(String(20), default="PROCESSING", index=True)
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    estimated_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    latency: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    warnings: Mapped[Optional[list]] = mapped_column(Json, nullable=True)
    few_shot: Mapped[Optional[dict]] = mapped_column(Json, nullable=True)   # the similar past decisions shown to the model, if any (L2); part of the content hash when present
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    requested_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.user_id"), nullable=True)
    created_at: Mapped[datetime] = _ts()
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class HumanReview(Base):
    __tablename__ = "human_reviews"
    __table_args__ = (
        ForeignKeyConstraint(["evaluation_id", "organisation_id"], ["evaluation_runs.evaluation_id", "evaluation_runs.organisation_id"]),
        UniqueConstraint("review_id", "organisation_id", name="uq_review_org"),
        Index("uq_one_open_review", "evaluation_id", "reviewer_id", unique=True, sqlite_where=text("review_status = 'REVIEW_IN_PROGRESS'"),
              postgresql_where=text("review_status = 'REVIEW_IN_PROGRESS'")),
    )
    review_id: Mapped[str] = _id()
    photo_id: Mapped[str] = mapped_column(String(36), index=True)
    evaluation_id: Mapped[str] = mapped_column(String(36), index=True)
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"))
    review_status: Mapped[str] = mapped_column(String(30), default="REVIEW_IN_PROGRESS")
    human_final_decision: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    overall_agreement_with_ai: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    reason_categories: Mapped[Optional[list]] = mapped_column(Json, nullable=True)
    reviewer_comments: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    idempotency_key: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    submission_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = _ts()


class CriterionCorrection(Base):
    """One correction of one criterion. A row is frozen by a trigger: changing your mind is a new revision, never an edit."""

    __tablename__ = "criterion_corrections"
    __table_args__ = (
        ForeignKeyConstraint(["review_id", "organisation_id"], ["human_reviews.review_id", "human_reviews.organisation_id"]),
        UniqueConstraint("review_id", "criterion_id", "criterion_type", "revision", name="uq_correction_revision"),
    )
    correction_id: Mapped[str] = _id()
    review_id: Mapped[str] = mapped_column(String(36), index=True)
    evaluation_id: Mapped[str] = mapped_column(ForeignKey("evaluation_runs.evaluation_id"), index=True)
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    criterion_id: Mapped[str] = mapped_column(String(120))
    criterion_type: Mapped[str] = mapped_column(String(30))
    ai_predicted_value: Mapped[Optional[Any]] = mapped_column(Json, nullable=True)
    human_corrected_value: Mapped[Optional[Any]] = mapped_column(Json, nullable=True)
    correction_reason: Mapped[list] = mapped_column(Json)
    reviewer_comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    supersedes_correction_id: Mapped[Optional[str]] = mapped_column(ForeignKey("criterion_corrections.correction_id"), nullable=True)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.user_id"))
    created_at: Mapped[datetime] = _ts()


class FeedbackEvent(Base):
    __tablename__ = "feedback_events"
    event_id: Mapped[str] = _id()
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    review_id: Mapped[Optional[str]] = mapped_column(String(36), index=True, nullable=True)
    evaluation_id: Mapped[Optional[str]] = mapped_column(String(36), index=True, nullable=True)
    photo_id: Mapped[Optional[str]] = mapped_column(String(36), index=True, nullable=True)
    event_type: Mapped[str] = mapped_column(String(40))
    actor_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.user_id"), nullable=True)
    payload: Mapped[dict] = mapped_column(Json, default=dict)
    created_at: Mapped[datetime] = _ts()


class LifecycleEvent(Base):
    __tablename__ = "lifecycle_events"
    event_id: Mapped[str] = _id()
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    photo_id: Mapped[str] = mapped_column(String(36), index=True)
    evaluation_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    stage: Mapped[str] = mapped_column(String(40))
    from_status: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    to_status: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    payload: Mapped[dict] = mapped_column(Json, default=dict)
    created_at: Mapped[datetime] = _ts()


class PhotoEmbedding(Base):
    """An image embedding of one photograph, made by one named model. Immutable: another model means another row. Used only to find similar photographs."""

    __tablename__ = "photo_embeddings"
    __table_args__ = (UniqueConstraint("photo_id", "model", name="uq_embedding_photo_model"),
                      ForeignKeyConstraint(["photo_id", "organisation_id"], ["photos.photo_id", "photos.organisation_id"]))
    embedding_id: Mapped[str] = _id()
    photo_id: Mapped[str] = mapped_column(String(36), index=True)
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    model: Mapped[str] = mapped_column(String(80))
    dim: Mapped[int] = mapped_column(Integer)
    vector: Mapped[list] = mapped_column(Json)
    created_at: Mapped[datetime] = _ts()


class ExampleFlag(Base):
    """Append-only: a person marks a photograph as misleading (never to be shown to the model as an example) or clears that mark. The latest row wins."""

    __tablename__ = "example_flags"
    __table_args__ = (ForeignKeyConstraint(["photo_id", "organisation_id"], ["photos.photo_id", "photos.organisation_id"]),)
    flag_id: Mapped[str] = _id()
    photo_id: Mapped[str] = mapped_column(String(36), index=True)
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    action: Mapped[str] = mapped_column(String(10))      # FLAG or UNFLAG
    reason: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(ForeignKey("users.user_id"))
    created_at: Mapped[datetime] = _ts()


class GapAnalysis(Base):
    """What the person said that the AI had not (L3). Append-only; one row per review and analyzer. Text-only: the analyzer never sees the photograph."""

    __tablename__ = "gap_analyses"
    __table_args__ = (UniqueConstraint("review_id", "analyzer", name="uq_gap_review_analyzer"),
                      ForeignKeyConstraint(["review_id", "organisation_id"], ["human_reviews.review_id", "human_reviews.organisation_id"]))
    gap_id: Mapped[str] = _id()
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    review_id: Mapped[str] = mapped_column(String(36), index=True)
    evaluation_id: Mapped[str] = mapped_column(String(36))
    photo_id: Mapped[str] = mapped_column(String(36), index=True)
    analyzer: Mapped[str] = mapped_column(String(60))
    model_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(10))            # DONE or FAILED
    agreement: Mapped[str] = mapped_column(String(20), default="")
    low_information: Mapped[bool] = mapped_column(Boolean, default=False)
    gaps: Mapped[list] = mapped_column(Json, default=list)
    summary: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = _ts()


class ImprovementCycle(Base):
    """One run of the learning loop (L4). The only mutable row of the loop: its status moves from RUNNING to a final one. Never deleted."""

    __tablename__ = "improvement_cycles"
    __table_args__ = (UniqueConstraint("organisation_id", "ordinal", name="uq_cycle_org_ordinal"), UniqueConstraint("cycle_id", "organisation_id", name="uq_cycle_org"))
    cycle_id: Mapped[str] = _id()
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)                       # 1, 2, 3 ... per organisation; the unique key also stops two workers running the same cycle
    decided_count: Mapped[int] = mapped_column(Integer)                 # decided photographs when it started
    trigger: Mapped[str] = mapped_column(String(10))                    # AUTO or MANUAL
    status: Mapped[str] = mapped_column(String(30), default="RUNNING")
    summary: Mapped[dict] = mapped_column(Json, default=dict)
    started_at: Mapped[datetime] = _ts()
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class ChangeRecord(Base):
    """One drafted change to the settings (L4), in the engine's overlay format. Append-only; its status history is in ChangeEvent."""

    __tablename__ = "change_records"
    __table_args__ = (UniqueConstraint("organisation_id", "label", name="uq_change_org_label"), UniqueConstraint("organisation_id", "fingerprint", name="uq_change_org_fingerprint"),
                      UniqueConstraint("change_id", "organisation_id", name="uq_change_org"),
                      ForeignKeyConstraint(["cycle_id", "organisation_id"], ["improvement_cycles.cycle_id", "improvement_cycles.organisation_id"]))
    change_id: Mapped[str] = _id()
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    cycle_id: Mapped[str] = mapped_column(String(36))
    label: Mapped[str] = mapped_column(String(20))                      # L4-0001: also the change_id inside the overlay
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)    # the same change is never drafted twice
    kind: Mapped[str] = mapped_column(String(30))
    target: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    loosening: Mapped[bool] = mapped_column(Boolean, default=False)
    record: Mapped[dict] = mapped_column(Json)
    rationale: Mapped[str] = mapped_column(Text, default="")
    evidence: Mapped[dict] = mapped_column(Json, default=dict)
    created_at: Mapped[datetime] = _ts()


class ChangeEvent(Base):
    """Append-only status history of a change: DRAFTED, HELD, BLOCKED, PASSED_NOT_APPLIED, APPLIED, MONITORED_OK, ROLLED_BACK. The latest event is its status."""

    __tablename__ = "change_events"
    __table_args__ = (ForeignKeyConstraint(["change_id", "organisation_id"], ["change_records.change_id", "change_records.organisation_id"]),
                      UniqueConstraint("change_id", "seq", name="uq_change_event_seq"))
    event_id: Mapped[str] = _id()
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    change_id: Mapped[str] = mapped_column(String(36), index=True)
    seq: Mapped[int] = mapped_column(Integer)                           # 1, 2, 3 ... per change: the strict order of its history (the status is the highest)
    status: Mapped[str] = mapped_column(String(24))
    reason: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict] = mapped_column(Json, default=dict)
    rulebook_version_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.user_id"), nullable=True)    # a person for a manual rollback, none for the loop itself
    created_at: Mapped[datetime] = _ts()


class ChangeTest(Base):
    """Append-only: one test of a change on the stored human decisions (replay of the stored answers, no model call)."""

    __tablename__ = "change_tests"
    __table_args__ = (ForeignKeyConstraint(["change_id", "organisation_id"], ["change_records.change_id", "change_records.organisation_id"]),)
    test_id: Mapped[str] = _id()
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    change_id: Mapped[str] = mapped_column(String(36), index=True)
    cycle_id: Mapped[str] = mapped_column(String(36))
    method: Mapped[str] = mapped_column(String(20), default="replay")
    metrics: Mapped[dict] = mapped_column(Json, default=dict)
    flips: Mapped[dict] = mapped_column(Json, default=dict)
    guardrails: Mapped[dict] = mapped_column(Json, default=dict)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = _ts()


class LearningExample(Base):
    __tablename__ = "learning_examples"
    __table_args__ = (UniqueConstraint("human_review_id", name="uq_learning_one_per_review"),
                      ForeignKeyConstraint(["photo_id", "organisation_id"], ["photos.photo_id", "photos.organisation_id"]))
    learning_example_id: Mapped[str] = _id()
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    photo_id: Mapped[str] = mapped_column(String(36))
    original_evaluation_id: Mapped[str] = mapped_column(ForeignKey("evaluation_runs.evaluation_id"))
    human_review_id: Mapped[str] = mapped_column(ForeignKey("human_reviews.review_id"))
    verified_decision: Mapped[str] = mapped_column(String(20))
    verified_tags: Mapped[list] = mapped_column(Json, default=list)
    verified_gate_results: Mapped[dict] = mapped_column(Json, default=dict)
    verified_reasons: Mapped[list] = mapped_column(Json, default=list)
    source: Mapped[str] = mapped_column(String(40), default="HUMAN_REVIEW")
    active_for_retrieval: Mapped[bool] = mapped_column(Boolean, default=False)  # currently usable as an example: on while the NGO has similar-photo examples switched on
    low_information: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())   # the written reason was too short or generic to teach anything: never shown as an example
    created_at: Mapped[datetime] = _ts()


class NoteDraft(Base):
    """One drafted prompt note (L5): what the drafter proposed for one gate after one decision, and how the checks judged it. Append-only; its status history is in NoteEvent."""

    __tablename__ = "note_drafts"
    __table_args__ = (UniqueConstraint("draft_id", "organisation_id", name="uq_note_org"), UniqueConstraint("review_id", "gate_id", name="uq_note_review_gate"),
                      ForeignKeyConstraint(["review_id", "organisation_id"], ["human_reviews.review_id", "human_reviews.organisation_id"]))
    draft_id: Mapped[str] = _id()
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    review_id: Mapped[str] = mapped_column(String(36), index=True)
    evaluation_id: Mapped[str] = mapped_column(String(36))
    photo_id: Mapped[str] = mapped_column(String(36))
    gate_id: Mapped[str] = mapped_column(String(10))
    outcome: Mapped[str] = mapped_column(String(24))                    # GATE_NOTE, SETTING_SUGGESTION, NO_CHANGE or ESCALATE
    direction: Mapped[str] = mapped_column(String(8), default="NONE")   # LOOSEN, TIGHTEN or NONE: whether the note makes the gate pass more or less often
    note_text: Mapped[str] = mapped_column(Text, default="")
    rationale: Mapped[str] = mapped_column(Text, default="")
    contradicts_client_wording: Mapped[bool] = mapped_column(Boolean, default=False)
    client_gate: Mapped[bool] = mapped_column(Boolean, default=False)   # the gate asks a question from the client's own rules
    flags: Mapped[list] = mapped_column(Json, default=list)             # e.g. LOOSENS_CLIENT_GATE
    checks: Mapped[dict] = mapped_column(Json, default=dict)            # {"passed": bool, "failed": [...]}
    source: Mapped[dict] = mapped_column(Json, default=dict)            # the mapped record the draft was written from (summary, rule note, confidence)
    model_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = _ts()


class NoteEvent(Base):
    """Append-only status history of a drafted note: DRAFTED, REJECTED, HELD, APPLIED, CONFIRMED, SUPERSEDED, WITHDRAWN. The highest seq is its status."""

    __tablename__ = "note_events"
    __table_args__ = (ForeignKeyConstraint(["draft_id", "organisation_id"], ["note_drafts.draft_id", "note_drafts.organisation_id"]),
                      UniqueConstraint("draft_id", "seq", name="uq_note_event_seq"))
    event_id: Mapped[str] = _id()
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    draft_id: Mapped[str] = mapped_column(String(36), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict] = mapped_column(Json, default=dict)
    prompt_version_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.user_id"), nullable=True)   # a person for a manual withdrawal, none for the system itself
    created_at: Mapped[datetime] = _ts()


class RuleEdit(Base):
    """One drafted rewrite of a rule (L5): what the drafter proposed for one rule after one decision, and how the checks judged it. Append-only; its status history is in RuleEditEvent."""

    __tablename__ = "rule_edits"
    __table_args__ = (UniqueConstraint("edit_id", "organisation_id", name="uq_ruleedit_org"), UniqueConstraint("review_id", "target_id", name="uq_ruleedit_review_target"),
                      ForeignKeyConstraint(["review_id", "organisation_id"], ["human_reviews.review_id", "human_reviews.organisation_id"]))
    edit_id: Mapped[str] = _id()
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    review_id: Mapped[Optional[str]] = mapped_column(String(36), index=True, nullable=True)   # none for a rewrite an administrator entered by hand
    evaluation_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    photo_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    target_id: Mapped[str] = mapped_column(String(64))                  # a criterion id (its question) or a client rule id such as R004 (its text)
    target_kind: Mapped[str] = mapped_column(String(16))                # RULE_QUESTION, SCENE_FACT or CLIENT_RULE
    outcome: Mapped[str] = mapped_column(String(16))                    # UPDATE_RULE, NO_CHANGE or ERROR
    direction: Mapped[str] = mapped_column(String(8), default="NONE")   # LOOSEN, TIGHTEN, NEUTRAL or NONE: whether the new wording makes the rule pass more or less often
    previous_text: Mapped[str] = mapped_column(Text, default="")        # the wording in force when this was drafted
    new_text: Mapped[str] = mapped_column(Text, default="")
    rationale: Mapped[str] = mapped_column(Text, default="")
    keeps_meaning: Mapped[bool] = mapped_column(Boolean, default=False)
    gate_id: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)   # the gate built on this rule, if any
    hard_gate: Mapped[bool] = mapped_column(Boolean, default=False)     # that gate can reject a photograph
    client_sourced: Mapped[bool] = mapped_column(Boolean, default=False)   # the rule comes from the client's own rules
    flags: Mapped[list] = mapped_column(Json, default=list)             # e.g. LOOSENS_GATE, CLIENT_RULE
    checks: Mapped[dict] = mapped_column(Json, default=dict)            # {"passed": bool, "failed": [...]}
    source: Mapped[dict] = mapped_column(Json, default=dict)            # the mapped record the draft was written from (summary, rule note, confidence)
    model_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = _ts()


class RuleEditEvent(Base):
    """Append-only status history of a rule rewrite: DRAFTED, REJECTED, HELD, NO_CHANGE, APPLIED, CONFIRMED, SUPERSEDED, WITHDRAWN. The highest seq is its status."""

    __tablename__ = "rule_edit_events"
    __table_args__ = (ForeignKeyConstraint(["edit_id", "organisation_id"], ["rule_edits.edit_id", "rule_edits.organisation_id"]),
                      UniqueConstraint("edit_id", "seq", name="uq_ruleedit_event_seq"))
    event_id: Mapped[str] = _id()
    organisation_id: Mapped[str] = mapped_column(ForeignKey("organisations.organisation_id"), index=True)
    edit_id: Mapped[str] = mapped_column(String(36), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict] = mapped_column(Json, default=dict)
    prompt_version_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.user_id"), nullable=True)   # a person for a manual withdrawal, none for the system itself
    created_at: Mapped[datetime] = _ts()
