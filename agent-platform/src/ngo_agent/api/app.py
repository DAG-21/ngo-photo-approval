"""FastAPI application for Phase 2A. Authentication, organisation isolation and the review workflow.

Run: uvicorn --factory ngo_agent.api.app:create_default_app --port 8100   (needs DATABASE_URL, AGENT_JWT_SECRET, AGENT_STORAGE_DIR)
Every route except /auth/login and /health needs a bearer token. A record of another organisation is always 404 (never 403).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, Request, UploadFile
from layered_engine.prompt_builder import build_prompt
from layered_engine.rulebook import Rulebook
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from .. import evaluation_service as ev
from .. import example_service as ex_svc
from .. import gap_service as gaps
from .. import loop_activity
from .. import loop_service as loop
from .. import note_service as notes
from .. import reason_summary
from .. import rule_service as rule_svc
from .. import review_service as rv
from .. import versions as vs
from ..db import make_engine, make_session_factory
from ..enums import ROLE_RANK, Role
from ..errors import Conflict, Forbidden, Invalid, NotFound, ServiceError, Unauthorised
from ..evaluation_view import addressable_items, publication_blockers
from ..reason_quality import is_low_information
from ..rule_drafter import rule_catalogue
from ..models import (CriterionCorrection, EvaluationRun, FeedbackEvent, HumanReview, LearningExample, LifecycleEvent, Organisation, Photo, PhotoEmbedding, PromptVersion,
                      RulebookVersion, User)
from ..security import AuthError, RateLimiter, decode_token, hash_password, issue_token, jwt_secret, verify_password
from ..storage import LocalStorage, ObjectStorage

log = logging.getLogger("ngo_agent")
STATIC = Path(__file__).parent / "static"
_DUMMY_HASH = hash_password("not-a-real-password-for-timing")


# ------------------------------------------------------------------------------------------------------------------------------ request bodies
class LoginIn(BaseModel):
    email: str
    password: str


class CorrectionIn(BaseModel):
    criterion_type: str
    criterion_id: str
    human_corrected_value: Any = None
    correction_reason: list[str] = Field(default_factory=list)
    reviewer_comment: Optional[str] = None


class CorrectionsIn(BaseModel):
    corrections: list[CorrectionIn]


class SubmitIn(BaseModel):
    decision: str
    reason_categories: list[str] = Field(default_factory=list)   # optional extra
    comment: Optional[str] = None   # the written reason: required for every decision, whether or not the person agrees with the AI
    publication_blocker_statement: Optional[str] = None   # required to approve a photograph whose evaluation lists publication blockers


class FewShotIn(BaseModel):
    enabled: bool
    k: Optional[int] = None
    min_similarity: Optional[float] = None
    confirm_cost_increase: bool = False


class FlagIn(BaseModel):
    flagged: bool
    reason: str = ""


class LoopSettingsIn(BaseModel):
    enabled: Optional[bool] = None
    paused: Optional[bool] = None
    every_n: Optional[int] = None
    min_rows: Optional[int] = None
    min_pattern: Optional[int] = None
    pattern_share: Optional[float] = None
    holdout_pct: Optional[int] = None
    min_gain: Optional[int] = None
    min_rejected_for_loosening: Optional[int] = None
    monitor_min: Optional[int] = None
    monitor_window: Optional[int] = None
    error_margin: Optional[int] = None
    confirm_automatic_changes: bool = False


class RollbackIn(BaseModel):
    reason: str


class NoteSettingsIn(BaseModel):
    mode: Optional[str] = None
    paused: Optional[bool] = None
    max_notes: Optional[int] = None
    max_note_chars: Optional[int] = None
    watch_window: Optional[int] = None
    watch_min_before: Optional[int] = None
    watch_margin: Optional[float] = None
    confirm_ngo_informed: bool = False


class RuleSettingsIn(BaseModel):
    mode: Optional[str] = None
    paused: Optional[bool] = None
    max_edits: Optional[int] = None
    max_chars: Optional[int] = None
    watch_window: Optional[int] = None
    watch_min_before: Optional[int] = None
    watch_margin: Optional[float] = None
    confirm_ngo_informed: bool = False


class AnswerIn(BaseModel):
    accept: bool


class ManualRuleIn(BaseModel):
    rule_id: str
    wording: str
    reason: str
    direction: str = "NEUTRAL"


class ReEvalIn(BaseModel):
    rulebook_version_id: Optional[str] = None
    prompt_version_id: Optional[str] = None
    allow_unapproved: bool = False


class UserIn(BaseModel):
    email: str
    display_name: str = ""
    password: str
    role: str


class VersionIn(BaseModel):
    kind: str
    version_number: str
    configuration: dict[str, Any] = Field(default_factory=dict)
    change_description: str = ""


# ---------------------------------------------------------------------------------------------------------------------------------- serialisers
def iso(d: Any) -> Optional[str]:
    return d.isoformat() if d is not None else None


def ser_photo(p: Photo) -> dict[str, Any]:
    return {"photo_id": p.photo_id, "organisation_id": p.organisation_id, "original_filename": p.original_filename, "file_hash": p.file_hash, "photo_status": p.photo_status,
            "uploaded_at": iso(p.uploaded_at), "uploaded_by": p.uploaded_by, "event_id": p.event_id, "capture_date": p.capture_date, "file_format": p.file_format,
            "image_width": p.image_width, "image_height": p.image_height}


def ser_run_summary(r: EvaluationRun) -> dict[str, Any]:
    return {"evaluation_id": r.evaluation_id, "photo_id": r.photo_id, "processing_status": r.processing_status, "ai_recommendation": r.ai_recommendation, "quality_score": r.quality_score,
            "model_id": r.model_id, "model_provider": r.model_provider, "evaluation_profile": r.evaluation_profile, "rulebook_version_id": r.rulebook_version_id,
            "prompt_version_id": r.prompt_version_id, "created_at": iso(r.created_at), "completed_at": iso(r.completed_at), "error": r.error}


def ser_run(r: EvaluationRun, s: Session) -> dict[str, Any]:
    d = ser_run_summary(r)
    rbv, pmv = s.get(RulebookVersion, r.rulebook_version_id), s.get(PromptVersion, r.prompt_version_id)
    dec = r.decision_output or {}
    d.update({"rulebook_version": rbv.version_number if rbv else None, "prompt_version": pmv.version_number if pmv else None, "layer_1": r.layer_1_output, "layer_2": r.layer_2_output,
              "layer_3": r.layer_3_output, "layer_4": r.layer_4_output, "decision": {k: dec.get(k) for k in ("recommendation", "decision_reasons", "human_review_questions", "failed_gates",
                                                                                                         "uncertain_gates", "not_applicable_gates", "warnings", "policy_validation_required")},
              "warnings": r.warnings, "input_tokens": r.input_tokens, "output_tokens": r.output_tokens, "estimated_cost": r.estimated_cost, "latency": r.latency,
              "content_hash": r.content_hash, "publication_blockers": publication_blockers(r), "few_shot": r.few_shot,
              "correctable_items": [{"criterion_type": t, "criterion_id": i, "ai_value": v} for (t, i), v in sorted(addressable_items(r).items())] if r.processing_status == "AI_EVALUATED" else []})
    return d


def ser_correction(c: CriterionCorrection) -> dict[str, Any]:
    return {"correction_id": c.correction_id, "review_id": c.review_id, "evaluation_id": c.evaluation_id, "criterion_id": c.criterion_id, "criterion_type": c.criterion_type,
            "ai_predicted_value": c.ai_predicted_value, "human_corrected_value": c.human_corrected_value, "correction_reason": c.correction_reason, "reviewer_comment": c.reviewer_comment,
            "revision": c.revision, "supersedes_correction_id": c.supersedes_correction_id, "created_by": c.created_by, "created_at": iso(c.created_at)}


def ser_review(r: HumanReview, s: Session, with_detail: bool = True) -> dict[str, Any]:
    d = {"review_id": r.review_id, "photo_id": r.photo_id, "evaluation_id": r.evaluation_id, "organisation_id": r.organisation_id, "reviewer_id": r.reviewer_id,
         "review_status": r.review_status, "human_final_decision": r.human_final_decision, "overall_agreement_with_ai": r.overall_agreement_with_ai,
         "reason_categories": r.reason_categories, "reviewer_comments": r.reviewer_comments, "reviewed_at": iso(r.reviewed_at), "created_at": iso(r.created_at)}
    if with_detail:
        d["low_information_reason"] = r.review_status == "FEEDBACK_SUBMITTED" and is_low_information(r.reviewer_comments)
        d["gap_analysis"] = [gaps.ser_gap(g) for g in gaps.gaps_of_review(s, r.review_id)]
        d["corrections"] = [ser_correction(c) for c in rv.review_corrections(s, r)]
        d["events"] = [{"event_type": e.event_type, "actor_id": e.actor_id, "payload": e.payload, "created_at": iso(e.created_at)}
                       for e in s.scalars(select(FeedbackEvent).where(FeedbackEvent.review_id == r.review_id).order_by(FeedbackEvent.created_at))]
    return d


def ser_version(v: Any) -> dict[str, Any]:
    return {"version_id": v.version_id, "version_number": v.version_number, "approval_status": v.approval_status, "configuration": v.configuration,
            "change_description": v.change_description, "created_by": v.created_by, "approved_by": v.approved_by, "created_at": iso(v.created_at), "effective_at": iso(v.effective_at)}


# --------------------------------------------------------------------------------------------------------------------------------------- app
def create_app(factory: sessionmaker[Session], storage: ObjectStorage, evaluator: Any, secret: str, run_inline: bool = False,
               activity_ids: Optional[set[str]] = None, max_upload_mb: int = 18, embedder: Any = None, gap_analyzer: Any = None, rulebook: Any = None,
               note_drafter: Any = None, rule_drafter: Any = None, reason_summarizer: Any = None) -> FastAPI:
    """`embedder`: embeds each uploaded photograph (locally) so it can be found as a similar example; None means no embedding and so no examples.
    `gap_analyzer`: after each decision, compares what the AI said with what the person wrote (text only); None means no analysis.
    `note_drafter`: after each decision, drafts a prompt note for a gate the reviewer disputed (L5); None means notes cannot be turned on. It also needs the model-based
    gap analyzer, whose mapping it works from, and an administrator must turn it on per organisation (it is off by default).
    `rule_drafter`: after each decision, rewrites a rule the reviewer disagreed with the AI about (L5, rule rewrites); None means rewrites cannot be turned on. It works from the
    model-based gap analyzer's mapping too, and an administrator must turn it on per organisation (it is off by default).
    `reason_summarizer`: clubs the AI's findings that say the same thing about the same subject into one sentence on the review screen (a display aid; None means the screen shows the findings as written).
    `rulebook`: the engine's rulebook, used by the learning loop to replay stored answers (loaded once if not given)."""
    loop_rb = rulebook or Rulebook()
    app = FastAPI(title="NGO Photo Approval Agent: Phase 2A", version="0.1.0", docs_url="/docs")
    limiter = RateLimiter(limit=5, window_s=300)
    ip_limiter = RateLimiter(limit=30, window_s=300)

    @app.exception_handler(ServiceError)
    async def _service_error(_r: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"error": exc.code, "detail": str(exc)})

    @app.exception_handler(IntegrityError)
    async def _integrity(_r: Request, exc: IntegrityError) -> JSONResponse:
        msg = str(exc.orig)
        if "immutable" in msg or "append-only" in msg or "cannot" in msg:
            return JSONResponse(status_code=409, content={"error": "immutable", "detail": "this record cannot be changed"})
        return JSONResponse(status_code=409, content={"error": "conflict", "detail": "the request conflicts with existing data"})

    @app.exception_handler(Exception)
    async def _unexpected(_r: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error: %s", type(exc).__name__)  # type only: never the message, which may contain a path or a value
        return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "unexpected error"})

    @app.middleware("http")
    async def _headers(request: Request, call_next: Callable) -> Response:
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/static"):
            resp.headers["Cache-Control"] = "no-cache"   # revalidate (ETag) on every load, so an upgraded server never serves a stale script to a new API
        elif not request.url.path.startswith(("/docs", "/openapi")):
            resp.headers.setdefault("Cache-Control", "no-store")
        if request.url.path in ("/", "/index.html"):
            resp.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; frame-ancestors 'none'"
        return resp

    def db() -> Session:  # one session per request
        s = factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def current_user(authorization: Optional[str] = Header(default=None), s: Session = Depends(db)) -> User:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise Unauthorised("authentication required")
        try:
            claims = decode_token(authorization[7:].strip(), secret)
        except AuthError as exc:
            raise Unauthorised(str(exc)) from exc
        u = s.get(User, claims["sub"])
        # the user is re-read on every request: deactivation, a role change or a move between organisations takes effect at once
        if u is None or not u.is_active or u.organisation_id != claims["org"]:
            raise Unauthorised("invalid or expired token")
        return u

    def org_path(organisation_id: str, user: User) -> str:
        if organisation_id != user.organisation_id:
            raise NotFound("organisation not found")  # not 403: the existence of another organisation is not revealed
        return organisation_id

    def schedule(bg: BackgroundTasks, evaluation_id: str) -> None:
        if run_inline:
            ev.execute_evaluation(factory, storage, evaluator, evaluation_id, embedder)
        else:
            bg.add_task(ev.execute_evaluation, factory, storage, evaluator, evaluation_id, embedder)

    def schedule_gap(bg: BackgroundTasks, review_id: str) -> None:
        if gap_analyzer is None:
            return
        if run_inline:
            gaps.analyze_review(factory, review_id, gap_analyzer)
        else:
            bg.add_task(gaps.analyze_review, factory, review_id, gap_analyzer)

    def schedule_notes(bg: BackgroundTasks, review_id: str) -> None:
        """After the gap analysis (it runs first: background tasks keep their order), draft a note for each gate the reviewer disputed and watch the notes already live."""
        if note_drafter is None:
            return
        if run_inline:
            notes.after_decision(factory, review_id, note_drafter, loop_rb)
        else:
            bg.add_task(notes.after_decision, factory, review_id, note_drafter, loop_rb)

    def schedule_rules(bg: BackgroundTasks, review_id: str) -> None:
        """After the gap analysis, rewrite each rule the reviewer disputed and watch the rewrites already live."""
        if rule_drafter is None:
            return
        if run_inline:
            rule_svc.after_decision(factory, review_id, rule_drafter, loop_rb)
        else:
            bg.add_task(rule_svc.after_decision, factory, review_id, rule_drafter, loop_rb)

    def schedule_cycle(bg: BackgroundTasks, organisation_id: str) -> None:
        """The learning loop checks, after every decision, whether a cycle is due (it never raises and does nothing unless the NGO has turned it on)."""
        if run_inline:
            loop.maybe_run_cycle(factory, organisation_id, loop_rb)
        else:
            bg.add_task(loop.maybe_run_cycle, factory, organisation_id, loop_rb)

    # ------------------------------------------------------------------------------------------------------------------------- open routes
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/auth/login")
    def login(body: LoginIn, request: Request, s: Session = Depends(db)) -> dict[str, Any]:
        email, ip = body.email.strip().lower(), (request.client.host if request.client else "?")
        if limiter.blocked(email) or ip_limiter.blocked(ip):
            return JSONResponse(status_code=429, content={"error": "too_many_attempts", "detail": "too many failed logins; wait a few minutes"})
        u = s.scalar(select(User).where(User.email == email))
        ok = verify_password(body.password, u.password_hash if u else _DUMMY_HASH)  # same work whether or not the account exists
        if not (u and u.is_active and ok):
            limiter.record_failure(email)
            ip_limiter.record_failure(ip)
            raise Unauthorised("invalid email or password")
        limiter.reset(email)
        return {"access_token": issue_token(u.user_id, u.organisation_id, u.role, secret), "token_type": "bearer", "organisation_id": u.organisation_id, "role": u.role,
                "user_id": u.user_id, "display_name": u.display_name}

    @app.get("/me")
    def me(user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        org = s.get(Organisation, user.organisation_id)
        return {"user_id": user.user_id, "email": user.email, "display_name": user.display_name, "role": user.role, "organisation_id": org.organisation_id,
                "organisation_name": org.name, "active_rulebook_version_id": org.active_rulebook_version_id, "active_prompt_version_id": org.active_prompt_version_id}

    # ---------------------------------------------------------------------------------------------------------------------------- photos
    @app.post("/photos/upload", status_code=201)
    async def upload(background: BackgroundTasks, file: UploadFile = File(...), event_id: Optional[str] = Form(default=None), capture_date: Optional[str] = Form(default=None),
                     user: User = Depends(current_user), s: Session = Depends(db)) -> JSONResponse:
        rv.need_role(user, Role.REVIEWER)
        limit = max_upload_mb * 1024 * 1024
        data = await file.read(limit + 1)
        if len(data) > limit:
            return JSONResponse(status_code=413, content={"error": "payload_too_large", "detail": f"the image exceeds {max_upload_mb} MB"})
        photo, created = ev.upload_photo(s, storage, user, file.filename or "upload", data, event_id, capture_date, embedder=embedder)
        run = None
        if created:
            run = ev.request_evaluation(s, user, photo)
        s.commit()
        if run is not None:
            schedule(background, run.evaluation_id)
        s.refresh(photo)
        return JSONResponse(status_code=201 if created else 200, content={"photo": ser_photo(photo), "created": created, "duplicate": not created,
                                                                           "evaluation_id": run.evaluation_id if run else None})

    @app.get("/photos")
    def list_photos(status: Optional[str] = None, limit: int = 100, offset: int = 0, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        q = select(Photo).where(Photo.organisation_id == user.organisation_id).order_by(Photo.uploaded_at.desc()).limit(min(limit, 500)).offset(offset)
        if status:
            q = q.where(Photo.photo_status == status)
        out = []
        for p in s.scalars(q):
            last = s.scalar(select(EvaluationRun).where(EvaluationRun.photo_id == p.photo_id).order_by(EvaluationRun.created_at.desc()).limit(1))
            d = ser_photo(p)
            d["latest_evaluation"] = ser_run_summary(last) if last else None
            out.append(d)
        return {"photos": out}

    @app.get("/photos/{photo_id}")
    def get_photo(photo_id: str, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        p = rv.get_photo(s, user, photo_id)
        runs = ev_runs(s, p)
        return {"photo": ser_photo(p), "evaluations": [ser_run_summary(r) for r in runs],
                "lifecycle": [{"stage": e.stage, "from_status": e.from_status, "to_status": e.to_status, "evaluation_id": e.evaluation_id, "created_at": iso(e.created_at)}
                              for e in s.scalars(select(LifecycleEvent).where(LifecycleEvent.photo_id == p.photo_id).order_by(LifecycleEvent.created_at))]}

    def ev_runs(s: Session, p: Photo) -> list[EvaluationRun]:
        return list(s.scalars(select(EvaluationRun).where(EvaluationRun.photo_id == p.photo_id).order_by(EvaluationRun.created_at.desc())))

    @app.get("/photos/{photo_id}/image")
    def photo_image(photo_id: str, user: User = Depends(current_user), s: Session = Depends(db)) -> Response:
        p = rv.get_photo(s, user, photo_id)
        mime = ev.FORMATS[p.file_format][1]
        return Response(content=storage.get(p.image_storage_path), media_type=mime, headers={"Cache-Control": "private, no-store"})

    @app.post("/photos/{photo_id}/evaluations", status_code=202)
    def re_evaluate(photo_id: str, body: ReEvalIn, background: BackgroundTasks, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        rv.need_role(user, Role.REVIEWER)
        p = rv.get_photo(s, user, photo_id)
        if body.allow_unapproved or body.rulebook_version_id or body.prompt_version_id:
            rv.need_role(user, Role.ORG_ADMIN)  # choosing a version is an administrator's act
        run = ev.request_evaluation(s, user, p, body.rulebook_version_id, body.prompt_version_id, body.allow_unapproved)
        s.commit()
        schedule(background, run.evaluation_id)
        return {"evaluation_id": run.evaluation_id, "processing_status": run.processing_status}

    # --------------------------------------------------------------------------------------------------------------------------- evaluations
    @app.get("/evaluations/{evaluation_id}")
    def get_evaluation(evaluation_id: str, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        run = rv.get_evaluation(s, user, evaluation_id)
        d = ser_run(run, s)
        mine = s.scalar(select(HumanReview).where(HumanReview.evaluation_id == run.evaluation_id, HumanReview.reviewer_id == user.user_id).order_by(HumanReview.created_at.desc()))
        d["my_review"] = ser_review(mine, s) if mine else None
        d["reviews"] = [ser_review(r, s, with_detail=False) for r in s.scalars(select(HumanReview).where(HumanReview.evaluation_id == run.evaluation_id).order_by(HumanReview.created_at))]
        d["integrity_ok"] = ev.verify_integrity(run, storage) if run.processing_status == "AI_EVALUATED" else None
        return d

    @app.get("/evaluations/{evaluation_id}/summary")
    def evaluation_summary(evaluation_id: str, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        """The AI's findings for the review screen with those that say the same thing about the same subject clubbed into one sentence. A display aid: when no model is set up, or it fails,
        `available` is false and the screen shows the findings as written."""
        run = rv.get_evaluation(s, user, evaluation_id)
        items = reason_summary.findings(run)
        s.rollback()                                                                                   # no transaction stays open while a model is called
        if reason_summarizer is None:
            return {"available": False}
        try:
            res = reason_summarizer.summarize(evaluation_id, items)
        except Exception as exc:  # noqa: BLE001  (the screen simply shows the original sentences)
            log.warning("clubbing the findings of evaluation %s failed: %s", evaluation_id, type(exc).__name__)
            return {"available": False}
        return {"available": True, **res}

    @app.post("/evaluations/{evaluation_id}/review", status_code=201)
    def open_review(evaluation_id: str, user: User = Depends(current_user), s: Session = Depends(db)) -> JSONResponse:
        review, created = rv.open_review(s, user, evaluation_id)
        s.commit()
        return JSONResponse(status_code=201 if created else 200, content=ser_review(review, s))

    @app.patch("/evaluations/{evaluation_id}/corrections")
    def patch_corrections(evaluation_id: str, body: CorrectionsIn, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        out = rv.add_corrections(s, user, evaluation_id, [c.model_dump() for c in body.corrections], activity_ids)
        s.commit()
        return {"corrections": [ser_correction(c) for c in out]}

    # ---------------------------------------------------------------------------------------------------------------------------- feedback
    @app.post("/feedback/{review_id}/submit")
    def submit(review_id: str, body: SubmitIn, background: BackgroundTasks, idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
               user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        review, replayed = rv.submit_review(s, user, review_id, body.decision, body.reason_categories, body.comment, idempotency_key,
                                         blocker_statement=body.publication_blocker_statement)
        s.commit()
        if not replayed:
            schedule_gap(background, review.review_id)   # after the decision is safely stored; it can never change or block it
            schedule_notes(background, review.review_id)
            schedule_rules(background, review.review_id)
            schedule_cycle(background, review.organisation_id)
        return {"review": ser_review(review, s), "replayed": replayed}

    @app.get("/feedback/{review_id}")
    def get_feedback(review_id: str, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        return ser_review(rv.get_review(s, user, review_id), s)

    # ---------------------------------------------------------------------------------------------------------------------------- organisation
    @app.get("/organisations/{organisation_id}/feedback")
    def org_feedback(organisation_id: str, status: Optional[str] = None, limit: int = 100, offset: int = 0, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        org_path(organisation_id, user)
        return {"reviews": [ser_review(r, s, with_detail=False) for r in rv.list_feedback(s, user, status, limit, offset)]}

    @app.get("/organisations/{organisation_id}/evaluation-history")
    def org_history(organisation_id: str, photo_id: Optional[str] = None, limit: int = 100, offset: int = 0, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        org_path(organisation_id, user)
        return {"evaluations": [ser_run_summary(r) for r in rv.evaluation_history(s, user, photo_id, limit, offset)]}

    @app.get("/organisations/{organisation_id}/learning-examples")
    def org_learning(organisation_id: str, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        rows = s.scalars(select(LearningExample).where(LearningExample.organisation_id == user.organisation_id).order_by(LearningExample.created_at.desc()))
        return {"learning_examples": [{"learning_example_id": e.learning_example_id, "photo_id": e.photo_id, "verified_decision": e.verified_decision, "active_for_retrieval": e.active_for_retrieval,
                                       "human_review_id": e.human_review_id, "created_at": iso(e.created_at)} for e in rows]}

    # --------------------------------------------------------------------------------------------------------------------------- learning loop (L4)
    @app.get("/organisations/{organisation_id}/learning-loop")
    def get_loop(organisation_id: str, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        return loop.overview(s, s.get(Organisation, organisation_id))

    @app.put("/organisations/{organisation_id}/learning-loop")
    def put_loop(organisation_id: str, body: LoopSettingsIn, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        """Turn the loop on or off, pause it, or tune it. Turning it on needs `confirm_automatic_changes`: settings then change by themselves."""
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        org = s.get(Organisation, organisation_id)
        new = loop.save_settings(org, body.model_dump(exclude={"confirm_automatic_changes"}, exclude_none=True), body.confirm_automatic_changes)
        s.commit()
        return new

    @app.post("/organisations/{organisation_id}/learning-loop/run")
    def run_loop(organisation_id: str, user: User = Depends(current_user)) -> dict[str, Any]:
        """Run a cycle now, instead of waiting for the next five decisions (administrators). Does nothing while the loop is off or paused."""
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        return {"status": loop.maybe_run_cycle(factory, organisation_id, loop_rb, trigger="MANUAL") or "NOT_RUN"}

    @app.get("/organisations/{organisation_id}/changelog")
    def get_changelog(organisation_id: str, status: Optional[str] = None, limit: int = 100, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        """Every change the loop drafted, in plain language, with its evidence, its test and its history; `status=HELD` is the list held for developers."""
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        return {"changes": loop.changelog(s, organisation_id, status, limit)}

    @app.post("/organisations/{organisation_id}/changes/{change_id}/rollback")
    def rollback(organisation_id: str, change_id: str, body: RollbackIn, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        v = loop.rollback_change(s, s.get(Organisation, organisation_id), user, change_id, body.reason)
        s.commit()
        return {"active_rulebook_version": v.version_number}

    # ------------------------------------------------------------------------------------------------------------------------ prompt notes (L5)
    @app.get("/organisations/{organisation_id}/prompt-notes")
    def get_notes(organisation_id: str, limit: int = 100, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        """The settings, the notes now in the prompt and every drafted note with its checks and history (administrators)."""
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        return notes.overview(s, s.get(Organisation, organisation_id), note_drafter is not None, min(limit, 500))

    @app.put("/organisations/{organisation_id}/prompt-notes")
    def put_notes(organisation_id: str, body: NoteSettingsIn, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        """Turn notes off, to drafts only, or to drafts that are applied; pause them; tune the limits. Raising the mode needs `confirm_ngo_informed`."""
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        new = notes.save_settings(s.get(Organisation, organisation_id), body.model_dump(exclude={"confirm_ngo_informed"}, exclude_none=True), body.confirm_ngo_informed, note_drafter is not None)
        s.commit()
        return new

    @app.post("/organisations/{organisation_id}/prompt-notes/{draft_id}/withdraw")
    def withdraw_note(organisation_id: str, draft_id: str, body: RollbackIn, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        """Take a note out of the prompt, with a reason (administrators)."""
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        pv = notes.withdraw_note(s, s.get(Organisation, organisation_id), user, draft_id, body.reason, loop_rb)
        s.commit()
        return {"active_prompt_version": pv.version_number}

    # ------------------------------------------------------------------------------------------------------------------------ rule rewrites (L5)
    @app.get("/organisations/{organisation_id}/rule-updates")
    def get_rule_updates(organisation_id: str, limit: int = 100, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        """The settings, the rules now carrying a rewritten wording and every drafted rewrite with its checks and history (administrators)."""
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        org = s.get(Organisation, organisation_id)
        profile = s.get(RulebookVersion, org.active_rulebook_version_id).configuration["profile"]
        return rule_svc.overview(s, org, rule_drafter is not None, min(limit, 500), rule_catalogue(loop_rb, profile))

    @app.put("/organisations/{organisation_id}/rule-updates")
    def put_rule_updates(organisation_id: str, body: RuleSettingsIn, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        """Turn rule rewrites off, to drafts only, or to drafts that are applied; pause them; tune the limits. Raising the mode needs `confirm_ngo_informed`."""
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        new = rule_svc.save_settings(s.get(Organisation, organisation_id), body.model_dump(exclude={"confirm_ngo_informed"}, exclude_none=True), body.confirm_ngo_informed, rule_drafter is not None)
        s.commit()
        return new

    @app.get("/organisations/{organisation_id}/rule-updates/pending")
    def pending_rule_updates(organisation_id: str, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        """Rewrites waiting for a person to confirm that a rule that was changed earlier may be changed again (reviewers and administrators)."""
        org_path(organisation_id, user)
        rv.need_role(user, Role.REVIEWER)
        return {"pending": rule_svc.pending(s, organisation_id)}

    @app.post("/organisations/{organisation_id}/rule-updates/{edit_id}/answer")
    def answer_rule_update(organisation_id: str, edit_id: str, body: AnswerIn, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        """Answer the question "this rule was changed earlier: change it again?" (reviewers and administrators). Yes applies the waiting wording, no keeps the wording in force."""
        org_path(organisation_id, user)
        rv.need_role(user, Role.REVIEWER)
        pv = rule_svc.answer(s, s.get(Organisation, organisation_id), user, edit_id, body.accept, loop_rb)
        s.commit()
        return {"applied": pv is not None, "active_prompt_version": pv.version_number if pv else None}

    @app.post("/organisations/{organisation_id}/rule-updates/manual", status_code=201)
    def manual_rule_update(organisation_id: str, body: ManualRuleIn, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        """An administrator changes one rule's wording by hand. It takes effect at once, whatever the rule-update mode, and is kept in the same history; the automatic watch never withdraws it
        and an automatic rewrite never replaces it."""
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        org = s.get(Organisation, organisation_id)
        profile = s.get(RulebookVersion, org.active_rulebook_version_id).configuration["profile"]
        pv = rule_svc.manual_edit(s, org, user, body.rule_id.strip(), body.wording, body.reason, body.direction, loop_rb, rule_catalogue(loop_rb, profile))
        s.commit()
        return {"active_prompt_version": pv.version_number}

    @app.post("/organisations/{organisation_id}/rule-updates/{edit_id}/withdraw")
    def withdraw_rule_update(organisation_id: str, edit_id: str, body: RollbackIn, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        """Take a rewrite out of the prompt (the rule goes back to its original wording), with a reason (administrators)."""
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        pv = rule_svc.withdraw_edit(s, s.get(Organisation, organisation_id), user, edit_id, body.reason, loop_rb)
        s.commit()
        return {"active_prompt_version": pv.version_number}

    @app.get("/organisations/{organisation_id}/loop-activity")
    def loop_activity_view(organisation_id: str, limit: int = 30, offset: int = 0, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        """What the learning loop did with each decided photograph, newest first: what the AI said, what the reviewer decided and wrote, where they disagreed, the rule that point relates to
        and what it was changed to (administrators). Read-only; it calls no model."""
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        org = s.get(Organisation, organisation_id)
        profile = s.get(RulebookVersion, org.active_rulebook_version_id).configuration["profile"]
        return loop_activity.activity(s, org, loop_rb, profile, limit, offset)

    @app.get("/organisations/{organisation_id}/prompt")
    def current_prompt(organisation_id: str, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        """The instructions the AI is given for every photograph right now, exactly as built (administrators): the fixed prompt with any rewritten rules and notes applied, with the
        rewrites listed beside the original wording. Similar-photo examples, when any are found, are added at evaluation time and are not part of this text."""
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        org = s.get(Organisation, organisation_id)
        pv, rbv = s.get(PromptVersion, org.active_prompt_version_id), s.get(RulebookVersion, org.active_rulebook_version_id)
        cfg, conf = rbv.configuration, pv.configuration or {}
        profile = loop_rb.resolve(cfg["profile"], without=cfg.get("without", []))
        built = build_prompt(loop_rb, profile, notes=conf.get("notes") or None, rule_text=conf.get("rule_text") or None)
        base = rule_catalogue(loop_rb, cfg["profile"])
        rewrites = [{"rule_id": rid, "kind": base[rid].kind, "gate_id": base[rid].gate_id, "original": base[rid].base, "now": text} for rid, text in sorted((conf.get("rule_text") or {}).items()) if rid in base]
        return {"version": pv.version_number, "hash": built.hash, "registered_hash": conf.get("prompt_hash"), "matches_registered": built.hash == conf.get("prompt_hash"), "profile": profile.name,
                "rulebook_version": built.rulebook_version, "rules_asked": len(built.criterion_ids), "rewrites": rewrites, "notes": conf.get("notes") or {}, "system": built.system_text,
                "user": built.user_text, "characters": len(built.system_text) + len(built.user_text)}

    @app.get("/organisations/{organisation_id}/gap-analyses")
    def org_gaps(organisation_id: str, limit: int = 100, offset: int = 0, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        """What staff said that the AI had not, per decision (administrators): the raw material of the learning loop."""
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        return {"gap_analyses": [gaps.ser_gap(g) for g in gaps.list_gaps(s, user, limit, offset)]}

    @app.get("/organisations/{organisation_id}/versions")
    def list_versions(organisation_id: str, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        org_path(organisation_id, user)
        org = s.get(Organisation, organisation_id)
        return {"active_rulebook_version_id": org.active_rulebook_version_id, "active_prompt_version_id": org.active_prompt_version_id,
                "rulebook_versions": [ser_version(v) for v in s.scalars(select(RulebookVersion).where(RulebookVersion.organisation_id == organisation_id).order_by(RulebookVersion.created_at))],
                "prompt_versions": [ser_version(v) for v in s.scalars(select(PromptVersion).where(PromptVersion.organisation_id == organisation_id).order_by(PromptVersion.created_at))]}

    @app.post("/organisations/{organisation_id}/versions", status_code=201)
    def create_version(organisation_id: str, body: VersionIn, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        if body.kind not in ("rulebook", "prompt"):
            raise Invalid("kind must be rulebook or prompt")
        v = vs.create_version(s, body.kind, organisation_id, body.version_number, body.configuration, body.change_description, user.user_id)
        s.commit()
        return ser_version(v)

    @app.post("/organisations/{organisation_id}/versions/{kind}/{version_id}/approve")
    def approve_version(organisation_id: str, kind: str, version_id: str, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        v = vs.approve_version(s, kind, vs.get_version(s, kind, organisation_id, version_id), user)
        s.commit()
        return ser_version(v)

    @app.post("/organisations/{organisation_id}/versions/{kind}/{version_id}/activate")
    def activate_version(organisation_id: str, kind: str, version_id: str, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        vs.activate_version(s, kind, vs.get_version(s, kind, organisation_id, version_id))
        s.commit()
        return list_versions(organisation_id, user, s)

    @app.post("/organisations/{organisation_id}/users", status_code=201)
    def create_user(organisation_id: str, body: UserIn, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        if body.role not in ROLE_RANK:
            raise Invalid(f"role must be one of {sorted(ROLE_RANK)}")
        email = body.email.strip().lower()
        if "@" not in email:
            raise Invalid("a valid email is required")
        if s.scalar(select(User).where(User.email == email)):
            raise Conflict("that email is already registered")
        try:
            h = hash_password(body.password)
        except ValueError as exc:
            raise Invalid(str(exc)) from exc
        u = User(organisation_id=organisation_id, email=email, display_name=body.display_name, password_hash=h, role=body.role)
        s.add(u)
        s.commit()
        return {"user_id": u.user_id, "email": u.email, "role": u.role}

    # ------------------------------------------------------------------------------------------------------------------------- example store (L2)
    @app.get("/organisations/{organisation_id}/few-shot")
    def get_few_shot(organisation_id: str, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        org = s.get(Organisation, organisation_id)
        embedded = s.scalar(select(func.count()).select_from(PhotoEmbedding).where(PhotoEmbedding.organisation_id == organisation_id))
        usable = s.scalar(select(func.count()).select_from(LearningExample).where(LearningExample.organisation_id == organisation_id, LearningExample.active_for_retrieval.is_(True)))
        return {**ex_svc.settings(org), "embedder": getattr(embedder, "name", None), "photos_embedded": embedded, "examples_available": usable,
                "photos_marked_misleading": len(ex_svc.flagged_ids(s, organisation_id))}

    @app.put("/organisations/{organisation_id}/few-shot")
    def put_few_shot(organisation_id: str, body: FewShotIn, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        org_path(organisation_id, user)
        rv.need_role(user, Role.ORG_ADMIN)
        if body.enabled and embedder is None:
            raise Conflict("no embedder is configured on this server, so similar photographs cannot be found; set AGENT_EMBEDDER (dinov2, clip or thumbnail)")
        new = ex_svc.save_settings(s, s.get(Organisation, organisation_id), body.enabled, body.k, body.min_similarity, body.confirm_cost_increase)
        s.commit()
        return new

    @app.get("/photos/{photo_id}/similar")
    def similar(photo_id: str, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        """The past decisions that would be shown to the model for this photograph right now (without the images)."""
        rv.need_role(user, Role.REVIEWER)
        photo = rv.get_photo(s, user, photo_id)
        cfg = ex_svc.settings(s.get(Organisation, user.organisation_id))
        examples, record = ex_svc.examples_for_evaluation(s, storage, photo, cfg, embedder) if cfg["enabled"] else ([], None)
        s.commit()
        return {"enabled": cfg["enabled"], "examples": (record or {}).get("examples", [])}

    @app.post("/photos/{photo_id}/example-flag", status_code=201)
    def example_flag(photo_id: str, body: FlagIn, user: User = Depends(current_user), s: Session = Depends(db)) -> dict[str, Any]:
        """An administrator marks a photograph as misleading as an example (or clears the mark). Append-only; the latest mark wins."""
        rv.need_role(user, Role.ORG_ADMIN)
        row = ex_svc.set_flag(s, user, rv.get_photo(s, user, photo_id), body.flagged, body.reason)
        s.commit()
        return {"photo_id": photo_id, "flagged": body.flagged, "flag_id": row.flag_id}

    if STATIC.exists():
        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC / "index.html")

        app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def create_default_app() -> FastAPI:
    """Production wiring from the environment. The real evaluator needs the same two authorisations as the engine (cloud switch and API key)."""
    from layered_engine.providers import load_vision_config, make_provider

    from ..evaluators import LayeredEngineEvaluator

    engine = make_engine()
    factory = make_session_factory(engine)
    storage = LocalStorage(os.environ.get("AGENT_STORAGE_DIR", "./storage"))
    cfg = load_vision_config()   # NGO_VISION=haiku selects Claude Haiku 5.5 (anthropic_config.yaml); unset: Gemini, as before
    provider = make_provider(cfg)
    evaluator = LayeredEngineEvaluator(provider, cfg, provider_name="anthropic-claude" if cfg["provider"] == "anthropic" else "google-gemini")  # raises AuthorizationError at start-up if cloud inference is not authorised: fails closed
    from ..embeddings import make_embedder
    from ..llm_gap_analyzer import make_gap_analyzer

    # AGENT_GAP_ANALYZER=baseline (default, offline keyword matching, no model) or llm (a model reads the reviewers' reasons: they are sent to the cloud as text).
    # AGENT_GAP_LLM=gemini (default) or groq (needs NGO_ALLOW_GROQ_INFERENCE=true and GROQ_API_KEY: its own authorisation, because Groq is a different company).
    # AGENT_GAP_MAX_CALLS optionally stops it after that many calls per process.
    profile = os.environ.get("AGENT_PROFILE", "layered_proposed_v1")   # the policy profile the analyst and the rule drafter list rules from (the organisation's own versions decide what is evaluated)
    gap = make_gap_analyzer(os.environ.get("AGENT_GAP_ANALYZER", "baseline"), provider=provider, cfg=cfg, llm=os.environ.get("AGENT_GAP_LLM", "gemini"),
                            max_calls=int(os.environ["AGENT_GAP_MAX_CALLS"]) if os.environ.get("AGENT_GAP_MAX_CALLS") else None, profile=profile)
    # AGENT_NOTE_DRAFTER=off (default) or llm: after each decision a model drafts a prompt note for a gate the reviewer disputed (L5). It reuses the gap analyzer's model and
    # works from its mapping, so it needs AGENT_GAP_ANALYZER=llm. Even then nothing is drafted until an administrator turns notes on for the organisation (off by default).
    drafter = None
    if os.environ.get("AGENT_NOTE_DRAFTER", "off").strip().lower() == "llm":
        from ..llm_gap_analyzer import LlmGapAnalyzer
        from ..note_drafter import LlmNoteDrafter

        if not isinstance(gap, LlmGapAnalyzer):
            raise ValueError("AGENT_NOTE_DRAFTER=llm needs AGENT_GAP_ANALYZER=llm: the drafter works from the model-based mapping of the reviewers' reasons")
        drafter = LlmNoteDrafter(gap.model, Rulebook())
    # AGENT_RULE_DRAFTER=off (default) or llm: after each decision a model rewrites a rule the reviewer disagreed with the AI about (L5, rule rewrites; the client's own rules included).
    # Same requirements as the note drafter; an administrator must turn it on per organisation (off by default).
    rule_drafter = None
    if os.environ.get("AGENT_RULE_DRAFTER", "off").strip().lower() == "llm":
        from ..llm_gap_analyzer import LlmGapAnalyzer
        from ..rule_drafter import LlmRuleDrafter

        if not isinstance(gap, LlmGapAnalyzer):
            raise ValueError("AGENT_RULE_DRAFTER=llm needs AGENT_GAP_ANALYZER=llm: the drafter works from the model-based mapping of the reviewers' reasons")
        rule_drafter = LlmRuleDrafter(gap.model, Rulebook(), profile=profile)
    # AGENT_REASON_SUMMARY=off (default) or llm: a model clubs the AI's findings that say the same thing into one sentence on the review screen (the AI's own findings are sent as text; display only).
    reason_summarizer = None
    if os.environ.get("AGENT_REASON_SUMMARY", "off").strip().lower() == "llm":
        from ..llm_gap_analyzer import LlmGapAnalyzer
        from ..reason_summary import ReasonSummarizer

        if not isinstance(gap, LlmGapAnalyzer):
            raise ValueError("AGENT_REASON_SUMMARY=llm needs AGENT_GAP_ANALYZER=llm: it reuses that model")
        reason_summarizer = ReasonSummarizer(gap.model)
    return create_app(factory, storage, evaluator, jwt_secret(), embedder=make_embedder(os.environ.get("AGENT_EMBEDDER", "clip")),   # local model, loaded lazily
                      gap_analyzer=gap, note_drafter=drafter, rule_drafter=rule_drafter, reason_summarizer=reason_summarizer)
