"""Upload, evaluation and re-evaluation. Every evaluation is a new EvaluationRun; a run is written once and then frozen (database trigger)."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
from typing import Any, Optional

from PIL import Image, UnidentifiedImageError
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from . import example_service
from .embeddings import Embedder
from .enums import AI_DEFERRING, PhotoStatus, ProcessingStatus, VersionApproval
from .errors import Conflict, Forbidden, Invalid, NotFound
from .evaluators import EvaluationFailure, Evaluator
from .models import EvaluationRun, LifecycleEvent, Organisation, Photo, PromptVersion, RulebookVersion, User, now
from .storage import ObjectStorage, image_key, raw_response_key, sha256_hex

log = logging.getLogger("ngo_agent.evaluation")
MAX_UPLOAD_BYTES = 18 * 1024 * 1024
FORMATS = {"JPEG": (".jpg", "image/jpeg"), "PNG": (".png", "image/png"), "WEBP": (".webp", "image/webp")}
try:  # the NGO's photographs include HEIC files; supported when pillow-heif is installed
    import pillow_heif

    pillow_heif.register_heif_opener()
    FORMATS["HEIF"] = (".heic", "image/heic")
except ImportError:  # pragma: no cover
    pass
_BEFORE_REVIEW = {PhotoStatus.UPLOADED.value, PhotoStatus.PROCESSING.value, PhotoStatus.FAILED.value, PhotoStatus.AI_EVALUATED.value,
                  PhotoStatus.PENDING_REVIEW.value, PhotoStatus.HUMAN_REVIEW_REQUIRED.value}
_SECRET = re.compile(r"(AIza[0-9A-Za-z_\-]{20,}|Bearer\s+[A-Za-z0-9._\-]+)")


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def log_lifecycle(s: Session, photo: Photo, stage: str, to_status: Optional[str] = None, evaluation_id: Optional[str] = None, payload: Optional[dict] = None,
                  change_status: bool = True) -> None:
    """Append a lifecycle event and (when allowed) move the photo's workflow status. The status is the workflow, not a decision."""
    frm = photo.photo_status
    if to_status and change_status and to_status != frm:
        photo.photo_status = to_status
    s.add(LifecycleEvent(organisation_id=photo.organisation_id, photo_id=photo.photo_id, evaluation_id=evaluation_id, stage=stage, from_status=frm,
                         to_status=photo.photo_status, payload=payload or {}))


def upload_photo(s: Session, storage: ObjectStorage, user: User, filename: str, data: bytes, event_id: Optional[str] = None, capture_date: Optional[str] = None,
                 extra_metadata: Optional[dict] = None, embedder: Optional[Embedder] = None) -> tuple[Photo, bool]:
    """Validate, store once (per organisation and content hash) and register. Returns (photo, created); an identical upload returns the existing photo.
    With an `embedder`, the photograph is also embedded (locally) so it can later be found as a similar example; a failure there never fails the upload."""
    if not data:
        raise Invalid("the uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise Invalid(f"the image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB; it is not resized automatically")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            fmt, (w, h) = im.format, im.size
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise Invalid("the file is not a readable image") from exc
    if fmt not in FORMATS:
        raise Invalid(f"unsupported image format {fmt}; allowed: {', '.join(FORMATS)}")
    h256 = sha256_hex(data)
    existing = s.scalar(select(Photo).where(Photo.organisation_id == user.organisation_id, Photo.file_hash == h256))
    if existing:
        return existing, False
    key = storage.put(image_key(user.organisation_id, h256, FORMATS[fmt][0]), data)
    safe_name = re.sub(r"[^\w.\- ]", "_", filename or "upload")[:200]
    p = Photo(organisation_id=user.organisation_id, image_storage_path=key, original_filename=safe_name, file_hash=h256, uploaded_by=user.user_id, event_id=event_id,
              capture_date=capture_date, file_format=fmt, image_width=w, image_height=h, photo_status=PhotoStatus.UPLOADED.value, photo_metadata=extra_metadata or {})
    s.add(p)
    s.flush()
    s.add(LifecycleEvent(organisation_id=p.organisation_id, photo_id=p.photo_id, stage="PHOTO_UPLOADED", from_status=None, to_status=p.photo_status,
                         payload={"file_hash": h256, "bytes": len(data)}))
    example_service.store_embedding(s, p, data, embedder)
    return p, True


def _resolve_versions(s: Session, org: Organisation, rulebook_version_id: Optional[str], prompt_version_id: Optional[str], allow_unapproved: bool,
                      user: User) -> tuple[RulebookVersion, PromptVersion, list[str]]:
    notes: list[str] = []
    rid, pid = rulebook_version_id or org.active_rulebook_version_id, prompt_version_id or org.active_prompt_version_id
    if not rid or not pid:
        raise Conflict("this organisation has no active rulebook and prompt version")
    rv = s.scalar(select(RulebookVersion).where(RulebookVersion.version_id == rid, RulebookVersion.organisation_id == org.organisation_id))
    pv = s.scalar(select(PromptVersion).where(PromptVersion.version_id == pid, PromptVersion.organisation_id == org.organisation_id))
    if rv is None or pv is None:
        raise NotFound("version not found")
    for v, kind in ((rv, "rulebook"), (pv, "prompt")):
        if v.approval_status != VersionApproval.APPROVED.value:
            if not allow_unapproved:
                raise Forbidden(f"the {kind} version {v.version_number!r} is {v.approval_status}; only APPROVED versions can be used")
            if user.role != "ORG_ADMIN":
                raise Forbidden("only an administrator can run an evaluation with an unapproved version")
            notes.append(f"EXPERIMENTAL: the {kind} version {v.version_number!r} is {v.approval_status}, not NGO-approved")
    return rv, pv, notes


def request_evaluation(s: Session, user: User, photo: Photo, rulebook_version_id: Optional[str] = None, prompt_version_id: Optional[str] = None,
                       allow_unapproved: bool = False) -> EvaluationRun:
    """Create a new EvaluationRun in PROCESSING. Never touches an earlier run."""
    if photo.organisation_id != user.organisation_id:
        raise NotFound("photo not found")
    org = s.get(Organisation, user.organisation_id)
    rv, pv, notes = _resolve_versions(s, org, rulebook_version_id, prompt_version_id, allow_unapproved, user)
    earlier = s.scalar(select(func.count()).select_from(EvaluationRun).where(EvaluationRun.photo_id == photo.photo_id))
    run = EvaluationRun(photo_id=photo.photo_id, organisation_id=photo.organisation_id, prompt_version_id=pv.version_id, rulebook_version_id=rv.version_id,
                        evaluation_profile=rv.configuration.get("profile", ""), processing_status=ProcessingStatus.PROCESSING.value, requested_by=user.user_id, warnings=notes)
    s.add(run)
    s.flush()
    if earlier:
        log_lifecycle(s, photo, "RE_EVALUATION_REQUESTED", None, run.evaluation_id, {"previous_runs": earlier, "rulebook_version": rv.version_number, "prompt_version": pv.version_number})
    log_lifecycle(s, photo, "AI_EVALUATION_STARTED", PhotoStatus.PROCESSING.value if photo.photo_status in _BEFORE_REVIEW else None, run.evaluation_id,
                  {"rulebook_version_id": rv.version_id, "prompt_version_id": pv.version_id}, change_status=photo.photo_status in _BEFORE_REVIEW)
    return run


def _hash_content(validated: dict, decision: dict, raw_sha: str, model_id: str, profile: str, prompt_version_id: str, rulebook_version_id: str,
                  few_shot: Optional[dict] = None) -> str:
    body = {"validated": validated, "decision": decision, "raw": raw_sha, "model": model_id, "profile": profile,
            "prompt_version_id": prompt_version_id, "rulebook_version_id": rulebook_version_id}
    if few_shot:   # only when examples were shown, so the hash of every run made without them is exactly what it always was
        body["few_shot"] = few_shot
    return hashlib.sha256(canonical(body).encode()).hexdigest()


def verify_integrity(run: EvaluationRun, storage: Optional[ObjectStorage] = None) -> bool:
    """True when the stored content still matches the hash written at completion (and, if storage is given, the raw response file too)."""
    if run.processing_status != ProcessingStatus.AI_EVALUATED.value or not run.raw_response_sha256:
        return False
    if storage and run.raw_response_location and sha256_hex(storage.get(run.raw_response_location)) != run.raw_response_sha256:
        return False
    return _hash_content(run.validated_response, run.decision_output, run.raw_response_sha256, run.model_id, run.evaluation_profile, run.prompt_version_id,
                         run.rulebook_version_id, run.few_shot) == run.content_hash


def execute_evaluation(factory: sessionmaker[Session], storage: ObjectStorage, evaluator: Evaluator, evaluation_id: str, embedder: Optional[Embedder] = None) -> str:
    """Run the evaluator for a PROCESSING run and write the result exactly once. Returns the final processing status. Safe to call twice.

    When the organisation has switched similar-photo examples on and an embedder is available, the most similar past staff decisions are shown to the model
    and recorded on the run. With no eligible example the evaluation is exactly what it was before examples existed."""
    with factory() as s:
        run = s.get(EvaluationRun, evaluation_id)
        if run is None:
            raise NotFound("evaluation not found")
        if run.processing_status != ProcessingStatus.PROCESSING.value:
            return run.processing_status
        photo = s.get(Photo, run.photo_id)
        rv = s.get(RulebookVersion, run.rulebook_version_id)
        pv = s.get(PromptVersion, run.prompt_version_id)
        cfg, expected_hash = dict(rv.configuration), pv.configuration.get("prompt_hash")
        if pv.configuration.get("notes"):
            cfg["prompt_notes"] = dict(pv.configuration["notes"])   # the reviewers' notes of this prompt version (L5); absent for every earlier version
        if pv.configuration.get("rule_text"):
            cfg["rule_text"] = dict(pv.configuration["rule_text"])   # the rules rewritten from the reviewers' feedback; absent for every earlier version
        org_id, photo_id, key, fmt = run.organisation_id, photo.photo_id, photo.image_storage_path, photo.file_format
        meta = {"width": photo.image_width, "height": photo.image_height}
        examples, shown = [], None
        fs_cfg = example_service.settings(s.get(Organisation, org_id))
        if fs_cfg["enabled"]:
            try:
                examples, shown = example_service.examples_for_evaluation(s, storage, photo, fs_cfg, embedder)
            except Exception as exc:  # noqa: BLE001  (a problem finding examples must never stop an evaluation: it runs without them)
                examples, shown = [], None
                log.warning("examples were not used for evaluation %s: %s", evaluation_id, type(exc).__name__)
            s.commit()   # a lazily made embedding is kept
    out, error = None, None
    try:  # the evaluator runs with no database transaction open: a slow model call must not hold a lock
        if examples:
            out = evaluator.evaluate(storage.get(key), FORMATS[fmt][1], cfg, meta, examples=examples)
        else:
            out = evaluator.evaluate(storage.get(key), FORMATS[fmt][1], cfg, meta)
    except EvaluationFailure as exc:
        error = str(exc)
    except Exception as exc:  # noqa: BLE001  (provider, network, storage: recorded as a failed run, never as an approval)
        error = f"{type(exc).__name__}: {exc}"
    with factory() as s:
        run = s.get(EvaluationRun, evaluation_id)
        photo = s.get(Photo, photo_id)
        if run.processing_status != ProcessingStatus.PROCESSING.value:  # another worker finished it first
            return run.processing_status
        run.completed_at = now()
        run.few_shot = shown   # what the model was shown (None when nothing was), recorded for the success and the failure alike
        if out is None:
            run.processing_status = ProcessingStatus.FAILED.value
            run.error = _SECRET.sub("[removed]", (error or "unknown error"))[:500]
            log_lifecycle(s, photo, "AI_EVALUATION_FAILED", PhotoStatus.FAILED.value, evaluation_id, {"error": run.error}, change_status=photo.photo_status in _BEFORE_REVIEW)
            s.commit()
            return run.processing_status
        v, d = out.validated, out.decision
        raw_key = storage.put(raw_response_key(org_id, evaluation_id), out.raw_text.encode())
        raw_sha = sha256_hex(out.raw_text.encode())
        warnings = list(run.warnings or []) + list(out.warnings)
        if expected_hash and out.prompt_hash != expected_hash:
            warnings.append(f"prompt hash {out.prompt_hash} differs from the registered prompt version ({expected_hash})")
        if cfg.get("rulebook_version") and out.rulebook_version != cfg["rulebook_version"]:
            warnings.append(f"rulebook version {out.rulebook_version} differs from the registered version ({cfg['rulebook_version']})")
        run.model_provider, run.model_id, run.evaluation_profile = out.model_provider, out.model_id, out.profile
        run.raw_response_location, run.raw_response_sha256, run.validated_response, run.decision_output = raw_key, raw_sha, v, d
        run.layer_1_output = v.get("layer_1_scene_understanding")
        run.layer_2_output = {"criteria": v.get("layer_2_criteria"), "gates": d.get("gates"), "failed_gates": d.get("failed_gates"), "uncertain_gates": d.get("uncertain_gates"),
                              "not_applicable_gates": d.get("not_applicable_gates")}
        run.layer_3_output = {"dimensions": v.get("layer_3_quality_dimensions"), "quality": d.get("quality")}
        run.layer_4_output = {"visual_flags": v.get("layer_4_visual_flags"), "publication": d.get("publication")}
        run.ai_recommendation = d.get("recommendation")
        run.quality_score = (d.get("quality") or {}).get("overall_score")
        run.input_tokens, run.output_tokens = out.usage.get("prompt_tokens"), out.usage.get("output_tokens")
        run.estimated_cost, run.latency = out.cost_usd, out.usage.get("latency_s")
        run.warnings = warnings + list(d.get("warnings") or [])
        run.content_hash = _hash_content(v, d, raw_sha, out.model_id, out.profile, run.prompt_version_id, run.rulebook_version_id, shown)
        run.processing_status = ProcessingStatus.AI_EVALUATED.value
        s.flush()
        move = photo.photo_status in _BEFORE_REVIEW
        log_lifecycle(s, photo, "AI_EVALUATION_COMPLETED", PhotoStatus.AI_EVALUATED.value, evaluation_id, {"model_id": out.model_id, "profile": out.profile}, change_status=move)
        log_lifecycle(s, photo, "AI_RECOMMENDATION_GENERATED", None, evaluation_id, {"ai_recommendation": run.ai_recommendation, "quality_score": run.quality_score})
        waiting = PhotoStatus.HUMAN_REVIEW_REQUIRED.value if run.ai_recommendation in AI_DEFERRING else PhotoStatus.PENDING_REVIEW.value
        log_lifecycle(s, photo, "WAITING_FOR_HUMAN_REVIEW", waiting, evaluation_id, {}, change_status=move)
        s.commit()
        return run.processing_status


def fail_stale_runs(factory: sessionmaker[Session], older_than_s: int = 900) -> int:
    """Recovery for a crash or restart: a run still PROCESSING after `older_than_s` seconds is marked FAILED (never approved), so the photo can be re-evaluated.
    Run it at start-up and from a scheduler. Returns how many runs were closed."""
    from datetime import timedelta

    cutoff = now() - timedelta(seconds=older_than_s)
    n = 0
    with factory() as s:
        for run in s.scalars(select(EvaluationRun).where(EvaluationRun.processing_status == ProcessingStatus.PROCESSING.value)):
            created = run.created_at if run.created_at.tzinfo else run.created_at.replace(tzinfo=cutoff.tzinfo)
            if created < cutoff:
                photo = s.get(Photo, run.photo_id)
                run.processing_status, run.completed_at, run.error = ProcessingStatus.FAILED.value, now(), "the evaluation did not finish (service restarted or timed out)"
                log_lifecycle(s, photo, "AI_EVALUATION_FAILED", PhotoStatus.FAILED.value, run.evaluation_id, {"error": run.error}, change_status=photo.photo_status in _BEFORE_REVIEW)
                n += 1
        s.commit()
    return n
