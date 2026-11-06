"""Cloud tuning workflow: preflight, (authorized) upload, (authorized) submit, poll, record.

Safety model - a paid job is submitted only when ALL hold:
  1. the caller passes ``submit=True`` (CLI: ``--submit-cloud-job``),
  2. ``cloud.allow_paid_tuning`` is true (env NGO_ALLOW_PAID_TUNING),
  3. every blocking preflight check passes,
  4. no job with the same idempotency key (dataset bytes + base model + hyper-parameters) exists
     that has not failed.
Uploading images to GCS additionally requires ``cloud.allow_cloud_upload``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ModelConfig, ProjectConfig
from ..exceptions import (
    AuthorizationError,
    ConfigError,
    ModelUnavailableError,
    ProviderError,
    TuningError,
)
from ..models.cloud_client import map_exception
from .dataset_builder import PLACEHOLDER_BUCKET, validate_jsonl_file
from .model_registry import FAILED_STATES, RUNNING_STATES, SUCCESS_STATES, ModelRegistry

logger = logging.getLogger(__name__)

MULTI_REGION_ONLY_PREFIXES = ("gemini-3",)  # docs: Gemini 3.x tuning/serving on "us"/"eu" multi-region only


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    blocking: bool = True


@dataclass
class PreflightReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str, blocking: bool = True) -> None:
        self.checks.append(Check(name, ok, detail, blocking))

    @property
    def can_submit(self) -> bool:
        return all(c.ok for c in self.checks if c.blocking)

    def render(self) -> str:
        rows = []
        for c in self.checks:
            mark = "OK  " if c.ok else ("FAIL" if c.blocking else "WARN")
            rows.append(f"[{mark}] {c.name}: {c.detail}")
        rows.append(f"=> {'READY to submit' if self.can_submit else 'NOT READY (blocking checks failed)'}")
        return "\n".join(rows)


def idempotency_key(train_path: Path, val_path: Path, base_model: str, hyper: dict[str, Any], track: str) -> str:
    h = hashlib.sha256()
    for p in (train_path, val_path):
        h.update(hashlib.sha256(p.read_bytes()).digest())
    h.update(json.dumps({"m": base_model, "h": hyper, "t": track}, sort_keys=True).encode())
    return h.hexdigest()[:20]


def hyperparameters(model_cfg: ModelConfig) -> dict[str, Any]:
    t = model_cfg.tuning
    return {"epoch_count": t.epoch_count, "adapter_size": t.adapter_size,
            "learning_rate_multiplier": t.learning_rate_multiplier}


# --------------------------------------------------------------------------- preflight


def run_preflight(
    project: ProjectConfig,
    model_cfg: ModelConfig,
    tuning_dir: Path,
    track: str,
    base_model: str | None = None,
    registry: ModelRegistry | None = None,
    check_credentials: bool = True,
    client: Any | None = None,
    rule_ids: list[str] | None = None,
) -> PreflightReport:
    """Offline checks by default; pass ``client`` to additionally verify the base model online."""
    rep = PreflightReport()
    cloud = project.cloud
    base_model = base_model or model_cfg.models["primary"].id

    rep.add("auth_mode_supports_tuning", cloud.auth_mode == "vertex",
            "vertex mode" if cloud.auth_mode == "vertex" else
            "NGO_AUTH_MODE=api_key supports inference only; tuning needs a Google Cloud project (NGO_AUTH_MODE=vertex)")
    rep.add("project_id", bool(cloud.project_id), cloud.project_id or "GOOGLE_CLOUD_PROJECT not set")
    rep.add("location", bool(cloud.location), cloud.location or "GOOGLE_CLOUD_LOCATION not set")
    if base_model.startswith(MULTI_REGION_ONLY_PREFIXES):
        rep.add("region_supported", cloud.location in {"us", "eu"},
                f"{base_model} tuning is documented for the 'us'/'eu' multi-region endpoints; configured: {cloud.location!r}")
    rep.add("base_model_configured", bool(base_model), base_model)
    ref = next((m for m in model_cfg.models.values() if m.id == base_model), None)
    rep.add("base_model_verified_in_docs", bool(ref and ref.verified),
            "ID confirmed in official docs" if ref and ref.verified else
            f"{base_model!r} is not marked verified in configs/model.yaml; confirm it before spending", blocking=False)
    rep.add("bucket", bool(cloud.gcs_bucket) and cloud.gcs_bucket != PLACEHOLDER_BUCKET,
            cloud.gcs_bucket or "NGO_GCS_BUCKET not set")
    rep.add("authorize_upload", cloud.allow_cloud_upload,
            "upload authorized" if cloud.allow_cloud_upload else "NGO_ALLOW_CLOUD_UPLOAD is not true (no images may leave this machine)")
    rep.add("authorize_paid_tuning", cloud.allow_paid_tuning,
            "paid tuning authorized" if cloud.allow_paid_tuning else "NGO_ALLOW_PAID_TUNING is not true")

    train, val = tuning_dir / "train.jsonl", tuning_dir / "validation.jsonl"
    files_ok = train.exists() and val.exists()
    rep.add("dataset_files", files_ok, "train.jsonl and validation.jsonl present" if files_ok
            else "run scripts/build_tuning_data.py first")
    if files_ok:
        for p in (train, val):
            r = validate_jsonl_file(p, model_cfg, track, rule_ids, allow_placeholder=False)
            rep.add(f"validate_{p.name}", r["invalid"] == 0 and r["examples"] > 0,
                    f"{r['examples']} examples, {r['invalid']} invalid {r['first_problems'][:1] or ''}")
        n_train = sum(1 for _ in train.open(encoding="utf-8"))
        rep.add("min_examples", n_train >= model_cfg.tuning.limits.recommended_min_examples,
                f"{n_train} training examples (docs recommend >= {model_cfg.tuning.limits.recommended_min_examples})",
                blocking=False)

    if check_credentials:
        try:
            import google.auth

            google.auth.default()
            rep.add("credentials", True, "Application Default Credentials found")
        except Exception as exc:  # noqa: BLE001
            rep.add("credentials", False, f"no usable Google credentials ({type(exc).__name__}); "
                    "run `gcloud auth application-default login` or set GOOGLE_APPLICATION_CREDENTIALS")

    if client is not None:
        try:
            client.models.get(model=base_model)
            rep.add("base_model_online", True, f"{base_model} reachable")
        except Exception as exc:  # noqa: BLE001
            rep.add("base_model_online", False, f"{base_model}: {map_exception(exc)}")

    if registry is not None and files_ok:
        key = idempotency_key(train, val, base_model, hyperparameters(model_cfg), track)
        existing = registry.find_by_key(key)
        rep.add("no_duplicate_job", existing is None,
                "no job with identical data/settings" if existing is None else
                f"identical job already exists: {existing['job_name']} ({existing['state']})")
    return rep


# --------------------------------------------------------------------------- upload


def upload_to_gcs(
    project: ProjectConfig, tuning_dir: Path, storage_client: Any | None = None, workers: int = 12
) -> dict[str, str]:
    """Upload planned images + JSONL files. Requires allow_cloud_upload.

    Idempotent and resumable: one bucket listing finds what is already there (same name and size), and only the
    rest is uploaded, in parallel (``workers`` threads) - one-file-at-a-time was ~0.6 photos/s.
    """
    from concurrent.futures import ThreadPoolExecutor

    cloud = project.cloud
    if not cloud.allow_cloud_upload:
        raise AuthorizationError("Upload to Cloud Storage is not authorized (NGO_ALLOW_CLOUD_UPLOAD).")
    if not cloud.gcs_bucket or cloud.gcs_bucket == PLACEHOLDER_BUCKET:
        raise ConfigError("NGO_GCS_BUCKET is not configured")
    if storage_client is None:
        from google.cloud import storage

        storage_client = storage.Client(project=cloud.project_id)
    bucket = storage_client.bucket(cloud.gcs_bucket)
    plan = pd.read_csv(tuning_dir / "upload_plan.csv")
    prefix = f"gs://{cloud.gcs_bucket}/"
    for uri in plan.gcs_uri:
        if not uri.startswith(prefix):
            raise TuningError("upload plan references a different bucket than configured; rebuild the tuning data")
    images_prefix = f"{cloud.gcs_prefix.strip('/')}/images/"
    present = {b.name: b.size for b in storage_client.list_blobs(cloud.gcs_bucket, prefix=images_prefix)}
    todo = [r for r in plan.itertuples() if present.get(r.gcs_uri[len(prefix):]) != r.bytes]
    skipped = len(plan) - len(todo)

    def put(row: Any) -> None:
        bucket.blob(row.gcs_uri[len(prefix):]).upload_from_filename(row.local_path, content_type=row.mime_type)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(put, todo))  # re-raises the first failure
    uploaded = len(todo)
    uris: dict[str, str] = {}
    for name in ("train", "validation"):
        key = f"{cloud.gcs_prefix.strip('/')}/datasets/{name}.jsonl"
        bucket.blob(key).upload_from_filename(str(tuning_dir / f"{name}.jsonl"), content_type="application/jsonl")
        uris[name] = f"gs://{cloud.gcs_bucket}/{key}"
    logger.info("uploaded %d images (%d already present) and 2 dataset files", uploaded, skipped)
    return uris


# --------------------------------------------------------------------------- submit / poll


def make_client(project: ProjectConfig) -> Any:  # pragma: no cover - needs network/credentials
    from google import genai
    from google.genai import types

    return genai.Client(
        vertexai=True, project=project.cloud.project_id, location=project.cloud.location,
        http_options=types.HttpOptions(api_version="v1beta1"),  # as in the official tuning example
    )


def submit_job(
    project: ProjectConfig,
    model_cfg: ModelConfig,
    tuning_dir: Path,
    track: str,
    dataset_uris: dict[str, str],
    registry: ModelRegistry,
    submit: bool,
    client: Any | None = None,
    base_model: str | None = None,
    force_new: bool = False,
    rule_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Submit (or, idempotently, return) a tuning job. ``submit`` must be True."""
    base_model = base_model or model_cfg.models["primary"].id
    if not submit:
        raise AuthorizationError("Submission requires an explicit --submit-cloud-job flag.")
    if not project.cloud.allow_paid_tuning:
        raise AuthorizationError("Paid tuning is not authorized (NGO_ALLOW_PAID_TUNING is not true).")

    report = run_preflight(project, model_cfg, tuning_dir, track, base_model, registry=None,
                           check_credentials=client is None, rule_ids=rule_ids)
    if not report.can_submit:
        raise TuningError("preflight failed:\n" + report.render())

    hyper = hyperparameters(model_cfg)
    key = idempotency_key(tuning_dir / "train.jsonl", tuning_dir / "validation.jsonl", base_model, hyper, track)
    existing = registry.find_by_key(key)
    if existing is not None and not force_new:
        logger.warning("identical tuning job already exists (%s); not submitting a duplicate", existing["job_name"])
        return existing

    from google.genai import types

    client = client or make_client(project)
    cfg_kwargs: dict[str, Any] = {
        "tuned_model_display_name": f"{model_cfg.tuning.display_name_prefix}-track{track}-{key[:8]}",
        "validation_dataset": types.TuningValidationDataset(gcs_uri=dataset_uris["validation"]),
    }
    for k, v in hyper.items():
        if v is not None:
            cfg_kwargs[k] = v
    try:
        job = client.tunings.tune(
            base_model=base_model,
            training_dataset=types.TuningDataset(gcs_uri=dataset_uris["train"]),
            config=types.CreateTuningJobConfig(**cfg_kwargs),
        )
    except Exception as exc:  # noqa: BLE001
        raise TuningError(f"tuning job submission failed: {map_exception(exc)}") from exc
    return registry.record_submission(
        job.name, key, track=track, base_model=base_model, train_uri=dataset_uris["train"],
        val_uri=dataset_uris["validation"], display_name=cfg_kwargs["tuned_model_display_name"],
        state=_state_name(job), hyperparameters=hyper,
    )


def _state_name(job: Any) -> str:
    s = getattr(job, "state", None)
    return getattr(s, "name", None) or str(s)


def poll_job(
    registry: ModelRegistry,
    job_name: str,
    client: Any,
    interval_s: float = 60.0,
    timeout_s: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Poll until a terminal state; record tuned model/endpoint on success; raise on failure/timeout."""
    start = clock()
    while True:
        try:
            job = client.tunings.get(name=job_name)
        except Exception as exc:  # noqa: BLE001
            mapped = map_exception(exc)
            if not mapped.retryable:
                raise TuningError(f"polling failed: {mapped}") from exc
            logger.warning("transient polling error: %s", mapped)
            job = None
        if job is not None:
            state = _state_name(job)
            if state in SUCCESS_STATES:
                tm = getattr(job, "tuned_model", None)
                model, endpoint = getattr(tm, "model", None), getattr(tm, "endpoint", None)
                if not (model or endpoint):
                    registry.update(job_name, state=state, error="job succeeded but no tuned model reported")
                    raise TuningError("job succeeded but the provider reported no tuned model/endpoint")
                return registry.update(job_name, state=state, tuned_model=model, tuned_endpoint=endpoint)
            if state in FAILED_STATES:
                err = getattr(getattr(job, "error", None), "message", None) or "no error message"
                registry.update(job_name, state=state, error=str(err)[:500])
                raise TuningError(f"tuning job ended in {state}: {err}")
            if state in RUNNING_STATES:
                registry.update(job_name, state=state)
            else:
                logger.warning("unrecognized job state %s", state)
        if timeout_s is not None and clock() - start > timeout_s:
            raise TuningError(f"timed out waiting for {job_name}")
        sleep(interval_s)


def refresh_job(registry: ModelRegistry, job_name: str, client: Any) -> dict[str, Any]:
    """One-shot status check: fetch, record state (and tuned model on success); never waits or raises
    for a failed/running job (the returned entry carries state + error)."""
    try:
        job = client.tunings.get(name=job_name)
    except Exception as exc:  # noqa: BLE001
        raise TuningError(f"status check failed: {map_exception(exc)}") from exc
    state = _state_name(job)
    fields: dict[str, Any] = {"state": state}
    if state in SUCCESS_STATES:
        tm = getattr(job, "tuned_model", None)
        fields.update(tuned_model=getattr(tm, "model", None), tuned_endpoint=getattr(tm, "endpoint", None))
    elif state in FAILED_STATES:
        fields["error"] = str(getattr(getattr(job, "error", None), "message", None) or "no error message")[:500]
    return registry.update(job_name, **fields)


def cancel_job(registry: ModelRegistry, job_name: str, client: Any) -> dict[str, Any]:
    try:
        client.tunings.cancel(name=job_name)
    except Exception as exc:  # noqa: BLE001
        raise TuningError(f"cancel failed: {map_exception(exc)}") from exc
    return registry.update(job_name, state="JOB_STATE_CANCELLING")


__all__ = ["run_preflight", "upload_to_gcs", "submit_job", "poll_job", "refresh_job", "cancel_job", "PreflightReport",
           "idempotency_key", "ProviderError", "ModelUnavailableError"]
