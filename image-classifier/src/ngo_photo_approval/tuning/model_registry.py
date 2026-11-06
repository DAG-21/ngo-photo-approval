"""Persistent registry of tuning jobs and tuned models (JSON file, atomic writes).

The registry is the single source of truth for job ids, idempotency keys and the *active* tuned
model. Tuned endpoints are only ever recorded from a provider response, never guessed.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..exceptions import TuningError

SUCCESS_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}
FAILED_STATES = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
RUNNING_STATES = {"JOB_STATE_QUEUED", "JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_UPDATING",
                  "JOB_STATE_PAUSED", "JOB_STATE_CANCELLING", "JOB_STATE_UNSPECIFIED"}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ModelRegistry:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    # -- persistence ---------------------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"jobs": [], "active_job": None}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TuningError(f"registry file is corrupt: {self.path.name}") from exc

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # -- queries -------------------------------------------------------------------------
    def jobs(self) -> list[dict[str, Any]]:
        return self._load()["jobs"]

    def find_by_key(self, idempotency_key: str, exclude_failed: bool = True) -> dict[str, Any] | None:
        for job in reversed(self.jobs()):
            if job["idempotency_key"] == idempotency_key and not (
                exclude_failed and job.get("state") in FAILED_STATES
            ):
                return job
        return None

    def get(self, job_name: str) -> dict[str, Any] | None:
        return next((j for j in self.jobs() if j["job_name"] == job_name), None)

    # -- mutations -----------------------------------------------------------------------
    def record_submission(self, job_name: str, idempotency_key: str, **fields: Any) -> dict[str, Any]:
        data = self._load()
        entry = {"job_name": job_name, "idempotency_key": idempotency_key, "created_at": _now(),
                 "updated_at": _now(), "state": "JOB_STATE_PENDING", "tuned_model": None,
                 "tuned_endpoint": None, "error": None, **fields}
        data["jobs"].append(entry)
        self._save(data)
        return entry

    def update(self, job_name: str, **fields: Any) -> dict[str, Any]:
        data = self._load()
        for job in data["jobs"]:
            if job["job_name"] == job_name:
                job.update(fields, updated_at=_now())
                self._save(data)
                return job
        raise TuningError(f"unknown job {job_name!r}")

    def activate(self, job_name: str) -> dict[str, Any]:
        job = self.get(job_name)
        if job is None:
            raise TuningError(f"unknown job {job_name!r}")
        if job.get("state") not in SUCCESS_STATES or not (job.get("tuned_endpoint") or job.get("tuned_model")):
            raise TuningError("only a succeeded job with a recorded tuned model/endpoint can be activated")
        data = self._load()
        data["active_job"] = job_name
        self._save(data)
        return job

    def active(self) -> dict[str, Any] | None:
        data = self._load()
        return self.get(data["active_job"]) if data.get("active_job") else None
