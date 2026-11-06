"""FastAPI service: POST /predict, GET /health, GET /ready.

Run:  uvicorn ngo_photo_approval.api.main:app --host 127.0.0.1 --port 8000
(the default factory builds the service from configs/ and environment variables).

Uploads are size-limited while being read, type-checked by declared content type AND by decoding,
handled through a private temp file that is always deleted, and never echo local paths.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from ..config import (
    ProjectConfig,
    load_model_config,
    load_policy_config,
    load_project_config,
    setup_logging,
)
from ..exceptions import (
    AuthorizationError,
    ConfigError,
    InvalidImageError,
    ModelArtifactError,
    ModelUnavailableError,
    ProviderAuthError,
    ProviderError,
    ResponseValidationError,
    TransientProviderError,
)
from ..rules.loader import load_rule_set
from ..service.prediction_service import PredictionService, build_service

logger = logging.getLogger(__name__)

_SUFFIX = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


def _error(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "detail": detail})


def default_service_factory() -> tuple[PredictionService, ProjectConfig]:
    project = load_project_config()
    setup_logging(project.logging["level"])
    service = build_service(project, load_model_config(), load_policy_config(), load_rule_set())
    return service, project


def create_app(service_factory: Callable[[], tuple[PredictionService, ProjectConfig]] | None = None) -> FastAPI:
    app = FastAPI(title="NGO Photo Approval Engine", version="0.1.0")
    factory = service_factory or default_service_factory
    state: dict[str, Any] = {}

    def get() -> tuple[PredictionService, ProjectConfig]:
        if "service" not in state:
            state["service"], state["project"] = factory()
        return state["service"], state["project"]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> JSONResponse:
        try:
            service, _ = get()
        except (ConfigError, ModelArtifactError, AuthorizationError) as exc:
            return _error(503, "not_ready", str(exc))
        info = service.readiness()
        ok = bool(info["analyzer_available"] and info["probability_model_loaded"])
        return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, **info})

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), program: str | None = Form(default=None)) -> JSONResponse:
        try:
            service, project = get()
        except (ConfigError, ModelArtifactError, AuthorizationError) as exc:
            return _error(503, "service_unavailable", str(exc))

        ctype = (file.content_type or "").lower()
        if ctype not in project.api.allowed_content_types:
            return _error(415, "unsupported_media_type", f"allowed types: {', '.join(project.api.allowed_content_types)}")
        limit = int(project.api.max_upload_mb * 1024 * 1024)
        data = await file.read(limit + 1)
        if len(data) > limit:
            return _error(413, "payload_too_large", f"image exceeds {project.api.max_upload_mb} MB")
        if not data:
            return _error(422, "empty_upload", "uploaded file is empty")

        try:
            resp = service.predict_bytes(data, suffix=_SUFFIX[ctype], program=program)
        except InvalidImageError as exc:
            return _error(422, "invalid_image", str(exc))
        except AuthorizationError as exc:
            return _error(503, "cloud_inference_not_authorized", str(exc))
        except ModelArtifactError as exc:
            return _error(503, "model_artifacts_unavailable", str(exc))
        except ModelUnavailableError as exc:
            return _error(503, "model_unavailable", str(exc))
        except TransientProviderError as exc:
            return _error(503, "provider_temporarily_unavailable", str(exc))
        except ProviderAuthError:
            return _error(502, "provider_auth_error", "the model provider rejected the service credentials")
        except ProviderError as exc:
            return _error(502, "provider_error", str(exc))
        except ResponseValidationError as exc:
            return _error(502, "invalid_model_response", str(exc))
        return JSONResponse(content=resp.model_dump(mode="json"))

    return app


def _lazy_app() -> FastAPI:
    return create_app()


app = _lazy_app()
