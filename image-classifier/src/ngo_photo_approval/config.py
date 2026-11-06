"""Typed configuration loading (YAML files + environment overrides)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .exceptions import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

logger = logging.getLogger(__name__)


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_value(name: str) -> str | None:
    """Env value, or None if unset/blank or still a template placeholder such as REPLACE_WITH_YOUR_BUCKET_NAME."""
    v = os.environ.get(name)
    if v is None:
        return None
    v = v.strip()
    return None if (not v or v.upper().startswith("REPLACE_WITH")) else v


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a mapping at the top level")
    return data


# --------------------------------------------------------------------------- project.yaml


class PathsConfig(BaseModel):
    data_dir: Path = Path("data/raw")
    rules_document: Path | None = None
    output_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    docs_dir: Path = Path("docs")


class QualityConfig(BaseModel):
    blur_laplacian_var_low: float = 60.0
    brightness_low: float = 40.0
    brightness_high: float = 220.0
    contrast_low: float = 25.0
    min_side_px_warn: int = 480


class DatasetConfig(BaseModel):
    folder_label_map: dict[str, str]
    trainable_labels: list[str] = ["approved", "not_approved"]
    supported_image_extensions: list[str] = [".jpg", ".jpeg", ".png", ".webp"]
    known_non_image_extensions: list[str] = [".mp4", ".mov", ".avi"]
    inventory_pending: bool = True
    near_duplicate_hamming_threshold: int = 4
    # Which cross-label duplicate clusters to drop from supervised data: "exact" (byte-identical
    # files with conflicting final labels), "near" (also near-duplicates), or "none".
    label_conflict_exclusion: Literal["exact", "near", "none"] = "exact"
    quality: QualityConfig = QualityConfig()

    @model_validator(mode="after")
    def _pending_not_trainable(self) -> DatasetConfig:
        if "pending" in self.trainable_labels:
            raise ValueError("'pending' must never be a trainable label")
        bad = set(self.folder_label_map.values()) - {"approved", "not_approved", "pending"}
        if bad:
            raise ValueError(f"folder_label_map has unknown normalized labels: {sorted(bad)}")
        return self


class SplitConfig(BaseModel):
    seed: int = 42
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15
    use_filename_groups: bool = True

    @model_validator(mode="after")
    def _sum_to_one(self) -> SplitConfig:
        if abs(self.train + self.validation + self.test - 1.0) > 1e-6:
            raise ValueError("split ratios must sum to 1.0")
        return self


class CloudConfig(BaseModel):
    provider: str = "google_vertex"
    # "vertex": Google Cloud project + Application Default Credentials (inference AND tuning).
    # "api_key": Gemini Developer API key from the GEMINI_API_KEY env var (inference only; no tuning).
    auth_mode: Literal["vertex", "api_key"] = "vertex"
    project_id: str | None = None
    location: str = "us"
    gcs_bucket: str | None = None
    gcs_prefix: str = "ngo-photo-approval"
    allow_cloud_inference: bool = False
    allow_cloud_upload: bool = False
    allow_paid_tuning: bool = False


class ApiConfig(BaseModel):
    max_upload_mb: float = 15
    allowed_content_types: list[str] = ["image/jpeg", "image/png", "image/webp"]


class RulesSourceConfig(BaseModel):
    """How rules.yaml is tied to the client's rules document."""

    # A rule is mandatory ONLY if the document itself uses one of these words for it (or its section).
    mandatory_markers: list[str] = ["mandatory", "compulsory", "must", "non-negotiable", "without fail"]
    # Refuse to load rules.yaml if it is out of sync with the document (fail closed).
    enforce_source_sync: bool = True


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rules: RulesSourceConfig = RulesSourceConfig()
    paths: PathsConfig = PathsConfig()
    dataset: DatasetConfig
    splits: SplitConfig = SplitConfig()
    cloud: CloudConfig = CloudConfig()
    review: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    api: ApiConfig = ApiConfig()
    logging: dict[str, Any] = Field(default_factory=lambda: {"level": "INFO"})

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else (PROJECT_ROOT / path)

    @property
    def data_dir(self) -> Path:
        return self.resolve(self.paths.data_dir)

    @property
    def output_dir(self) -> Path:
        return self.resolve(self.paths.output_dir)

    @property
    def artifacts_dir(self) -> Path:
        return self.resolve(self.paths.artifacts_dir)


def load_project_config(path: Path | None = None) -> ProjectConfig:
    """Load project.yaml and apply environment overrides (.env is read if present)."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    raw = load_yaml(path or CONFIG_DIR / "project.yaml")
    cfg = ProjectConfig.model_validate(raw)

    if os.environ.get("NGO_DATA_DIR"):
        cfg.paths.data_dir = Path(os.environ["NGO_DATA_DIR"])
    if _env_value("GOOGLE_CLOUD_PROJECT"):
        cfg.cloud.project_id = _env_value("GOOGLE_CLOUD_PROJECT")
    if _env_value("GOOGLE_CLOUD_LOCATION"):
        cfg.cloud.location = _env_value("GOOGLE_CLOUD_LOCATION")  # type: ignore[assignment]
    if os.environ.get("NGO_AUTH_MODE"):
        mode = os.environ["NGO_AUTH_MODE"].strip().lower()
        if mode not in {"vertex", "api_key"}:
            raise ConfigError("NGO_AUTH_MODE must be 'vertex' or 'api_key'")
        cfg.cloud.auth_mode = mode  # type: ignore[assignment]
    if _env_value("NGO_GCS_BUCKET"):
        cfg.cloud.gcs_bucket = _env_value("NGO_GCS_BUCKET")
    if _env_value("NGO_GCS_PREFIX"):
        cfg.cloud.gcs_prefix = _env_value("NGO_GCS_PREFIX")  # type: ignore[assignment]
    for env_name, attr in (
        ("NGO_ALLOW_CLOUD_INFERENCE", "allow_cloud_inference"),
        ("NGO_ALLOW_CLOUD_UPLOAD", "allow_cloud_upload"),
        ("NGO_ALLOW_PAID_TUNING", "allow_paid_tuning"),
    ):
        val = _env_bool(env_name)
        if val is not None:
            setattr(cfg.cloud, attr, val)
    if os.environ.get("NGO_MAX_UPLOAD_MB"):
        cfg.api.max_upload_mb = float(os.environ["NGO_MAX_UPLOAD_MB"])
    if os.environ.get("NGO_LOG_LEVEL"):
        cfg.logging["level"] = os.environ["NGO_LOG_LEVEL"]
    return cfg


# --------------------------------------------------------------------------- thresholds.yaml


class ThresholdSelectionConfig(BaseModel):
    max_false_approval_rate: float = 0.05
    min_validation_recall: float = 0.0
    candidate_grid_step: float = 0.01


class CalibrationConfig(BaseModel):
    method: Literal["sigmoid", "isotonic"] = "sigmoid"
    min_samples_for_calibration: int = 200
    min_samples_per_class: int = 50
    max_ece_for_calibrated_claim: float = 0.08
    min_validation_roc_auc: float = 0.60  # a no-skill model must not be presented as a usable probability


class PolicyConfig(BaseModel):
    approval_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    enforce_mandatory_rules: bool = True
    missing_rule_result_policy: Literal["reject", "review"] = "reject"
    uncertain_result_policy: Literal["reject", "review"] = "reject"
    allow_review_status: bool = False
    threshold_selection: ThresholdSelectionConfig = ThresholdSelectionConfig()
    calibration: CalibrationConfig = CalibrationConfig()


def load_policy_config(path: Path | None = None) -> PolicyConfig:
    return PolicyConfig.model_validate(load_yaml(path or CONFIG_DIR / "thresholds.yaml"))


# --------------------------------------------------------------------------- model.yaml


class ModelRef(BaseModel):
    id: str
    verified: bool = False


class TunedModelRef(BaseModel):
    endpoint: str | None = None
    model_resource: str | None = None
    version: str | None = None


class GenerationConfig(BaseModel):
    temperature: float = 0.0
    max_output_tokens: int = 2048
    media_resolution: str = "MEDIA_RESOLUTION_HIGH"
    request_timeout_s: float = 120


class ImageInputConfig(BaseModel):
    max_side_px: int = 3072
    jpeg_quality: int = 92
    max_bytes: int = 18 * 1024 * 1024
    # Images for TUNING and for the tuned model: every image is resized so its LONG side is exactly this many
    # pixels (down- or up-scaled) and re-encoded identically (metadata stripped). In this dataset resolution and
    # file size alone separate the labels (provenance), so they must not be learnable. None disables it.
    tuning_normalize_long_side_px: int | None = 1280
    tuning_jpeg_quality: int = 90


class RetryConfig(BaseModel):
    max_attempts: int = 4
    initial_backoff_s: float = 2.0
    max_backoff_s: float = 30.0
    rate_limit_rps: float = 2.0


class TuningLimits(BaseModel):
    max_images_per_example: int = 30
    max_image_bytes: int = 20 * 1024 * 1024
    max_tokens_per_example: int = 131072
    max_training_file_bytes: int = 1024**3
    max_multimodal_examples: int = 300_000
    recommended_min_examples: int = 100
    allowed_mime_types: list[str] = ["image/jpeg", "image/png", "image/webp"]


class TuningConfig(BaseModel):
    task_track: Literal["A", "B"] = "A"
    epoch_count: int | None = None
    adapter_size: int | None = None
    learning_rate_multiplier: float | None = None
    display_name_prefix: str = "ngo-photo-approval"
    limits: TuningLimits = TuningLimits()


class PricingConfig(BaseModel):
    input_per_million_tokens_usd: float | None = None
    output_per_million_tokens_usd: float | None = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    provider: str = "google_vertex"
    models: dict[str, ModelRef]
    tuned_model: TunedModelRef = TunedModelRef()
    generation: GenerationConfig = GenerationConfig()
    image_input: ImageInputConfig = ImageInputConfig()
    retry: RetryConfig = RetryConfig()
    tuning: TuningConfig = TuningConfig()
    pricing: PricingConfig = PricingConfig()


def load_model_config(path: Path | None = None) -> ModelConfig:
    return ModelConfig.model_validate(load_yaml(path or CONFIG_DIR / "model.yaml"))


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
