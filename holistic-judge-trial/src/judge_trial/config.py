"""Trial configuration. The main project is used strictly read-only."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

TRIAL_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = TRIAL_ROOT / "config" / "trial.yaml"
PROMPTS_DIR = TRIAL_ROOT / "prompts"


class ModelCfg(BaseModel):
    id: str = "gemini-3.5-flash-lite"


class GenerationCfg(BaseModel):
    temperature: float = 0.0
    max_output_tokens: int = 1024
    media_resolution: str = "MEDIA_RESOLUTION_HIGH"
    request_timeout_s: float = 120


class ImagesCfg(BaseModel):
    normalized_dir: str = "data/derived/tuning_images_norm1280"


class PoolsCfg(BaseModel):
    dev_size_per_class: int = 150
    confirm_size_per_class: int = 150
    seed: int = 42
    example_similarity_max: float | None = 0.88


class ExamplesCfg(BaseModel):
    k_approved: int = 4
    k_not_approved: int = 4
    seed: int = 42
    reasons_file: str | None = None
    fixed_not_approved: list[str] | None = None  # use these train photos as the disapproved examples instead of cluster medoids


class CacheCfg(BaseModel):
    mode: Literal["auto", "on", "off"] = "auto"
    ttl_seconds: int = 3600
    min_prefix_tokens: int = 2048


class RunCfg(BaseModel):
    workers: int = 4
    max_calls_default: int = 400


class CostCfg(BaseModel):
    tokens_per_image: int = 1120
    expected_output_tokens: int = 250
    input_per_million_usd: float | None = None
    cached_input_per_million_usd: float | None = None
    output_per_million_usd: float | None = None
    cache_storage_per_million_tokens_per_hour_usd: float | None = None


class TrialConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    main_project: str = "../image-classifier"
    model: ModelCfg = ModelCfg()
    generation: GenerationCfg = GenerationCfg()
    images: ImagesCfg = ImagesCfg()
    pools: PoolsCfg = PoolsCfg()
    examples: ExamplesCfg = ExamplesCfg()
    cache: CacheCfg = CacheCfg()
    run: RunCfg = RunCfg()
    cost: CostCfg = CostCfg()

    @property
    def main_root(self) -> Path:
        p = Path(self.main_project)
        return (p if p.is_absolute() else TRIAL_ROOT / p).resolve()

    @property
    def normalized_dir(self) -> Path:
        p = Path(self.images.normalized_dir)
        return p if p.is_absolute() else self.main_root / p

    @property
    def examples_path(self) -> Path:
        return TRIAL_ROOT / "examples" / "examples.json"

    def pool_path(self, name: str) -> Path:
        return TRIAL_ROOT / "pools" / f"{name}.csv"


def load_trial_config(path: Path | None = None) -> TrialConfig:
    raw = yaml.safe_load((path or CONFIG_PATH).read_text(encoding="utf-8")) or {}
    return TrialConfig.model_validate(raw)


def ensure_main_importable(cfg: TrialConfig) -> None:
    """Make the main project's package importable (read-only use) when it is not installed in this interpreter."""
    try:
        import ngo_photo_approval  # noqa: F401
    except ImportError:
        src = cfg.main_root / "src"
        if not src.is_dir():
            raise RuntimeError(f"main project not found at {cfg.main_root}") from None
        sys.path.insert(0, str(src))
