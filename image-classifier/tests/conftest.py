"""Shared fixtures. All images are SYNTHETIC; no client data is used in tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ngo_photo_approval.config import (
    ModelConfig,
    PolicyConfig,
    ProjectConfig,
    load_model_config,
    load_policy_config,
)
from ngo_photo_approval.models.cloud_client import BackendRequest, BackendResponse
from ngo_photo_approval.schemas import Rule, RuleSet, UsageInfo


def make_image(path: Path, seed: int = 0, size: tuple[int, int] = (320, 240), fmt: str = "JPEG") -> Path:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format=fmt)
    return path


def make_dataset(root: Path) -> Path:
    """approved(12) / disapproved(8) / pending-approval(3), one exact duplicate, one corrupt file."""
    n = 0
    for k in range(1, 13):
        make_image(root / "approved" / f"prog_a-{(k - 1) // 3 + 1} ({k}).jpg", seed=n)
        n += 1
    for k in range(1, 9):
        make_image(root / "disapproved" / f"prog_b-{(k - 1) // 4 + 1} ({k}).jpg", seed=n)
        n += 1
    for k in range(1, 4):
        make_image(root / "pending-approval" / f"prog_c-1 ({k}).jpg", seed=n)
        n += 1
    # exact duplicate of an approved image, inside approved
    (root / "approved" / "prog_a-9 (1).jpg").write_bytes((root / "approved" / "prog_a-1 (1).jpg").read_bytes())
    (root / "disapproved" / "prog_b-9 (1).jpg").write_bytes(b"this is not an image")
    (root / "pending-approval" / "clip-1.mp4").write_bytes(b"\x00\x00\x00 ftypmp42")
    return root


def make_rules(extra: list[dict] | None = None) -> RuleSet:
    base = [
        dict(rule_id="R001", name="banner", description="banner present", requirement_type="mandatory"),
        dict(rule_id="R002", name="uniform", description="uniform present", requirement_type="mandatory"),
        dict(rule_id="R003", name="beneficiaries", description="people present", requirement_type="mandatory"),
        dict(rule_id="R004", name="nice_lighting", description="good lighting", requirement_type="scored",
             uncertain_policy="score_only"),
    ]
    return RuleSet(rule_set_version="test-1.0", rules=[Rule(**r) for r in base + (extra or [])])


@pytest.fixture
def rule_set() -> RuleSet:
    return make_rules()


@pytest.fixture
def policy() -> PolicyConfig:
    return PolicyConfig(approval_threshold=0.80)


@pytest.fixture
def model_cfg() -> ModelConfig:
    cfg = load_model_config()
    cfg.retry.initial_backoff_s = 0.0
    cfg.retry.rate_limit_rps = 0.0
    return cfg


@pytest.fixture
def real_policy() -> PolicyConfig:
    return load_policy_config()


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    return make_dataset(tmp_path / "raw")


@pytest.fixture
def project(tmp_path: Path, dataset_dir: Path) -> ProjectConfig:
    return ProjectConfig.model_validate({
        "paths": {"data_dir": str(dataset_dir), "output_dir": str(tmp_path / "out"),
                  "artifacts_dir": str(tmp_path / "art")},
        "dataset": {"folder_label_map": {"approved": "approved", "disapproved": "not_approved",
                                         "pending-approval": "pending"}},
        "api": {"max_upload_mb": 1},
        "review": {"enabled": False},
    })


def checks_json(statuses: dict[str, str], extra: list[dict] | None = None) -> str:
    items = [{"rule_id": k, "status": v, "evidence": f"evidence for {k}"} for k, v in statuses.items()]
    return json.dumps({"checks": items + (extra or [])})


class FakeBackend:
    """Scripted VisionBackend: yields responses (strings) or raises exceptions in order."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.calls.append(request)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return BackendResponse(text=item, usage=UsageInfo(prompt_tokens=100, output_tokens=20, total_tokens=120,
                                                          latency_s=0.01))
