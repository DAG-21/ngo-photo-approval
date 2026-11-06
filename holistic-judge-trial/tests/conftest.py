from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judge_trial.backend import BackendResult  # noqa: E402
from judge_trial.config import ensure_main_importable, load_trial_config  # noqa: E402


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """No test may ever create a real Gemini client."""
    try:
        from google import genai
    except ImportError:
        return

    def boom(*_a, **_k):
        raise AssertionError("tests must never create a real Gemini client")

    monkeypatch.setattr(genai, "Client", boom)


@pytest.fixture
def cfg(tmp_path):
    c = load_trial_config().model_copy(deep=True)
    (tmp_path / "imgs").mkdir()
    c.images.normalized_dir = str(tmp_path / "imgs")
    return c


@pytest.fixture
def rule_set(cfg):
    ensure_main_importable(cfg)
    from ngo_photo_approval.rules.loader import load_rule_set

    return load_rule_set()


def make_image(cfg, image_id: str) -> None:
    Image.new("RGB", (32, 24), (10, 20, 30)).save(cfg.normalized_dir / f"{image_id}.jpg", "JPEG")


def good_answer(score: int = 80, decision: str = "APPROVED") -> str:
    return json.dumps({"observations": "ok", "reasons": [{"rule_id": "R010", "effect": "supports", "note": "sharp"}],
                       "score": score, "decision": decision, "cannot_judge": False})


class FakeBackend:
    """Stands in for Gemini. ``script`` is a list of response texts, used in order (use workers=1)."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.calls = self.prepared = self.closed = 0
        self.cache_active = False

    def prepare(self, prompt):
        self.prepared += 1

    def judge(self, image_bytes, mime_type):
        self.calls += 1
        text = self.script.pop(0) if self.script else good_answer()
        if isinstance(text, Exception):
            raise text
        return BackendResult(text=text, usage={"prompt_tokens": 100})

    def close(self):
        self.closed += 1
