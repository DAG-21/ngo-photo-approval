from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layered_engine.providers import ProviderResult, load_gemini_config  # noqa: E402
from layered_engine.rulebook import ResolvedProfile, Rulebook  # noqa: E402
from layered_engine.schemas import CriterionAnswer, DimensionAssessment, ModelAssessment, SceneFacts, SceneUnderstanding, Status  # noqa: E402

LEGACY, PROPOSED = "checklist5_legacy", "layered_proposed_v1"

FOOD = dict(f_photo="YES", f_single="YES", f_volunteers="YES", f_banner="YES", f_banner_prominent="YES", f_beneficiaries="YES", f_animals="NO",
            f_item_given="YES", f_handover="YES", f_volunteer_giver="YES", f_logo="YES", f_medical="NO", f_orientation="FRONT", f_group="GROUP",
            f_activity_type="FOOD_DISTRIBUTION")

DIMS = ("activity_communication", "technical_quality", "composition", "naturalness_storytelling", "brand_presentation", "visual_polish")


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


@pytest.fixture(scope="session")
def rb() -> Rulebook:
    return Rulebook()


@pytest.fixture(scope="session")
def legacy(rb) -> ResolvedProfile:
    return rb.resolve(LEGACY)


@pytest.fixture(scope="session")
def proposed(rb) -> ResolvedProfile:
    return rb.resolve(PROPOSED)


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_gemini_config()


def make_assessment(profile: ResolvedProfile, facts: Optional[dict] = None, answers: Optional[dict[str, str]] = None, dims: Optional[dict[str, Any]] = None,
                    flags: Optional[list[dict]] = None, default_dim: int = 4, image_id: str = "img_test", omit: tuple[str, ...] = (), evidence: Optional[dict[str, str]] = None) -> ModelAssessment:
    """A fully compliant food-distribution photograph, with overrides. Every asked criterion is answered YES unless overridden."""
    answers, evidence, dims = answers or {}, evidence or {}, dims or {}
    f = {**FOOD, **(facts or {})}
    crit = [CriterionAnswer(criterion_id=c.criterion_id, status=Status(answers.get(c.criterion_id, "YES")), evidence=evidence.get(c.criterion_id, "seen in the photograph"))
            for c in profile.asked_criteria() if c.criterion_id not in omit]
    dimensions = {}
    for d in DIMS:
        v = dims.get(d, default_dim)
        dimensions[d] = DimensionAssessment(applicable=False) if v is None else DimensionAssessment(applicable=True, score=v, explanation="x")
    return ModelAssessment(image_id=image_id, layer_1_scene_understanding=SceneUnderstanding(activity_description="volunteers serving food", scene_facts=SceneFacts(**f)),
                           layer_2_criteria=crit, layer_3_quality_dimensions=dimensions, layer_4_visual_flags=flags or [])


def verified_metadata(**over) -> dict:
    base = dict(consent_status="VERIFIED", rights_status="VERIFIED", caption_status="VERIFIED", credits_rights_status="VERIFIED")
    base.update(over)
    return base


class FakeProvider:
    """Returns a scripted response for each call. Never touches the network."""

    def __init__(self, texts=None, model_id: str = "fake-model"):
        self.texts = list(texts or [])
        self.model_id = model_id
        self.calls = 0
        self.seen: list[tuple[str, str]] = []

    def assess(self, image_bytes, mime_type, system_text, user_text, schema):
        self.calls += 1
        self.seen.append((system_text, user_text))
        item = self.texts.pop(0) if self.texts else None
        if isinstance(item, Exception):
            raise item
        return ProviderResult(text=item if item is not None else "{}", usage={"prompt_tokens": 1000, "output_tokens": 500}, model_id=self.model_id)


def response_text(assessment: ModelAssessment) -> str:
    d = assessment.model_dump(mode="json")
    for k in ("image_id", "model_recommendation", "model_overall_score", "schema_version"):
        d.pop(k, None)
    return json.dumps(d)


def write_image(path: Path, size=(64, 48)) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 130, 140)).save(path, "JPEG")
    return path
