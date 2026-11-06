from __future__ import annotations

import json
import math

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from ngo_photo_approval.data.preprocessing import normalize_for_tuning
from ngo_photo_approval.exceptions import ModelArtifactError
from ngo_photo_approval.scoring.feature_builder import (
    FEATURE_SETS,
    build_features,
    feature_names,
    image_logit,
)
from ngo_photo_approval.scoring.image_score import (
    ImageScorer,
    tensor_from_normalized_file,
    tensor_from_original,
)

from ..conftest import make_image


def test_image_feature_sets_and_names(rule_set):
    assert {"image", "image+rules"} <= set(FEATURE_SETS)
    assert feature_names(rule_set, "image") == ["img_logit"]
    names = feature_names(rule_set, "image+rules")
    assert names[0] == "img_logit" and names[1:] == feature_names(rule_set, "rules")


def test_image_features_need_an_image_score_and_stay_finite(rule_set):
    with pytest.raises(ValueError, match="image score"):
        build_features(rule_set, "image")
    f = build_features(rule_set, "image", image_score=0.75)
    assert list(f) == ["img_logit"] and f["img_logit"] == pytest.approx(math.log(3))
    assert math.isfinite(image_logit(0.0)) and math.isfinite(image_logit(1.0))
    assert image_logit(0.2) < 0 < image_logit(0.8)
    # rule-only feature sets are unchanged and ignore an image score
    assert list(build_features(rule_set, "rules")) == feature_names(rule_set, "rules")


def test_new_photo_preprocessing_matches_the_training_embeddings_exactly(tmp_path, model_cfg):
    """The training embeddings came from the normalized copies on disk; a new photo is normalized on the fly.
    Both must yield the identical CLIP input, or the image score is silently skewed."""
    cfg = model_cfg.image_input
    cfg.tuning_normalize_long_side_px = 480
    original = make_image(tmp_path / "photo.jpg", seed=3, size=(1600, 900))
    derived = tmp_path / "photo_norm.jpg"
    derived.write_bytes(normalize_for_tuning(original, cfg).data)  # what the training export wrote to disk
    a = tensor_from_normalized_file(derived)
    b = tensor_from_original(original, cfg)
    assert a.shape == (3, 224, 224) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)


def test_tall_photo_is_squashed_not_center_cropped(tmp_path, model_cfg):
    """Framing/cropping at the edges must stay visible: a marker at the very top must still change the input."""
    from PIL import Image

    cfg = model_cfg.image_input
    cfg.tuning_normalize_long_side_px = 400
    base = Image.new("RGB", (200, 800), (128, 128, 128))
    marked = base.copy()
    marked.paste((255, 0, 0), (0, 0, 200, 40))  # red band across the top edge
    base.save(tmp_path / "a.jpg", quality=95)
    marked.save(tmp_path / "b.jpg", quality=95)
    diff = np.abs(tensor_from_original(tmp_path / "a.jpg", cfg) - tensor_from_original(tmp_path / "b.jpg", cfg))
    assert diff[:, :20, :].max() > 1.0  # the top rows differ clearly: they were not cropped away


class FakeEmbedder:
    def embed_tensors(self, tensors: np.ndarray) -> np.ndarray:
        m = tensors.reshape(len(tensors), -1).mean(axis=1, keepdims=True)
        return np.hstack([m * k for k in (1.0, -2.0, 0.5, 3.0)])


def make_artifact(tmp_path, norm):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + rng.normal(scale=0.5, size=60) > 0).astype(int)
    d = tmp_path / "image_score"
    d.mkdir()
    joblib.dump(LogisticRegression().fit(X, y), d / "model.joblib")
    (d / "metadata.json").write_text(json.dumps({"model_version": "img-test", "normalize_long_side_px": norm}))
    return d


def test_image_scorer_scores_a_photo_between_0_and_1(tmp_path, model_cfg):
    cfg = model_cfg.image_input
    cfg.tuning_normalize_long_side_px = 320
    scorer = ImageScorer.load(make_artifact(tmp_path, 320), cfg, embedder=FakeEmbedder())
    p = scorer.score(make_image(tmp_path / "x.jpg", seed=1, size=(640, 480)))
    assert 0.0 <= p <= 1.0 and scorer.version == "img-test"


def test_image_scorer_refuses_a_changed_normalization_or_missing_artifacts(tmp_path, model_cfg):
    cfg = model_cfg.image_input
    cfg.tuning_normalize_long_side_px = 320
    d = make_artifact(tmp_path, 1280)  # trained on 1280 px images, but the config now says 320
    with pytest.raises(ModelArtifactError, match="normalization changed"):
        ImageScorer.load(d, cfg, embedder=FakeEmbedder())
    with pytest.raises(ModelArtifactError, match="not found"):
        ImageScorer.load(tmp_path / "nothing", cfg, embedder=FakeEmbedder())
