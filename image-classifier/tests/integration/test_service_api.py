from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ngo_photo_approval.api.main import create_app
from ngo_photo_approval.exceptions import (
    AuthorizationError,
    InvalidImageError,
    ModelArtifactError,
    ProviderError,
    ResponseValidationError,
    TransientProviderError,
)
from ngo_photo_approval.models.baseline_analyzer import RuleAnalyzer
from ngo_photo_approval.schemas import Decision, PredictionResponse, ProbabilityType, ReasonCode
from ngo_photo_approval.scoring.feature_builder import feature_names
from ngo_photo_approval.scoring.predict import ProbabilityEstimate
from ngo_photo_approval.service.prediction_service import (
    PredictionService,
    items_from_directory,
    load_batch_results,
)

from ..conftest import FakeBackend, checks_json, make_image

ALL_PASS = {"R001": "PASS", "R002": "PASS", "R003": "PASS", "R004": "PASS"}


class ConstScorer:
    """Scorer stub returning a fixed probability, to isolate policy behaviour."""

    def __init__(self, rule_set, p=0.95, ptype=ProbabilityType.CALIBRATED):
        self.p, self.ptype = p, ptype
        self.metadata = {"calibration_status": "calibrated" if ptype == ProbabilityType.CALIBRATED else "uncalibrated_classifier"}
        self._names = feature_names(rule_set, "rules")

    def check_features(self, expected):
        assert expected == self._names

    def score(self, features):
        assert list(features) == self._names
        return ProbabilityEstimate(
            raw_classifier_probability=self.p,
            calibrated_probability=self.p if self.ptype == ProbabilityType.CALIBRATED else None,
            probability=self.p, probability_type=self.ptype, scoring_model_version="const-1")


def make_service(rule_set, policy, project, model_cfg, script, scorer="default", p=0.95, ptype=ProbabilityType.CALIBRATED):
    backend = FakeBackend(script)
    analyzer = RuleAnalyzer(backend, rule_set, model_cfg, "gemini-test")
    sc = ConstScorer(rule_set, p, ptype) if scorer == "default" else scorer
    return PredictionService(rule_set, policy, project, model_cfg, analyzer, sc, "rules", model_version="mv-1"), backend


@pytest.fixture
def img(tmp_path):
    return make_image(tmp_path / "photo.jpg", seed=11)


# ------------------------------------------------------------------ service


def test_happy_path_returns_validated_contract(rule_set, policy, project, model_cfg, img):
    svc, _ = make_service(rule_set, policy, project, model_cfg, [checks_json(ALL_PASS)], p=0.87)
    r = svc.predict_path(img, image_id="example_001")
    assert isinstance(r, PredictionResponse)
    assert r.decision == Decision.APPROVE and r.approval_probability == 0.87 and r.approval_percentage == 87.0
    assert r.probability_type == ProbabilityType.CALIBRATED and r.threshold == 0.80
    assert r.hard_rule_failures == [] and [c.status.value for c in r.checks] == ["PASS"] * 4
    assert (r.rule_set_version, r.model_id, r.model_version) == ("test-1.0", "gemini-test", "mv-1")
    assert r.heuristic_score == 100.0 and r.reason_codes == [ReasonCode.APPROVED]
    assert any("unconfirmed" in w for w in r.warnings)  # test rule set is not client-confirmed


def test_mandatory_failure_overrides_high_probability(rule_set, policy, project, model_cfg, img):
    svc, _ = make_service(rule_set, policy, project, model_cfg, [checks_json({**ALL_PASS, "R001": "FAIL"})], p=0.91)
    r = svc.predict_path(img)
    assert r.decision == Decision.REJECT and r.hard_rule_failures == ["R001"]
    assert r.approval_probability == 0.91 and r.approval_percentage == 91.0  # probability is not zeroed
    assert ReasonCode.MANDATORY_RULE_FAILED in r.reason_codes


def test_uncertain_mandatory_rule_rejects(rule_set, policy, project, model_cfg, img):
    svc, _ = make_service(rule_set, policy, project, model_cfg, [checks_json({**ALL_PASS, "R002": "UNCERTAIN"})], p=0.99)
    r = svc.predict_path(img)
    assert r.decision == Decision.REJECT and ReasonCode.MANDATORY_RULE_UNCERTAIN in r.reason_codes


def test_below_threshold_rejects_and_exact_threshold_approves(rule_set, policy, project, model_cfg, img):
    svc, _ = make_service(rule_set, policy, project, model_cfg, [checks_json(ALL_PASS)], p=0.79)
    assert svc.predict_path(img).decision == Decision.REJECT
    svc, _ = make_service(rule_set, policy, project, model_cfg, [checks_json(ALL_PASS)], p=0.80)
    assert svc.predict_path(img).decision == Decision.APPROVE


def test_uncalibrated_output_is_labelled_uncalibrated(rule_set, policy, project, model_cfg, img):
    svc, _ = make_service(rule_set, policy, project, model_cfg, [checks_json(ALL_PASS)], p=0.9,
                          ptype=ProbabilityType.UNCALIBRATED_CLASSIFIER)
    r = svc.predict_path(img)
    assert r.probability_type == ProbabilityType.UNCALIBRATED_CLASSIFIER
    assert any("uncalibrated" in w for w in r.warnings)


def test_no_probability_model_never_approves(rule_set, policy, project, model_cfg, img):
    svc, _ = make_service(rule_set, policy, project, model_cfg, [checks_json(ALL_PASS)], scorer=None)
    r = svc.predict_path(img)
    assert r.decision == Decision.REJECT and r.approval_probability is None and r.probability_type == ProbabilityType.NONE
    assert ReasonCode.PROBABILITY_UNAVAILABLE in r.reason_codes and r.heuristic_score == 100.0


def test_review_requires_both_switches(rule_set, policy, project, model_cfg, img):
    policy.allow_review_status = True
    policy.uncertain_result_policy = "review"
    from ngo_photo_approval.schemas import Rule, RuleSet

    rs = RuleSet(rule_set_version="r", rules=[Rule(rule_id="R001", name="a", description="d",
                                                   requirement_type="mandatory", uncertain_policy="review")])
    script = [checks_json({"R001": "UNCERTAIN"})]

    def run():
        backend = FakeBackend(script)
        svc = PredictionService(rs, policy, project, model_cfg, RuleAnalyzer(backend, rs, model_cfg, "m"),
                                ConstScorer(rs, 0.99) if False else None, "rules")
        return svc.predict_path(img).decision

    assert run() == Decision.REJECT  # project.review.enabled is false -> REVIEW unavailable
    project.review["enabled"] = True
    assert run() == Decision.REVIEW


def test_failures_raise_typed_errors_and_never_approve(rule_set, policy, project, model_cfg, img, tmp_path):
    svc, _ = make_service(rule_set, policy, project, model_cfg, ["not json"])
    with pytest.raises(ResponseValidationError):
        svc.predict_path(img)
    svc, _ = make_service(rule_set, policy, project, model_cfg, [checks_json({"R001": "PASS"})])
    with pytest.raises(ResponseValidationError):
        svc.predict_path(img)
    svc, _ = make_service(rule_set, policy, project, model_cfg, [TimeoutError()])
    with pytest.raises(TransientProviderError):
        svc.predict_path(img)
    svc, _ = make_service(rule_set, policy, project, model_cfg, [RuntimeError("x")])
    with pytest.raises(ProviderError):
        svc.predict_path(img)
    svc, _ = make_service(rule_set, policy, project, model_cfg, [checks_json(ALL_PASS)])
    with pytest.raises(InvalidImageError):
        svc.predict_path(tmp_path / "ghost.jpg")


def test_service_without_analyzer_requires_authorization(rule_set, policy, project, model_cfg, img):
    svc = PredictionService(rule_set, policy, project, model_cfg, None, None)
    with pytest.raises(AuthorizationError):
        svc.predict_path(img)
    assert svc.readiness()["analyzer_available"] is False


def test_feature_mismatch_between_scorer_and_rule_set_detected(rule_set, policy, project, model_cfg):
    class Wrong(ConstScorer):
        def check_features(self, expected):
            raise ModelArtifactError("different feature set")

    with pytest.raises(ModelArtifactError):
        PredictionService(rule_set, policy, project, model_cfg, None, Wrong(rule_set))


def test_output_contains_no_local_paths(rule_set, policy, project, model_cfg, img):
    svc, _ = make_service(rule_set, policy, project, model_cfg, [checks_json(ALL_PASS)])
    blob = svc.predict_path(img).model_dump_json()
    assert str(img.parent) not in blob and "photo.jpg" not in blob


# ------------------------------------------------------------------ batch


def test_batch_resume_skips_done_retries_errors(rule_set, policy, project, model_cfg, tmp_path):
    good1, good2 = make_image(tmp_path / "d" / "a.jpg", 1), make_image(tmp_path / "d" / "b.jpg", 2)
    (tmp_path / "d" / "c.jpg").write_bytes(b"corrupt")
    items = items_from_directory(tmp_path / "d", [".jpg"])
    assert len(items) == 3
    out = tmp_path / "out" / "pred.jsonl"

    svc, backend = make_service(rule_set, policy, project, model_cfg, [checks_json(ALL_PASS)])
    s1 = svc.predict_batch(items, out)
    assert s1 == {"processed": 3, "skipped_done": 0, "ok": 2, "errors": 1}
    assert len(backend.calls) == 2
    recs = {r["image_id"]: r for r in load_batch_results(out)}
    assert sorted(r["status"] for r in recs.values()) == ["error", "error"][:0] + ["error", "ok", "ok"]

    # simulate a torn last line from a crash, then fix the corrupt image and resume
    with open(out, "a", encoding="utf-8") as f:
        f.write('{"status": "ok", "image_id": "tor')
    make_image(tmp_path / "d" / "c.jpg", 3)
    svc2, backend2 = make_service(rule_set, policy, project, model_cfg, [checks_json(ALL_PASS)])
    s2 = svc2.predict_batch(items, out)
    assert s2 == {"processed": 1, "skipped_done": 2, "ok": 1, "errors": 0} and len(backend2.calls) == 1
    final = load_batch_results(out)
    assert len(final) == 3 and all(r["status"] == "ok" for r in final)
    assert good1.exists() and good2.exists()


def test_batch_stops_on_authorization_problem(rule_set, policy, project, model_cfg, tmp_path):
    img = make_image(tmp_path / "d" / "a.jpg")
    backend = FakeBackend([AuthorizationError("nope")])
    svc = PredictionService(rule_set, policy, project, model_cfg, RuleAnalyzer(backend, rule_set, model_cfg, "m"), None)
    with pytest.raises(AuthorizationError):
        svc.predict_batch(items_from_directory(img.parent, [".jpg"]), tmp_path / "o.jsonl")


# ------------------------------------------------------------------ API


def jpeg_bytes(size=(64, 48)):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 80, 40)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def client_factory(rule_set, policy, project, model_cfg):
    def build(script=None, scorer="default", ready=True):
        svc, _ = make_service(rule_set, policy, project, model_cfg, script or [checks_json(ALL_PASS)], scorer=scorer)
        return TestClient(create_app(lambda: (svc, project)), raise_server_exceptions=False)
    return build


def test_health_and_ready(client_factory):
    c = client_factory()
    assert c.get("/health").json() == {"status": "ok"}
    r = c.get("/ready")
    assert r.status_code == 200 and r.json()["ready"] is True and r.json()["rule_set_version"] == "test-1.0"
    assert client_factory(scorer=None).get("/ready").status_code == 503


def test_ready_reports_factory_failure(project):
    def broken():
        raise ModelArtifactError("probability model artifacts not found")

    r = TestClient(create_app(broken)).get("/ready")
    assert r.status_code == 503 and r.json()["error"] == "not_ready"
    assert TestClient(create_app(broken)).post("/predict", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")}).status_code == 503


def test_predict_success_and_response_validates(client_factory):
    r = client_factory().post("/predict", files={"file": ("secret_name.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    resp = PredictionResponse.model_validate(r.json())
    assert resp.decision == Decision.APPROVE and 0 <= resp.approval_percentage <= 100
    assert "secret_name" not in r.text and "ngo_upload_" not in r.text


def test_predict_rejects_unsupported_type(client_factory):
    r = client_factory().post("/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415 and r.json()["error"] == "unsupported_media_type"


def test_predict_rejects_oversized_upload(client_factory):
    r = client_factory().post("/predict", files={"file": ("big.jpg", b"\xff\xd8" + b"0" * (1024 * 1024 + 10), "image/jpeg")})
    assert r.status_code == 413


def test_predict_rejects_empty_and_corrupt_images(client_factory):
    c = client_factory()
    assert c.post("/predict", files={"file": ("a.jpg", b"", "image/jpeg")}).status_code == 422
    r = c.post("/predict", files={"file": ("a.jpg", b"definitely not a jpeg", "image/jpeg")})
    assert r.status_code == 422 and r.json()["error"] == "invalid_image"


@pytest.mark.parametrize(("script", "status", "code"), [
    ([TimeoutError()], 503, "provider_temporarily_unavailable"),
    ([type("E", (Exception,), {"code": 401})()], 502, "provider_auth_error"),
    ([type("E", (Exception,), {"code": 404})()], 503, "model_unavailable"),
    ([type("E", (Exception,), {"code": 400})()], 502, "provider_error"),
    (["this is not json"], 502, "invalid_model_response"),
    ([AuthorizationError("cloud inference is not authorized")], 503, "cloud_inference_not_authorized"),
])
def test_service_failures_map_to_meaningful_errors(client_factory, script, status, code):
    r = client_factory(script).post("/predict", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == status and r.json()["error"] == code
    assert "decision" not in r.json()  # a failure never carries an APPROVE


def test_temp_files_are_always_removed(client_factory, tmp_path, monkeypatch):
    spool = tmp_path / "spool"
    spool.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(spool))
    c = client_factory()
    assert c.post("/predict", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")}).status_code == 200
    assert c.post("/predict", files={"file": ("a.jpg", b"corrupt!", "image/jpeg")}).status_code == 422
    c2 = client_factory(["not json"])
    assert c2.post("/predict", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")}).status_code == 502
    assert list(Path(spool).iterdir()) == []


def test_program_form_field_is_passed_through(client_factory):
    r = client_factory().post("/predict", data={"program": "arham_aahar"},
                              files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    json.loads(r.text)
