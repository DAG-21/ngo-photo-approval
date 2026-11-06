from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from ngo_photo_approval.config import CalibrationConfig
from ngo_photo_approval.evaluation.calibration_metrics import expected_calibration_error, reliability_bins
from ngo_photo_approval.evaluation.error_analysis import build_error_analysis, classify_error
from ngo_photo_approval.evaluation.metrics import (
    classification_metrics,
    evaluate_predictions,
    rule_level_metrics,
    select_threshold,
)
from ngo_photo_approval.exceptions import DataError, ModelArtifactError
from ngo_photo_approval.rules.evaluator import evaluate_rules
from ngo_photo_approval.schemas import ProbabilityType, RuleCheck
from ngo_photo_approval.scoring import train as train_mod
from ngo_photo_approval.scoring.calibration import calibration_status
from ngo_photo_approval.scoring.feature_builder import build_features, feature_names
from ngo_photo_approval.scoring.predict import ProbabilityScorer
from ngo_photo_approval.scoring.train import train_probability_model


def synthetic(n: int, rule_set, seed: int = 0):
    """Features for n images whose approval depends (noisily) on rule outcomes."""
    rng = np.random.default_rng(seed)
    names = feature_names(rule_set, "rules")
    X, y = [], []
    ids = ["R001", "R002", "R003", "R004"]
    for _ in range(n):
        latent = rng.random() < 0.7
        statuses = {}
        for rid in ids:
            p_pass = 0.93 if latent else 0.35
            r = rng.random()
            statuses[rid] = "PASS" if r < p_pass else ("UNCERTAIN" if r < p_pass + 0.03 else "FAIL")
        res = evaluate_rules(rule_set, [RuleCheck(rule_id=k, status=v) for k, v in statuses.items()],
                             __import__("ngo_photo_approval.config", fromlist=["PolicyConfig"]).PolicyConfig())
        f = build_features(rule_set, "rules", res)
        X.append([f[k] for k in names])
        y.append(int(latent) if rng.random() > 0.05 else int(not latent))  # 5% label noise
    return np.array(X), np.array(y), names


# ------------------------------------------------------------------ features


def test_feature_vector_stable_and_complete(rule_set):
    res = evaluate_rules(rule_set, [RuleCheck(rule_id=r, status="PASS") for r in ("R001", "R002", "R003", "R004")],
                         __import__("ngo_photo_approval.config", fromlist=["x"]).PolicyConfig())
    f = build_features(rule_set, "rules", res)
    assert list(f) == feature_names(rule_set, "rules")
    assert f["R001_pass"] == 1 and f["R001_fail"] == 0 and f["frac_pass"] == 1


def test_missing_and_failed_analysis_become_missing_indicators(rule_set):
    f = build_features(rule_set, "rules", None)
    assert f["R001_missing"] == 1 and f["R001_pass"] == 0 and f["n_missing"] == 4
    from ngo_photo_approval.config import PolicyConfig

    res = evaluate_rules(rule_set, [RuleCheck(rule_id="R001", status="PASS")], PolicyConfig())
    f = build_features(rule_set, "rules", res)
    assert f["R002_missing"] == 1 and f["R001_missing"] == 0


def test_quality_features_require_metrics(rule_set):
    with pytest.raises(ValueError):
        build_features(rule_set, "rules+quality", None, None)
    q = dict(blur_laplacian_var=100.0, brightness=120.0, contrast=40.0, megapixels=2.0, aspect_ratio=1.5)
    f = build_features(rule_set, "quality", None, q)
    assert set(f) == {"q_log_blur", "q_brightness", "q_contrast", "q_aspect_ratio"}


# ------------------------------------------------------------------ training / calibration


def test_trainer_has_no_way_to_receive_test_data():
    params = set(inspect.signature(train_probability_model).parameters)
    assert not {p for p in params if "test" in p.lower()}
    assert {"X_train", "X_val"} <= params


def test_trains_calibrates_and_labels_honestly(rule_set, tmp_path):
    X, y, names = synthetic(900, rule_set)
    groups = np.arange(len(y)) // 4
    tr, va = slice(0, 600), slice(600, 900)
    cfg = CalibrationConfig(min_samples_for_calibration=200, min_samples_per_class=50, max_ece_for_calibrated_claim=0.20)
    meta = train_probability_model(X[tr], y[tr], groups[tr], X[va], y[va], names, cfg, tmp_path / "m", seed=1)
    assert meta["uses_test_data"] is False
    assert meta["calibration_status"] == "calibrated", meta["calibration_reasons"]
    scorer = ProbabilityScorer.load(tmp_path / "m")
    est = scorer.score(dict(zip(names, X[0], strict=True)))
    assert 0.0 <= est.probability <= 1.0 and 0.0 <= est.raw_classifier_probability <= 1.0
    assert est.probability_type == ProbabilityType.CALIBRATED and est.calibrated_probability is not None
    assert (tmp_path / "m" / "model.joblib").exists() and (tmp_path / "m" / "metadata.json").exists()


def test_small_data_cannot_be_labelled_calibrated(rule_set, tmp_path):
    X, y, names = synthetic(150, rule_set, seed=3)
    groups = np.arange(len(y)) // 3
    cfg = CalibrationConfig()  # needs >= 200 train samples and 50/class
    meta = train_probability_model(X[:90], y[:90], groups[:90], X[90:], y[90:], names, cfg, tmp_path / "s", seed=1)
    assert meta["calibration_status"] == "uncalibrated_classifier" and meta["calibration_reasons"]
    est = ProbabilityScorer.load(tmp_path / "s").score(dict(zip(names, X[0], strict=True)))
    assert est.probability_type == ProbabilityType.UNCALIBRATED_CLASSIFIER
    assert est.calibrated_probability is None and est.probability == est.raw_classifier_probability


def test_high_ece_blocks_calibrated_claim():
    cfg = CalibrationConfig(max_ece_for_calibrated_claim=0.05)
    assert calibration_status(1000, 500, 500, 0.04, cfg)[0] == "calibrated"
    status, why = calibration_status(1000, 500, 500, 0.12, cfg)
    assert status == "uncalibrated_classifier" and "ECE" in why[0]
    assert calibration_status(1000, 500, 500, None, cfg)[0] == "uncalibrated_classifier"


def test_single_class_training_rejected(rule_set, tmp_path):
    X, _, names = synthetic(60, rule_set)
    with pytest.raises(DataError):
        train_probability_model(X, np.ones(60, int), np.arange(60), X, np.ones(60, int), names,
                                CalibrationConfig(), tmp_path / "x")


def test_scorer_rejects_feature_mismatch_and_missing_artifacts(rule_set, tmp_path):
    with pytest.raises(ModelArtifactError):
        ProbabilityScorer.load(tmp_path / "nothing")
    X, y, names = synthetic(300, rule_set)
    train_probability_model(X[:200], y[:200], np.arange(200) // 4, X[200:], y[200:], names, CalibrationConfig(),
                            tmp_path / "m")
    scorer = ProbabilityScorer.load(tmp_path / "m")
    with pytest.raises(ModelArtifactError):
        scorer.check_features(names[:-1])
    with pytest.raises(ModelArtifactError):
        scorer.score({"R001_pass": 1.0})


def test_training_is_deterministic(rule_set, tmp_path):
    X, y, names = synthetic(400, rule_set)
    g = np.arange(400) // 4
    a = train_probability_model(X[:250], y[:250], g[:250], X[250:], y[250:], names, CalibrationConfig(), tmp_path / "a", seed=5)
    b = train_probability_model(X[:250], y[:250], g[:250], X[250:], y[250:], names, CalibrationConfig(), tmp_path / "b", seed=5)
    assert a["model_version"] == b["model_version"] and a["validation"] == b["validation"]
    assert train_mod.DEFAULT_C_GRID


# ------------------------------------------------------------------ calibration metrics


def test_ece_zero_for_perfectly_calibrated_and_large_for_overconfident():
    y = np.array([1] * 8 + [0] * 2)
    assert expected_calibration_error(y, np.full(10, 0.8)) == pytest.approx(0.0, abs=1e-9)
    assert expected_calibration_error(y, np.full(10, 0.99)) > 0.15
    assert sum(b["n"] for b in reliability_bins(y, np.linspace(0, 1, 10))) == 10


# ------------------------------------------------------------------ evaluation metrics


def test_classification_metrics_far_frr():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    m = classification_metrics(y_true, y_pred)
    assert m["confusion_matrix"] == {**m["confusion_matrix"], "tp": 3, "fn": 1, "fp": 1, "tn": 3}
    assert m["false_approval_rate"] == 0.25 and m["false_rejection_rate"] == 0.25
    assert m["approval_precision"] == 0.75 and m["accuracy"] == 0.75


def test_unresolved_predictions_counted_conservatively():
    df = pd.DataFrame({
        "label": ["approved", "approved", "not_approved", "not_approved", "approved"],
        "decision": ["APPROVE", "ERROR", "REVIEW", "REJECT", "REJECT"],
        "probability": [0.9, None, None, 0.1, 0.2],
    })
    out = evaluate_predictions(df)
    assert out["processing"]["automatic"] == 3 and out["processing"]["errors"] == 1 and out["processing"]["review"] == 1
    assert out["processing"]["automatic_rate"] == pytest.approx(0.6)
    assert out["strict"]["confusion_matrix"]["fp"] == 0  # unresolved never counts as an approval
    assert out["strict"]["confusion_matrix"]["fn"] == 2  # ERROR on an approved image counts as a miss
    assert out["probability"]["n"] == 3


def test_threshold_selection_respects_false_approval_limit():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(200), np.zeros(100)].astype(int)
    p = np.r_[rng.beta(6, 2, 200), rng.beta(2, 6, 100)]
    sel = select_threshold(y, p, max_far=0.05, step=0.01)
    assert sel["feasible"] and sel["validation_far"] <= 0.05
    # a lower threshold would break the limit or give no more recall
    looser = select_threshold(y, p, max_far=0.5, step=0.01)
    assert looser["threshold"] <= sel["threshold"] and looser["validation_recall"] >= sel["validation_recall"]


def test_infeasible_threshold_is_reported_not_hidden():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    sel = select_threshold(y, p, max_far=0.0, min_recall=0.9)
    assert sel["feasible"] is False and "note" in sel


def test_rule_level_metrics_use_only_verified_labels():
    pred = pd.DataFrame({"image_id": ["a", "b", "c"], "rule_id": ["R001"] * 3, "status": ["PASS", "FAIL", "PASS"]})
    truth = pd.DataFrame({"image_id": ["a", "b", "c"], "rule_id": ["R001"] * 3,
                          "status": ["PASS", "PASS", "FAIL"], "verified": [True, True, False]})
    m = rule_level_metrics(pred, truth)["R001"]
    assert m["n"] == 2 and m["accuracy"] == 0.5  # row c ignored: not verified


def test_error_analysis_tags_and_blank_manual_cause():
    preds = pd.DataFrame({
        "image_id": ["a", "b", "c"], "label": ["not_approved", "approved", "approved"],
        "decision": ["APPROVE", "REJECT", "APPROVE"], "probability": [0.9, 0.2, 0.95],
        "hard_rule_failures": ["", "R001", ""], "uncertain_rules": ["R002", "", ""],
    })
    audit = pd.DataFrame({"image_id": ["a", "b", "c"], "flag_blurry": [True, False, False]})
    err = build_error_analysis(preds, audit)
    assert set(err.error_type) == {"false_approval", "false_rejection"}
    fa = err[err.error_type == "false_approval"].iloc[0]
    assert "uncertain:R002" in fa.auto_tags and "quality:blurry" in fa.auto_tags and fa.manual_cause == ""
    assert classify_error("approved", "APPROVE") is None and classify_error("approved", "ERROR") == "unresolved_error"


def test_no_skill_model_is_never_called_calibrated(rule_set, tmp_path):
    """Base-rate predictions have tiny ECE but no discrimination: must not be labelled calibrated."""
    rng = np.random.default_rng(0)
    names = feature_names(rule_set, "rules")
    X = rng.random((600, len(names)))
    y = (rng.random(600) < 0.75).astype(int)  # labels independent of features
    meta = train_probability_model(X[:400], y[:400], np.arange(400) // 4, X[400:], y[400:], names,
                                   CalibrationConfig(max_ece_for_calibrated_claim=0.3), tmp_path / "ns")
    assert meta["calibration_status"] == "uncalibrated_classifier"
    assert any("discrimination" in r for r in meta["calibration_reasons"])
    assert calibration_status(1000, 500, 500, 0.01, CalibrationConfig(), validation_auc=0.5)[0] == "uncalibrated_classifier"
    assert calibration_status(1000, 500, 500, 0.01, CalibrationConfig(), validation_auc=0.8)[0] == "calibrated"
