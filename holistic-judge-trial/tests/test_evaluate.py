from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from judge_trial.evaluate import (
    evaluate_frame, expected_calibration_error, false_approvals, load_run_frame, recall_at_far, render_report, threshold_at_far,
)


def test_threshold_allows_floor_of_five_percent_and_ties_never_exceed_it():
    neg = np.arange(1, 41)  # 40 disapproved photos, scores 1..40 => at most 2 may be accepted
    thr = threshold_at_far(neg)
    assert thr == 38 and (neg > thr).sum() == 2
    assert threshold_at_far(np.arange(1, 11)) == 10  # 10 negatives => floor(0.5) = 0 allowed => accept nothing
    tied = np.full(40, 90)
    assert threshold_at_far(tied) == 90 and (tied > 90).sum() == 0  # ties are rejected, never accepted
    assert threshold_at_far(np.array([])) == -np.inf


def test_recall_at_far_counts_false_positives_and_negatives():
    y = np.array([0] * 40 + [1] * 10)
    s = np.concatenate([np.arange(1, 41), np.arange(31, 41)])
    r = recall_at_far(y, s)
    assert r["false_positives"] == 2 and r["false_approval_rate"] == pytest.approx(2 / 40)
    assert r["false_negatives"] == int(10 - (s[y == 1] > 38).sum())
    assert r["approved_recall"] == pytest.approx((s[y == 1] > 38).mean())


def test_ece_is_zero_for_a_calibrated_model_and_large_for_an_overconfident_one():
    rng = np.random.RandomState(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02
    assert expected_calibration_error(y, np.where(p > 0.5, 0.99, 0.01)) > 0.2


def make_frame(n=120, signal=True, seed=0):
    rng = np.random.RandomState(seed)
    y = np.array([1, 0] * (n // 2))
    score = np.clip((70 if signal else 50) + (20 if signal else 0) * (2 * y - 1) + rng.normal(0, 8, n), 0, 100).round()
    return pd.DataFrame({"image_id": [f"i{k}" for k in range(n)], "label": np.where(y == 1, "approved", "not_approved"),
                         "program": "p", "group_id": [f"g{k // 2}" for k in range(n)], "ok": True, "score": score,
                         "decision": np.where(score >= 50, "APPROVED", "DISAPPROVED"), "cannot_judge": False,
                         "observations": "x", "reasons": [[]] * n})


META = {"prompt_hash": "h", "variant": "D", "model_id": "m", "calls_logged": 120, "failed": 0, "photos_in_pool": 120, "not_attempted": 0}


def test_a_judge_with_signal_beats_chance_and_one_without_does_not():
    good = evaluate_frame(make_frame(signal=True), META, n_boot=50)
    flat = evaluate_frame(make_frame(signal=False), META, n_boot=50)
    assert good["auc"] > 0.9 and good["bootstrap"]["auc_ci95"][0] > 0.8
    assert flat["auc"] < 0.65 and flat["bootstrap"]["auc_ci95"][0] < 0.6


def test_failures_are_excluded_and_reported_never_counted_as_wrong_answers():
    df = make_frame()
    df["ok"] = df["ok"].astype(object)
    df.loc[:9, "ok"] = False
    df.loc[10:14, "ok"] = np.nan  # never attempted
    res = evaluate_frame(df, {**META, "failed": 10, "not_attempted": 5}, n_boot=20)
    assert res["n_scored"] == 105 and any("failed" in w for w in res["warnings"])


def test_small_negative_class_is_flagged_and_baseline_comparison_is_paired():
    df = make_frame(60)  # 30 disapproved < 50
    base = np.linspace(0, 1, len(df))
    res = evaluate_frame(df, META, baseline=base, n_boot=30)
    assert any("imprecise" in w for w in res["warnings"]) and "baseline_embeddings" in res
    assert res["bootstrap"]["auc_minus_baseline_ci95"] is not None
    text = render_report(res)
    assert "Embeddings" in text and "WARNING" in text


def test_report_distinguishes_false_positives_from_false_negatives():
    res = evaluate_frame(make_frame(), META, n_boot=20)
    text = render_report(res)
    assert "False positives" in text and "False negatives" in text
    assert res["as_decided"]["false_positives"] >= 0 and "false_negative_rate" in res["as_decided"]


def test_false_approval_list_shows_highest_scored_disapproved_photos_first():
    df = make_frame()
    fa = false_approvals(df, threshold=60, top=5)
    assert (fa["score"].diff().dropna() <= 0).all() and set(df.set_index("image_id").loc[fa.image_id, "label"]) == {"not_approved"}


def test_one_class_only_gives_no_ranking_metrics():
    df = make_frame()
    df = df[df.label == "approved"]
    res = evaluate_frame(df, META, n_boot=10)
    assert "auc" not in res and res["warnings"]


def test_mixed_prompts_in_one_folder_are_rejected(tmp_path):
    (tmp_path / "results.jsonl").write_text(
        json.dumps({"image_id": "a", "prompt_hash": "h1", "variant": "D", "model_id": "m", "ok": True}) + "\n"
        + json.dumps({"image_id": "b", "prompt_hash": "h2", "variant": "D", "model_id": "m", "ok": True}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="several prompts"):
        load_run_frame(tmp_path, pd.DataFrame({"image_id": ["a", "b"]}))


def test_later_rows_supersede_earlier_ones_for_a_retried_photo(tmp_path):
    rows = [{"image_id": "a", "prompt_hash": "h", "variant": "D", "model_id": "m", "ok": False},
            {"image_id": "a", "prompt_hash": "h", "variant": "D", "model_id": "m", "ok": True, "score": 70}]
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    df, meta = load_run_frame(tmp_path, pd.DataFrame({"image_id": ["a", "b"], "label": ["approved", "approved"]}))
    assert meta["failed"] == 0 and meta["not_attempted"] == 1 and df.set_index("image_id").loc["a", "score"] == 70
