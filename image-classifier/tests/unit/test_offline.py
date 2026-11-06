from __future__ import annotations

import json

import pandas as pd

from ngo_photo_approval.config import PolicyConfig
from ngo_photo_approval.models.baseline_analyzer import AnalysisResult
from ngo_photo_approval.models.batch_runner import run_batch
from ngo_photo_approval.schemas import RuleCheck, VisualAnalysis
from ngo_photo_approval.scoring.offline import feature_table, load_analyses

from ..conftest import make_image


def analysis_record(image_id: str, statuses: dict[str, str], ok: bool = True) -> dict:
    if not ok:
        return AnalysisResult(image_id=image_id, ok=False, error_type="provider_error", error="x", model_id="m",
                              rule_set_version="v").model_dump(mode="json")
    a = VisualAnalysis(image_id=image_id, checks=[RuleCheck(rule_id=k, status=v) for k, v in statuses.items()])
    return AnalysisResult(image_id=image_id, ok=True, analysis=a, model_id="m", rule_set_version="v").model_dump(mode="json")


def test_batch_runner_resumes_and_retries_failures(tmp_path):
    imgs = {f"i{n}": make_image(tmp_path / f"{n}.jpg", n) for n in range(4)}
    out = tmp_path / "res" / "a.jsonl"
    calls: list[str] = []
    fail_once = {"i2"}

    def call(iid, path, program):
        calls.append(iid)
        if iid in fail_once:
            fail_once.discard(iid)
            return AnalysisResult(image_id=iid, ok=False, error_type="provider_error", error="t", model_id="m",
                                  rule_set_version="v")
        return AnalysisResult(image_id=iid, ok=True, analysis=VisualAnalysis(checks=[]), model_id="m", rule_set_version="v")

    items = [(i, p, None) for i, p in imgs.items()]
    s1 = run_batch(items, call, out, workers=3)
    assert s1["ok"] == 3 and s1["failed"] == 1 and sorted(calls) == ["i0", "i1", "i2", "i3"]

    with open(out, "a", encoding="utf-8") as f:  # torn line from a crash
        f.write('{"image_id": "i9", "ok": tr')
    calls.clear()
    s2 = run_batch(items, call, out, workers=3)
    assert calls == ["i2"] and s2["skipped_done"] == 3 and s2["ok"] == 1
    final = load_analyses(out)
    assert {k: v["ok"] for k, v in final.items() if k != "i9"} == {"i0": True, "i1": True, "i2": True, "i3": True}


def test_feature_table_uses_saved_analyses_and_counts_drops(rule_set):
    split = pd.DataFrame({
        "image_id": ["a", "b", "c", "d"], "label": ["approved", "not_approved", "approved", "not_approved"],
        "group_id": ["g1", "g2", "g3", "g4"], "program": ["p"] * 4})
    full = {"R001": "PASS", "R002": "PASS", "R003": "PASS", "R004": "PASS"}
    analyses = {"a": analysis_record("a", full), "b": analysis_record("b", {**full, "R001": "FAIL"}),
                "c": analysis_record("c", {}, ok=False)}  # d has no analysis at all
    X, y, groups, ids, drops = feature_table(split, rule_set, PolicyConfig(), "rules", {}, analyses)
    assert ids == ["a", "b"] and list(y) == [1, 0] and list(groups) == ["g1", "g2"]
    assert drops == {"no_analysis": 1, "analysis_failed": 1, "no_quality": 0}
    assert X.shape[0] == 2 and X[0].tolist() != X[1].tolist()


def test_feature_table_labels_come_from_human_decision_only(rule_set):
    split = pd.DataFrame({"image_id": ["a"], "label": ["approved"], "group_id": ["g"], "program": ["p"]})
    q = {"a": dict(blur_laplacian_var=50.0, brightness=100.0, contrast=40.0, megapixels=1.0, aspect_ratio=1.3)}
    X, y, _, ids, drops = feature_table(split, rule_set, PolicyConfig(), "quality", q)
    assert ids == ["a"] and y[0] == 1 and X.shape == (1, 4) and sum(drops.values()) == 0
    X2, _, _, ids2, drops2 = feature_table(split, rule_set, PolicyConfig(), "quality", {})
    assert ids2 == [] and drops2["no_quality"] == 1


def test_load_analyses_ignores_garbage_and_missing(tmp_path):
    assert load_analyses(tmp_path / "nope.jsonl") == {}
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"image_id": "a", "ok": True}) + "\nnot json\n", encoding="utf-8")
    assert list(load_analyses(p)) == ["a"]
