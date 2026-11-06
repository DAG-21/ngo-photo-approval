from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from judge_trial.checklist import ANSWERS, check_ids, checklist_schema, load_checks, parse_checklist, rule_summary
from judge_trial.checklist_eval import badness, check_table, cv_auc, load_answers, tag_agreement
from judge_trial.prompt import build_prompt
from judge_trial.runner import execute, make_plan, read_results
from judge_trial.schema import ResponseError

from conftest import FakeBackend, make_image

VERSION = "checklist1"


@pytest.fixture
def spec():
    return load_checks(VERSION)


def answer_text(spec, overrides=None, default="YES"):
    overrides = overrides or {}
    checks = {i: {"answer": overrides.get(i, default), "evidence": "x" if overrides.get(i, default) in ("NO", "CANNOT_TELL") else ""}
              for i in check_ids(spec)}
    return json.dumps({"scene": "volunteers serving food", "checks": checks, "cannot_judge": False})


def test_every_enabled_rule_is_covered_and_ids_are_unique(spec, rule_set):
    ids = check_ids(spec)
    assert len(ids) == len(set(ids)) and all(c["q"].strip() for c in spec["checks"])
    questioned = {c["rule"] for c in spec["checks"]} - {"FACT"}
    assert questioned | set(spec["covers"]) == {r.rule_id for r in rule_set.active()}


def test_checklist_prompt_has_all_questions_and_no_score_or_decision(cfg, rule_set, spec):
    p = build_prompt(cfg, "K", rule_set, [], VERSION)
    assert p.check_ids == check_ids(spec) and not p.examples
    assert all(f"- {i}:" in p.system_text for i in p.check_ids)
    assert "score" not in p.schema["properties"] and "decision" not in p.schema["properties"]
    assert "Do not add a score" in p.system_text
    assert "Original wording" not in p.system_text and "Avoid photographs in which the volunteer" in p.system_text  # the client's text is shown
    team = [c["id"] for c in spec["checks"] if c["source"] == "team"]
    assert team and all("team guidance, not in the client's document" in line for line in p.system_text.splitlines()
                        if any(line.startswith(f"- {t}:") for t in team))


def test_checklist_hash_changes_when_a_question_changes(cfg, rule_set, tmp_path, monkeypatch):
    a = build_prompt(cfg, "K", rule_set, [], VERSION)
    import judge_trial.prompt as P

    real = P.load_checks

    def edited(version):
        s = real(version)
        s["checks"][0] = {**s["checks"][0], "q": s["checks"][0]["q"] + " (changed)"}
        return s

    monkeypatch.setattr(P, "load_checks", edited)
    assert build_prompt(cfg, "K", rule_set, [], VERSION).hash != a.hash


def test_missing_question_for_a_new_rule_stops_the_build(cfg, rule_set, monkeypatch):
    import judge_trial.prompt as P

    real = P.load_checks

    def trimmed(version):
        s = real(version)
        s["checks"] = [c for c in s["checks"] if c["rule"] != "R011"]
        return s

    monkeypatch.setattr(P, "load_checks", trimmed)
    with pytest.raises(ValueError, match="no questions"):
        build_prompt(cfg, "K", rule_set, [], VERSION)


def test_parser_requires_every_check_and_valid_answers(spec):
    ids = check_ids(spec)
    ok = parse_checklist(answer_text(spec, {ids[5]: "NO"}), ids)
    assert ok.checks[ids[5]]["answer"] == "NO" and ok.checks[ids[0]]["answer"] == "YES"
    bad = json.loads(answer_text(spec))
    del bad["checks"][ids[3]]
    with pytest.raises(ResponseError, match="missing"):
        parse_checklist(json.dumps(bad), ids)
    bad = json.loads(answer_text(spec))
    bad["checks"][ids[1]]["answer"] = "MAYBE"
    with pytest.raises(ResponseError, match="bad answer"):
        parse_checklist(json.dumps(bad), ids)
    bad = json.loads(answer_text(spec))
    bad["checks"]["INVENTED"] = {"answer": "YES", "evidence": ""}
    with pytest.raises(ResponseError, match="unknown"):
        parse_checklist(json.dumps(bad), ids)
    with pytest.raises(ResponseError):
        parse_checklist("not json", ids)


def test_schema_lists_every_check_as_required(spec):
    s = checklist_schema(check_ids(spec))
    assert list(s["properties"]["checks"]["required"]) == check_ids(spec)
    assert s["properties"]["checks"]["properties"][check_ids(spec)[0]]["properties"]["answer"]["enum"] == list(ANSWERS)


def test_rule_summary_gives_partial_scores_and_labels(spec):
    ids = check_ids(spec)
    banner = [c["id"] for c in spec["checks"] if c["rule"] == "R001"]
    ans = {i: {"answer": "YES", "evidence": ""} for i in ids}
    ans[banner[3]] = {"answer": "NO", "evidence": "creased"}
    ans[banner[1]] = {"answer": "CANNOT_TELL", "evidence": "cut"}
    s = rule_summary(ans, spec)["R001"]
    assert s["fraction_met"] == pytest.approx(2 / 3) and s["status"] == "FAIL" and s["unknown"] == 1
    ans2 = {i: {"answer": "NOT_APPLICABLE", "evidence": ""} for i in ids}
    r = rule_summary(ans2, spec)["R001"]
    assert r["status"] == "NOT_APPLICABLE" and r["fraction_met"] is None
    assert "FACT" not in rule_summary(ans, spec)


def test_runner_stores_checklist_answers_and_no_label(cfg, rule_set, spec, tmp_path):
    p = build_prompt(cfg, "K", rule_set, [], VERSION)
    pool = pd.DataFrame([{"image_id": f"img_{i}", "label": "approved"} for i in range(3)])
    for i in pool.image_id:
        make_image(cfg, i)
    _, todo = make_plan(cfg, p, "dev", pool, tmp_path / "run")
    script = [answer_text(spec, {"R001_FLAT": "NO"}), "garbage", answer_text(spec)]
    summary = execute(cfg, p, todo, FakeBackend(script), tmp_path / "run", max_calls=10, workers=1)
    rows = read_results(tmp_path / "run")
    assert summary == {"ok": 2, "failed": 1}
    good = [r for r in rows if r["ok"]]
    assert all("label" not in r and "score" not in r and "decision" not in r for r in rows)
    assert good[0]["checks"]["R001_FLAT"]["answer"] == "NO" and good[0]["prompt_hash"] == p.hash


# --------------------------------------------------------------------------------------------- analysis
def synthetic(spec, n=120, seed=0):
    rng = np.random.default_rng(seed)
    ids = check_ids(spec)
    y = pd.Series(rng.integers(0, 2, n), index=[f"p{i}" for i in range(n)])
    prog = pd.Series(rng.choice(["a", "b", "c"], n), index=y.index)
    ans = pd.DataFrame("YES", index=y.index, columns=ids)
    noisy = (rng.random(n) < np.where(y == 0, 0.7, 0.1))
    ans.loc[noisy, "R009_LIT"] = "NO"
    ans["R001_FLAT"] = "NOT_APPLICABLE"
    ans.loc[rng.random(n) < 0.05, "R010_SHARP"] = "CANNOT_TELL"
    return ans, y, prog


def test_check_table_finds_the_discriminating_check(spec):
    ans, y, prog = synthetic(spec)
    t = check_table(ans, y, spec, prog).set_index("check")
    assert t.loc["R009_LIT", "gap"] > 0.4 and abs(t.loc["R007_LEVEL", "gap"]) < 0.01
    assert t.loc["R009_LIT", "gap_within_program"] > 0.3
    assert np.isnan(t.loc["R001_FLAT", "no_rate_approved"])  # never graded: not applicable everywhere


def test_badness_encodes_no_unknown_and_everything_else(spec):
    ans, _, _ = synthetic(spec)
    b = badness(ans, spec)
    assert set(b["R009_LIT"].unique()) <= {0.0, 1.0} and 0.5 in set(b["R010_SHARP"].unique())
    assert (b["R001_FLAT"] == 0).all() and "F_BANNER" not in b.columns


def test_cv_auc_is_high_for_a_real_signal_and_chance_for_noise(spec):
    ans, y, prog = synthetic(spec, n=200)
    groups = pd.Series([f"g{i // 2}" for i in range(len(y))], index=y.index)
    good = cv_auc(badness(ans, spec), y, groups)
    assert good["auc"] > 0.7  # one binary check with a true AUC of 0.8 cannot do much better out of fold
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(rng.integers(0, 2, (200, 8)).astype(float), index=y.index)
    assert 0.35 < cv_auc(noise, y, groups)["auc"] < 0.65
    assert np.isnan(cv_auc(badness(ans.iloc[:10], spec), y.iloc[:10], groups.iloc[:10])["auc"])  # too few photos


def test_tag_agreement_compares_with_the_base_rate_on_approved_photos(spec):
    ans, y, _ = synthetic(spec)
    dis = [i for i in y.index if y[i] == 0][:6]
    tags = pd.DataFrame({"primary": ["R009"] * 6, "second": [None] * 6, "third": [None] * 6}, index=dis)
    out = tag_agreement(ans, y, tags, spec)
    assert len(out) == 6 and out.attrs["approved_rule_no_rate"]["R009"] < 0.3
    assert out["primary_has_NO"].dropna().astype(bool).mean() > 0.4


def test_load_answers_uses_the_latest_ok_row(spec, tmp_path):
    ids = check_ids(spec)
    row = lambda img, ok, ans: {"image_id": img, "ok": ok, "checks": {i: {"answer": ans, "evidence": ""} for i in ids}} if ok else {"image_id": img, "ok": False}  # noqa: E731
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in [row("a", True, "YES"), row("a", True, "NO"), row("b", False, "")]),
                                            encoding="utf-8")
    a = load_answers(tmp_path, spec)
    assert list(a.index) == ["a"] and a.loc["a", "R009_LIT"] == "NO"


def test_back_view_exception_clears_the_back_view_checks(spec):
    ans, _, _ = synthetic(spec, n=10)
    ans["R008_BACK"] = "NO"
    ans["T_FRONT"] = "NO"
    ans["T_BACK_EXCEPTION"] = ["YES"] * 5 + ["NO"] * 5
    b = badness(ans, spec)
    assert (b["R008_BACK"].iloc[:5] == 0).all() and (b["T_FRONT"].iloc[:5] == 0).all()
    assert (b["R008_BACK"].iloc[5:] == 1).all() and (b["T_BACK_EXCEPTION"].iloc[:5] == 0).all()


@pytest.fixture
def spec2():
    return load_checks("checklist2")


def test_checklist2_adds_the_requested_checks_and_keeps_every_rule_covered(cfg, rule_set, spec2):
    ids = set(check_ids(spec2))
    assert len(check_ids(spec2)) == len(ids)
    assert {"R007_HEIGHT", "R007_VERTICALS", "R007_DISTORTION", "R007_FOREGROUND", "R007_CUTOFF", "R007_ORIENTATION"} <= ids        # camera angle
    assert {"R015_BEN_PRESENT", "R015_BEN_FRONT", "R015_BEN_POSE", "R015_BEN_PROMINENT", "R015_BEN_INTERACT"} <= ids                  # beneficiary
    assert {"R006_COVERAGE", "R006_ZOOMED_OUT", "R006_SIZE"} <= ids                                                                       # coverage
    assert {"R009_OVEREXPOSED", "R009_UNDEREXPOSED", "R009_PEOPLE_LIT"} <= ids and "R009_EXPOSURE" not in ids                              # exposure
    questioned = {c["rule"] for c in spec2["checks"]} - {"FACT"}
    assert questioned | set(spec2["covers"]) == {r.rule_id for r in rule_set.active()}
    present = next(c for c in spec2["checks"] if c["id"] == "R015_BEN_PRESENT")
    assert present["na"] == ""            # a missing beneficiary is graded, never 'not applicable'
    p1, p2 = build_prompt(cfg, "K", rule_set, [], VERSION), build_prompt(cfg, "K", rule_set, [], "checklist2")
    assert p1.hash != p2.hash and p2.check_ids == check_ids(spec2)
    assert "Do not add a score" in p2.system_text and "never NOT_APPLICABLE" in p2.system_text
