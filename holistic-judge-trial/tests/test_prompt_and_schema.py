from __future__ import annotations

import json

import pytest

from judge_trial.examples import load_examples
from judge_trial.prompt import build_prompt, render_text, restated_block
from judge_trial.schema import ResponseError, parse_answer, response_schema

from conftest import good_answer

EX = [{"image_id": f"img_{i}", "label": "approved" if i % 2 else "not_approved", "reason": None} for i in range(4)]


def test_variant_a_has_persona_and_every_rule_verbatim_and_nothing_else(cfg, rule_set):
    p = build_prompt(cfg, "A", rule_set, EX)
    for r in rule_set.active():
        assert r.original_text in p.system_text and r.rule_id in p.system_text  # the client's wording, untouched
    assert "grouped and restated" not in p.system_text and "Team guidance" not in p.system_text
    assert p.examples == [] and "social media manager" in p.system_text


def test_b_adds_restatement_and_c_adds_guidance_once_and_separately(cfg, rule_set):
    b = build_prompt(cfg, "B", rule_set, EX)
    c = build_prompt(cfg, "C", rule_set, EX)
    assert "grouped and restated" in b.system_text and "Team guidance" not in b.system_text
    assert c.system_text.count("from the front of the people") == 1  # five rules carry it; it is stated once
    assert "NOT in the client's document" in c.system_text
    assert all(r.original_text in c.system_text for r in rule_set.active())


def test_d_carries_examples_with_labels_and_never_invented_reasons(cfg, rule_set):
    d = build_prompt(cfg, "D", rule_set, EX)
    assert len(d.examples) == 4 and "Reviewer decision: APPROVED." in render_text(d)
    assert "Reason (tagged" not in render_text(d)  # no reason file => label only
    with_reason = build_prompt(cfg, "D", rule_set, [{**EX[1], "reason": "back to camera"}, *EX[2:]])
    assert "the original reviewers' own reason was not recorded): back to camera" in render_text(with_reason)


def test_variant_d_without_examples_is_refused(cfg, rule_set):
    with pytest.raises(ValueError, match="needs examples"):
        build_prompt(cfg, "D", rule_set, [])


def test_hash_is_stable_and_tracks_every_change(cfg, rule_set):
    assert build_prompt(cfg, "C", rule_set, EX).hash == build_prompt(cfg, "C", rule_set, EX).hash
    assert build_prompt(cfg, "C", rule_set, EX).hash != build_prompt(cfg, "B", rule_set, EX).hash
    assert build_prompt(cfg, "D", rule_set, EX).hash != build_prompt(cfg, "D", rule_set, EX[:3]).hash
    assert build_prompt(cfg, "D", rule_set, EX).hash != build_prompt(cfg, "D", rule_set, [{**EX[0], "reason": "x"}, *EX[1:]]).hash


def test_restatement_fails_closed_when_rules_and_groups_disagree(rule_set):
    rules = rule_set.active()
    ids = [r.rule_id for r in rules]
    with pytest.raises(ValueError, match="not grouped"):
        restated_block(rules, {"All": ids[:-1]})
    with pytest.raises(ValueError, match="unknown"):
        restated_block(rules, {"All": [*ids, "R999"]})
    with pytest.raises(ValueError):
        restated_block(rules, {"A": ids, "B": ids[:1]})  # a rule listed twice


def test_schema_lists_every_rule_id_and_orders_fields(rule_set):
    ids = [r.rule_id for r in rule_set.active()]
    s = response_schema(ids)
    assert s["properties"]["reasons"]["items"]["properties"]["rule_id"]["enum"] == [*ids, "OTHER"]
    assert s["propertyOrdering"][:2] == ["observations", "reasons"] and s["propertyOrdering"][2] == "score"


def test_parser_accepts_a_good_answer_and_rejects_everything_else():
    ids = ["R010"]
    a = parse_answer(good_answer(73), ids)
    assert a.score == 73 and a.decision == "APPROVED" and a.reasons[0]["rule_id"] == "R010"
    for bad in ["not json", "[]", '{"score": 5}']:
        with pytest.raises(ResponseError):
            parse_answer(bad, ids)
    for score in (101, -1, 5.5, "80", True):
        with pytest.raises(ResponseError):
            parse_answer(good_answer().replace('"score": 80', f'"score": {json.dumps(score)}'), ids)
    with pytest.raises(ResponseError, match="bad decision"):
        parse_answer(good_answer().replace("APPROVED", "MAYBE"), ids)
    with pytest.raises(ResponseError, match="bad reason"):
        parse_answer(good_answer().replace("R010", "R999"), ids)


def test_reasons_file_fills_in_real_reasons_only(cfg, tmp_path):
    import json

    cfg.examples_path.parent.mkdir(parents=True, exist_ok=True)
    original = cfg.examples_path.read_text(encoding="utf-8") if cfg.examples_path.exists() else None
    try:
        cfg.examples_path.write_text(json.dumps(EX[:2]), encoding="utf-8")
        (tmp_path / "reasons.csv").write_text("image_id,reason\nimg_1,backs to camera\n", encoding="utf-8")
        cfg.examples.reasons_file = str(tmp_path / "reasons.csv")
        loaded = {e["image_id"]: e["reason"] for e in load_examples(cfg)}
        assert loaded == {"img_0": None, "img_1": "backs to camera"}
    finally:
        if original is None:
            cfg.examples_path.unlink()
        else:
            cfg.examples_path.write_text(original, encoding="utf-8")


def test_summary_reason_is_required_only_when_the_prompt_asks_for_it():
    ids = ["R010"]
    plain = json.loads(good_answer())
    assert parse_answer(json.dumps(plain), ids).reason is None            # v1 prompts: no reason field
    assert "reason" not in response_schema(ids)["properties"]
    s = response_schema(ids, summary_reason=True)
    assert s["propertyOrdering"][:4] == ["observations", "reasons", "reason", "score"] and "reason" in s["required"]
    with pytest.raises(ResponseError, match="missing"):
        parse_answer(json.dumps(plain), ids, summary_reason=True)         # asked for but absent: a failed call
    with pytest.raises(ResponseError, match="reason"):
        parse_answer(json.dumps({**plain, "reason": "  "}), ids, summary_reason=True)
    assert parse_answer(json.dumps({**plain, "reason": " Sharp. "}), ids, summary_reason=True).reason == "Sharp."


def test_v2_prompt_asks_for_the_reason_and_has_its_own_hash(cfg):
    from judge_trial.prompt import build_for_variant

    v1, v2 = build_for_variant(cfg, "A", "v1"), build_for_variant(cfg, "A", "v2")
    assert not v1.summary_reason and v2.summary_reason
    assert "- reason:" in v2.system_text and "- reason:" not in v1.system_text
    assert v1.hash != v2.hash


def test_v3_replaces_the_front_angle_guidance_with_the_team_exception(cfg):
    from judge_trial.prompt import build_for_variant

    v2, v3 = build_for_variant(cfg, "C", "v2"), build_for_variant(cfg, "C", "v3")
    assert "Exception:" not in v2.system_text and "Exception:" in v3.system_text
    assert "NOT in the client's document" in v3.system_text
    assert v3.system_text.count("not from behind them") == 1                        # stated once, not once per rule
    assert "Avoid photographs in which the volunteer" in v3.system_text              # the client's own wording is untouched
    assert v2.hash != v3.hash


def test_guidance_replacement_refuses_to_hide_other_guidance():
    from types import SimpleNamespace as NS

    from judge_trial.prompt import guidance_block

    rules = [NS(additional_guidance="old"), NS(additional_guidance="something else")]
    with pytest.raises(ValueError, match="no longer matches"):
        guidance_block(rules, (["old"], ["new"]))
    assert "- new" in guidance_block([NS(additional_guidance="old")], (["old"], ["new"]))
