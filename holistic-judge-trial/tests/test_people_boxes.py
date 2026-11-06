from __future__ import annotations

import json

import numpy as np
import pytest

from judge_trial.people_boxes import BoxError, box_metrics, build_boxes_prompt, norm_box, parse_boxes, read_rows, run

from conftest import FakeBackend, make_image


def good(people=None, activity=(100, 100, 900, 900)):
    return json.dumps({"people": people if people is not None else [{"role": "volunteer", "box": [200, 200, 800, 500]}],
                       "activity_box": list(activity), "note": ""})


def test_parser_accepts_valid_and_rejects_malformed_boxes():
    assert parse_boxes(good())["people"][0]["role"] == "volunteer"
    assert parse_boxes(good(people=[]))["people"] == []
    for bad in ('{"people": [{"role": "boss", "box": [0,0,1,1]}], "activity_box": [0,0,0,0]}',
                '{"people": [{"role": "other", "box": [0,0,1]}], "activity_box": [0,0,0,0]}',
                '{"people": [{"role": "other", "box": [500,0,100,10]}], "activity_box": [0,0,0,0]}',
                '{"people": [{"role": "other", "box": [0,0,1001,10]}], "activity_box": [0,0,0,0]}',
                '{"people": []}', "not json"):
        with pytest.raises(BoxError):
            parse_boxes(bad)


def test_prompt_is_hashed_and_asks_for_boxes_only():
    p = build_boxes_prompt()
    assert p.hash and "never identify anyone" in p.system_text and "do not judge" in p.system_text
    assert set(p.schema["required"]) == {"people", "activity_box", "note"}


def test_metrics_find_a_dark_area_around_people_and_the_coverage():
    y = np.full((100, 100), 150.0)
    y[40:100, 0:50] = 60.0                       # a dark patch where the person stands and around them
    row = {"people": [{"role": "volunteer", "box": [500, 100, 900, 400]}], "activity_box": [400, 0, 1000, 500], "note": ""}
    m = box_metrics(y, row)
    assert m["y_people"] < 80 and m["people_minus_global"] < -20 and m["y_rest"] > 140
    assert m["people_minus_rest"] < -60
    assert m["tallest_person"] == pytest.approx(0.4, abs=0.02) and m["activity_cover"] == pytest.approx(0.3)
    none = box_metrics(y, {"people": [], "activity_box": [0, 0, 0, 0], "note": ""})
    assert none["n_people"] == 0 and np.isnan(none["activity_cover"]) and "y_people" not in none
    assert norm_box([0, 0, 1000, 1000], 10, 20) == (0, 0, 10, 20)


def test_run_is_resumable_and_never_stores_labels(cfg, tmp_path):
    ids = [f"img_{i}" for i in range(4)]
    for i in ids:
        make_image(cfg, i)
    out = tmp_path / "b.jsonl"
    p = build_boxes_prompt()
    s1 = run(cfg, ids, out, FakeBackend([good(), "garbage", good(), good()]), p, max_calls=10, workers=1)
    assert s1["ok"] == 3 and s1["failed"] == 1
    fake = FakeBackend([good()])
    s2 = run(cfg, ids, out, fake, p, max_calls=10, workers=1)
    assert fake.calls == 1 and s2["skipped_done"] == 3                     # only the failed photo is retried
    assert all("label" not in r for r in read_rows(out))
    with pytest.raises(RuntimeError, match="max-calls"):
        run(cfg, [f"x{i}" for i in range(5)], tmp_path / "c.jsonl", FakeBackend(), p, max_calls=2)
