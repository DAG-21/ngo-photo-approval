from __future__ import annotations

import json

import pandas as pd
import pytest

from judge_trial.backend_cost import estimate_cost
from judge_trial.prompt import build_prompt
from judge_trial.runner import RunRefused, execute, make_plan, read_results

from conftest import FakeBackend, good_answer, make_image


@pytest.fixture
def prompt(cfg, rule_set):
    return build_prompt(cfg, "A", rule_set, [])


@pytest.fixture
def pool(cfg):
    rows = [{"image_id": f"img_{i:02d}", "label": "approved" if i % 2 else "not_approved"} for i in range(10)]
    for r in rows:
        make_image(cfg, r["image_id"])
    return pd.DataFrame(rows)


def test_planning_never_calls_the_backend(cfg, prompt, pool, tmp_path):
    fake = FakeBackend()
    plan, todo = make_plan(cfg, prompt, "dev", pool, tmp_path / "run")
    assert plan.n_todo == 10 and len(todo) == 10 and fake.calls == fake.prepared == 0
    assert not (tmp_path / "run" / "results.jsonl").exists()


def test_pilot_limit_keeps_both_classes(cfg, prompt, pool, tmp_path):
    _, todo = make_plan(cfg, prompt, "dev", pool, tmp_path / "run", limit=4)
    assert len(todo) == 4 and set(pool.set_index("image_id").loc[todo.image_id, "label"]) == {"approved", "not_approved"}


def test_execute_writes_one_row_per_photo_and_closes_the_backend(cfg, prompt, pool, tmp_path):
    fake = FakeBackend()
    _, todo = make_plan(cfg, prompt, "dev", pool, tmp_path / "run")
    summary = execute(cfg, prompt, todo, fake, tmp_path / "run", max_calls=50, workers=1)
    rows = read_results(tmp_path / "run")
    assert summary == {"ok": 10, "failed": 0} and len(rows) == 10 and fake.prepared == 1 and fake.closed == 1
    assert {r["prompt_hash"] for r in rows} == {prompt.hash} and all("label" not in r for r in rows)  # labels never stored


def test_resume_skips_finished_photos_and_retries_failed_ones(cfg, prompt, pool, tmp_path):
    run = tmp_path / "run"
    _, todo = make_plan(cfg, prompt, "dev", pool, run)
    script = [good_answer()] * 7 + ["garbage", RuntimeError("boom"), good_answer()]
    execute(cfg, prompt, todo, FakeBackend(script), run, max_calls=50, workers=1)
    assert sum(not r["ok"] for r in read_results(run)) == 2
    plan, todo2 = make_plan(cfg, prompt, "dev", pool, run)
    assert plan.n_done == 8 and plan.n_todo == 2  # only the two failures come back
    fake = FakeBackend()
    execute(cfg, prompt, todo2, fake, run, max_calls=50, workers=1)
    assert fake.calls == 2


def test_bad_answers_are_recorded_as_failures_not_guessed(cfg, prompt, pool, tmp_path):
    run = tmp_path / "run"
    _, todo = make_plan(cfg, prompt, "dev", pool.head(2), run)
    execute(cfg, prompt, todo, FakeBackend(["not json", good_answer().replace('"score": 80', '"score": 400')]), run, 50, 1)
    rows = read_results(run)
    assert [r["ok"] for r in rows] == [False, False] and all("bad response" in r["error"] for r in rows)


def test_call_cap_stops_the_run_before_the_first_call(cfg, prompt, pool, tmp_path):
    fake = FakeBackend()
    _, todo = make_plan(cfg, prompt, "dev", pool, tmp_path / "run")
    with pytest.raises(RunRefused, match="max-calls"):
        execute(cfg, prompt, todo, fake, tmp_path / "run", max_calls=5, workers=1)
    assert fake.calls == fake.prepared == 0


def test_a_run_folder_cannot_mix_two_prompts(cfg, rule_set, prompt, pool, tmp_path):
    run = tmp_path / "run"
    _, todo = make_plan(cfg, prompt, "dev", pool, run)
    execute(cfg, prompt, todo, FakeBackend(), run, 50, 1)
    other = build_prompt(cfg, "B", rule_set, [])
    with pytest.raises(RunRefused, match="different prompt"):
        make_plan(cfg, other, "dev", pool, run)


def test_backend_is_closed_even_when_the_run_breaks(cfg, prompt, pool, tmp_path):
    class Exploding(FakeBackend):
        def prepare(self, p):
            super().prepare(p)
            raise KeyboardInterrupt

    fake = Exploding()
    _, todo = make_plan(cfg, prompt, "dev", pool, tmp_path / "run")
    with pytest.raises(KeyboardInterrupt):
        execute(cfg, prompt, todo, fake, tmp_path / "run", 50, 1)
    assert fake.closed == 0  # prepare itself failed: nothing to clean up, and no photo was sent
    assert fake.calls == 0


def test_a_torn_last_line_does_not_break_resume(cfg, prompt, pool, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    good = {"image_id": "img_00", "prompt_hash": prompt.hash, "ok": True}
    (run / "results.jsonl").write_text(json.dumps(good) + '\n{"image_id": "img_01", "prom', encoding="utf-8")
    plan, _ = make_plan(cfg, prompt, "dev", pool, run)
    assert plan.n_done == 1 and plan.n_todo == 9


def test_cost_estimate_is_unknown_until_prices_are_set_then_computed(cfg, rule_set):
    d = build_prompt(cfg, "D", rule_set, [{"image_id": f"e{i}", "label": "approved", "reason": None} for i in range(8)])
    cfg.cost.input_per_million_usd = cfg.cost.cached_input_per_million_usd = cfg.cost.output_per_million_usd = None  # the real config now has prices
    est = estimate_cost(d, cfg, 100)
    assert est["usd"] is None and est["caching_expected"] and est["prefix_tokens"] >= 8 * cfg.cost.tokens_per_image
    assert est["input_tokens_no_cache"] > est["input_tokens_with_cache_fresh"]
    cfg.cost.input_per_million_usd, cfg.cost.cached_input_per_million_usd, cfg.cost.output_per_million_usd = 1.0, 0.1, 4.0
    usd = estimate_cost(d, cfg, 100)["usd"]
    assert usd["with_cache"] < usd["without_cache"]
    a = build_prompt(cfg, "A", rule_set, [])
    assert estimate_cost(a, cfg, 100)["caching_expected"] is False  # a short prompt is not worth caching in auto mode


def test_random_sample_is_seeded_sized_and_exclusive_with_limit(cfg, prompt, pool, tmp_path):
    _, a = make_plan(cfg, prompt, "dev", pool, tmp_path / "r1", random_n=5, seed=3)
    _, b = make_plan(cfg, prompt, "dev", pool, tmp_path / "r2", random_n=5, seed=3)
    _, c = make_plan(cfg, prompt, "dev", pool, tmp_path / "r3", random_n=5, seed=4)
    assert len(a) == 5 and a["image_id"].tolist() == b["image_id"].tolist() and a["image_id"].tolist() != c["image_id"].tolist()
    assert len(make_plan(cfg, prompt, "dev", pool, tmp_path / "r4", random_n=10_000)[1]) == len(pool)  # never more than exist
    with pytest.raises(RunRefused, match="not both"):
        make_plan(cfg, prompt, "dev", pool, tmp_path / "r5", limit=4, random_n=4)


def test_ids_list_runs_exactly_those_photos_and_refuses_strangers(cfg, prompt, pool, tmp_path):
    _, todo = make_plan(cfg, prompt, "dev", pool, tmp_path / "r1", ids=["img_01", "img_04"])
    assert sorted(todo["image_id"]) == ["img_01", "img_04"]
    with pytest.raises(RunRefused, match="not in this pool"):
        make_plan(cfg, prompt, "dev", pool, tmp_path / "r2", ids=["img_01", "img_999"])
    with pytest.raises(RunRefused, match="on its own"):
        make_plan(cfg, prompt, "dev", pool, tmp_path / "r3", ids=["img_01"], limit=2)
