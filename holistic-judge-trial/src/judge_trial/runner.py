"""Planning and running a judging pass over a pool of photos.

``make_plan`` is pure and offline. ``execute`` is the only function that calls a backend; the run script reaches it
only with ``--execute``. Results are appended to results.jsonl one photo at a time, so an interrupted run resumes
without repeating photos, and a run can never mix two different prompts.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .backend import JudgeBackend
from .backend_cost import caching_expected, estimate_cost
from .config import TRIAL_ROOT, TrialConfig
from .data import image_path
from .prompt import BuiltPrompt
from .checklist import parse_checklist
from .schema import ResponseError, parse_answer


class RunRefused(RuntimeError):
    """A safety check stopped the run before any photo was sent."""


@dataclass
class RunPlan:
    variant: str
    pool: str
    model_id: str
    prompt_hash: str
    n_examples: int
    n_photos: int
    n_done: int
    n_todo: int
    n_calls_planned: int
    estimate: dict
    run_dir: str


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-")


def run_dir_for(cfg: TrialConfig, pool: str, variant: str, tag: str | None) -> Path:
    name = f"{variant}__{slug(cfg.model.id)}" + (f"__{slug(tag)}" if tag else "")
    return TRIAL_ROOT / "runs" / pool / name


def read_results(run_dir: Path) -> list[dict]:
    path = run_dir / "results.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn last line from an interrupted run
    return rows


def _done_ids(rows: list[dict], prompt_hash: str) -> set[str]:
    stale = {r["prompt_hash"] for r in rows} - {prompt_hash}
    if stale:
        raise RunRefused(f"this run folder already holds results from a different prompt ({sorted(stale)}); "
                         "use --tag to start a separate run")
    return {r["image_id"] for r in rows if r.get("ok")}


def make_plan(cfg: TrialConfig, prompt: BuiltPrompt, pool_name: str, pool: pd.DataFrame, run_dir: Path,
              limit: int | None = None, random_n: int | None = None, seed: int = 42,
              ids: list[str] | None = None) -> tuple[RunPlan, pd.DataFrame]:
    todo_frame = pool.sort_values("image_id")
    if ids is not None:
        if limit is not None or random_n is not None:
            raise RunRefused("use --ids-file on its own, not with --limit or --random")
        missing = sorted(set(ids) - set(todo_frame["image_id"]))
        if missing:
            raise RunRefused(f"{len(missing)} photos in the ids file are not in this pool (e.g. {missing[:3]})")
        todo_frame = todo_frame[todo_frame["image_id"].isin(ids)]
    if limit is not None and random_n is not None:
        raise RunRefused("use either --limit or --random, not both")
    if random_n is not None:
        # a simple random draw (classes are NOT balanced on purpose); the same seed always gives the same photos
        todo_frame = todo_frame.sample(n=min(random_n, len(todo_frame)), random_state=seed).sort_values("image_id")
    if limit is not None:
        # a pilot keeps both classes represented: half from each, in a fixed order
        half = max(1, limit // 2)
        todo_frame = pd.concat([g.head(half) for _, g in todo_frame.groupby("label")]).sort_values("image_id")
    done = _done_ids(read_results(run_dir), prompt.hash)
    todo = todo_frame[~todo_frame["image_id"].isin(done)]
    plan = RunPlan(prompt.variant, pool_name, cfg.model.id, prompt.hash, len(prompt.examples), len(todo_frame),
                   int(todo_frame["image_id"].isin(done).sum()), len(todo),
                   len(todo) + (1 if len(todo) and caching_expected(prompt, cfg) else 0),  # +1 = creating the cache
                   estimate_cost(prompt, cfg, len(todo)), str(run_dir))
    return plan, todo


def execute(cfg: TrialConfig, prompt: BuiltPrompt, todo: pd.DataFrame, backend: JudgeBackend, run_dir: Path,
            max_calls: int, workers: int | None = None) -> dict:
    if len(todo) > max_calls:
        raise RunRefused(f"{len(todo)} calls planned but --max-calls is {max_calls}; raise it deliberately or use --limit")
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "results.jsonl"
    ids = list(todo["image_id"])
    ok = failed = 0
    backend.prepare(prompt)
    try:
        def one(image_id: str) -> dict:
            row = {"image_id": image_id, "variant": prompt.variant, "prompt_hash": prompt.hash,
                   "model_id": cfg.model.id, "cache_active": backend.cache_active, "ts": time.time()}
            try:
                res = backend.judge(image_path(cfg, image_id).read_bytes(), "image/jpeg")
                ans = (parse_checklist(res.text, prompt.check_ids) if prompt.check_ids
                       else parse_answer(res.text, prompt.rule_ids, prompt.summary_reason))
                row.update(ok=True, usage=res.usage, **asdict(ans))
            except ResponseError as exc:
                row.update(ok=False, error=f"bad response: {exc}", raw=res.text[:500], usage=res.usage)
            except Exception as exc:  # noqa: BLE001  (provider errors etc.; the message is already scrubbed upstream)
                row.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:200]}")
            return row

        with ThreadPoolExecutor(max_workers=workers or cfg.run.workers) as pool, out.open("a", encoding="utf-8") as fh:
            futures = [pool.submit(one, i) for i in ids]
            for n, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                ok, failed = ok + bool(row["ok"]), failed + (not row["ok"])
                if n % 25 == 0 or n == len(ids):
                    print(f"  {n}/{len(ids)} done  ({ok} ok, {failed} failed)", flush=True)
    finally:
        backend.close()
    return {"ok": ok, "failed": failed}


def write_plan(plan: RunPlan, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan.json").write_text(json.dumps(asdict(plan), indent=2), encoding="utf-8")
