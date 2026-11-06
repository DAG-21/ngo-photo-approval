"""Running the engine: HISTORICAL_ANNOTATION and NEW_PHOTO modes, resumable, with every artefact preserved.

HISTORICAL_ANNOTATION judges an existing photograph without knowing its historical label: no label is an input to anything in this
module (the labels are joined later by export.py, from a separate file). NEW_PHOTO is the same assessment plus verified metadata.
An API failure or a response that does not validate is recorded as a failure; it is never turned into an evaluation.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from pydantic import ValidationError

from .decision_engine import evaluate_final_decision
from .ingestion import IngestionError, load_image
from .prompt_builder import BuiltPrompt, asked_fact_fields, build_prompt
from .rulebook import ResolvedProfile, Rulebook
from .schema_export import build_response_schema
from .schemas import Metadata, ModelAssessment, missing_facts

HISTORICAL = "HISTORICAL_ANNOTATION"
NEW_PHOTO = "NEW_PHOTO"


@dataclass
class Engine:
    rulebook: Rulebook
    profile: ResolvedProfile
    prompt: BuiltPrompt
    schema: dict[str, Any]
    provider: Any  # a VisionProvider (None for dry runs and offline decisions)
    config: dict[str, Any]

    @property
    def model_id(self) -> str:
        return getattr(self.provider, "model_id", None) or self.config["model_id"]


def build_engine(config: dict[str, Any], provider: Any = None, profile_name: Optional[str] = None, rulebook_dir: Optional[Path] = None) -> Engine:
    rb = Rulebook(rulebook_dir) if rulebook_dir else Rulebook()
    profile = rb.resolve(profile_name or config["evaluation"]["profile"])
    prompt = build_prompt(rb, profile)
    return Engine(rb, profile, prompt, build_response_schema(profile, prompt.criterion_ids), provider, config)


def parse_response(text: str, image_id: str, profile: Optional[ResolvedProfile] = None) -> ModelAssessment:
    """`profile` (optional): only the optional scene facts that profile asks are required. Without it every fact is required, as it always was."""
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("the response is not a JSON object")
    raw["image_id"] = image_id
    for k in ("model_recommendation", "model_overall_score"):
        raw.pop(k, None) if raw.get(k) is None else None
    try:
        assessment = ModelAssessment.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        raise ValueError(f"response does not match the schema at {'.'.join(str(p) for p in first['loc'])}: {first['msg']}") from exc
    lacking = missing_facts(assessment.layer_1_scene_understanding.scene_facts, asked_fact_fields(profile) if profile is not None else None)
    if lacking:
        raise ValueError(f"response does not match the schema at layer_1_scene_understanding.scene_facts.{lacking[0]}: Field required")
    return assessment


def _cost(usage: Mapping[str, Any], cfg: Mapping[str, Any]) -> Optional[float]:
    from .costs import cost_usd   # thinking billed as output, cached input at the cached rate, prices per model

    return cost_usd(usage, cfg)


def evaluate_one(engine: Engine, image_id: str, image_path: Path | str, mode: str, metadata: Optional[Mapping[str, Any]] = None, raw_dir: Optional[Path] = None) -> dict[str, Any]:
    """One photograph, end to end. Always returns a row; ok=False carries the error."""
    row: dict[str, Any] = {
        "image_id": image_id, "input_ref": Path(image_path).name, "mode": mode, "model_id": engine.model_id, "prompt_version": engine.prompt.template_version,
        "prompt_hash": engine.prompt.hash, "rulebook_version": engine.profile.rulebook_version, "profile": engine.profile.name,
        "timestamp": datetime.now(timezone.utc).isoformat(), "ok": False}
    try:
        meta = dict(metadata or {})
        bad = [k for k in meta if k.lower() in Metadata.FORBIDDEN_KEYS]
        if bad:
            raise ValueError(f"metadata must not carry historical labels: {bad}")
        if mode == HISTORICAL and any(k in meta for k in ("consent_status", "rights_status", "caption_status")):
            raise ValueError("historical annotation takes no publication metadata")
        img = load_image(image_path)
        meta = {**{k: v for k, v in img.metadata.items() if k in ("width", "height")}, **meta}
        res = engine.provider.assess(img.data, img.mime_type, engine.prompt.system_text, engine.prompt.user_text, engine.schema)
        row.update(usage=res.usage, cost_usd=_cost(res.usage, engine.config), raw_text=res.text)
        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"{image_id}__{engine.prompt.hash}.json").write_text(res.text or "", encoding="utf-8")
        assessment = parse_response(res.text, image_id, engine.profile)
        decision = evaluate_final_decision(assessment, engine.profile, meta, versions={"prompt_version": engine.prompt.template_version, "prompt_hash": engine.prompt.hash, "model_id": engine.model_id})
        row.update(ok=True, validated=assessment.model_dump(mode="json"), decision=decision.model_dump(mode="json"), metadata_used=meta, ingestion_warnings=img.warnings)
    except (IngestionError, ValueError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001  (provider errors: already scrubbed upstream)
        row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return row


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line from an interrupted run
    return rows


def done_keys(rows: list[dict[str, Any]], engine: Engine) -> set[str]:
    return {r["image_id"] for r in rows if r.get("ok") and r.get("prompt_hash") == engine.prompt.hash and r.get("model_id") == engine.model_id
            and r.get("profile") == engine.profile.name and r.get("rulebook_version") == engine.profile.rulebook_version}


@dataclass
class RunPlan:
    n_photos: int
    n_done: int
    n_todo: int
    n_calls: int
    max_calls: int
    mode: str
    model_id: str
    profile: str


def plan_run(engine: Engine, entries: list[dict[str, Any]], out_dir: Path, mode: str, max_calls: Optional[int] = None) -> tuple[RunPlan, list[dict[str, Any]]]:
    done = done_keys(read_rows(out_dir / "results.jsonl"), engine)
    todo = [e for e in entries if e["image_id"] not in done]
    cap = max_calls if max_calls is not None else int(engine.config["processing"]["max_calls_default"])
    return RunPlan(len(entries), len(entries) - len(todo), len(todo), len(todo), cap, mode, engine.model_id, engine.profile.name), todo


def run_batch(engine: Engine, entries: list[dict[str, Any]], out_dir: Path, mode: str, max_calls: Optional[int] = None, workers: Optional[int] = None) -> dict[str, int]:
    """The only function that calls the provider for many photos. Refuses if the plan exceeds max_calls."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plan, todo = plan_run(engine, entries, out_dir, mode, max_calls)
    if plan.n_calls > plan.max_calls:
        raise RuntimeError(f"{plan.n_calls} calls planned but --max-calls is {plan.max_calls}; raise it deliberately")
    ok = failed = 0
    workers = workers or int(engine.config["processing"]["max_concurrent_requests"])
    with ThreadPoolExecutor(max_workers=workers) as pool, (out_dir / "results.jsonl").open("a", encoding="utf-8") as fh:
        futures = [pool.submit(evaluate_one, engine, e["image_id"], e["path"], mode, e.get("metadata"), out_dir / "raw") for e in todo]
        for fut in as_completed(futures):
            row = fut.result()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            ok, failed = ok + bool(row["ok"]), failed + (not row["ok"])
    return {"ok": ok, "failed": failed, "skipped_done": plan.n_done}
