"""L2, the example store: find the staff's past decisions on photographs similar to a new one, so the model can be shown them as examples.

* Every photograph gets an image embedding when it is uploaded. A photograph can serve as an example only if a person has decided it AND its review became a
  learning example (so a policy-change reason, a self-declared reviewer error and an approval over a publication blocker never do), it is not flagged
  as misleading, and the NGO has switched the feature on.
* Search is inside one organisation only. Nothing here reads another organisation's photographs, embeddings, decisions or reasons.
* Selection is balanced: the best approved and the best rejected example are both included when both exist, so the model does not simply copy the majority.
* Off by default. Turning it on adds three photographs to every evaluation (measured on 20 pilot photographs: about 46% more input tokens, about 8% more cost at the
  placeholder price) and sends past photographs and staff reasons to the model, so it needs an explicit confirmation.

Limitation: the search compares the new photograph with every eligible stored embedding in Python (fine for thousands of photographs). A vector index
(pgvector or similar) is the next step when the archive is large.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from layered_engine import few_shot
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .embeddings import Embedder
from .errors import Invalid
from .models import ExampleFlag, LearningExample, Organisation, Photo, PhotoEmbedding, User, now

log = logging.getLogger("ngo_agent.examples")
DEFAULTS = {"enabled": False, "k": 3, "min_similarity": 0.0}
MAX_K = 5
MIN_FLAG_REASON = 5


# ------------------------------------------------------------------------------------------------------------------------------------ settings
def settings(org: Organisation) -> dict[str, Any]:
    raw = {**DEFAULTS, **((org.configuration or {}).get("few_shot") or {})}
    return {"enabled": bool(raw["enabled"]), "k": int(raw["k"]), "min_similarity": float(raw["min_similarity"])}


def save_settings(s: Session, org: Organisation, enabled: bool, k: Optional[int] = None, min_similarity: Optional[float] = None, confirm_cost_increase: bool = False) -> dict[str, Any]:
    cur = settings(org)
    if enabled and not cur["enabled"] and not confirm_cost_increase:
        raise Invalid("turning on similar-photo examples adds three photographs to every evaluation (measured on 20 pilot photographs: about 46% more input tokens and about 8% more cost at the placeholder price), and sends the NGO's past photographs and staff "
                      "reasons to the model; set confirm_cost_increase to true to turn it on")
    k = cur["k"] if k is None else k
    ms = cur["min_similarity"] if min_similarity is None else min_similarity
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= MAX_K:
        raise Invalid(f"k must be a whole number from 1 to {MAX_K}")
    if not isinstance(ms, (int, float)) or isinstance(ms, bool) or not 0.0 <= ms <= 1.0:
        raise Invalid("min_similarity must be a number from 0 to 1")
    new = {"enabled": bool(enabled), "k": k, "min_similarity": float(ms)}
    org.configuration = {**(org.configuration or {}), "few_shot": new}
    org.updated_at = now()
    # an example is retrievable exactly while the feature is on (the flag keeps its meaning: "currently usable as an example")
    s.execute(update(LearningExample).where(LearningExample.organisation_id == org.organisation_id).values(active_for_retrieval=bool(enabled)))
    return new


def few_shot_enabled(s: Session, organisation_id: str) -> bool:
    org = s.get(Organisation, organisation_id)
    return bool(org and settings(org)["enabled"])


# ---------------------------------------------------------------------------------------------------------------------------------- embeddings
def store_embedding(s: Session, photo: Photo, data: bytes, embedder: Optional[Embedder]) -> Optional[PhotoEmbedding]:
    """Embed a photograph and keep the embedding. Never raises: a photograph that cannot be embedded is simply not available as an example."""
    if embedder is None:
        return None
    existing = s.scalar(select(PhotoEmbedding).where(PhotoEmbedding.photo_id == photo.photo_id, PhotoEmbedding.model == embedder.name))
    if existing:
        return existing
    try:
        vec = embedder.embed(data)
    except Exception as exc:  # noqa: BLE001
        log.warning("photo %s was not embedded by %s: %s", photo.photo_id, getattr(embedder, "name", "?"), type(exc).__name__)
        return None
    row = PhotoEmbedding(photo_id=photo.photo_id, organisation_id=photo.organisation_id, model=embedder.name, dim=len(vec), vector=[round(float(x), 6) for x in vec])
    s.add(row)
    s.flush()
    return row


def _vector(s: Session, photo: Photo, embedder: Embedder, storage: Any) -> Optional[np.ndarray]:
    row = s.scalar(select(PhotoEmbedding).where(PhotoEmbedding.photo_id == photo.photo_id, PhotoEmbedding.model == embedder.name))
    if row is None:   # uploaded before an embedder was configured: embed now, from the stored file
        try:
            row = store_embedding(s, photo, storage.get(photo.image_storage_path), embedder)
        except Exception:  # noqa: BLE001
            row = None
    return _unit(np.asarray(row.vector, dtype=np.float32)) if row is not None else None


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


# -------------------------------------------------------------------------------------------------------------------------------------- flags
def flagged_ids(s: Session, organisation_id: str) -> set[str]:
    """Photographs currently marked misleading: the latest row for each photograph wins."""
    state: dict[str, bool] = {}
    for pid, action in s.execute(select(ExampleFlag.photo_id, ExampleFlag.action).where(ExampleFlag.organisation_id == organisation_id).order_by(ExampleFlag.created_at)):
        state[pid] = action == "FLAG"
    return {p for p, f in state.items() if f}


def set_flag(s: Session, user: User, photo: Photo, flagged: bool, reason: str = "") -> ExampleFlag:
    if photo.organisation_id != user.organisation_id:
        raise Invalid("photo not found")
    reason = (reason or "").strip()
    if flagged and len(reason) < MIN_FLAG_REASON:
        raise Invalid(f"say why this photograph is misleading as an example (at least {MIN_FLAG_REASON} characters)")
    row = ExampleFlag(photo_id=photo.photo_id, organisation_id=photo.organisation_id, action="FLAG" if flagged else "UNFLAG", reason=reason, created_by=user.user_id)
    s.add(row)
    s.flush()
    return row


# ------------------------------------------------------------------------------------------------------------------------------------ search
@dataclass
class Candidate:
    photo_id: str
    similarity: float
    decision: str
    example: Any = None


def select_examples(cands: list[Candidate], k: int, min_similarity: float = 0.0) -> list[Candidate]:
    """The `k` most similar candidates above `min_similarity`, balanced: the best approved and the best rejected are both included when both exist.
    Returned least similar first, so the most similar example sits next to the new photograph in the request."""
    pool = sorted((c for c in cands if c.similarity >= min_similarity), key=lambda c: -c.similarity)
    chosen: list[Candidate] = []
    for decision in ("APPROVE", "REJECT"):
        best = next((c for c in pool if c.decision == decision), None)
        if best is not None:
            chosen.append(best)
    chosen = sorted(chosen, key=lambda c: -c.similarity)[:k]
    for c in pool:
        if len(chosen) >= k:
            break
        if c not in chosen:
            chosen.append(c)
    return sorted(chosen, key=lambda c: c.similarity)


def find_similar(s: Session, organisation_id: str, query: np.ndarray, model: str, k: int, min_similarity: float, exclude_ids: set[str]) -> list[Candidate]:
    flagged = flagged_ids(s, organisation_id)
    latest: dict[str, LearningExample] = {}
    for ex in s.scalars(select(LearningExample).where(LearningExample.organisation_id == organisation_id, LearningExample.active_for_retrieval.is_(True),
                                                      LearningExample.low_information.is_(False)).order_by(LearningExample.created_at.desc())):
        if ex.photo_id not in latest and ex.photo_id not in exclude_ids and ex.photo_id not in flagged:
            latest[ex.photo_id] = ex          # one example per photograph: its most recent decision
    if not latest:
        return []
    rows = s.execute(select(PhotoEmbedding.photo_id, PhotoEmbedding.vector).where(PhotoEmbedding.organisation_id == organisation_id, PhotoEmbedding.model == model,
                                                                                 PhotoEmbedding.photo_id.in_(list(latest)))).all()
    if not rows:
        return []
    matrix = np.asarray([r[1] for r in rows], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    sims = (matrix / np.where(norms > 0, norms, 1.0)) @ _unit(np.asarray(query, dtype=np.float32))
    cands = [Candidate(pid, float(sim), latest[pid].verified_decision, latest[pid]) for (pid, _), sim in zip(rows, sims)]
    return select_examples(cands, k, min_similarity)


def _tags(ex: LearningExample) -> list[str]:
    """The tags a person added. The AI's own observations that nobody removed are not staff tags and are not shown as such."""
    return [t["tag"] for t in (ex.verified_tags or []) if isinstance(t, dict) and t.get("tag") and t.get("kind") == "ADDED_BY_REVIEWER"]


def _reasons(ex: LearningExample) -> str:
    rs = ex.verified_reasons or []
    final = next((r.get("comment") for r in rs if r.get("criterion_id") == "final_decision" and r.get("comment")), "")
    others = [f"{r['criterion_id']}: {r['comment']}" for r in rs if r.get("criterion_id") != "final_decision" and r.get("comment")]
    return " ".join(x for x in (final, ("Also: " + "; ".join(others)) if others else "") if x).strip()


def examples_for_evaluation(s: Session, storage: Any, photo: Photo, cfg: dict[str, Any], embedder: Optional[Embedder]) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
    """(examples for the model, the audit record of them). Both empty/None when the pool is empty: then the evaluation is exactly what it was before L2.
    The photograph being evaluated is never its own example (so a re-evaluation or a test cannot cheat)."""
    from .evaluation_service import FORMATS  # lazy: evaluation_service imports this module

    if embedder is None:
        return [], None
    q = _vector(s, photo, embedder, storage)
    if q is None:
        return [], None
    examples: list[dict[str, Any]] = []
    for c in find_similar(s, photo.organisation_id, q, embedder.name, cfg["k"], cfg["min_similarity"], {photo.photo_id}):
        p = s.get(Photo, c.photo_id)
        if p is None or p.organisation_id != photo.organisation_id:    # defence in depth: never another organisation's photograph
            continue
        try:
            data = storage.get(p.image_storage_path)
        except Exception:  # noqa: BLE001  (an unreadable example is skipped, never fatal)
            continue
        examples.append({"image_bytes": data, "mime_type": FORMATS[p.file_format][1], "decision": c.decision, "tags": _tags(c.example), "reasons": _reasons(c.example),
                         "similarity": round(c.similarity, 6), "photo_id": c.photo_id, "learning_example_id": c.example.learning_example_id})
    if not examples:
        return [], None
    desc = few_shot.describe(examples)
    for d, e in zip(desc, examples):
        d["learning_example_id"] = e["learning_example_id"]
    return examples, {"format": few_shot.FORMAT_VERSION, "format_hash": few_shot.format_hash(), "embedding_model": embedder.name, "k": cfg["k"],
                      "min_similarity": cfg["min_similarity"], "examples": desc}
