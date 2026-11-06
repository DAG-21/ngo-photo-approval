# Phase 2A: technical audit of the existing project (2026-11-02)

Scope of the audit: `image-classifier/` (original project, read-only), `holistic-judge-trial/`, `layered-engine/` (the four-layer engine and the V1/V2 pilots). Everything below was read from the files, not assumed.

## 1. What exists

| Area | What is there | Phase 2A consequence |
|---|---|---|
| Evaluator | `layered-engine`: rulebook (YAML, 127 criteria, 18 gates, versions 1.1.0 and 2.0.0-proposed), deterministic Python decision engine, Gemini 3.8 Flash provider (`providers.py`, fails closed without authorisation), Pydantic response models, resumable batch runner. File-based: results are JSONL under `runs/` | The evaluator is reused as it is. Phase 2A wraps it; it does not rebuild it. |
| Versioning of rulebook / prompt | rulebook version string, profile name, prompt template version and 16-hex prompt hash are written on every result row. Frozen V1 snapshot in `evaluators/v1_frozen/`; V2 is a profile that inherits V1 and records each change | Recorded in the new `rulebook_versions` / `prompt_versions` tables so that every evaluation points to exact versions. |
| Evaluation schema | `ModelAssessment` (layers 1-4) and `Decision` (gates, quality, publication, reasons) are Pydantic models and are stored in each result row (`validated`, `decision`) | Stored verbatim, per layer, in the immutable `evaluation_runs` row. |
| API | `image-classifier/src/ngo_photo_approval/api/main.py`: FastAPI with `POST /predict`, `/health`, `/ready`. It serves the **original** pipeline (rules layer, calibrated scoring, embeddings), not the layered engine. No auth, no database, no organisations. Upload handling (size and type limits, decoding, temp file always deleted) is good and is reused as a pattern | A new service is needed. The old `/predict` is left untouched. |
| Database | **None.** No SQLAlchemy, Alembic, Postgres driver, or SQL file anywhere in the three project folders | New schema, new migrations. Nothing to reuse and nothing to duplicate. |
| Image storage | Local folders only (`data/...`, normalised 1280-px copies); photos are addressed by opaque ids; an explicitly authorised GCS upload step exists for tuning data | New object-storage abstraction with a local-disk implementation now and a cloud implementation later. |
| Frontend | **None** in the NGO project (the only React/Vite material found is an unrelated folder, `grok-sign-lang-app`) | A small review UI is built and served by the same service; no build chain. |
| Auth / multi-tenancy | None | New: users, roles, JWT, organisation scoping on every query. |
| Tests | `layered-engine`: 153 passing; `image-classifier`: unit and API integration tests | The new package gets its own suite and the same "no real network call" guard. |
| Security notes | `image-classifier/docs/SECURITY_PRIVACY.md`: three independent authorisation switches for anything leaving the machine, opaque ids, no paths in API errors, no secrets or client data in git | Carried over: the new service keeps the same rules (details in `PHASE2A_REPORT.md`). |
| Environment | Python 3.11, FastAPI 0.162, Pydantic 2.14, pytest. Docker is installed but its engine was **not running**, so PostgreSQL could not be started on this machine | Code is written for PostgreSQL and tested on SQLite; see Known limitations. |

## 2. Completed, incomplete, known issues

* Completed and reusable: the four-layer evaluation, versioned profiles, V1/V2 comparison, immutable-by-convention JSONL results.
* Incomplete: no persistent store, no service for the evaluator, no review workflow, no users.
* Known issues that matter here: the V1/V2 pilot showed that relaxing gates raised false approvals (2 to 6 of 25), and five of the seven V2 changes conflict with NGO decisions of 2026-11-02. **Phase 2A therefore stores the rulebook version on every evaluation and does not make any V2 change active.** The default registered rulebook version is V1 (`layered_proposed_v1`, rulebook 1.1.0, prompt hash `11427c79d96c569a`). V2 is registered as a `DRAFT` version that cannot be activated until an administrator approves it.

## 3. What Phase 2A adds, and how it extends the existing code

New folder `agent-platform/` (package `ngo_agent`), importing `layered_engine` for evaluation. `image-classifier/` is not modified.

```
upload -> object storage + photos row -> EvaluationRun (PROCESSING) -> Evaluator (layered_engine + Gemini)
       -> EvaluationRun filled once (AI_EVALUATED) -> HumanReview -> CriterionCorrection / FeedbackEvent
       -> validation workflow -> LearningExample (inactive for retrieval)
```

Decision on technology: SQLAlchemy 2 + Alembic, written so the same models run on PostgreSQL (production) and SQLite (tests, demo). LangGraph is **not** introduced: Phase 2A has no multi-step agent control flow that a framework would help with, and the prompt asks not to rebuild a working workflow to adopt one.

## 4. Missing functionality required for Phase 2A, and the plan

1. Data model and migration (section in `PHASE2A_DESIGN.md`).
2. Immutable evaluation records enforced in the database itself (triggers), not only in application code.
3. Review, correction, event and validation workflow with a controlled correction-reason taxonomy.
4. Authentication, roles, organisation isolation.
5. Evaluation service that runs the layered engine and always writes a new `EvaluationRun`.
6. API and a review screen.
7. Tests for the fourteen listed behaviours and the seven demonstration scenarios.
