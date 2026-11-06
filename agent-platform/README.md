# NGO photo approval agent: Phase 2A (feedback infrastructure and human review)

Wraps the layered evaluator (`../layered-engine`) in a service where an NGO employee uploads a photograph, sees the AI's assessment, corrects it, submits a final decision, and every action is recorded without ever altering what the AI said.

Read in this order: `docs/PHASE2A_TECHNICAL_AUDIT.md` (what existed), `docs/PHASE2A_DESIGN.md` (data model, immutability, workflow, access control), `docs/PHASE2A_REPORT.md` (what was built, tests, limits).

## Quick start (local, SQLite)
```bash
# tests (no network, no model call)
python -m pytest -q

# the same tests on PostgreSQL (needs `pip install -e .[postgres]`; each test gets its own schema, dropped afterwards)
TEST_DATABASE_URL=postgresql+psycopg://user@host:5432/dbname python -m pytest -q
# add TEST_USE_ALEMBIC=1 to build every test schema with the production migration instead of create_all

# demonstration of scenarios A-G on real stored pilot evaluations (no model call); add --serve for the review screen on http://127.0.0.1:8100
python scripts/demo_phase2a.py [--serve]

# a real deployment
export DATABASE_URL=postgresql+psycopg://user:pass@host/db     # PostgreSQL in production
export AGENT_JWT_SECRET=<at least 32 random characters>
export AGENT_STORAGE_DIR=/var/lib/ngo-agent
export NGO_ALLOW_CLOUD_INFERENCE=true GEMINI_API_KEY=...        # the same two authorisations as the engine; without them start-up fails closed
alembic upgrade head
uvicorn --factory ngo_agent.api.app:create_default_app --port 8100
```
Create an organisation and its first administrator with `ngo_agent.bootstrap.create_organisation` (it also registers the engine's V1 as the approved, active rulebook and prompt, and V2 as a DRAFT that cannot be activated until an administrator approves it).

## What is guaranteed
* An `EvaluationRun` is written once and frozen by database triggers (PostgreSQL and SQLite); re-evaluating a photograph creates a new run. A content hash detects tampering even if triggers are disabled.
* `workflow status`, `ai_recommendation` and `human_final_decision` are three separate fields.
* The server reads the AI's value from the stored run; a client cannot supply it.
* **The person's decision is final; there is no validation step.** The person supports the AI's verdict, or disregards it, and in **either** case writes the reasons in the reason box (required): when they agree it still records which points matter, since the AI may have missed some. A submitted review cannot be changed. Corrections are append-only and versioned (a changed mind is a new revision).
* A submitted review becomes a learning example straight away, **inactive for retrieval**, except when its reason is `POLICY_CHANGE` or `HUMAN_REVIEW_ERROR`, or it approved a photograph over a publication blocker.
* Every query is scoped to the caller's organisation; another organisation's record is 404.
* Nothing is published automatically. Photo approval and publication readiness stay separate.
* A person cannot approve a photograph whose evaluation lists a publication blocker (for example a legible number plate) without a written statement of why that is acceptable. The server enforces it, the statement is stored with the blockers as they were (a `BLOCKER_ACKNOWLEDGED` event), and the review screen shows it afterwards.

## Similar past decisions as examples (L2, off by default)
Each uploaded photograph is embedded locally (`AGENT_EMBEDDER`: `clip` default, `dinov2`, `thumbnail`, `none`; weights must already be in the local Hugging Face cache, nothing is downloaded). An administrator can switch on `PUT /organisations/{id}/few-shot` (it needs `confirm_cost_increase: true`: each evaluation carries three more photographs (measured: about 46% more input tokens, about 8% more cost) and sends the NGO's past photographs and staff reasons to the model). The 3 most similar past staff decisions of the same NGO are then shown to the model as examples, balanced between approved and rejected, and recorded on the evaluation. `GET /photos/{id}/similar` previews them; administrators can mark a photograph misleading. Details and measurements: `docs/PHASE3_LEARNING_LOOP_DESIGN.md`, section 13.

## Gap analysis (L3, offline)
After each decision the platform compares what the AI said with what the person wrote (text only, contact details removed) and stores the gaps (`gap_analyses`, append-only; administrators list them at `GET /organisations/{id}/gap-analyses`). The analyzer included needs no model; it is a keyword heuristic with known limits (docs/PHASE3_LEARNING_LOOP_DESIGN.md, section 14). A model-based analyzer that reads the rules and the reviewer's reasons instead is built but off by default (`AGENT_GAP_ANALYZER=llm`, with `AGENT_GAP_LLM=gemini` or `groq`; section 18): it sends the reasons, as text, to the cloud model. Groq has its own switch and key (`NGO_ALLOW_GROQ_INFERENCE`, `GROQ_API_KEY`) because it is a different company from Google. Rule updates (L5, section 20, which supersedes the prompt notes of section 19.1): after each decision a model may rewrite a rule the reviewer disagreed with the AI about, the client's own rules included; off by default, then drafts only, then (if an administrator chooses) applied, flagged when it loosens a gate that can reject, and withdrawn automatically when reviewers start overriding the AI more often (`AGENT_RULE_DRAFTER=llm`, or `serve_fresh.py --llm-gaps --llm-rules`). The rulebook files are never edited, so the original wording is always kept. Reasons that are too short or generic are kept but never shown to the model as examples.

## Learning loop (L4, off by default)
An administrator can turn on an automatic update cycle per NGO (`PUT /organisations/{id}/learning-loop`, with `confirm_automatic_changes: true`). After every 5 decided photographs it drafts changes to the quality thresholds and to whether a gate rejects by itself, tests them by replaying the stored answers against the staff's decisions, applies at most one that passes, watches the photographs that follow, and rolls it back by itself if it does worse. A change that would approve even one photograph staff rejected is blocked, and a change that loosens the rules without enough evidence is held for developers. Changelog: `GET /organisations/{id}/changelog`; pause with `paused: true`; undo the latest change with `POST .../changes/{id}/rollback`. It makes no model call. Administrators drive it from the "Learning loop" screens (overview and controls, the changelog with the held list and rollback, and the similar-photo examples switch); details and limits: `docs/PHASE3_LEARNING_LOOP_DESIGN.md`, sections 15 and 16.

## Layout
`src/ngo_agent`: `models.py` (schema), `triggers.py` (immutability), `evaluation_service.py` (upload, evaluation, re-evaluation, recovery), `review_service.py` (review, corrections, final submission, learning example), `versions.py`, `security.py`, `storage.py`, `evaluators.py`, `embeddings.py` and `example_service.py` (the example store), `gap_service.py` and `reason_quality.py` (gap analysis), `loop_service.py` (the learning loop), `api/app.py` and `api/static/` (review screen in `app.js`, administrator screens in `admin.js`). `alembic/` has the migrations (0001 creates the schema; 0002 removes the validation workflow; 0003 adds the example store; 0004 adds gap analysis; 0005 adds the learning loop). Tests are in `tests/`.
