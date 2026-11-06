# Implementation and testing report

Date: 2026-11-02. Folder: `layered-engine/` (new). Nothing outside this folder was modified (checked on code, config, scripts, docs and prompts of
`image-classifier` and `holistic-judge-trial`).

## What was implemented
* A **rulebook** in configuration: 126 criteria (65 preserved from checklist5 plus 61 new), 18 gates (G001-G015 for both profiles; G016-G018 in the layered profile only), the 16 client rules verbatim, an activity
  taxonomy, named applicability conditions, exceptions, quality scoring, publication checks and two policy profiles (`checklist5_legacy`, `layered_proposed_v1`).
* A **validator** (`scripts/validate_rulebook.py`) that fails on unknown sources/statuses, undefined conditions, changed client wording, a missing legacy
  id, or a profile override that is not recorded as a policy change. It lists the 7 policy changes that touch a mandatory requirement (4 are the NGO's decisions, 3 remain proposals) (`docs/VALIDATION_REPORT.md`).
* **Layer 1-4 engine** in Python: applicability (three-valued; unknown means "applies"), exception handling (acts only on NO), gate evaluation (missing or
  conflicting answers become CANNOT_TELL), quality scoring (formula, renormalisation, caps, severe-weakness rules), publication readiness (metadata plus visual
  flags) and a decision engine with a fixed precedence.
* A **dynamic prompt builder** and **response-schema exporter** built from the configuration (activities, criteria, dimensions and flags all come from YAML).
* A **provider-neutral interface** with a Gemini implementation: fails closed without the authorization switch and key, retries transient errors, scrubs the key,
  sends the original image bytes. Strict Pydantic models for the response.
* **Historical-annotation and new-photo modes** with resumable JSONL runs, raw-response preservation, versioned rows, cost tracking, and a **JSONL/CSV export** that joins
  labels afterwards and keeps model observations, human findings and the historical decision apart.
* Docs: `README.md`, `RULEBOOK_CHANGELOG.md`, this report, the pre-implementation assessment. Three **synthetic** sample outputs in `samples/`.

## Files
Created (all new, inside `layered-engine/`): `config/` (gemini_config, prompt_sections, rulebook/*.yaml, schemas/*.json), `src/layered_engine/` (schemas, rulebook,
expressions, applicability, exception_handler, gate_evaluator, quality_scorer, publication, decision_engine, prompt_builder, schema_export, providers, ingestion,
annotation, export), `scripts/` (build_rulebook, validate_rulebook, run_engine, make_samples), `tests/` (8 files), `samples/`, `docs/`, `README.md`, `RULEBOOK_CHANGELOG.md`.
Changed in existing projects: **none**.

## Tests
135 passed, 0 failed, all offline (a guard makes any real Gemini client creation fail). Coverage includes: every one of the 15 gates for NO, CANNOT_TELL and N/A in the
legacy profile; applicability edge cases (education with no handover, animal care, no beneficiary, no banner, unknown facts); G10/G11 legacy vs proposed; G13, G14, G15
and the direct-gaze exception; the back-view exception needing all three conditions; the medical-PPE exception; G1 components; the scoring formula and renormalisation;
thresholds; severe weakness; no double counting; a soft score never overriding a gate; missing metadata as PENDING; visual blockers; platform dimensions; fifteen edge-case
fixtures; fail-closed authorization; retry behaviour; resumability and the call cap; the label never reaching the model; the enriched export.
Mutation check: deliberately breaking five behaviours (unknown treated as pass, N/A treated as failure, missing answer treated as pass, a high score overriding a gate, the legacy profile
letting quality decide) made the suite fail each time; the code was restored and the suite is green.

## What was NOT tested or done
* **Live Gemini testing so far = Pilot 1 only** (`docs/PILOT_REPORT.md`): 10 disapproved photographs under both profiles. The first live attempt failed completely (schema limits rejected by the API, a client-creation race, an output limit that
  the model's thinking used up); all three were fixed and the retry produced valid, complete answers. Not yet measured: false rejections on approved photos, accuracy against the labels, run-to-run variation, and the real price.
* The fifteen "edge-case fixtures" are **synthetic assessments**, not photographs. They test the rule engine, not Gemini.
* The comparison experiment (checklist5 prompt vs this architecture on the same photographs; gate agreement, false approvals, N/A error rate, cost, latency) has not been run.
* Integration tests with real photographs are pending the first pilot.

## Unresolved policy decisions (need the NGO)
See the last section of `RULEBOOK_CHANGELOG.md`. In short: G010/G011 hard vs soft, G012 applicability, the G001 components, the medical-PPE result, the quality thresholds, and every NEW-* and P-* criterion (all PROPOSED).
Decided by the NGO on 2026-11-02 (rulebook 1.1.0): face scope stays strict; banners optional but visible and distinguishable when present; animals count as a subject; beneficiary direct gaze always disapproved for now, volunteer gaze allowed; back views disallowed with no exception. The profile `layered_proposed_v1` therefore carries `policy_validation_required = true` on every decision.

## Known weaknesses of the design
* Face requirements are strict (every visible beneficiary) until the computer-vision layer exists; faces are judged only by the model's visual impression.
* G016-G018 rely on the model recognising a back view and distinguishing a banner from its background. Untested on real photographs.
* Everything depends on the model's answers. In the trials so far Gemini's rule answers separated approved from disapproved photos poorly (AUC about 0.5-0.6) and changed between identical runs.
  This architecture makes the evaluation explicit and auditable; it does not by itself make the answers more accurate.
* About 6,300 tokens of instructions plus ~100 answers per photograph; cost per photo is unmeasured. Context caching is not used.
* Quality thresholds, platform sizes and the dimension caps are uncalibrated placeholders.

## How to run, and the first pilot
```bash
cd layered-engine
../image-classifier/.venv/Scripts/python.exe -m pytest -q                      # 135 tests, offline
../image-classifier/.venv/Scripts/python.exe scripts/validate_rulebook.py
../image-classifier/.venv/Scripts/python.exe scripts/run_engine.py prompt
# pilot: a CSV with image_id,path (opaque ids; no label column), then
../image-classifier/.venv/Scripts/python.exe scripts/run_engine.py historical pilot.csv --run-name pilot1 --max-calls 10               # dry run
../image-classifier/.venv/Scripts/python.exe scripts/run_engine.py historical pilot.csv --run-name pilot1 --max-calls 10 --execute --env-file ../image-classifier/.env
../image-classifier/.venv/Scripts/python.exe scripts/run_engine.py export runs/pilot1 --labels labels.csv
```
Suggested first pilot: 10 photographs with known tagged reasons, profile `layered_proposed_v1`, then the same 10 under `checklist5_legacy`, and compare the gates that fired with the human tags.
Check first: JSON validity, whether all 85 criteria come back, how often a gate comes back CANNOT_TELL or "missing", and the real tokens and cost per photo.
