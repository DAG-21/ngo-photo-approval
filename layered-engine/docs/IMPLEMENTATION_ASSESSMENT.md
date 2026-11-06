# Implementation assessment (written before any code)

## Where this is built
A new sibling folder, `layered-engine/`. Reason: you asked earlier to keep the main project clean and separate. It reads the
other two folders (client rules, checklist5) at *generation time only*; at run time it depends on nothing outside itself.

## What exists today
| Area | Where | State |
|---|---|---|
| Client rules (16) | `image-classifier/rules/photo-approval-requirement-rules.txt` -> `configs/rules.yaml` | verbatim text kept; no rule is marked mandatory in the document |
| Main project engine | `image-classifier/src/ngo_photo_approval/` | rules layer (PASS/FAIL/UNCERTAIN), calibrated scorer, embeddings, FastAPI, tuning prep. **Untouched.** |
| Holistic judge + checklists | `holistic-judge-trial/` | prompts v1-v3, checklist1-5, `policy.py` (15 gates), `checklist.py`, caching backend, Excel exports, 100+ offline tests |
| Checklist5 | `holistic-judge-trial/prompts/checklist5/{checks,policy}.yaml` | 65 questions, 15 gates, `unknown_is_fail: true` |
| Gemini access | `holistic-judge-trial/src/judge_trial/backend.py`, main `cloud_client.py` | fail-closed authorization, retry, rate limit, usage |
| Verified model IDs | `image-classifier/configs/model.yaml` | `gemini-3.5-flash-lite`, `gemini-3.5-flash`. `gemini-3.8-flash` was checked against the key's model list on 2026-11-02 (metadata call, no photos): available. |

## Preserve / change / new
* **Preserve:** the 16 client rules and wording; all 65 checklist5 IDs; the 15 gates and `unknown_is_fail`; fail-closed cloud authorization;
  the rule that labels are never shown to Gemini during annotation; resumable JSONL runs.
* **Change (in the new folder only):** gate evaluation moves from "the model's NO" to a *registry + applicability + exceptions* engine run in Python;
  scoring becomes a separate soft layer; publication checks become a separate layer.
* **New modules:** `rulebook` (loader/validator), `applicability`, `exceptions`, `gates`, `scoring`, `publication`, `decision`, `prompt_builder`,
  `schemas` (Pydantic), `providers` (neutral interface + Gemini), `annotation` (historical + new-photo modes), `export`.

## Conflicts and duplicates found
1. `R014/R015/R016` restate other rules. They are kept as `SUMMARY` criteria that are never scored (no double counting).
2. `R015_*` (beneficiary questions) sit under final-QC Q2 in checklist5 but several are team rules about beneficiaries. They keep their IDs; their `parent_rule_id` stays `R015`.
3. `R004_VOL_GAZE` was removed in checklist5 (a volunteer may look at the camera). The paste's G15 says only beneficiaries. Kept removed; recorded in the changelog.
4. The paste lists "G1-G15" and the checklist5 IDs for the same gates. The gate registry carries both (`G001`..`G015` -> checklist5 ID).
5. Several scene facts the paste adds (F_ORIENTATION, F_PRIMARY_SUBJECTS...) overlap with existing questions (`T_FRONT`, `R015_BEN_PRESENT`). They are Layer 1 *facts*; the old questions stay as diagnostics.
6. The paste's proposed profile *changes* mandatory gates (G4 scope, G9, G10, G11, G12, G14, G15). Each change is recorded as a policy change and listed by the validation report.

## Mapping of existing IDs to layers
* **Layer 1 facts:** `F_PHOTO F_SINGLE F_VOLUNTEERS F_BANNER` + 13 new scene facts.
* **Layer 2 gates (legacy):** the 15 checklist5 gates. In the proposed profile 11 stay hard, 4 change (`R006_COVERAGE`, `R006_ZOOMED_OUT` -> soft; `R003_*` scope; `R006`/`R015_BEN_RECEIVING` applicability).
* **Layer 2 diagnostics (not gates):** every other checklist5 question: kept, evaluated, reported, and used for scoring evidence, never double counted.
* **Layer 3:** six quality dimensions fed by the diagnostics and `NEW-008..024`.
* **Layer 4:** `P001-P011` and `NEW-001..007` (external verification fields).

## Known risk, stated up front
The measured results so far (AUC 0.49-0.59 for the rules/gate approaches against human labels, 0.77 for embeddings) mean this
architecture will make the evaluation **explicit and auditable**; it is not evidence that it will *match the human labels* better.
The experiment in the paste (checklist5 vs four-layer on the same photos) is how that gets measured.
