# Layered engine: four-layer NGO photo evaluation

Gemini **observes**. Python **decides**. The model looks at one photograph and reports facts, criterion answers with evidence, quality grades and
visual flags. Python decides which requirements apply, enforces the mandatory gates, computes the quality score and produces the only
authoritative decision. The model's own recommendation or totals are recorded and never used.

> Status: built and tested offline (135 tests; rulebook 1.1.0 with the NGO's decisions of 2026-11-02). Pilot 1 (10 disapproved photographs, both profiles, Gemini 3.8 Flash) has been run: see `docs/PILOT_REPORT.md`. It found and fixed three defects that only the live API
> exposed, and showed complete, well-evidenced answers, but it contained no approved photographs, so false rejections are unmeasured. See "Current limitations".

## 1. Objective
Evaluate a photograph against the NGO's published standards in a way that is auditable, configurable and explainable, rather than predicting whether
it resembles historically approved photographs. Earlier trials (in `../holistic-judge-trial`) found that rule-based Gemini answers did not track the
human labels (AUC 0.49 to 0.59, against about 0.77 for an embeddings model). This engine makes the evaluation explicit; whether it matches the labels
better is an experiment (section 18), not a given.

## 2. The four layers
| Layer | Purpose | Who decides |
|---|---|---|
| 1. Scene understanding and applicability | what is happening, who is present, which requirements can apply | Gemini reports facts; **Python** computes applicability |
| 2. Mandatory gates | strict yes/no on every applicable non-negotiable requirement | Gemini answers; **Python** enforces |
| 3. Photography and editorial quality | six graded dimensions, a score out of 100 | Gemini grades 0-4; **Python** computes the score |
| 4. Publication readiness | consent, rights, caption, event facts, platform, visual flags | metadata only; Gemini can raise visual flags, never verify anything |

## 3. Rule source hierarchy
Every criterion carries `source` (CLIENT, TEAM or PROPOSED), `parent_rule_id`, `layer`, `decision_effect`, `applicability`, `approval_status` and more.
The client's 16 rules are in `config/rulebook/rule_sources.yaml` word for word; improved operational wording is stored separately in
`rulebook_v1.yaml`. The client's document marks no rule as mandatory: gates are team policy applied on top. PROPOSED rules (24 `NEW-*`, `P0xx`, the
extra scene facts and G1 components) are configured but only active in the layered profile, and are listed for NGO approval.

## 4. Activity taxonomy
`config/rulebook/activity_taxonomy.yaml`: 11 activities. Add a line to add one; the prompt and response schema pick it up (tested).

## 5. Mandatory gate logic (Layer 2)
Statuses: YES, NO, CANNOT_TELL, NOT_APPLICABLE. For each gate Python: (1) computes applicability from the Layer-1 facts (three-valued; an UNKNOWN fact
counts as "applies"); (2) replaces a missing answer with CANNOT_TELL; (3) turns a model "NOT_APPLICABLE" for something that applies into CANNOT_TELL;
(4) ignores a NO on something that does not apply (noted); (5) applies exceptions to NO answers only. Then: any applicable hard gate NO -> **REJECT**;
any CANNOT_TELL -> **HUMAN_REVIEW** (`unknown_is_fail` is preserved as "unknown is never a pass"). NOT_APPLICABLE is skipped.
Exceptions (`exceptions.yaml`): only medical PPE is active (it escalates to a human). The T2 back-view exception is **withdrawn** and the direct-gaze exception is **dormant** (both are kept in the file, active in no profile). Layered gates added by the NGO's decisions: **G016** no back views (R008_BACK, T_FRONT, R015_BEN_FRONT; the photo must be front or side), **G017** a present banner must be clearly visible, **G018** a present banner must be distinguishable from its surroundings (R001_DISTINCT). Beneficiary direct gaze (G015) is disapproved in every case for now; a volunteer's is not a violation.
Gate G1 can be split into components; the parent passes only if every applicable component passes.

## 6. Soft scoring (Layer 3)
`quality_scoring.yaml`. Six dimensions (weights 25/20/20/15/10/10), 0-4 each; `score = sum((s/4) * weight)` renormalised over applicable dimensions. A
dimension at 0 forces RESELECT, at 1 caps the result at HUMAN_REVIEW. Thresholds 85/70/55 are **uncalibrated proposals**. A soft score is only looked
at after the gates, so it can never override a gate failure. A former hard gate treated as soft (G10, G11, non-prominent banner) caps one dimension once.

## 7. Publication readiness (Layer 4)
`publication_checks.yaml`. Missing metadata is PENDING, never VERIFIED and never a violation. Visual flags (confidential information, manipulation, unsafe
activity, degrading representation) block at MEDIUM+ confidence; a possible minor only requires verification and is never stated as a fact about age.
`photo_approval_status` and `publication_readiness_status` are separate: a photo can be approved and still `APPROVED_PENDING_PUBLICATION_CHECKS`.

## 8-9. Gemini configuration and environment
`config/gemini_config.yaml` (model `gemini-3.8-flash`, temperature 0.1, JSON output, retries, concurrency). Environment: `GEMINI_API_KEY` and
`NGO_ALLOW_CLOUD_INFERENCE=true` (both required; a missing one stops the run before any client exists). `--env-file PATH` reads only those two keys.
The key is never logged. The photograph is sent as the original bytes (not resized).

## 10. Input and output
Input: an image (and, for new photos, verified `metadata`: `consent_status`, `rights_status`, `caption_status`, `credits_rights_status`,
`child_safeguarding_status`, `event_details_status`, `platform`, `direct_gaze_approved`...; values VERIFIED / PENDING / BLOCKED / NOT_REQUIRED).
Output (one JSONL row per photo): the validated model assessment, the Python decision (`recommendation`, `photo_approval_status`,
`publication_readiness_status`, reasons, every gate with its evidence, quality, publication checks), the raw response (saved before parsing), usage and cost,
and the model, prompt, rulebook and profile versions. Schemas: `config/schemas/*.json` (regenerate with `run_engine.py schemas`). `samples/` holds three
**synthetic** examples of the full structure.

## 11-15. Changing the rulebook
* **Add a rule:** add an entry under `criteria:` in `rulebook_v1.yaml` (source, parent, layer, effect, applicability, status). If it must be a gate, add it under `gates:`. Run `scripts/validate_rulebook.py`, then add a `RULEBOOK_CHANGELOG.md` entry.
* **Modify a gate:** change its wording in the criterion (`description_operational`) or its applicability name; for a profile-specific change use `overrides:` in `policy_profiles.yaml`, which **must** carry a `policy_change` record (the validator enforces it).
* **Add an exception:** an entry in `exceptions.yaml` (applies_to, required criteria/facts/metadata, effect, active_in).
* **Change a scoring weight or threshold:** `quality_scoring.yaml`. Weights must sum to 100 (tested).
* **Switch profile:** `evaluation.profile` in `gemini_config.yaml` or `--profile`. `checklist5_legacy` keeps the established 65 questions / 15 hard gates / strict scope; `layered_proposed_v1` is the recommended policy.
Changing a criterion, exception, activity or dimension changes the prompt and its hash. Thresholds and weights do not appear in the prompt, so **bump `rulebook_version` in `rulebook_v1.yaml` whenever you change the configuration**: every result records the model, prompt hash, rulebook version and profile, and a finished photo is only skipped when all of them match.

## 16-17. Running it
```bash
python scripts/validate_rulebook.py                     # offline
python scripts/run_engine.py prompt --profile layered_proposed_v1
python scripts/run_engine.py new IMAGE.jpg --image-id NGO_IMG_000001 --metadata meta.json              # dry run
python scripts/run_engine.py new IMAGE.jpg --image-id NGO_IMG_000001 --metadata meta.json --execute   # real call
python scripts/run_engine.py historical LIST.csv --run-name pilot1 --max-calls 20                       # dry run (LIST.csv: image_id,path; no label column)
python scripts/run_engine.py historical LIST.csv --run-name pilot1 --max-calls 20 --execute --env-file ../image-classifier/.env
python scripts/run_engine.py export runs/pilot1 --labels labels.csv --human human.csv                   # joins labels AFTER annotation
```
Use the main project's virtualenv (`../image-classifier/.venv/Scripts/python.exe`). Everything is a dry run unless `--execute` is given, and `--execute` also needs
the authorization switch and key. Runs are resumable: a finished photo is skipped only if the model, profile, rulebook and prompt hash all match.
**Historical annotation** never gives Gemini a label (the builder cannot receive one, the list loader rejects a label column, metadata with label keys is refused);
labels are joined afterwards by `export` and stored in their own field, apart from model observations and human-confirmed findings.

## 17b. Evaluator V1 and V2 (profiles) and how to compare them
* `layered_proposed_v1` is the evaluator exactly as run in pilots 2 and 3 (frozen in `evaluators/v1_frozen/`; a test compares its prompt and response schema byte for byte). `checklist5_legacy` is the original 15-gate interpretation.
* `layered_proposed_v2` inherits V1 and applies seven proposed changes (V2-01 ... V2-07, see `RULEBOOK_CHANGELOG.md` 2.0.0-proposed). All are `PROPOSED_PENDING_NGO_APPROVAL`; none is NGO policy.
* Switch profile: `--profile layered_proposed_v2` on `scripts/run_engine.py`. Switch one change off: `Rulebook().resolve("layered_proposed_v2", without=["V2-03"])`.
* Re-decide stored answers under another policy without any model call (exact for hard/soft, applicability and aggregation changes; it cannot show a change of wording): `layered_engine.comparison.redecide`.
* V1 vs V2 on the same photos, gate-level breakdown, ablation stages, leave-one-out, upper bounds: `python scripts/v1_v2_comparison.py --excel reports/v1_vs_v2_comparison.xlsx` (offline). Read `docs/V1_V2_COMPARISON.md` before quoting any number: it is a development comparison on photos that motivated the changes.
* Metrics: positive = approved; engine "approved" only when it auto-approves; false approval = disapproved photo auto-approved; false rejection = approved photo not auto-approved. With 25 photos per class one photo is 4 points: do not read small differences as improvement.

## 18. Tests
`../image-classifier/.venv/Scripts/python.exe -m pytest -q` : 135 tests, all offline (a guard makes any attempt to create a real Gemini client fail).
They cover every gate (NO, CANNOT_TELL, N/A, in both profiles), conditional applicability, the exceptions, G10/G11 legacy vs proposed, scoring, thresholds,
publication, the fifteen edge-case scenarios, fail-closed authorization, retries, resumability and the label separation. A mutation check (breaking five
safety behaviours on purpose) was caught by the suite each time.
The planned experiment (checklist5 vs this architecture on the same photographs) needs real calls and has **not** been run.

## 19. Current limitations
* **Only 10 photographs have been run** (all disapproved). The first live attempt failed completely (the API rejects `minItems`/`maxItems` in the schema); after the fix all answers were valid and complete, but accuracy against the human labels, false rejections on approved photos and run-to-run variation are unmeasured. A photo takes about a minute (the model also thinks) and costs roughly 1.5 to 2 US cents at placeholder prices.
* The face requirements stay strict (every visible beneficiary) until the computer-vision layer can locate and judge faces; until then the model's judgement of 'clear, sharp, lit' is all there is.
* G016-G018 depend on the model telling front from back and a banner from its background, which has not been tested on real photographs.
* The quality thresholds, the platform sizes and the one-third coverage idea are uncalibrated assumptions.
* The decisions are only as good as the model's answers; earlier trials showed Gemini's rule answers separate approved from disapproved photos poorly and vary between runs.
* Not implemented: computer vision, embeddings/retrieval, fine-tuning, batch selection (P002, P003, NEW-023), cropping checks (P006).
* Prices in `gemini_config.yaml` are placeholders from an earlier search summary, not the official price list.

## 20. Roadmap (design hooks exist; nothing built)
Computer vision evidence can be added as `CriterionAnswer(source="cv")` without touching the rule engine. Few-shot retrieval would extend the prompt builder.
Fine-tuning would reuse the enriched JSONL/CSV. Batch selection would add a layer over per-photo decisions.
