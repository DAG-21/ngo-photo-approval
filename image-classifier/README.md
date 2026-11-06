# NGO Photo Approval Engine

An explainable, auditable engine that decides whether an NGO photograph may be published on social media.
It combines **explicit rule evaluation** (mandatory rules can never be outvoted), a **cloud vision-language model
(Gemini)** for visual checks, a **calibrated probability layer** trained on the NGO's historical decisions, and a
**leakage-resistant evaluation pipeline**.

> **Read this first - current status** (details in [docs/LIMITATIONS.md](docs/LIMITATIONS.md))
>
> * Implemented and tested offline: dataset audit, splits, rule engine, decision policy, probability + calibration,
>   cloud client (mocked), tuning-data builder, guarded tuning-job manager, prediction service, batch mode, FastAPI.
> * **Rules:** the text file `rules/photo-approval-requirement-rules.txt` is the single source of truth. It is formalized in
>   `configs/rules.yaml` as 16 rules (wording verbatim). A rule is mandatory **only if the text says so**; it says so for none, so
>   all rules are scored (evaluated on every photo, none rejects alone). If you edit the text, the system refuses to run until
>   `python scripts/sync_rules.py --write` re-syncs it - see [docs/RULE_SPECIFICATION.md](docs/RULE_SPECIFICATION.md).
> * **Blocked on inputs:** no VLM run, upload or tuning job has been executed - no Gemini credential is configured and the
>   NGO's data-transfer authorization gate is closed. Put an API key in `.env` (section 7) to start. Only the offline quality-feature baseline has measured results, and it shows
>   **no skill** (validation ROC-AUC ~0.51). Notably, image resolution/file size alone separated the labels (AUC ~0.80): the
>   dataset has a provenance confound (see docs/DATASET_AUDIT.md section 5), so those properties are deliberately not features.
> * `pending` images are tracked but excluded from every training, calibration and evaluation step.

## 1. How a decision is made

```
photo ──► validate/decode ──► VLM rule analysis (PASS/FAIL/UNCERTAIN per rule, structured + validated)
                                     │
                                     ▼
                         rule engine (deterministic, provider-independent)
                          mandatory fail / uncertain / missing  ──► REJECT (or REVIEW if enabled)
                                     │ rules allow
                                     ▼
              features (rule outcomes + image quality) ──► logistic regression ──► calibration
                                     │
                                     ▼
                    probability ≥ threshold ? APPROVE : REJECT      (threshold from configs/thresholds.yaml)
```

* A **mandatory rule failure always rejects**, even at 99 % probability. The probability itself is *not* zeroed:
  "policy rejection" and "estimated historical approval chance" are different things and are reported separately.
* Missing, duplicated, unknown or `UNCERTAIN` results can never count as a pass for a mandatory rule.
* Any provider/validation failure raises a typed error (or an error row in batch mode); it never becomes an approval.

### Three different numbers - never confuse them

| Field | Meaning | Where it appears |
|---|---|---|
| `heuristic_score` (0-100) | share of active rules that passed. A summary, **not** a probability. | response |
| `raw_classifier_probability` (0-1) | logistic-regression output; trained but not calibrated. | response |
| `approval_probability` (0-1) + `approval_percentage` (0-100) with `probability_type` | the number used for the threshold. `probability_type` is `calibrated` **only** if calibration passed the data-size and validation-ECE checks in `configs/thresholds.yaml`; otherwise `uncalibrated_classifier` (or `none` if no model is loaded). | response |

A model's self-reported confidence, if returned, is stored as `self_reported_confidence_uncalibrated` and is never used.

## 2. Repository layout

```
configs/    project.yaml  rules.yaml  model.yaml  thresholds.yaml        (all policy/provider settings)
src/ngo_photo_approval/
  data/      inventory, audit, duplicates, preprocessing, splitting
  rules/     loader, evaluator (rule engine), policy (threshold decision)
  models/    cloud_client (Gemini + retry/rate limit), baseline_analyzer, tuned_model, response_parser, batch_runner
  tuning/    dataset_builder (JSONL), job_manager (preflight/upload/submit/poll), model_registry
  scoring/   feature_builder, train, calibration, predict, offline
  evaluation/ metrics, calibration_metrics, error_analysis
  service/   prediction_service (single, bytes, batch)      api/main.py (FastAPI)
scripts/    audit_dataset  prepare_dataset  run_baseline  build_tuning_data  manage_tuning_job
            annotate_rules  train_probability_model  evaluate_models  run_inference
docs/       overview, rule spec, dataset audit, model selection, training guide, evaluation report,
            cloud setup, security & privacy, limitations
tests/      unit/ + integration/  (synthetic images only; no cloud access needed)
data/raw/   the client's dataset (read-only, git-ignored)       data/{manifests,splits,tuning,results}  derived, git-ignored
```

## 3. Setup

Python 3.11+ (developed on 3.11.5).

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
Copy-Item .env.example .env        # then edit; .env is git-ignored
```
Linux / macOS:
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env
```
Below, `python` means the venv interpreter (`.\.venv\Scripts\python.exe` on Windows, `python` once activated elsewhere).
Everything up to and including "run the tests" works with **no Google Cloud account**.

```bash
python -m pytest            # 205 tests, no network, no cloud
python -m ruff check .
```

## 4. Dataset and label interpretation

`NGO_DATA_DIR` (default `data/raw`) must contain one folder per outcome:

| Folder | Normalized label | Used for training/calibration/evaluation |
|---|---|---|
| `approved/` | `approved` | yes (positive) |
| `disapproved/` | `not_approved` | yes (negative) |
| `pending-approval/` | `pending` | **no** - tracked separately; reserved for later manual validation. Never treated as rejected. |

The original files are never moved, renamed or modified; all derived data goes to `data/`.
Files are named `<class_folder>_<n><ext>` (`approved_1.jpg` ... `approved_3448.jpg`, `disapproved_1.jpg`, `pending-approval_1.jpg`, ...), numbered 1..count within each class
by `python scripts/rename_dataset.py --apply`. The original names (which carried the program and submission number, e.g. `arham_aahar-10 (56).jpg`) are kept in
`data/manifests/rename_map.csv` (copy: `data/raw/_rename_map.csv`); image ids, programs and groups are derived from them, so splits did not change.
`python scripts/rename_dataset.py` is a dry run, `--revert` restores every original name, and new files added later are numbered on from the last index.
The program prefix is **not** a label.
See [docs/DATASET_AUDIT.md](docs/DATASET_AUDIT.md) for counts, duplicates, label-conflict clusters and format findings.

## 5. Run the pipeline

### 5.1 Audit and split (offline)
```bash
python scripts/audit_dataset.py         # inventory + audit + duplicates -> data/manifests/*.csv, docs/DATASET_AUDIT.md
python scripts/prepare_dataset.py       # grouped, stratified 70/15/15 split (seed 42) -> data/splits/*.csv
```
The first audit decodes ~10 GB of photos (several minutes); re-runs reuse a cache keyed by size+mtime.
Splits group exact duplicates, near-duplicates and same-submission filenames together, so no group crosses
train/validation/test. The test set is only read by `evaluate_models.py --final-test`.

### 5.1b Why were photos disapproved? (human tagging)
```bash
pip install -e .[tagging]
python scripts/make_tagging_sheet.py build                    # -> data/annotations/disapproval_tagging_sheet.xlsx
python scripts/make_tagging_sheet.py import <returned>.xlsx   # -> data/annotations/disapproval_reasons.csv + summary
```
The workbook has one row per unique not-approved photo (train + validation only; thumbnails embedded; random order so any prefix is a fair sample), dropdowns for the reason (from the rules plus a few non-rule reasons), and a progress counter. It contains photos of real people: share it only with authorised taggers.

### 5.2 Offline baseline (no cloud)
```bash
python scripts/run_baseline.py --mode technical
```
Trains the quality-feature logistic regression on *train*, selects a threshold on *validation*, reports validation
metrics. This is a weak baseline that cannot enforce rules; it exists to measure how much the hybrid adds.

### 5.3 Untuned VLM baseline (needs authorization)
Dry run first (prints image count, rough token estimate, authorization state, sends nothing):
```bash
python scripts/run_baseline.py --mode vlm --splits validation --sample 100
```
To execute you must (a) have the NGO's written data-transfer authorization, (b) set in `.env`:
`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION=us`, `NGO_ALLOW_CLOUD_INFERENCE=true`, and authenticate (section 7), then:
```bash
python scripts/run_baseline.py --mode vlm --splits train validation --sample 300 --execute
python scripts/train_probability_model.py --feature-set rules --analysis-model gemini-3.5-flash-lite --name vlm_gemini-3.5-flash-lite
```
Results are appended to `data/results/vlm/<model>/<split>.jsonl`; re-running resumes. Start with a small sample, check
cost/latency, then expand.

### 5.4 Tuning data, dry run, and (authorized) job submission
```bash
python scripts/build_tuning_data.py --track A                 # writes data/tuning/*.jsonl; uploads nothing
python scripts/manage_tuning_job.py preflight                 # offline readiness report
```
Submitting needs **all** of: `NGO_ALLOW_CLOUD_UPLOAD=true`, `NGO_ALLOW_PAID_TUNING=true`, `NGO_GCS_BUCKET`, passing preflight,
**and** the explicit flag:
```bash
python scripts/manage_tuning_job.py submit --submit-cloud-job
python scripts/manage_tuning_job.py status --wait
python scripts/manage_tuning_job.py activate --job <job name>
```
Repeating `submit` with identical data and settings returns the existing job instead of paying twice (`--force-new` overrides).
Track B (rule-level) needs human-verified annotations - see [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md).

### 5.5 Evaluate and choose a threshold
```bash
python scripts/evaluate_models.py --split validation          # candidates are auto-discovered
python scripts/evaluate_models.py --split test --final-test   # ONCE, at the very end
```
The threshold is chosen on **validation** by `train_probability_model.py` (highest approval recall subject to
`max_false_approval_rate` in `configs/thresholds.yaml`; written to `artifacts/probability_model/<name>/threshold_selection.json`).
Copy the chosen value into `approval_threshold` deliberately; it is never changed automatically. If no threshold meets the
false-approval limit the selection reports `feasible: false` instead of silently relaxing it.

### 5.6 Inference and API
```bash
python scripts/run_inference.py --image path/to/photo.jpg
python scripts/run_inference.py --dir path/to/folder --out data/results/batch.jsonl --csv data/results/batch.csv
uvicorn ngo_photo_approval.api.main:app --host 127.0.0.1 --port 8000
```
`POST /predict` (multipart `file`, optional form field `program`), `GET /health`, `GET /ready`.
Limits come from `configs/project.yaml` (`api.max_upload_mb`, allowed content types). Uploads use a private temp file that is
always deleted; error bodies never contain local paths. Example response:
```json
{"image_id": "upload_ab12cd34ef56", "decision": "APPROVE", "approval_probability": 0.87, "approval_percentage": 87.0,
 "probability_type": "calibrated", "threshold": 0.8, "hard_rule_failures": [], "reason_codes": ["APPROVED"],
 "checks": [{"rule_id": "R001", "status": "PASS", "evidence": "..."}], "rule_set_version": "...",
 "model_id": "gemini-3.5-flash-lite", "model_version": null, "heuristic_score": 100.0}
```
(numbers illustrative). Status codes: 415 type, 413 size, 422 invalid image, 502 provider/model-output failure, 503 unavailable/unauthorized.

## 6. Configuration quick reference

| File | Controls |
|---|---|
| `configs/project.yaml` | dataset path/label map, supported formats, split ratios + seed, cloud project/region, authorization gates, API limits, review switch |
| `configs/rules.yaml` | the rule set (id, wording, mandatory vs scored, check method, uncertain policy, program scope, confirmation flags) |
| `configs/thresholds.yaml` | `approval_threshold`, mandatory enforcement, missing/uncertain policies, review, threshold-selection and calibration criteria |
| `configs/model.yaml` | model IDs, generation/retry/rate-limit settings, image-input limits, tuning hyper-parameters and documented limits, pricing |

`REVIEW` outcomes require **both** `review.enabled: true` (project.yaml) **and** `allow_review_status: true` (thresholds.yaml).

## 7. Gemini credentials (summary; see [docs/CLOUD_SETUP.md](docs/CLOUD_SETUP.md))
**Gemini API key (inference/baseline only):** put it in the git-ignored **`.env`** file in the project root, on the line `GEMINI_API_KEY=` (no quotes), keep `NGO_AUTH_MODE=api_key`,
and set `NGO_ALLOW_CLOUD_INFERENCE=true` only after the NGO authorizes sending photos to Google. Never put the key in YAML/code/chat. Tuning is not available in this mode.
Verify the key without sending any photo: `python scripts/check_gemini_key.py` (metadata-only calls; never prints the key).

**Google Cloud project (needed for tuning; recommended for production):** `NGO_AUTH_MODE=vertex` with Application Default Credentials: `gcloud auth application-default login`, or `GOOGLE_APPLICATION_CREDENTIALS` pointing to a
service-account key stored **outside** the repo. Gemini 3.x tuning/serving uses the `us` (or `eu`) multi-region endpoint.

## 8. Cost methodology
Cost is reported as **unknown** until prices are entered in `configs/model.yaml -> pricing` (they change; copy from the current official pricing page).
Per-call token usage returned by the API is stored with every analysis; `estimate_cost` multiplies it by the configured prices.
Rough planning figure used by dry runs: ~800 input / ~250 output tokens per image per rule-analysis call.
Tuning cost scales with training tokens x epochs; check the official tuning pricing before `submit`.

## 9. Privacy
Images may show beneficiaries and children. Nothing leaves the machine unless explicit flags are set; no facial recognition or
identity/beneficiary inference is requested (prompts forbid it); logs contain IDs, decisions and token counts only. Using a cloud
provider does not by itself satisfy the NGO's legal obligations. See [docs/SECURITY_PRIVACY.md](docs/SECURITY_PRIVACY.md).

## 10. Known limitations
See [docs/LIMITATIONS.md](docs/LIMITATIONS.md): unconfirmed rules, no cloud run yet, labels reflect human decisions that may not map 1:1 to the rules,
heavy duplication, program/label confounding, and small validation/test group counts.
