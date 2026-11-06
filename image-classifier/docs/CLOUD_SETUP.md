# Cloud Setup (Google Cloud - Gemini Enterprise Agent Platform / Vertex AI)

Nothing in this repository contacts Google Cloud unless the gates below are opened. Offline work never needs an account.

## 1. Facts verified against the official docs (2026-10-29)
* Supervised tuning supports text, image, document, audio and video; image examples use `fileData{mimeType,fileUri}` with a `gs://` (or public HTTP) URI.
* The supported-model list names *Gemini 3.5 Flash* and *Gemini 3.5 Flash-Lite*; the tuning example uses `base_model="gemini-3.5-flash"`. The Flash-Lite
  **ID is not printed verbatim** in the pages we could read, so it is marked `verified: false` in `configs/model.yaml`; preflight warns until you confirm it.
* For Gemini 3.5 / 3.1 Flash-Lite, tuning and serving are limited to the `us` and `eu` multi-region endpoints (`GOOGLE_CLOUD_LOCATION=us`).
* Limits used by the builder: 30 images/example, 20 MB/image, 131,072 tokens/example, 1 GB per JSONL, up to 300K multimodal examples; adapter sizes 1/2/4/8/16.
* The docs we could extract did not state: exact per-model image-tuning support matrix, the list of terminal job states beyond SUCCEEDED/FAILED/CANCELLED/EXPIRED
  (we use the SDK's `JobState` enum), pricing, and the exact inference snippet for a tuned model (it is called "the same way as base Gemini"). Re-verify these before spending.
* Pricing is **not** hard-coded anywhere. Enter it in `configs/model.yaml -> pricing`.

## 2a. Using a Gemini API key (inference only)
If you have a Gemini Developer API key (Google AI Studio) instead of a Google Cloud project:

1. Open the project's **`.env`** file (project root; it is git-ignored and was created for you with the transfer gate closed).
2. Paste the key after `GEMINI_API_KEY=` - no quotes, no spaces. Keep `NGO_AUTH_MODE=api_key`.
3. Only when the NGO has authorized sending its photos to Google, set `NGO_ALLOW_CLOUD_INFERENCE=true`.
4. Check with a dry run (prints `api key present`, sends nothing): `python scripts/run_baseline.py --mode vlm --splits validation --sample 20`, then add `--execute` to send the sample.

Never put the key in `configs/*.yaml`, source files, docs, commit messages, or chat. The code reads it from the environment at call time and never logs or stores it.
A stray `GEMINI_API_KEY` in your shell is ignored unless `NGO_AUTH_MODE=api_key` is set explicitly, so data can't silently switch destination.

Limits of API-key mode:
* **Tuning is not available** (`manage_tuning_job.py preflight` fails with `auth_mode_supports_tuning`); tuning needs `NGO_AUTH_MODE=vertex` with a Google Cloud project, bucket and credentials.
* Data handling differs between the Gemini Developer API and Google Cloud (Vertex): retention, whether content may be used to improve products (this has historically depended on paid vs free tier), and
  regional processing. **Verify the current terms before sending photos that may show children or beneficiaries**; this has not been checked here.
* Model IDs on the Developer API can differ from Vertex. `gemini-3.5-flash` is confirmed for Vertex tuning in the docs; check that your key can call it (a 404 maps to `model_unavailable`).

## 2. Authentication
Preferred for production (Application Default Credentials):
```bash
gcloud auth application-default login
gcloud config set project <PROJECT_ID>
```
Service account (automation): create a key file **outside** the repo and set `GOOGLE_APPLICATION_CREDENTIALS` in your shell or `.env`. `.gitignore` already excludes `.env`,
`*credentials*.json`, `*service-account*.json`, `*.key`, `*.pem`. Prefer short-lived credentials/workload identity over key files.

## 3. Least-privilege IAM (suggested; confirm with your cloud admin)
| Task | Role (suggested) |
|---|---|
| Inference + tuning jobs | `roles/aiplatform.user` on the project |
| Upload/read tuning images + JSONL | `roles/storage.objectAdmin` **on the one bucket only** (or `objectCreator` + `objectViewer`) |
| The tuning service must read the bucket | grant the Vertex AI service agent `roles/storage.objectViewer` on that bucket |
Do not use project-wide Owner/Editor. Enable uniform bucket-level access and public-access prevention; **never** make image URLs public.

## 4. Bucket
Create a private bucket in the NGO's chosen jurisdiction (the docs say bucket region is flexible but recommend the tuning project). Set
`NGO_GCS_BUCKET` and rebuild the tuning data so the JSONL URIs point at it. Consider a lifecycle rule to delete `images/` and `datasets/` after the job completes
(see docs/SECURITY_PRIVACY.md for the retention process).

## 5. Authorization gates (all default to **false**)
| Variable | Unlocks |
|---|---|
| `NGO_ALLOW_CLOUD_INFERENCE` | sending images to Gemini for baseline runs, inference and the API |
| `NGO_ALLOW_CLOUD_UPLOAD` | uploading dataset images + JSONL to GCS |
| `NGO_ALLOW_PAID_TUNING` | submitting a tuning job (**plus** the `--submit-cloud-job` CLI flag) |
Set them only after the NGO has authorized the corresponding data transfer in writing.

## 6. Preflight
```bash
python scripts/manage_tuning_job.py preflight             # offline: config, authorization, JSONL validity, credentials, duplicate-job check
python scripts/manage_tuning_job.py preflight --online    # additionally checks the base model exists via the API
```
The report lists each blocking failure (missing project, bucket, authorization, placeholder URIs, invalid records, no credentials, identical job already submitted).

## 7. After a job succeeds
`status --wait` records `tuned_model` and `tuned_endpoint` from the provider response into `artifacts/tuning_registry.json`; `activate --job <name>` marks it active.
Put the endpoint in `configs/model.yaml -> tuned_model.endpoint` (or pass it explicitly) before using `--use-tuned`. Nothing is guessed.
