# Security and Privacy Notes

NGO photographs may show beneficiaries, workers and **children**. Using a cloud provider does not by itself satisfy the NGO's privacy, consent, or regulatory
obligations (e.g. DPDP/GDPR-style rules, child-protection policies, donor/beneficiary consent). Obtain the NGO's legal sign-off before enabling any cloud gate.

## What the code enforces
| Control | Implementation |
|---|---|
| No transfer without authorization | three independent gates (`NGO_ALLOW_CLOUD_INFERENCE`, `NGO_ALLOW_CLOUD_UPLOAD`, `NGO_ALLOW_PAID_TUNING`), all `false` by default; tests assert refusal |
| No accidental spend | tuning submission also needs `--submit-cloud-job`, a passing preflight, and is idempotent per (data, settings) |
| Raw data is read-only | scripts only read `NGO_DATA_DIR`; derived copies go to `data/derived/`; a test asserts the inventory does not modify files |
| No identity inference | prompts forbid facial recognition, identification and inferring beneficiary status from appearance; rules must carry an explicit visual definition |
| Opaque identifiers | image IDs are hashes; GCS object names use the ID, not the original filename; API errors/responses never contain local paths (tests) |
| Safe logs | logs carry image ID, decision, reason codes, token counts and latency only - never image bytes, prompts with images, credentials or full URLs |
| Safe uploads | API enforces size and content-type limits, decodes the image, uses a private temp file that is always deleted (tests cover success and failure paths) |
| Secrets | `.env`, key files and credentials are git-ignored; `.env.example` holds placeholders only |
| No client data in Git | `data/raw`, `data/manifests`, `data/splits`, `data/tuning`, `data/results`, `data/derived`, annotations, model artifacts and all image/video extensions are git-ignored |
| Error visibility | each failure type is distinct (`InvalidImageError`, `ProviderAuthError`, ...); nothing silently becomes an approval |

> The project folder is **not currently a git repository**; if you `git init`, review `git status` before the first commit to confirm no client file is staged.

## Data handled by this project, and where it lives
| Artifact | Contains | Location | Sensitivity |
|---|---|---|---|
| Raw photos | client images | `NGO_DATA_DIR` (outside repo preferred) | high |
| `data/derived/tuning_images/` | JPEG copies (HEIC/MPO converted, EXIF-rotated, resized) - ~3 GB | local | high - delete when no longer needed |
| `data/manifests/*.csv`, `data/splits/*.csv` | relative paths, hashes, sizes, quality metrics | local | medium (paths reveal program names) |
| `data/tuning/*.jsonl`, `upload_plan.csv` | URIs, labels, prompts (no pixels) | local | medium |
| `data/results/vlm/**`, `batch_predictions.jsonl` | structured model output incl. one-sentence evidence notes | local | medium - evidence text may describe a scene |
| GCS bucket (if used) | the images and JSONL | cloud | high |

## Retention and deletion (process; execute only when authorized)
1. After a tuning job succeeds and is verified, delete the bucket prefix `<NGO_GCS_PREFIX>/images/` and `datasets/` (or let a lifecycle rule do so). Tuned model artifacts are separate cloud resources; delete via the console/API when decommissioning.
2. Delete `data/derived/` (regenerable with `build_tuning_data.py`).
3. Delete or archive `data/results/**` according to the NGO's retention policy; analyses can be regenerated but cost money.
4. Temporary upload files are removed immediately by the service; no action needed.
5. Record each deletion (what, when, who authorized) alongside the model registry for provenance.
No automatic deletion is performed by the code, because deleting client data is irreversible and must be explicitly authorized.

## Provenance
Each trained probability model stores: model version, feature set/version, rule-set version + fingerprint, seed, row counts, calibration status and reasons.
Each analysis record stores the model id, prompt version, rule-set version, token usage and the structured response. Tuning jobs are stored with their
dataset-derived idempotency key, base model, hyper-parameters, URIs and resulting tuned model.

## API-key mode
`NGO_AUTH_MODE=api_key` sends images to the Gemini Developer API using the key in `.env`. The key is read from the environment at call time and is never logged, stored in configs, or included in
responses; `.env` is git-ignored. It must be rotated if it is ever pasted into a chat, commit or ticket. Data-use/retention terms for this API differ from Google Cloud/Vertex and have **not**
been verified here: confirm them (especially for photos of children and beneficiaries) before opening the `NGO_ALLOW_CLOUD_INFERENCE` gate.

## Residual risks to review with the NGO
* VLM evidence strings are free text produced by a third-party model; they could mention people in a scene. Consider not persisting them.
* Hashes/near-duplicate analysis are local, but any cloud step transmits full-resolution pixels (up to 3072 px per side by default).
* Provider data-use terms (training on customer data, regional processing, retention) must be confirmed for the chosen project/region.
