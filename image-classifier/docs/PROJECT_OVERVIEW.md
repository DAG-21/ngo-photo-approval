# Project Overview

## Objective
Given a photograph taken during NGO activities, estimate whether it will be approved for publication, return a probability
(0-100 %) **and** a decision, and keep rule-level evidence so each decision can be explained and audited.

## Design principles (as implemented)
1. **Rules are explicit.** A deterministic rule engine (`rules/evaluator.py`) consumes validated structured results; it
   never calls a provider and never reads free text. Mandatory failures reject regardless of any score.
2. **The model proposes, the policy disposes.** The VLM only reports PASS/FAIL/UNCERTAIN per rule with a short evidence note.
3. **Honest probabilities.** A separate scikit-learn layer (logistic regression + cross-fitted calibration) produces the
   probability. It is labelled `calibrated` only when data size and measured validation ECE justify it.
4. **No fabricated ground truth.** `pending` is never a negative label; VLM suggestions are never verified labels;
   missing client rules are documented, not invented.
5. **No data movement or spend by default.** Cloud inference, upload and paid tuning each have a separate authorization gate
   (config/env), and tuning also requires an explicit CLI flag; everything has a dry-run.
6. **Evidence before complexity.** An untuned baseline and a quality-only baseline come first; fine-tuning is kept only if the
   measured results justify it.

## Components
| Layer | Modules | Notes |
|---|---|---|
| Data | `data/*` | inventory (hashes, dHash, quality), audit, duplicate clusters, grouped stratified splits |
| Rules | `rules/*` | loader/validation, evaluator, threshold policy |
| Visual analysis | `models/*` | Gemini client (google-genai, Vertex), prompt, strict response parser, retry/rate limit, usage/cost |
| Tuning | `tuning/*` | JSONL builder (documented schema), preflight/upload/submit/poll, idempotent registry |
| Probability | `scoring/*` | features, training, calibration, prediction (+ `offline.py` for dataset-level runs) |
| Evaluation | `evaluation/*`, `scripts/evaluate_models.py` | FAR/FRR, PR/ROC-AUC, Brier, ECE, reliability, error analysis |
| Serving | `service/*`, `api/*`, `scripts/run_inference.py` | single image, upload bytes, resumable batch, FastAPI |

## Reasonable engineering decisions taken (and where to change them)
* `disapproved` folder -> `not_approved` (`configs/project.yaml: dataset.folder_label_map`).
* Byte-identical images that carry **both** final labels are excluded from supervised data and listed for human review
  (`dataset.label_conflict_exclusion`).
* Near-duplicates are grouped, never deleted. Group = connected components of {same submission filename, same SHA-256, near-duplicate cluster}.
* Image input is passed through untouched unless it is HEIC/MPO, EXIF-rotated, over 3072 px or over 18 MiB, in which case a
  JPEG copy is made in `data/derived/` (originals are never modified). Small banners/logos keep their detail (`model.yaml: image_input`).
* Default decision policy is conservative: reject on mandatory failure, uncertainty or missing results; no REVIEW state.
* Track A (overall label) is the default tuning target because it needs only data we have; Track B (rule-level) needs verified annotations that do not exist yet.
