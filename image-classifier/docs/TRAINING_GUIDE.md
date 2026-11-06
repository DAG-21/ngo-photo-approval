# Training Guide and Experiment Plan

## 1. Split strategy (Phase 4)

* **Population:** only final-label images that decoded and survived the audit (`eligible_for_training`): 3,410 `approved` + 1,015 `not_approved`
  = 4,425. Excluded: 212 `pending` (not final), 155 images in byte-identical clusters with contradictory labels. (Counts are regenerated
  by `scripts/audit_dataset.py`; see `docs/DATASET_AUDIT.md`.)
* **Groups:** union-find over (i) same `label|program|submission` key (taken from the ORIGINAL filenames via `rename_map.csv`; files are now named `<class>_<n>`), (ii) same SHA-256, (iii) dHash near-duplicate cluster
  (Hamming <= 4 and similar aspect ratio). 680 groups. All members of a group go to the same split. Event IDs and usable EXIF capture times do
  not exist, so the filename key is an *assumed* proxy for "same event/burst" (unverified).
* **Assignment:** greedy, largest group first, to the split furthest from its per-class targets; seeded (`splits.seed = 42`), ratios
  70/15/15 from `configs/project.yaml`. Result: train 3,098 / validation 664 / test 663 images with ~77 % approved in each.
* **Guards:** `verify_no_leakage` (image id, group id, SHA-256, duplicate cluster disjoint across splits) and `verify_no_pending` run on every
  split build and on every tuning export; the tuning builder additionally proves no test image is exported.
* **Test discipline:** `evaluate_models.py --split test` refuses to run without `--final-test`; the probability trainer's signature has no test
  argument (enforced by a unit test); templates for rule annotation exclude test images.
* **Caveat:** group sizes are very uneven (validation has only ~50 groups), so validation/test metrics have high variance and a few large
  submissions dominate them. Treat small differences between candidates as noise. Cross-validation over groups is a sensible follow-up once
  the real VLM features exist.

## 2. Two tuning tracks (kept distinct)

### Track A - overall approval classifier (default; data available now)
| | |
|---|---|
| Task | image -> `{"decision": "approved" | "not_approved"}` |
| Labels | image-level human decisions; `pending` excluded |
| Training target text | `approved` -> `{"decision": "approved"}`; `not_approved` -> `{"decision": "not_approved"}` |
| Output | hard label only: **no probability, no rule evidence**, so by itself it cannot enforce mandatory rules or give a calibrated number |
| Class balance | 77/23 in train. Optional `--balance downsample` (train only, seeded) to reduce a bias toward "approved" |
| Evidence needed to keep it | beats the untuned VLM + rules + probability pipeline on the same held-out split, at an equal or lower false-approval rate |

### Track B - rule-level compliance analysis (preferred for auditability; **blocked**)
| | |
|---|---|
| Task | image -> PASS/FAIL/UNCERTAIN per active rule (+ evidence) |
| Labels | **human-verified** per-rule annotations only. Not derivable from the image-level label |
| Blocker | the client rules document is missing, so there are no confirmed rules to annotate against |
| Workflow | `python scripts/annotate_rules.py template` -> humans fill `status/evidence/annotator`, set `verified=true` -> save as `data/annotations/rule_annotations.csv` -> `annotate_rules.py validate` -> `build_tuning_data.py --track B` |
| Volume | the docs recommend starting at ~100 examples and scaling to thousands; verify rule-label balance per rule before tuning |
| Safeguards | unverified rows ignored; images lacking verified labels for **all** active rules are skipped *with a reason*; annotation of test images is rejected by `validate` |

### Image normalization for tuning (provenance guard)
Resolution and file size alone separate approved from not_approved in this dataset (ROC-AUC ~0.80; approved images are mostly full-resolution camera originals). A tuned model sees pixels, so
unguarded it could learn that shortcut instead of anything about the photos. Therefore tuning examples **and** tuned-model inference use the same normalization
(`image_input.tuning_normalize_long_side_px`, default 1280): the long side is set to exactly that many pixels (large images scaled down, small ones up), then re-encoded identically as JPEG (q90, 4:2:0, no EXIF).
The originals are untouched; normalized copies live in `data/derived/tuning_images_norm1280/` and are what gets uploaded. Train and serve must match: `TunedDecisionClassifier` calls the same `prepare_for_tuned_model`.
Trade-off: very small banner text can lose detail. This reduces the resolution/size/compression cues but cannot prove the model ignores other provenance traces, so the final evaluation should still be checked
on resolution-matched subsets. Changing the value changes the training data, so rebuild (`build_tuning_data.py`) before any new tuning job.

## 3. Comparison plan (same held-out split)
1. `technical_quality` - quality-feature LR (offline, available now).
2. `vlm_rules:<model>` - untuned Gemini rules + rule engine + probability model.
3. `tuned_decision:<model>` - Track A tuned model (hard labels).
4. `vlm_rules:<tuned model>` - Track B tuned rule evaluator + rule engine + probability model (needs verified annotations).

Primary criterion: false-approval rate at the client's tolerance (`threshold_selection.max_false_approval_rate`, default 5 %), then approval recall,
auto-processed share, calibration (Brier/ECE), latency and cost. Fine-tuning is retained only if it clearly improves these.

## 4. Label -> training-record mapping (Track A)
```
approved      -> {"contents":[{"role":"user","parts":[{"fileData":{"mimeType":"image/jpeg","fileUri":"gs://<bucket>/<prefix>/images/<image_id>.jpg"}},
                                                       {"text":"Was this photograph approved for publication? ..."}]},
                              {"role":"model","parts":[{"text":"{\"decision\": \"approved\"}"}]}], "systemInstruction":{...}, "generationConfig":{"mediaResolution":"MEDIA_RESOLUTION_HIGH"}}
not_approved  -> same, target {"decision": "not_approved"}
pending       -> never exported
```
Schema source: official docs (image tuning + prepare-data pages, checked 2026-10-29). File URIs use the opaque `image_id`, never the original file name.
Documented limits enforced: <= 30 images/example, <= 20 MB/image, image MIME types, <= 131,072 tokens/example (estimated), <= 1 GB per JSONL.
The documented guidance is to start with ~100 examples and provide a validation set.

## 5. Running the tuning workflow
```bash
python scripts/build_tuning_data.py --track A        # dry run: JSONL + upload_plan.csv + export_report.json (placeholder URIs until a bucket is set)
python scripts/manage_tuning_job.py preflight        # lists every missing prerequisite
```
After the NGO authorizes the transfer and a bucket/credentials exist (docs/CLOUD_SETUP.md), rebuild the JSONL (so URIs point at the bucket), then
`manage_tuning_job.py submit --submit-cloud-job`. Jobs are recorded in `artifacts/tuning_registry.json`; the tuned endpoint is written **only**
from the provider's response, and `activate` is refused for a job that has not succeeded.

## 6. Probability model
`scripts/train_probability_model.py` fits logistic regression (C chosen on validation Brier from {0.01, 0.1, 1, 10}); calibration is **cross-fitted on
train** with group-aware folds (so validation stays clean for threshold selection and ECE measurement); the test split is never read.
Features: per-rule pass/fail/uncertain/missing indicators + counts (from rule-engine outputs on saved VLM results) and/or deterministic quality
measures. Ground-truth annotations are never features. Calibrated status requires >= 200 training samples, >= 50 per class, and validation ECE <= 0.08.

A model is labelled `calibrated` only if, in addition, validation ROC-AUC >= 0.60: a no-skill model that outputs the base rate can have a tiny ECE but is not a usable probability.

Current measured state: only the quality-only model has been trained (blur, brightness, contrast, aspect ratio; **resolution/file size deliberately excluded** because they are provenance proxies - see DATASET_AUDIT.md section 5). On validation it has ROC-AUC ~0.51 (no skill) and is reported as `uncalibrated_classifier`. An earlier version that included megapixels scored 0.81 and was discarded for that reason.
