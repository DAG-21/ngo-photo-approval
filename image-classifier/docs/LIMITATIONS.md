# Limitations and Open Items

Status legend: **Implemented+tested** (unit/integration tests, no cloud) · **Run on real data** · **Blocked** (needs input/authorization) · **Not started**.

## Feature status
| Capability | Status |
|---|---|
| Dataset inventory, audit, duplicate clusters, grouped splits | Implemented+tested; **run on real data** |
| Rule engine + decision policy | Implemented+tested (mandatory override, uncertain/missing/duplicate/unknown, threshold edge cases) |
| Probability model + calibration + threshold selection | Implemented+tested; **run on real data for the quality-only model** |
| Gemini client, prompts, strict parsing, retry/rate-limit (mocked) | Implemented+tested; **never called against the real API** |
| Tuning data builder (Track A) | Implemented+tested; **run on real data (dry run, placeholder URIs)** |
| Tuning data builder (Track B) | Implemented+tested; **Blocked**: no verified rule annotations |
| Tuning job manager (preflight/upload/submit/poll/cancel/registry) | Implemented+tested with mocks; **no job submitted, no upload performed** |
| Prediction service, batch/resume, FastAPI | Implemented+tested with mocks |
| Untuned VLM baseline results | **Blocked**: no credentials / transfer authorization |
| Tuned-model results, final test-set comparison | **Not started** (depends on the above) |
| OCR / object-detector rule checks | extension point only |

## Inputs that are missing
1. **Interpretation gaps in the rules text.** `rules/photo-approval-requirement-rules.txt` is the source of truth (16 rules, none mandatory because the text never says "mandatory"). It does not define "volunteer", "wherever appropriate", "unless a specific photograph requires it", or numeric blur/light thresholds, and the plain-language restatements (`visual_evidence`, `applies_when`) have not been reviewed by the client, so `rule_set_confirmed` is false. With no mandatory rules the decision is the probability threshold alone: rule results inform the probability but never reject a photo by themselves.
2. **A Gemini credential and NGO data-transfer authorization** (API key for inference, or a Google Cloud project + bucket for tuning).
3. **Verified per-rule annotations** (needed for Track B and for any rule-level accuracy claim).
4. **Pricing** entered in `configs/model.yaml` (costs are reported as unknown until then).
5. Verification of the `gemini-3.5-flash-lite` model ID (not printed verbatim in the docs we could read) and of the image-tuning support matrix per model.

## Data caveats (measured; see DATASET_AUDIT.md)
* **Class imbalance** ~77 % approved / 23 % not approved. A classifier that always approves scores 77 % accuracy; accuracy is therefore not a headline metric - false-approval rate is.
* **Heavy duplication:** hundreds of byte-identical copies; 17 exact-duplicate clusters carry *both* labels (155 images excluded as contradictory ground truth, one 33-image cluster has 32 `disapproved` + 1 `approved`). Near-duplicates are kept but grouped.
* **Program/label confounding:** some programs occur under only one label (e.g. `chaas_distribution` only as not_approved/pending); a model can learn "program" instead of "rule compliance". Program is not an input feature of the probability model, but visual content correlates with it.
* **Resolution / file-size provenance confound:** approved images are mostly full-resolution camera originals (63-68 % >= 4 MP over the eligible set; validation split 65 %) while many not_approved images are small copies (11 % >= 4 MP in the validation split, 45 % overall). The split-to-split spread (driven by a few large duplicate groups) shows how unstable such statistics are. Resolution, file size and compression history are label proxies, so they are **not** used as features.
* **Format artifacts:** 111 HEIC and 82 MPO files (all in `approved`; 2 PNGs too) sit behind `.jpg` names. They are converted to JPEG before any model sees them so the format cannot act as a label proxy, but re-encoding is a (small) distribution difference worth re-checking once VLM results exist.
* **Labels are human decisions**, not rule outcomes. `disapproved` may include reasons outside the rule set (quality, duplicates, administrative); we cannot tell. Rule-level claims need verified rule annotations.
* **`pending` (182 JPG + 30 MP4)** is excluded everywhere; videos are not supported by the pipeline.
* **Unverified grouping assumption:** `program-N` is assumed to be one submission/event. If the real structure differs, leakage between splits is still possible for visually related but differently named photos.
* **Few groups in validation/test (~50-60):** metrics are high-variance; differences of a few points between candidates are not reliable.

## Modelling caveats
* **The quality-only baseline has no skill once provenance is removed.** With image resolution included it reached validation ROC-AUC 0.81 - but file size alone gave 0.80 on the same split, so it was learning "camera original vs small copy", not compliance. After dropping the resolution feature (blur/brightness/contrast/aspect only) validation ROC-AUC is **0.51** (chance). It is reported as `uncalibrated_classifier`, flagged "no useful discrimination", and excluded from recommendations. Its low FAR (3.9 %) comes with 24 % recall and is noise. A model must beat this honestly-constructed floor to be credited with understanding the photos.
* **Provenance risk applies to the VLM too.** A tuned VLM sees pixels and could learn resolution/compression cues. Mitigations to consider before tuning: set `configs/model.yaml -> image_input.max_side_px` to a common cap (e.g. 1600) so all images share a resolution ceiling (trade-off: small banners), and evaluate on a test subset matched for resolution. The client should say whether production photos will be camera originals or compressed copies.
* Validation is used for hyper-parameter (C) and threshold selection, so validation metrics are optimistic; the test split has not been scored.
* A Track A tuned model returns only a hard label - no probability, no rule trail.
* Unresolved outputs (errors/REVIEW) are counted as REJECT in the `strict` metrics; the `resolved` view excludes them. Both are reported.
* Automatic error-cause tags are mechanical (uncertain/failed rules, measured quality flags, analysis failures). Visual causes ("banner too small", "ambiguous rule", "inconsistent human decision") require human review; `error_analysis.csv` has an empty `manual_cause` column for that.

## Operational caveats
* The API holds a single in-process service; no auth, rate limiting or batch endpoint (deploy behind a gateway; batch via CLI).
* `data/derived/tuning_images` uses ~3 GB; the first full audit takes several minutes (10 GB of decoding).
* Provider behaviour (error codes, terminal job states, tuned endpoint call pattern) is mapped from the SDK surface and docs but untested against the live service.
