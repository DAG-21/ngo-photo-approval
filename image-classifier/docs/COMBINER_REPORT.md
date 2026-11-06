# Combiner report: learned image score + rule results

_Script: `scripts/build_combiner.py` (local; test split untouched). Numbers: `data/results/combiner/eval.json`. Artifacts: `artifacts/probability_model/{image_score,image_only,image_rules}/`._

## Design
1. **Image score (stage 1).** CLIP embedding of the resolution-normalized photo -> logistic regression on the human labels. Out-of-fold scores (group-aware 5-fold) are used downstream, so a photo's score never used its own label.
2. **Combiner (stage 2).** Logistic regression on `logit(image score)` + the rule indicators, then calibrated (cross-fitted). It learns from the labels how much the image signal and each rule count.
3. The final decision is still: calibrated probability >= threshold, and a rule only hard-rejects if the rules text says it is mandatory.

## Results (grouped 5-fold CV x 3 repeats on the 674 photos that have both signals: 535 approved / 139 not approved)
| Model | ROC-AUC | PR-AUC | Approved recall at 5% false approvals | Brier |
|---|---:|---:|---:|---:|
| Image score only | **0.763** | 0.918 | 24.1% | 0.139 |
| Rules only | 0.512 | 0.803 | 6.9% | 0.172 |
| Image + all 16 rules | 0.700 | 0.891 | 20.5% | 0.151 |
| Image + 3 rule-summary numbers | 0.737 | 0.897 | 14.0% | 0.141 |

* **Adding the rule results makes the model worse**, even in a lean form (-0.062 and -0.026 AUC; the sign is the same in every repeat). The combiner gives the image score a weight of about +0.90 and any single rule at most about 0.1-0.3 (standardized).
* **Why, as far as we can tell:** the rule results used here are from an older prompt (`rules-v3`, before the camera-from-the-front guidance) judged by Gemini 3.5 Flash-Lite, which passes nearly every rule (lighting, focus, dignity ~100%). Lenient, near-constant judgments carry little information. This says nothing about whether the SOP rules matter to the humans; it says these judgments do not predict their decisions.
* **Image-only on all 3,388 train+validation photos (the model to beat):** calibrated, cross-validated ROC-AUC 0.735, ECE 0.036 (passes the calibration criteria: enough data, ECE <= 0.08, AUC >= 0.6). At <= 5% false approvals only ~12% of approved photos pass (threshold 0.96, chosen on cross-validated scores).

## What was saved
* `image_score/` - stage-1 model (the image probability), tied to the normalization setting; loading refuses a changed `tuning_normalize_long_side_px`.
* `image_only/` - calibrated image-only model (feature set `image`). It ignores the rules entirely: do not use it as a decision model while any rule is mandatory.
* `image_rules/` - the combiner, flagged **provisional** and `beats_image_only_in_cv: false`.
The prediction service is not wired to these yet (no deployment path uses the image score today).

## Caveats
* The combiner comparison covers only 674 photos (139 not approved), so differences of a few AUC points are within noise; the consistent sign across repeats and the size of the gap to the lean model are the reasons to believe the direction.
* The stage-1 out-of-fold scores come from a different fold split than stage 2 (standard stacking practice; slightly optimistic for both).
* The C values (stage 2) were picked on the same CV, so every number is mildly optimistic. The recall figures use thresholds taken from the same out-of-fold scores.
* Everything is train+validation only. Nothing has been evaluated on the test split.

## What would change the picture
Better rule judgments (a stricter model or the new guidance, evaluated on all train photos), rule results for many more photos, or a human-tagged reason for each disapproval. Re-run `build_combiner.py --rule-version <new folder>` after any of these; the comparison is automatic.
