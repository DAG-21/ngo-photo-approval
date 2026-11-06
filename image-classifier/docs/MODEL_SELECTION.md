# Model Selection

## Recommendation (as of this build)
**No production model can be recommended yet - the evidence does not exist.** What is measured: a quality-features-only baseline on the validation split
(see `docs/EVALUATION_REPORT.md`) - it has **no discriminating skill** (ROC-AUC ~0.51) once the resolution/file-size provenance proxy is removed, so it sets the floor that any real candidate must beat. What is *not* measured: any VLM (untuned or tuned), because there are no cloud credentials or data-transfer authorization,
and the client's rule set is unconfirmed. Recommended next sequence, cheapest first:

1. Obtain written authorization to send images to Google (API key, or Cloud project/region). The rules are already fixed by the text document (RULE_SPECIFICATION.md); none is mandatory.
2. Run the untuned **`gemini-3.5-flash-lite`** (the model chosen for use) rule analysis on a **100-image stratified validation sample** (`run_baseline.py --mode vlm ... --execute`), check latency, token
   usage and malformed/UNCERTAIN rates, then price the full run. Compare `gemini-3.5-flash` on the same sample (`--model comparison`) for cost vs accuracy.
3. Train the rule-feature probability model on train-split analyses, select the threshold on validation, and compare with the quality baseline.
4. Only if the untuned pipeline misses the false-approval target, prepare Track A tuning (cheap, data exists) and, in parallel, Track B annotations.
5. Final, single evaluation of all retained candidates on the test split with `--final-test`.

## Candidate models (configurable in `configs/model.yaml`)
| Role | ID | Verified? | Notes |
|---|---|---|---|
| Primary (in use) | `gemini-3.5-flash-lite` | yes - "Gemini 3.5 Flash-Lite" is in the supported-tuning list; exact ID confirmed in the live model list of the configured API key; passed a text-only structured-output test with the real 16-rule prompt | cheaper; accuracy on photos not yet measured |
| Comparison | `gemini-3.5-flash` | yes - named in the supported-tuning list, in the official tuning example, and in the live model list | multimodal; tuning on `us`/`eu` multi-region endpoints |
| Probability layer | scikit-learn logistic regression + sigmoid/isotonic calibration | n/a | interpretable, small data; boosting only if justified |
| Quality baseline | OpenCV blur/brightness/contrast + size features -> logistic regression | n/a | free, offline |

## Decision rules for choosing between architectures
Evaluate on the **same held-out split**, thresholds fixed from validation:

1. Discard any candidate whose strict false-approval rate exceeds the client tolerance (default 5 %).
2. Among the rest, prefer higher strict approval recall (more photos auto-approved), then higher auto-processed share, then better calibration (Brier/ECE), then lower latency/cost.
3. Prefer candidates that **retain a rule-level audit trail** and can enforce mandatory rules (hybrid pipeline) over a hard-label-only tuned classifier, unless the latter is clearly better on (1)-(2).
4. Keep fine-tuning only if it improves (1)-(2) by more than the validation noise (a few points with ~50 groups) **and** the cost/complexity is justified; otherwise stay on the untuned model.
5. Treat any result obtained on validation as provisional; confirm once on test.

## What we deliberately did not assume
* That fine-tuning helps. The pipeline is built so an untuned model is a first-class option.
* That a VLM's own confidence is a probability. It is stored, labelled uncalibrated, and unused.
* That image-level labels teach rule-level behaviour. Track B requires verified rule annotations.
