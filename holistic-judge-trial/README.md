# Holistic-judge trial

One Gemini prompt (persona + the client's rules + worked examples) that looks at a photo and returns an approval
score, a decision and reasons. It replaces nothing in `../image-classifier`: this folder only **reads** that project
(splits, normalized photos, rules, CLIP embeddings) and writes only inside itself.

## Status: built and tested offline. No Gemini call has ever been made from here.

* Every script defaults to a **dry run**. A real call needs `--execute` on the command line **and** the main
  project's own authorization (`NGO_ALLOW_CLOUD_INFERENCE=true` plus credentials). Without both, no client is created.
* 48 offline tests pass; they use a fake backend and a guard that makes any attempt to create a real Gemini client fail.
* No results exist yet, so there are **no performance numbers** for this approach.

Run everything with the main project's virtualenv:

```bash
cd holistic-judge-trial
../image-classifier/.venv/Scripts/python.exe scripts/<script>.py ...
```

## What the prompt contains

Built from files by `src/judge_trial/prompt.py`; `python scripts/show_prompt.py --variant D` prints it exactly.

| Part | Source |
|---|---|
| Persona: social media manager of an NGO, judge the photo only, never identify anyone | `prompts/v1/persona.md` |
| The 16 rules, **word for word** | main project `configs/rules.yaml` (`original_text`; tied to the client's document) |
| The same rules grouped and restated | rules' own `description` / `applies_when`, grouped by `prompts/v1/variants.yaml`; adds no requirement |
| Team guidance (front, not back, angle) | the rules' `additional_guidance`, stated once and labelled "not in the client's document" |
| How to decide (nothing mandatory, N/A rules do not count, say cannot-judge) | `prompts/v1/decision_guide.md` |
| Output: observations, reasons per rule, score 0-100, decision, cannot_judge | `prompts/v1/output_instructions.md`, `schema.py` |
| 8 worked examples with the reviewers' decision | `examples/examples.json` (train split only) |

Prompt versions: **v1** the original; **v2** adds an overall plain-language `reason` to the answer; **v3** (= v2) replaces the
front-angle team guidance with `prompts/v3/team_guidance.md`, which adds an exception: a back view of volunteers is fine when
they wear the dress code, the NGO's logo and name show on their backs, and the beneficiaries are clearly visible. The exception
is the team's wording, shown to the model as "NOT in the client's document"; the client's own rule text and the main project's
`configs/rules.yaml` are unchanged. Select a version with `--version v3`.

**checklist1** (variant K) is a different style: the model answers about 40 itemised YES/NO/CANNOT_TELL/NOT_APPLICABLE questions (a breakdown of each rule,
plus a few scene facts, the team's front-angle and back-view questions, and two checks the taggers' notes suggested) and gives
**no score and no decision**. A rule's partial result (3 of 4 banner checks met = 0.75), its PASS/FAIL label and the
overall prediction are computed afterwards in code (`checklist.py`, `checklist_eval.py`), so they can be re-weighted without calling the
model again. The split into questions is our decomposition, not client wording, and needs the NGO's review.
`python scripts/evaluate_checklist.py --pool dev --tag <tag>` reports how often each check says NO per class (also within each programme),
a cross-validated AUC of a small model on the checks, and how the checks line up with the human tags.

**Blind tagging** (`scripts/blind_sheet.py`): an Excel sheet of the 82 tagged-and-paired photos (41 disapproved, 41 approved, shuffled, labels hidden, neutral file
names) for taggers to say whether they would approve each photo and what problem they see. It tests whether the first round's reasons
mean anything (taggers saw only disapproved photos then). `build` makes the sheet and a private `key.csv` with the labels; `import` and
`analyse` read the returned copy. Never share `key.csv` with taggers.

Four arms, to see what each addition buys: **A** persona + verbatim rules, **B** + restated rules, **C** + team
guidance, **D** + worked examples. Every result row carries the prompt hash, so a result can never be attributed to
the wrong prompt, and a run folder refuses to mix two prompts. A prompt version is frozen once scored: copy `v1` to `v2`.

## "Tell it once": context caching

For variant D the fixed part (instructions + 8 example photos, roughly 11k tokens, an estimate) is sent once into a
Gemini context cache; each photo is then judged against the cache by name and only the photo travels with the
request. Gemini is stateless, so the model still reads the cached content on each call; caching makes it cheaper, not
free, and the cache expires (default 1 h) and is deleted at the end of a run. If the cache cannot be created
(`cache.mode: auto`), the same content is sent inline instead; `mode: on` stops with an error rather than hiding that.

## Photo pools (`pools/*.csv`, built offline by `scripts/build_pools.py`)

| Pool | Photos | Use |
|---|---:|---|
| dev | 300 (150 approved / 150 disapproved) from **train** | build and compare prompt versions |
| confirm | 300 (150 / 150) from **train**, disjoint from dev | check a frozen prompt without overfitting to dev |
| validation | 507 (465 / **42**) from **validation** | one final look at a frozen prompt |

The **test split is never used.** Each unique photo content counts once. A photo is excluded from every pool if it is
an example, an exact or near duplicate of one, in the same *small* submission group, or looks near-identical to one
(CLIP cosine similarity >= 0.88, about 4% of photos). Large "groups" are deliberately not used for exclusion: the
project's `group_id` puts 512 of the 695 disapproved training photos into one component, and excluding it would have
removed an entire program (arham_aahar) from the pools.

## How a result is judged (`scripts/evaluate_trial.py`)

Compared with the CLIP + DINOv2 embeddings model on the **same photos** (its out-of-fold scores): ROC-AUC, approved
photos accepted at <= 5% false approvals, **false positives** (disapproved photos approved) and **false negatives**
(approved photos not approved), the judge's own APPROVED/DISAPPROVED decision, calibration (grouped cross-validation),
and 95% ranges from a bootstrap that resamples whole groups. The 5% operating point never lets ties push the
false-approval rate over the limit. Failed calls are excluded and reported, never counted as wrong answers.
Bar for "worth keeping", set before seeing any result: AUC of about 0.75 or better on dev **and** confirm, clearly
above the rules-only result (about 0.51); near 0.6 would point to the labels, not the prompt.

## Known limits (read before trusting any number)

* **Disapproved examples now carry a human reason; approved ones do not.** The four disapproved examples were picked by hand
  from the photos the NGO team tagged (R006 activity not clear, R005 posed, R009 poor lighting, R003 faces hidden) and carry
  that tag, worded as "tagged afterwards by the NGO team; the original reviewers' own reason was not recorded". The tags
  come from 198 photos tagged by unnamed people (two blocks that look like two taggers); treat them as a fair guide, not
  proof. Results from before this change (the 6-photo pilots and the 50-photo run) used the earlier label-only examples and
  are not comparable; the old examples and pools are kept in `archive/before_tagged_examples/`.
* **Validation has only 42 disapproved photos**: any rate there is imprecise, and the report says so. The dev and
  confirm pools are balanced, so their calibration numbers (ECE, Brier) do not reflect the real class mix; AUC and
  recall at a fixed false-approval rate do not depend on it.
* **Disapproved photos are concentrated in one program** (about three quarters of those in dev are arham_aahar).
  The judge is not told the program, but visual style can still correlate with it.
* **Assumptions to replace after the first real call:** 1,120 tokens per photo, a 2,048-token minimum for caching, and
  the empty price fields in `config/trial.yaml`. Dry runs print token counts; cost stays "unknown" until prices are filled in.
* Example photos are chosen by clustering (k-means medoids on CLIP embeddings), a reasonable but untested choice.

## Commands

```bash
python scripts/show_prompt.py --variant D                        # exactly what is sent (offline)
python scripts/run_trial.py --variant D --pool dev --limit 20    # dry run: plan, tokens, number of calls
python scripts/run_trial.py --variant D --pool dev --limit 20 --execute   # REAL calls: only when instructed
python scripts/evaluate_trial.py report   --variant D --pool dev # offline; reads results.jsonl
python scripts/evaluate_trial.py ablation --pool dev             # A / B / C / D side by side
python scripts/evaluate_trial.py noise --variant D --pool dev --tags a b   # repeat-run stability
```

## Suggested order once real calls are approved

1. 20-photo pilot of variant D (21 calls): checks the plumbing, whether the cache is created, and measured tokens.
2. Fill in prices from the official price list; re-run the dry run for a real cost figure.
3. Variants A to D on dev (4 x 300 calls, plus one cache creation for D) and `ablation`.
4. Repeat run on a small subset (`--tag`) to see how noisy the scores are.
5. Freeze the best prompt, run it on confirm, then once on validation.
