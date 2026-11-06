# Rule Specification

**Single source of truth:** `rules/photo-approval-requirement-rules.txt` ("Basic Photography SOP & Quality Control Guidelines").
Earlier example rules from the project brief have no standing here and have been removed.

**Machine-readable form:** `configs/rules.yaml` (currently `1.2-sop`). It only adds structure to the document, plus one clearly labelled team clarification (see "Additional guidance" below):

* `original_text` is copied verbatim and `source_ref` links each rule to its item in the document;
* **a rule is mandatory only if the document itself says so** (marker words `mandatory`, `compulsory`, `must`, `non-negotiable`, `without fail` - configurable in `configs/project.yaml -> rules.mandatory_markers`; "not mandatory"/"optional" cancel them);
* the document says so for **no** rule, so **all 16 rules are `scored`**. Each is evaluated against the exact wording on every photo and its PASS/FAIL/UNCERTAIN result and evidence are always reported, but no single rule rejects a photo on its own;
* `description`, `visual_evidence` and `applies_when` are plain-language restatements for the model. They must not add requirements the text does not contain.

## The 16 rules

| ID | `source_ref` in the document | Applies when |
|---|---|---|
| R001 | Banner Visibility & Presentation | a banner is present |
| R002 | Volunteer Dress Code | volunteers are visible |
| R003 | Beneficiary Visibility | a beneficiary is visible and showing their face is appropriate |
| R004 | Natural Expressions | people are visible |
| R005 | Natural & Candid Shots | always |
| R006 | Capture the Actual Seva | always |
| R007 | Proper Camera Angle | always |
| R008 | Volunteer Positioning | volunteers are visible |
| R009 | Proper Lighting | always |
| R010 | Focus & Image Clarity | always |
| R011 | Clean Background | always |
| R012 | Final QC Test Q4 (dignity) | always |
| R013 | Final QC Test Q5 (humanity and compassion) | always |
| R014 | Final QC Test Q1 (seva clearly visible?) | always |
| R015 | Final QC Test Q2 (volunteer and beneficiary captured appropriately?) | always |
| R016 | Final QC Test Q3 (natural, well-framed, properly lit?) | always |

Every enabled rule is requested from the model for every photo, with the client's wording quoted in the prompt as authoritative ("apply exactly").
For the Final QC Test questions, PASS means the answer is "yes".

## Conditional rules and N/A
R001-R004 and R008 are conditional in the document ("If a banner is present...", "wherever appropriate", ...). When the condition clearly does not hold the model returns `PASS` with evidence
starting `N/A:`; if it cannot tell, `UNCERTAIN`. No logic reads the evidence text. A photo with no volunteers therefore passes R002/R008 as N/A, while R006/R014 ask whether the seva is shown at all.

## How decisions use the rules (consequence of "nothing is mandatory")
* No rule is a hard gate, so the rule engine never rejects because a rule failed or was uncertain. FAIL/UNCERTAIN results are recorded and become **features** of the probability model
  (per-rule pass/fail/uncertain/missing indicators), which is trained on the NGO's historical decisions. The final decision is that probability against the approval threshold.
* An **incomplete or malformed analysis still rejects** (missing, duplicated or unknown rule results), so a photo can never be approved on partial information.
* If the client later writes "mandatory" (or a marker word) in the document, that rule becomes a hard gate automatically after `sync_rules.py --write`: a FAIL or UNCERTAIN on it rejects regardless of probability.

## Additional guidance (not in the document)
`additional_guidance` is a separate field for instructions from the NGO team that the client's text does not contain. It is sent to the model on its own line, labelled
"additional guidance from the NGO team (not part of the client wording)", next to the verbatim wording. `original_text` is never edited, and the document sync check ignores this field.

Current guidance (added 2026-10-30 on the team's instruction, after a manual review of `disapproved_3` showed the model passing a photo shot from behind the people):
> The photo must be taken from the front of the people, not from behind them. If the main people in the frame are seen mainly from behind (their backs towards the camera), this rule is not met (FAIL).

It applies to **R003** (beneficiary visibility), **R007** (camera angle), **R008** (volunteer positioning), **R015** (volunteer and beneficiary captured appropriately) and **R016** (natural, well-framed, lit).
Remove or move it by editing `additional_guidance` in `configs/rules.yaml`. If the client wants it to be an official rule, add the sentence to the rules document and run `sync_rules.py --write`; the guidance field can then be cleared.
This guidance goes beyond the client's wording, so it is not part of the "single source of truth" and should be confirmed with the client. Changing it changes the prompt (`rules-v4`) and the rule-set version, and
rule-check results are stored per prompt/rule-set version (`data/results/rule_check/<model>/<prompt>__<rule set>/`), so older results (v3, `1.1-sop`) are never mixed with new ones.

## When the document changes
The loader **refuses to start** (CLI, batch, API) if `rules.yaml` no longer matches the document: new items, changed wording, removed items, a rule linked to nothing, or a mandatory/scored
designation that disagrees with the document's own words.

```bash
python scripts/sync_rules.py             # lists every difference; exit code 1 when out of sync
python scripts/sync_rules.py --write     # new items -> new rules (next free id, scored unless the text says mandatory)
                                         # changed wording -> original_text/description updated, flagged for re-review
                                         # removed items -> rule disabled (ids and history are never deleted)
                                         # mandatory/scored re-derived from the document's own words; version suffix +YYYYMMDD
```
After `--write`, read the `notes` of added/changed rules and fill in `visual_evidence` / `applies_when` if useful. A changed rule set also changes the probability model's feature set, so
**retrain it** (`train_probability_model.py` refuses a stale model; the service refuses to load a model whose features no longer match). Saved VLM analyses made with older wording should be re-run.

## Not defined by the document (interpretation limits)
Who counts as a "volunteer" vs a "beneficiary" (the model may use activity context only, never personal attributes); what "wherever appropriate" and "unless a specific photograph requires it"
mean; numeric thresholds for blur or brightness (OpenCV measures feed the probability model, but nothing is enforced); program-specific differences. These stay UNCERTAIN-prone and are listed in each rule's `limitations`.

## Rule schema (`configs/rules.yaml`)
| Field | Meaning |
|---|---|
| `rule_id`, `name` | stable `R###` id / snake_case name |
| `original_text`, `source_ref` | client wording (verbatim) and its identifier in the document |
| `description`, `visual_evidence`, `applies_when` | restatements for the model; `applies_when` empty = always applies |
| `additional_guidance` | team instruction NOT in the document, sent separately and labelled as such (empty = none) |
| `requirement_type` | `mandatory` only if the document says so, else `scored`; enforced against the document at load time |
| `uncertain_policy` | `reject`/`review` for mandatory rules, `score_only` for scored ones (`score_only` is invalid for mandatory - schema-enforced) |
| `check_method`, `enabled`, `applies_to_programs` | `vlm` implemented; `python`/`ocr`/`detector` are extension points |
| `confirmed`, `mandatory_status_confirmed`, `needs_manual_annotation`, `limitations`, `notes` | governance and caveats |

## Evaluation methodology
* Rule-level accuracy is measured only against human-verified labels (`scripts/annotate_rules.py template` writes one row per image per rule; VLM suggestions sit in a separate column and are never ground truth).
* Image-level human decisions (approved / not approved) are not rule labels.
* `rule_set_version` is stored with every result and response; trained probability models store the rule-set fingerprint.
* `rule_set_confirmed` stays `false` until the client reviews the restatements (`description`, `visual_evidence`, `applies_when`); responses carry a warning meanwhile.
