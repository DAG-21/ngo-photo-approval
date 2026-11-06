# Pilot 1 (2026-11-02): 10 photographs, two profiles, Gemini 3.8 Flash

**Photographs:** 10 photos the NGO disapproved and the taggers gave a reason for (varied reasons), under opaque ids (`NGO_IMG_000001`...). No label, file name or tag was sent to the
model. They are the 1280 px normalised copies used in all earlier trials, not the originals. **All 10 were disapproved**, so this pilot can say nothing about approved photographs.
The human tags are in `pilot/key_PRIVATE.csv` (not shared with the model). Runs: `runs/pilot_layered_proposed_v1`, `runs/pilot_checklist5_legacy`; report script `scripts/pilot_report.py`.

## What went wrong first (and was fixed)
| Problem | Cause | Fix |
|---|---|---|
| **First attempt: 0 of 10 answers.** HTTP 400 on every call | the response schema carried `minItems`/`maxItems` on the 83-item criteria list; the API rejects that. (Found by sending the schema in pieces; the photo and prompt were fine.) | removed the limits; completeness is enforced in Python instead (a missing criterion becomes CANNOT_TELL). A test guards it. |
| 4 of those failures: "client has been closed" | several threads built a client at once and discarded one while in use | client creation is now behind a lock |
| 1 of 10 answers truncated (invalid JSON) | the output limit (8,192) also counts the model's own thinking (about 2,700 tokens) on top of a ~5,500-token answer | limit raised to 16,384 in `gemini_config.yaml`; that photo then succeeded on retry |
So the very first live run failed completely; the design had never been tested against the real API, and these three defects only showed up there.

## Results (after the fixes)
| | layered_proposed_v1 | checklist5_legacy |
|---|---|---|
| valid answers | 10 of 10 (one needed a retry after truncation) | 10 of 10 |
| criteria asked per photo | 82 | 61 |
| criteria missing from an answer | **0** | **0** |
| "not applicable" conflicts | 0 | 0 |
| gates the model could not establish (CANNOT_TELL) | 1 | 0 |
| prompt / output / thinking tokens per photo | ~7,400 / ~5,800 / ~1,500 | ~6,600 / ~4,500 / ~2,100 |
| time per photo | ~54 s | ~49 s |
| decisions | 8 REJECT, 2 HUMAN_REVIEW (both on a quality score of 65 and 69) | 9 REJECT, 1 APPROVE |
| photos where a gate matching the tagged reason fired | 3 of 10 | 4 of 10 |
Cost: about $0.18 for the layered run and $0.13 for the legacy run at the **placeholder** prices (which were quoted for Flash-Lite; the real price of this model is unknown).
At that price a photo costs roughly $0.015 and takes about a minute, so a 4,000-photo annotation would take many hours even with 5 parallel calls.

## What looks good
* Every answer was valid, complete and in the schema. The model answered all 82 criteria on every photo, used the facts consistently with the gates (for example no banner -> banner gates not applicable), and gave specific evidence.
* Evidence is concrete and mostly checkable: "the banner behind the plastic sheet is blurred, washed out, and illegible" (G018), "a central primary beneficiary is seen directly from behind" (G016), "the volunteer handing over the item is wearing a black jacket" (G013).
* The new gates did fire: G016 (back views) on 3 photos, banner gates on the photo tagged for its banner.

## What this does not show, and the worries
* **No approved photographs were included**, so there is no measure of false rejections. My own audit found 97% of 300 approved photos break at least one client rule. These gates are strict (G001 failed on 6 of 10, the face gates on 2 to 4, G013 on 3), so expect many *approved* photos to be rejected as well. Whether that matches what the NGO wants is a policy question the next pilot should measure.
* Only **3 of 10** photos had a gate fire for the reason the tagger gave; several tags (posed, cluttered background, dignity, focus) have no gate at all by design, and many gates fired that nobody tagged. With 10 photos and tags that are only a tagger's reading, this is a hint, not a result.
* The same photo gave different gate answers under the two prompts in 13 of 100 comparable gate statuses. That mixes the prompt change with ordinary run-to-run variation; it was not separated.
* Latency and thinking tokens make a large annotation expensive and slow; thinking cannot be switched off in the current configuration.

## Next steps I would take
1. A second pilot with about 10 approved and 10 disapproved photos, to measure false rejections and compare the gate rates between classes.
2. Repeat the same photos twice under the same profile to measure run-to-run variation.
3. Decide, with the NGO, whether the new strict gates (G016 to G018) and the existing face and angle gates should reject or send to human review while computer vision does not yet exist.

---

# Pilot 2 (2026-11-02): 10 approved + 10 disapproved, layered profile

**Photographs:** 10 the NGO approved (random from the 41 approved photos of the earlier paired set; I had audited these myself) and 10 disapproved ones with taggers' reasons, none reused from pilot 1.
Opaque ids, no labels or tags sent; same 1280 px copies. Run `runs/pilot2_layered`, script `scripts/pilot2_report.py`, human labels in `pilot/key2_PRIVATE.csv`. 20 of 20 answers valid and complete (0 missing criteria), about 52 s and $0.017 a photo at placeholder prices.

| | approved (10) | disapproved (10) |
|---|---|---|
| REJECT | **8** | 9 |
| HUMAN_REVIEW | 1 | 0 |
| auto-approved (incl. pending publication checks) | 1 | **1** (the costly error) |
| mean failed gates | 1.3 | 5.5 |
| mean quality score | 56.3 | 55.8 |

* **As the strict gates predicted, 8 of the 10 approved photos were rejected outright (9 of 10 not auto-approved).** At the moment the layered profile would send nearly every approved photo back. That is a policy outcome, not a bug: the gates measure things the human reviewers evidently tolerate.
* **The number of failed gates separates the classes** (AUC 0.88 on 10 v 10; 0.79 against all 20 disapproved photos of both pilots). Earlier approaches were at 0.49 to 0.59. With 10 photos per class the range is about +/-0.15 to 0.2, and run-to-run variation has not been measured. The quality score does not separate them at all (56 v 56; AUC 0.53).
* **Which gates carry the signal.** The face gates (G002, G004 to G008) and the back-view gate (G016) fired on **0 of 10 approved** photos but on 7 of 10 disapproved ones in pilot 2, and on 4 of 10 in pilot 1 (which I did not use to pick them). The noisy gates fire on both classes: G001 (camera-angle components) on 5 of 10 approved and 8 of 10 disapproved, G015 (beneficiary gaze) 3 v 4, G009 (banner creases) 1 v 1, G013 1 v 1.
* **Caveat on that finding:** I looked at the gate rates after seeing pilot 2, so "which gates separate" is a hypothesis to confirm on fresh photos, not a result. It is consistent with what you and the taggers said (faces are the main issue) and with what the model's evidence shows, but it comes from 20 photos, mostly one programme (Arham Aahar).
* **The approved photos failed on:** G001 (5), G015 (3), G017 (1), G009 (1), G012 (1), G013 (1), G014 (1). In my own audit most of these photos break R004, R005, R006 or R011 (looking at the camera, posed, activity not shown, clutter); the model's gates picked up only some of that (for example G015 gaze and G001 'activity visible').
* **One disapproved photo was auto-approved** (tag R006, "activity not clearly shown"; no gate fired). That is the costly error, once in 10.

## What I would do next
1. Re-test the hypothesis on fresh photos (about 20 approved and 20 disapproved, 40 calls, about $0.70 and 35 minutes): do the face and back-view gates keep firing on disapproved photos and stay silent on approved ones?
2. Measure run-to-run variation by running the same photos twice (otherwise a gate that looks discriminating may be noise).
3. If it holds, change the policy question for the NGO: let only the gates that separate the classes reject, and send the rest (G001, G015, G009, G013, G017) to human review rather than automatic rejection, until computer vision exists.

---

# Pilot 3 (2026-11-02): 50 random photos, 25 approved + 25 disapproved, layered profile

Random draw (seed 50) from the unique train and validation photos, none used in pilots 1 or 2, natural programme mix (Arham Aahar 27 of 50). Opaque ids, no labels sent. Run `runs/pilot3_layered`,
scripts `make_pilot3_list.py`, `pilot3_report.py`, Excel `reports/pilot3_50_photos.xlsx`. 50 of 50 answers valid (2 criteria missing or N/A-conflicting across all photos, both escalated by the engine), ~55 s and $0.017 a photo at placeholder prices.

| | approved (25) | disapproved (25) |
|---|---|---|
| REJECT | 20 | 22 |
| HUMAN_REVIEW | 1 | 1 |
| auto-approved | 4 | **2** (the costly error) |
| mean failed gates | 3.4 | 4.0 |

* **False rejection: 21 of 25 approved photos (84%) were not auto-approved. Agreement with the humans 27 of 50.** The strict gates reject almost everything, approved or not.
* **The pilot-2 finding did not replicate.** Failed-gate count separated the classes at AUC 0.88 in pilot 2 but only **0.57 (95% range 0.41 to 0.72, about chance)** here; the quality score 0.56. The face and back-view gates, which fired on 0 of 10 approved photos in pilot 2, fired on **11 of 25 approved** (and 14 of 25 disapproved) in pilot 3. Each face gate fires on 16 to 44% of approved photos.
* So pilot 2's apparent separation was largely small-sample luck (and probably its approved photos being a milder set: 8 of 10 were from the earlier programme-matched group and averaged 1.3 failed gates, against 3.4 in pilot 3). I flagged at the time that it was a hypothesis noticed after the fact; this is why that caution mattered.
* Pooling all three pilots (35 approved, 45 disapproved; pilot 1 contains only disapproved photos and was not a random draw) gives AUC 0.64 (0.52 to 0.75): a weak signal at most, not a usable classifier.
* **Tagged reasons:** of the 6 tagged disapproved photos, a gate matching the tagged reason fired on only some (see the script output); lighting-tagged photos (R009) were rejected for other reasons (G001, face gates, gaze), not for lighting.
* **Costly errors:** 2 of 25 disapproved photos (both Arham Aahar) were auto-approved.

## Conclusion so far
With the strict gates as set, the engine behaves as a **very strict filter that agrees with the humans mostly because most disapproved photos also fail some gate**, not because the gates reproduce the NGO's decisions: approved and disapproved photos fail a similar number of gates (3.4 v 4.0).
This matches the earlier trials (rule-based answers near chance against the labels) and my own audit (97% of approved photos break at least one client rule). The consistent message is that the written rules, applied this strictly, do not explain what the NGO approves.
The engine's strengths are different: complete, evidenced, auditable answers and a clean separation of what the model sees from what policy decides. Options worth discussing with the NGO: turning the noisiest gates into human-review triggers; learning gate weights from the labels (a model on the gate answers, evaluated on held-out groups, as with the embeddings); or using the evidence as an explanation layer on top of the embeddings model, which separated the classes much better (AUC about 0.77).
