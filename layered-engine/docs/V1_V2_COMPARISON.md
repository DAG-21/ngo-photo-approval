# V1 vs V2 on the same 50 photographs (pilot 3), 2026-11-02

**Read this first.** This is a *development* comparison. The V2 changes were chosen after looking at the gate statistics of these same 50 photographs, so V2 is not independently validated on them. Each class has 25 photographs: one photograph moves a rate by 4 points. Nothing here is NGO policy: every V2 change is `PROPOSED_PENDING_NGO_APPROVAL`.

* V1 = profile `layered_proposed_v1` (rulebook 1.1.0, prompt hash `11427c79d96c569a`), run once. Frozen in `evaluators/v1_frozen/`.
* V2 = profile `layered_proposed_v2` (rulebook 2.0.0-proposed, prompt `photo_evaluation_prompt_v2`), run once on the same 50 photographs, same model (`gemini-3.8-flash`), same settings. 56 calls (6 retries: 5 read time-outs and 1 response with a duplicated answer), all 50 answers valid, about $0.85 at the placeholder prices.
* Workbook: `reports/v1_vs_v2_comparison.xlsx` (Summary, Photos with thumbnails, Ablation stages, Gate level, Upper bound, V2 changes, Notes). JSON: `reports/v1_vs_v2_comparison.json`. Reproduce offline: `python scripts/v1_v2_comparison.py --excel reports/v1_vs_v2_comparison.xlsx`.
* Positive = approved. "Engine approved" only when the engine auto-approves. False approval = disapproved photo auto-approved (the costly error). False rejection = approved photo not auto-approved (rejected, sent to review or reselect).

## 1. Result as run

| | V1 | V2 |
|---|---:|---:|
| Approved photos auto-approved (TP) | 4 | 8 |
| Approved photos not auto-approved (FN) | 21 | 17 |
|   explicit REJECT / HUMAN_REVIEW / RESELECT | 20 / 1 / 0 | 7 / 8 / 2 |
| Disapproved photos stopped (TN) | 23 | 19 |
| **Disapproved photos auto-approved (FP)** | **2** | **6** |
| False-rejection rate (95% interval) | 84% (65-94) | 68% (48-83) |
| False-approval rate | 8% (2-25) | 24% (11-43) |
| Approval precision | 4/6 | 8/14 |
| Rejection precision (of explicit REJECTs) | 22/42 = 52% | 11/18 = 61% |
| Rejection recall (disapproved explicitly rejected) | 22/25 = 88% | 11/25 = 44% |
| Agreement with the historical decision | 27/50 | 27/50 |
| Human reviews | 2 | 15 |
| Mean failed gates, approved / disapproved | 3.4 / 4.0 | 0.8 / 1.6 |
| Separation by failed-gate count (AUC, 95% range) | 0.57 (0.41-0.72) | 0.59 (0.46-0.72) |
| Separation by quality score (AUC) | 0.56 | 0.58 |

What this says, without dressing it up:

* V2 lets more approved photographs through (4 to 8) and rejects far fewer of them outright (20 to 7). The cost is **three times as many false approvals (2 to 6)** and a collapse of rejection recall from 88% to 44%. Agreement with the historical decision does not move (27 of 50 in both).
* The intervals overlap almost completely. With 25 photographs per class, 4 to 8 approved photographs and 2 to 6 false approvals are not distinguishable from chance variation.
* Separation of the classes did not improve (AUC 0.57 to 0.59, ranges 0.41-0.72 and 0.46-0.72). V2 is more permissive for **both** classes, not more discriminating.
* Most photographs that stopped being rejected did not become approvals: they moved to **human review (15)** and **reselect (2)**, almost always because of the quality score (55 to 70, thresholds still uncalibrated). The quality layer has become the main decider, and it separates the classes no better than the gates did (AUC 0.58).

## 2. Which change did what

Two kinds of change must be kept apart. **Policy-only** changes alter what Python does with the answers (hard or soft, applicability, aggregation). They can be replayed exactly on stored answers, with no model call and no run-to-run noise. **Wording** changes alter what the model is asked (V2-02 primary/featured scope, V2-06 G001 wording, plus the new guidance); their effect can only be seen in a new run, which also contains ordinary run-to-run variation.

Replay (replaying V1's stored answers reproduced all 50 stored V1 decisions exactly, and V2's all 50 of its own):

| Stage (cumulative, the specification's order) | V1 answers: TP / FP | V2 answers: TP / FP |
|---|---:|---:|
| E1 original gates | 4 / 2 | 5 / 6 |
| E2 + G015 soft | 4 / 2 | 6 / 6 (rescues #11) |
| E3 + primary/featured scope | 4 / 2 (no effect on fixed answers) | 6 / 6 |
| E4 + conditional G012 | 4 / 2 | 6 / 6 |
| E5 + G001 wording, G016 restructure | 4 / 2 | 6 / 6 |
| E6 + G009 prominent-only, G017/G018 soft = full V2 | 6 / 3 (rescues #7, #11; new false approval #47) | 8 / 6 (rescues #7, #16) |

Leave-one-out on V2's own answers (one set of answers, so no noise): only **V2-01 (gaze soft)** rescues one approved photograph (#11) and only **V2-04 (banner visibility soft)** rescues two (#7, #16). V2-02, -03, -05, -06 and -07 change no verdict on these answers. **None of the policy-only changes creates a false approval on V2's answers**; on V1's answers, V2-04 creates one (#47, a disapproved photograph rejected only for banner visibility).

So of the 4 approved photographs that V2 rescued (#7, #11, #19, #25) and the 4 new false approvals (#29, #39, #47, #49):

* Policy-only: #7 and #11 rescued (banner, gaze), #47 new false approval (banner).
* Wording-driven (the model answered the face and angle questions differently): #19 and #25 rescued; #29, #49 new false approvals, and #39 partly (its face gates also passed in V2, and its banner gate became soft). The V2 evidence texts show the mechanism working as designed: in #29 a child hidden behind a parcel and in #49 a woman seen in profile are now treated as secondary or acceptable, so the face and angle gates pass. The same relaxation that frees approved photographs frees disapproved ones.
* Whether that is run-to-run noise or the new wording cannot be separated from one run each. On questions whose wording did not change, the two runs gave identical answers 92% of the time (3,174 of 3,450); the least stable were the angle questions (`R007_ANGLE` 36 of 50). That is not negligible: at about 8% of answers differing, a few photographs flipping is expected even with no change at all.

## 3. Gate-level breakdown

Denominator = photographs on which the gate applies. "Rej A/D" = photographs rejected with this gate among their failures (approved / disapproved). "Alone" = rejected for that gate only: the photographs a single relaxation can rescue.

| Gate | V1 failed, approved | V1 failed, disapproved | V2 failed, approved | V2 failed, disapproved | Rescued approved by the V2 change | Introduced false approvals | Reading |
|---|---:|---:|---:|---:|---|---|---|
| G001 angle | 13/25 | 14/25 | 6/25 | 9/25 | wording (V2-06): #19, #25 with face gates | #29, #49 with face gates | falls on both classes; 2 approved / 1 disapproved rejected for G001 alone in both |
| G002 faces seen | 9/21 | 12/21 | 3/21 | 6/21 | wording (V2-02) | with G001/G004 | same pattern |
| G004 face visible | 11/21 | 14/21 | 3/21 | 6/21 | wording (V2-02) | #29, #49 | halves for both classes |
| G005 obstruction | 11/21 | 12/21 | 2/21 | 5/21 | wording (V2-02) | #29, #49 | same |
| G006 face sharp | 8/21 | 8/21 | 2/21 | 3/21 | wording (V2-02) | | still no separation |
| G007 face size | 5/21 | 7/21 | 1/21 | 3/21 | wording | | |
| G008 face lit | 4/21 | 5/21 | 1/21 | 2/21 | wording | | |
| G009 banner folds | 1/12 | 1/12 | 0/11 | 0/10 | none | none | too rare to judge |
| G012 receiving | 2/14 | 5/17 | 2/15 | 2/19 | none | none | fires more on disapproved in V1 (5 vs 2), fewer in V2 |
| G013 giver in white | 1/15 | 3/18 | 0/14 | 2/18 | none | none | never rejects an approved photo in V2 |
| G015 gaze | 9/21 | 5/21 | 7/21 (soft) | 4/21 (soft) | #11 (V2-01) | none | the model still says NO as often; it just cannot reject. V2 tags gaze DIRECT on 10 approved and 6 disapproved |
| G016 back view | 3/25 | 7/23 | 0/21 | 1/21 | none | none | fell because it is now volunteer-only and because beneficiaries seen from behind are no longer counted: it used to fire mostly on disapproved photographs (7 vs 3), so this removes a signal that pointed the right way |
| G017 banner visible | 6/12 | 5/12 | 3/12 (soft) | 3/12 (soft) | #7, #16 (V2-04) | #47 (on V1's answers) | 2 approved and 1 disapproved rejected for this gate alone in V1 |
| G018 banner distinct | 0/12 | 0/12 | 0/12 | 0/12 | none | none | never fires |
| G010 / G011 coverage, zoomed out | soft in both | | soft in both | | none | none | see section 6 |

The two things the table shows most clearly:

1. **The face and angle gates fire on disapproved photographs more often than on approved ones in both runs** (V1: G004 14 vs 11, G002 12 vs 9, G001 14 vs 13; V2: G004 6 vs 3, G002 6 vs 3, G001 9 vs 6, G005 5 vs 2). The difference is small and inside the noise, but its direction is the one a useful gate would have. Relaxing these gates is what produced V2's extra false approvals.
2. **G015 gaze and G017 banner visibility fire about as often (or more) on approved photographs, so softening them rescues approved photographs at almost no cost** (G015: no false approval; G017: one). But they rescue only 3 photographs in total (#7, #11, #16), because the other approved photographs fail several other gates as well (mean 3.4 failed gates in V1).

Upper bound (gates deleted outright, a diagnostic, not a proposal): on V2's answers, removing the face gates and G001 together lets 10 of 25 approved photographs through at the price of 7 of 25 false approvals; removing every gate gives 10 and 8. Even with every gate deleted, only 10 of 25 approved photographs pass, so after the gates it is the quality score (uncalibrated) that keeps the rest from passing, and each gate relaxed admits disapproved photographs at a rate close to the approved ones it frees.

## 4. The two original false approvals (#27 and #35), and the new ones

Both were auto-approved by V1 and still are by V2. **No historical reason is recorded for either, and none is invented here.** What I can see in the photographs (my reading, not a human tag):

* **#27** (Arham Aahar): a volunteer in a white polo shirt hands a plate to a woman; five or six women queue behind. No banner. The volunteer shows no visible ID card. A large tray of plates fills the foreground, and strangers' arms are cropped at the right edge. Candidate causes a human might have used: no branding, no ID card (a requirement that is in the NGO's wider practice but is not among the 16 written rules), foreground clutter, awkward crop. Unconfirmed.
* **#35** (Arham Aahar): volunteers in white serve a child from behind a table in front of a banner. The child is in profile looking down at the plate (face partly turned), a volunteer at the right edge is cropped through the face, and the top of the banner is slightly creased. Candidate causes: the child's face, the cropped volunteer, the banner. The engine passed G004 on the child's profile. Unconfirmed.
* New in V2: #29 (human tag R009, lighting), #39, #47, #49 (Arham Aahar and others, untagged). #29 and #49 pass because V2 judged secondary faces leniently (section 2); #47 because banner visibility became soft.

A human reviewer should state the actual reasons for #27, #29, #35, #39, #47 and #49. Until then it is not known whether the engine missed a rule, the humans applied an unwritten one, or the historical labels are inconsistent.

## 5. Conditional applicability and uncertainty

* Hard gates marked NOT_APPLICABLE: V1 183, V2 134 (G015, G017 and G018 are no longer hard in V2). G012 is not applied in 19 (V1) and 16 (V2) of the photographs where no handover is expected; V1 had already narrowed G012 to distribution activities, so the specification's concern that G012 is applied too broadly was largely already addressed in V1 (it fired on 7 of 31 applicable photographs).
* CANNOT_TELL on a hard gate: 3 (V1), 2 (V2). Criteria missing or in conflict: 2 in each run. One V2 response repeated a criterion and was rejected by validation, then answered correctly on retry.
* Activity type differed between the two runs on 4 of 50 photographs.
* The workbook inconsistency about G010/G011 is a labelling error. Both gates are soft in V1 and V2 (`resolve("layered_proposed_v1")` has 16 hard gates without them); "active in: both profiles" meant the gate exists in both, not that it is hard in both. They are hard only in `checklist5_legacy`. The new workbook says hard or soft per profile.

## 6. What is and is not established

Established (exact, reproducible, no noise): the offline replays; which gates reject which photographs; that V1's decision for every photograph can be reproduced from its stored answers; that none of the policy-only V2 changes adds a false approval to V2's own answers; that V1 is unchanged (a test compares its prompt and schema byte for byte).

Not established: that V2 would do better or worse on photographs it has not seen; the size of run-to-run variation on the *decision* (only 92% answer agreement on unchanged questions is measured); whether the 4 wording-driven rescues and 3 wording-driven false approvals are the new wording or noise; the real price; whether the historical labels are right.

Quality layer: thresholds 55 and 70 are uncalibrated, and in V2 they decide the fate of 17 photographs. They should not be treated as a rejection rule.

## 7. Policy conflicts that need the NGO

Five of the seven V2 changes change or re-propose something the NGO decided or withdrew on 2026-11-02 (`RULEBOOK_CHANGELOG.md`): V2-01 (every direct beneficiary gaze disapproved), V2-02 (face scope strict until computer vision), V2-03 (G009 withdrawn narrowing), V2-04 (banner clearly visible when present), V2-05 (back views). The logo exception for back views was **not** reinstated. V2-06 and V2-07 are wording and applicability only. If the NGO does not re-approve a change, switch it off with `without=[...]`.

## 8. Recommendation

* Do **not** adopt V2 as a replacement. It raises false approvals from 2 to 6 for a gain in approved photographs that sits inside the noise, and agreement with the historical decision is unchanged.
* The only two V2 changes that rescued approved photographs without adding a false approval on V2's own answers are V2-01 (gaze soft) and V2-04 (banner visibility soft). They rescued 3 photographs between them. They are also the two that reverse the NGO's explicit decisions, so the NGO should decide them with this evidence in front of them.
* The face-scope wording (V2-02) is the large lever (it roughly halves face-gate failures) and acts equally on both classes; it should be decided as a policy choice, not as an accuracy improvement.
* Before any fresh validation on a larger sample: measure run-to-run noise (run V1 again on the same 50 photographs, about $0.85) so differences like these can be judged; obtain the real reasons for #27, #29, #35, #39, #47, #49; calibrate the quality thresholds. Fresh validation on 200-300 unseen photographs is worthwhile only after the NGO has fixed which V2 changes are policy, because validating a profile the NGO may reject measures nothing useful. Selection of the final profile must use a development partition and leave a final untouched test partition, not these 50 photographs.
