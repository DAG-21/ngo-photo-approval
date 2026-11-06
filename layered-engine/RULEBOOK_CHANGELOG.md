# Rulebook changelog

Every change to a rule records: old wording, new wording, reason, source, approval status, effective rulebook version.
"Old wording" for 1.0.0 refers to checklist5 (`holistic-judge-trial/prompts/checklist5`).

## 2.0.0-proposed (2026-11-02): evaluator V2, profile `layered_proposed_v2`
**Nothing here is NGO policy.** Every change below is `PROPOSED_PENDING_NGO_APPROVAL`. `layered_proposed_v1` (rulebook 1.1.0, prompt hash `11427c79d96c569a`) and `checklist5_legacy` are unchanged and frozen (`evaluators/v1_frozen/`, with a test that compares the V1 prompt and response schema byte for byte).
V2 inherits V1 and records each difference with a `change_id`, so any change can be switched off for an ablation: `Rulebook.resolve("layered_proposed_v2", without=["V2-03"])`.
Motivation: pilot 3 (50 photographs, 25 approved + 25 disapproved) rejected 20 of the 25 approved photographs. The evidence is a diagnostic sample; it motivated these changes, it does not prove them.

| Id | Change | Previous effect | New effect | Relation to the NGO's 2026-11-02 decisions |
|---|---|---|---|---|
| V2-01 | G015 gaze becomes a soft editorial assessment plus tag `f_gaze_tag` (DIRECT_CAMERA_GAZE / NATURAL_CANDID_GAZE / INTENTIONAL_PORTRAIT / UNCERTAIN_GAZE) | hard gate | no rejection; naturalness score | **reverses decision 4** (every beneficiary direct gaze is disapproved) |
| V2-02 | G002, G004-G008 and G001 component B judge PRIMARY or FEATURED beneficiaries, not every visible one (`beneficiary_visibility.subject_scope`, legacy value kept) | every visible beneficiary | primary / featured only | **reverses decision 1** (strict scope until computer vision) |
| V2-03 | G009 folds apply only when the banner is prominent | any banner | prominent banner | **re-proposes** what decision 2 withdrew |
| V2-04 | G017 / G018 banner visibility and distinctness become soft (brand presentation capped at 2 once) | hard gates | soft | **conflicts with decision 2** (clearly visible when present) |
| V2-05 | G016 = volunteer back view only (R008). T_FRONT and R015_BEN_FRONT become graded diagnostics | R008 + T1 + beneficiary front | R008 only | narrows decision 5; **the logo exception stays withdrawn** (not reinstated) |
| V2-06 | G001 component wording: D "nothing hidden" is limited to the activity itself; faces are left to the face gates | four overlapping components | four non-overlapping components | not covered by an NGO decision |
| V2-07 | G012 applies to distribution activities **and** visible handovers (a book or medicine handover filed under education or healthcare), judged for primary beneficiaries | distribution activities, every beneficiary | distributions and visible handovers, primary beneficiaries | not covered by an NGO decision |

Also in V2: taxonomy gains `POST_ACTIVITY` (V2 only); `EXC_MEDICAL_PPE` is active (PPE or a covering routes to CANNOT_TELL, never a pass); prompt version `photo_evaluation_prompt_v2`; G010 / G011 remain soft (they were already soft in V1; the pilot-3 workbook's "active in: both profiles" meant *present*, not *hard* in both). Not changed: G013 white top and R002, G003, G014, quality thresholds and weights (still UNCALIBRATED), publication layer.
Not implemented because the NGO withdrew them or they need data that does not exist: the T2 back-view exception, a computer-vision face layer, NEW-001..024 as a separate registry (the existing `NEW-*` and `P*` criteria are unchanged).

## 1.1.0 (2026-11-02): the NGO's five decisions
Source of every entry below: the NGO's instruction of 2026-11-02. Status TEAM_APPROVED unless noted. `checklist5_legacy` is unchanged by all of them.

| # | Decision | Old (1.0.0) | New (1.1.0) | Effect |
|---|---|---|---|---|
| 1 | Face scope: computer vision will help with faces later | layered profile narrowed G004-G008 to PRIMARY or FEATURED beneficiaries (proposal) | **withdrawn.** Scope stays ALL_VISIBLE_BENEFICIARIES in both profiles; wording identical to checklist5 | G004-G008 apply to every visible beneficiary; to be revisited when the CV layer exists |
| 2 | Banners: not in every photo, but when present clearly visible and differentiable | G009 (creases) narrowed to prominent banners only (proposal) | **withdrawn.** G009 applies whenever a banner is present. **New gates:** G017 `R001_VISIBLE` "clearly visible"; G018 new criterion `R001_DISTINCT` "clearly distinguishable from its surroundings: stands out from the background, text or logo readable". All three are N/A when there is no banner | a present banner that is creased, hard to see or blends in is REJECTED (layered profile) |
| 3 | Animals count as a subject | G014 widened to animals (PENDING_CONFIRMATION) | now TEAM_APPROVED. Wording of `R015_PERSON_PRESENT`: "a beneficiary, a volunteer, or an animal receiving care" | an animal-only photograph passes G014 |
| 4 | Direct gaze: volunteers allowed, beneficiaries not (rare future exceptions) | proposal: `EXC_DIRECT_GAZE` active when metadata says an approved portrait | exception **made dormant** (active in no profile; kept as a future hook). G015 unchanged: every beneficiary direct gaze is REJECTED, even if `direct_gaze_approved` is supplied. A volunteer's gaze is stated as "not a violation" in the prompt guidance (there is no volunteer-gaze criterion) | no way to allow a beneficiary's direct gaze for now |
| 5 | Back views disallowed; front-on position required; no backs | T1 was guidance only; T2 allowed a back view with dress code + logo on the back + beneficiaries visible | T2 **withdrawn** (`EXC_BACK_VIEW` DEPRECATED, `T_BACK_EXC_*` and `T_BACK_EXCEPTION` no longer asked in the layered profile). **New gate G016** with components `R008_BACK`, `T_FRONT`, `R015_BEN_FRONT`: the parent passes only if every applicable component passes | anyone shown mainly from behind (volunteer or beneficiary) is REJECTED; a side view is still acceptable |

Also in 1.1.0: the soft cap on `R001_FLAT` (brand_presentation) was removed from `quality_scoring.yaml` because a creased banner is a hard gate whenever a banner is present.
Scripted record of the YAML edit: `scripts/apply_ngo_decisions_2026_11_02.py`.

**Two judgement calls to confirm with the NGO**
* Decision 5 says "front on view" and "no backs". I read that as *front or side, never from behind* (a side view is allowed, as in the client's R008). If side views must also be rejected, T_FRONT and the question wording need tightening.
* Decision 5 reverses the earlier team-approved T2 exception (logo on the back). I applied the newer instruction.
* Decision 1: I read "we shall include computer vision later" as "do not narrow the face scope now". If you meant something else (for example, drop the face gates until CV exists), say so.

## 1.0.0 (2026-11-02)

### Carried over unchanged (65 questions, 15 gates)
All 65 checklist5 criterion ids and their wording are preserved (a test compares them). The 15 gates keep their ids as G001-G015, source TEAM, TEAM_APPROVED.
`checklist5_legacy` reproduces them exactly, including the strict "every visible beneficiary" scope, `unknown_is_fail`, and the hard G010/G011.

### Proposals in `layered_proposed_v1` that are still open (PROPOSED)
| Target | Old | New | Reason | Status |
|---|---|---|---|---|
| G001 R007_ANGLE | one question covering activity, faces and interaction | parent gate passes only if the four components pass (G001_ACTIVITY_VISIBLE, G001_BENEFICIARIES_VISIBLE, G001_INTERACTION_VISIBLE, G001_NOTHING_HIDDEN) | the question asked four things at once | PROPOSED |
| G010 R006_COVERAGE | hard gate | Layer 3 soft grading (activity_communication capped at 2) | zoomed-out framing is a quality matter | PROPOSED |
| G011 R006_ZOOMED_OUT | hard gate | Layer 3 soft grading (same cap; one defect counted once) | as G010 | PROPOSED |
| G012 R015_BEN_RECEIVING | applies whenever a beneficiary is visible | applies only in distribution activities (FOOD_DISTRIBUTION, DONATION_DISTRIBUTION) | education, healthcare, animal care and community service have no physical handover | PROPOSED |
| EXC_MEDICAL_PPE | none | in a medical scene, PPE-only obstruction on R003_FACE/R003_CLEAR becomes CANNOT_TELL (human review), not a violation | masks and shields are not a violation but are not waved through | PROPOSED |

### Added (PROPOSED, active only in `layered_proposed_v1`)
* Scene facts: F_BENEFICIARIES, F_BANNER_PROMINENT, F_ANIMALS, F_ACTIVITY_TYPE, F_ITEM_GIVEN, F_HANDOVER, F_VOLUNTEER_GIVER, F_ORIENTATION, F_LOGO, F_MEDICAL, F_GROUP, F_PRIMARY/SECONDARY/INCIDENTAL_SUBJECTS (the last three are the `subjects` list). The facts are collected in **both** profiles because applicability is computed in Python. (F_BANNER_PROMINENT is now informational only.)
* Diagnostics for the PPE exception (R003_OBSTRUCTION_PPE_ONLY) and the four G001 components.
* NEW-001..007 (safeguarding and publication: consent, child safeguarding, confidential information, dignified representation, manipulation, rights, safety), NEW-008..015 (technical), NEW-016..024 (editorial), P001..P011 (publication readiness). NEW-001, NEW-006 and most P-items need external information and are not answered by the model.

### Removed / not carried
* R004_VOL_GAZE was already removed in checklist5 (a volunteer may look into the camera). It stays removed. The client's R004 text still says neither should; this departure is team policy.
* R014, R015, R016 (final QC Q1-Q3) are kept as SUMMARY criteria and never scored on their own.

### Policy decisions still open for the NGO
G010/G011 hard vs soft; G012 applicability; the G001 components; EXC_MEDICAL_PPE result (human review vs pass); all quality thresholds; all NEW-* and P-* criteria.
