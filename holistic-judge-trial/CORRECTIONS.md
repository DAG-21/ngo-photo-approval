# Corrections register

Every correction you have asked for, where it lives, and the test that proves it is followed. A correction is only "done" when it has
an enforced gate in `prompts/checklist5/policy.yaml` (so a NO answer forces a disapproval in code) AND a stricter question in
`prompts/checklist5/checks.yaml` (so the model is asked precisely). `tests/test_policy.py` fails if this table and the policy disagree.

| # | Instruction | Questions (checks.yaml) | Gate that forces DISAPPROVED (policy.yaml) |
|---|---|---|---|
| 1 | Stress harder on camera angle; the angle must not hide the beneficiaries' faces | R007_ANGLE (stricter), R007_FACES_SEEN (new), R007_HEIGHT (stricter), 8 more R007 checks | R007_ANGLE, R007_FACES_SEEN |
| 2 | If something is being given to a beneficiary and that item is blurry, disapprove | R006_ITEM_SHARP (new), R006_HANDOVER (new) | R006_ITEM_SHARP |
| 3 | Beneficiaries' faces must be strictly, clearly visible | R003_FACE (every face, strict), R003_CLEAR, R003_CROP | R003_FACE, R003_CLEAR |
| 4 | A banner that is visible but very creased must be disapproved | R001_FLAT (clearly visible heavy folds or creases = NO; slight waviness is fine) | R001_FLAT |
| 5 | Beneficiary face first priority: visible and lit | R003_FACE_SHARP, R003_FACE_SIZE, R009_FACE_LIT (new); face questions placed first | R003_FACE_SHARP, R003_FACE_SIZE, R009_FACE_LIT |
| 6 | Zoomed-out photo where the activity is a small part: disapprove | R006_COVERAGE (about one third of the frame), R006_ZOOMED_OUT | R006_COVERAGE, R006_ZOOMED_OUT |
| 7 | If there is a beneficiary they must be receiving something from a volunteer, and that volunteer must wear a white top (otherwise discard); the beneficiary face must be clearly visible (see 3 and 5) | R015_BEN_RECEIVING (new), R002_GIVER_WHITE (new) | R015_BEN_RECEIVING, R002_GIVER_WHITE |
| 8 | The photo must contain a beneficiary OR a volunteer (replaces "a beneficiary must be visible") | R015_PERSON_PRESENT (new; R015_BEN_PRESENT stays as information only) | R015_PERSON_PRESENT |
| 9 | The beneficiary must not look straight into the camera, but a volunteer may | R004_BEN_GAZE (the volunteer-gaze question was removed) | R004_BEN_GAZE |
| E1 | (earlier, retired by 8) A photo with no beneficiary visible must count against it | R015_BEN_PRESENT | (no longer a gate) |
| E2 | (earlier) Add a point on overexposure | R009_OVEREXPOSED (separate from underexposure) | (scored, not a gate: not asked to disapprove) |

History: checklist3 worded #4 as "any crease, even a slight one", which was stricter than "very creased" and made the gate fire on 29 of about 34 banner photos; checklist4 corrects that one question.

History (2): checklist5 retires the "no beneficiary visible" gate (E1) in favour of 8, and removes the question that penalised volunteers for looking into the camera (the client's R004 text says neither should; you said a volunteer may).

How strictness is applied: the model is told to be strict and to answer YES only when a requirement is clearly and fully met; the
policy treats NO as a failure and, by default, CANNOT_TELL as a failure too (`unknown_is_fail`). NOT_APPLICABLE never fails.

Departures from the client's wording (the client's rules are unchanged in the prompt, shown verbatim above the questions):
the face questions no longer say "wherever appropriate"; the coverage threshold (about one third) is ours, not the client's.
