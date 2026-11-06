## What to return

Return one JSON object with these fields, in this order:

- observations: two to four short sentences describing what is visible that matters for the rules.
- reasons: up to five items, each with a rule_id (or OTHER if the point is not covered by a rule), an effect ("supports" if it speaks for approval, "against" if it speaks against), and a short note.
- reason: one or two plain sentences, written for a colleague who has not seen the rules, saying why this photograph should or should not be approved. Name the decisive point(s) and keep it consistent with the decision below.
- score: an integer from 0 to 100 for how likely the NGO's human reviewers approved this photograph. 0 means certainly rejected, 100 means certainly approved, 50 means you genuinely cannot tell. Use the whole range.
- decision: APPROVED if the score is 50 or more, otherwise DISAPPROVED.
- cannot_judge: true only if the photograph cannot be assessed (for example it is unreadable or not a photograph).
