## What to return

Return one JSON object:

- scene: one short sentence describing what the photograph shows.
- checks: an object with one entry for EVERY question id listed above, in the same order. Each entry has "answer" (YES, NO, CANNOT_TELL or NOT_APPLICABLE) and "evidence" (a short phrase for NO or CANNOT_TELL, otherwise an empty string).
- cannot_judge: true only if the photograph cannot be assessed at all (for example unreadable).

Do not add a score or an approve/disapprove decision.
