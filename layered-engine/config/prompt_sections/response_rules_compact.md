## How to respond (short format)

Return one JSON object that follows the response schema exactly. Keep it short.

- `activity_description`: at most 15 words.
- `scene_facts`: every fact is required. Use UNKNOWN when it cannot be established.
- `checks`: a status (YES, NO, CANNOT_TELL or NOT_APPLICABLE) for EVERY id listed in the schema's `checks`. They include every mandatory gate. Never leave one out.
- `other_criteria_not_yes`: every other criterion of section D counts as YES unless you list it here. List each one that is NO, CANNOT_TELL or NOT_APPLICABLE; do not list the ones that are YES.
- `evidence`: one entry for every criterion you answered NO or CANNOT_TELL, at most 15 words, saying what you saw. Nothing for YES answers.
- `quality`: one whole number per dimension of section F, from 0 to 4, or -1 when the dimension genuinely does not apply. No explanations.
- `visual_flags`: only the flags in section G, only when you see them. An empty list is normal.
- Do not give a final decision, an overall score or a recommendation.
