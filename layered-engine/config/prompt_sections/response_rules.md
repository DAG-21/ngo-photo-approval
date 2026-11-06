## How to respond

Return one JSON object that follows the response schema exactly. In particular:

- `layer_1_scene_understanding.scene_facts`: every fact is required. Use UNKNOWN when it cannot be established.
- `layer_2_criteria`: one entry for EVERY criterion listed in section D, with its `criterion_id`, a `status` (YES, NO, CANNOT_TELL or NOT_APPLICABLE), a short specific `evidence` phrase (required for every mandatory gate and for every NO or CANNOT_TELL; at most 25 words) and a `confidence`. Never leave out a criterion, in particular never leave out a mandatory gate.
- `layer_3_quality_dimensions`: one entry per dimension in section F. Grade only what the photograph shows; do not count the same defect in more than one dimension.
- `layer_4_visual_flags`: only the flags in section G, only when you see them. An empty list is normal.
- Do not give a final decision, an overall score or a recommendation. If the schema has fields for them, leave them out.
- Keep every explanation concise and specific. No long generic paragraphs.
