# NGO Photo Approval Engine

End-to-end implementation of the **NGO Photo Approval Engine** using **Gemini Vision**, **Label-Blind Dataset Enrichment**, **Taxonomy Normalisation**, **Programmatic Prevalence Analysis**, **Dynamic Few-Shot Retrieval**, and a **Zero-False-Positive Hierarchical Decision Layer**.

## Architecture & Directory Structure

```text
ngo-photo-approval-engine/
├── config/
│   ├── ngo_rules.json                 # 16 structured NGO rules (severity, decision_effect, visual assessability)
│   ├── tag_taxonomy.json              # Controlled canonical tag dictionary & synonym mappings
│   ├── evaluation_config.yaml         # Split ratios, burst window, retrieval settings, & enhanced rubric
│   └── gemini_config.yaml             # Low-variance generation parameters, retry backoff, & prompt versions
├── prompts/
│   ├── image_annotation_prompt.txt    # Prompt 1: Label-blind historical image evaluation
│   ├── taxonomy_normalization_prompt.txt # Prompt 2: Tag consolidation & synonym normalization
│   ├── dataset_analysis_prompt.txt    # Prompt 3: Historical pattern & rulebook gap analysis
│   └── photo_approval_prompt.txt      # Prompt 4: Production photo evaluation with dynamic few-shot references
├── src/
│   ├── schemas.py                     # Pydantic models validating all 16 rules & JSON outputs
│   ├── ingestion/prepare_dataset.py   # Phase 0: Randomized IDs (NGO_IMG_XXXXXX), EXIF, dHash burst grouping, splits
│   ├── image_processing/quality_and_exif.py # SHA-256, 64-bit dHash, dimensions, and EXIF parser
│   ├── gemini/client.py               # Label-blind byte sanitizer, exponential backoff, raw response archiving
│   ├── annotation/annotator.py        # Phase 1: 100+100 pilot selector and 4,000-image batch annotator
│   ├── annotation/enrichment.py       # Phase 4: Immutable historical label join + human validation export (JSONL/CSV)
│   ├── taxonomy/normalizer.py         # Phase 2: Canonical tag mapper preserving original raw observations
│   ├── evaluation/pattern_analyzer.py # Phase 3: Programmatic prevalence & rulebook gap calculator
│   ├── retrieval/vector_index.py      # Phase 5: Visual embedding index with event/burst exclusion & test isolation
│   ├── decision_engine/engine.py      # Phase 5: Priority 1-5 deterministic guardrails + Prompt 4 evaluation
│   ├── decision_engine/api.py         # FastAPI service & human-in-the-loop feedback endpoint
│   └── evaluation/experiments.py      # Phase 6: Ablation benchmark runner (Experiments A, B, C, and D)
├── data/
│   ├── original/{approved,rejected}/
│   ├── processed/{annotations,normalized_annotations,enriched_dataset}/
│   └── evaluation/{training_ids.csv,validation_ids.csv,test_ids.csv}
└── outputs/
    ├── raw_gemini_responses/
    ├── annotation_reports/
    ├── evaluation_reports/
    └── final_decisions/
```

## Quickstart

1. **Install dependencies and configure API key**:
   ```bash
   pip install -r requirements.txt
   export GEMINI_API_KEY="your-api-key"
   ```
2. **Run End-to-End Verification Pipeline**:
   ```bash
   python3 run_pipeline_verification.py
   ```
