"""
Phase 4: Human Validation and Enriched Dataset Creation
- Joins immutable `historical_decision` with normalized model annotations and image metadata.
- Keeps `potential_rejection_factors` (model-observed) strictly separate from `human_validation.confirmed_rejection_reasons`.
- Generates targeted human review queues (approved with high-severity flags, rejected with no obvious rule failures, borderline images).
- Exports the final enriched dataset to both JSONL (`enriched_dataset.jsonl`) and CSV (`enriched_dataset.csv`).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from src.schemas import (
    EnrichedDatasetRecord,
    EnrichedImageMetadata,
    HumanValidationRecord,
    ModelAssessmentSummary,
)


class DatasetEnricher:
    def __init__(
        self,
        metadata_csv: Path,
        normalized_annotations_dir: Path,
        enriched_output_dir: Path,
    ):
        self.metadata_df = pd.read_csv(metadata_csv)
        self.normalized_dir = normalized_annotations_dir
        self.enriched_output_dir = enriched_output_dir
        self.enriched_output_dir.mkdir(parents=True, exist_ok=True)

    def build_enriched_dataset(
        self,
        human_reviews_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[EnrichedDatasetRecord]:
        """
        Merges historical labels, model annotations, and optional human validation overrides
        without ever mutating the original historical approval label or overwriting model tags.
        """
        human_reviews_by_id = human_reviews_by_id or {}
        meta_lookup = self.metadata_df.set_index("image_id").to_dict(orient="index")
        enriched_records: List[EnrichedDatasetRecord] = []

        for img_id, meta in sorted(meta_lookup.items()):
            norm_file = self.normalized_dir / f"{img_id}.json"
            if not norm_file.exists():
                continue

            ann = json.loads(norm_file.read_text(encoding="utf-8"))
            rule_map = {r["rule_id"]: r["status"] for r in ann.get("rule_assessments", [])}

            cat_buckets: Dict[str, List[str]] = {
                "TECHNICAL": [],
                "COMPOSITION": [],
                "EDITORIAL": [],
                "BRAND": [],
                "SAFETY": [],
                "CONTEXT": [],
            }
            for t in ann.get("normalized_tags", ann.get("identified_tags", [])):
                cat = t.get("category", "TECHNICAL").upper()
                tid = t.get("canonical_tag_id", t.get("tag_id"))
                if cat in cat_buckets and tid not in cat_buckets[cat]:
                    cat_buckets[cat].append(tid)

            pot_factors = sorted({
                pf["factor"] for pf in ann.get("potential_rejection_factors", [])
            })

            audit_meta = ann.get("_audit_metadata", {})
            model_summary = ModelAssessmentSummary(
                recommendation=ann.get("visual_publication_assessment", {}).get("recommendation", "NEEDS_HUMAN_REVIEW"),
                annotation_version=str(ann.get("annotation_version", "1.0")),
                model_version=audit_meta.get("model_version", "gemini-2.5-flash"),
            )

            h_data = human_reviews_by_id.get(img_id, {})
            human_val = HumanValidationRecord(
                reviewed=bool(h_data.get("reviewed", False)),
                confirmed_tags=h_data.get("confirmed_tags", []),
                confirmed_rejection_reasons=h_data.get("confirmed_rejection_reasons", []),
                reviewer_notes=h_data.get("reviewer_notes"),
            )

            record = EnrichedDatasetRecord(
                image_id=img_id,
                historical_decision=meta["historical_decision"],
                rule_assessments=rule_map,
                technical_tags=cat_buckets["TECHNICAL"],
                composition_tags=cat_buckets["COMPOSITION"],
                editorial_tags=cat_buckets["EDITORIAL"],
                brand_tags=cat_buckets["BRAND"],
                safety_tags=cat_buckets["SAFETY"],
                context_tags=cat_buckets["CONTEXT"],
                potential_rejection_factors=pot_factors,
                model_assessment=model_summary,
                human_validation=human_val,
                metadata=EnrichedImageMetadata(
                    event_id=str(meta["event_id"]) if pd.notna(meta.get("event_id")) else None,
                    capture_date=str(meta["capture_date"]) if pd.notna(meta.get("capture_date")) else None,
                    image_width=int(meta["image_width"]),
                    image_height=int(meta["image_height"]),
                ),
            )
            enriched_records.append(record)

        # Export to JSONL and CSV
        jsonl_path = self.enriched_output_dir / "enriched_dataset.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in enriched_records:
                f.write(json.dumps(rec.to_dict()) + "\n")

        flat_rows = []
        for rec in enriched_records:
            d = rec.to_dict()
            flat_rows.append({
                "image_id": d["image_id"],
                "historical_decision": d["historical_decision"],
                "model_recommendation": d["model_assessment"]["recommendation"],
                "technical_tags": "|".join(d["technical_tags"]),
                "composition_tags": "|".join(d["composition_tags"]),
                "editorial_tags": "|".join(d["editorial_tags"]),
                "brand_tags": "|".join(d["brand_tags"]),
                "safety_tags": "|".join(d["safety_tags"]),
                "context_tags": "|".join(d["context_tags"]),
                "potential_rejection_factors": "|".join(d["potential_rejection_factors"]),
                "human_reviewed": d["human_validation"]["reviewed"],
                "confirmed_rejection_reasons": "|".join(d["human_validation"]["confirmed_rejection_reasons"]),
                "event_id": d["metadata"]["event_id"],
                "image_width": d["metadata"]["image_width"],
                "image_height": d["metadata"]["image_height"],
            })
        pd.DataFrame(flat_rows).to_csv(self.enriched_output_dir / "enriched_dataset.csv", index=False)
        return enriched_records
