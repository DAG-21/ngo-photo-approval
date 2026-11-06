"""
Phase 2: Taxonomy Normalisation and Tag Consolidation (Prompt 2)
- Step 2.1: Aggregates all raw `identified_tags` across the annotated dataset with counts & sample evidence.
- Step 2.2: Consolidates synonyms into canonical tags while preserving raw-to-canonical mappings
  and keeping low-frequency safeguarding/compliance tags intact.
  Never overwrites original raw annotation files; writes normalized records to `data/processed/normalized_annotations/`.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
from src.gemini.client import GeminiVisionClient
from src.schemas import NormalizedTaxonomyOutput


class TaxonomyNormalizer:
    def __init__(
        self,
        base_taxonomy_path: Path,
        prompt_template_path: Path,
        annotations_dir: Path,
        normalized_dir: Path,
        client: GeminiVisionClient = None,
    ):
        self.base_taxonomy = json.loads(base_taxonomy_path.read_text(encoding="utf-8"))
        self.prompt_template = prompt_template_path.read_text(encoding="utf-8")
        self.annotations_dir = annotations_dir
        self.normalized_dir = normalized_dir
        self.normalized_dir.mkdir(parents=True, exist_ok=True)
        self.client = client

    def build_builtin_mapping(self) -> Dict[str, Dict[str, str]]:
        """Builds lookup mapping from `config/tag_taxonomy.json`."""
        mapping: Dict[str, Dict[str, str]] = {}
        for category, tag_list in self.base_taxonomy.get("categories", {}).items():
            for t in tag_list:
                canon_id = t["canonical_tag_id"]
                mapping[canon_id] = {"canonical_tag_id": canon_id, "parent_category": category}
                for raw_syn in t.get("merged_source_tags", []):
                    mapping[raw_syn.upper()] = {"canonical_tag_id": canon_id, "parent_category": category}
        return mapping

    def aggregate_raw_tags(self) -> List[Dict[str, Any]]:
        """Collects frequency counts and example evidence for every raw tag across all annotations."""
        counts: Dict[str, int] = defaultdict(int)
        categories: Dict[str, str] = {}
        examples: Dict[str, List[str]] = defaultdict(list)

        for fpath in sorted(self.annotations_dir.glob("NGO_IMG_*.json")):
            data = json.loads(fpath.read_text(encoding="utf-8"))
            for tag in data.get("identified_tags", []):
                tid = tag["tag_id"].strip().upper()
                counts[tid] += 1
                categories[tid] = tag.get("category", "UNKNOWN")
                if len(examples[tid]) < 3 and tag.get("evidence"):
                    examples[tid].append(tag["evidence"])

        summary = []
        for tid, count in sorted(counts.items(), key=lambda x: -x[1]):
            summary.append({
                "raw_tag_id": tid,
                "parent_category": categories[tid],
                "occurrences": count,
                "example_evidence": examples[tid],
            })
        return summary

    def normalize_annotations(
        self,
        custom_mapping: Dict[str, str] = None,
    ) -> Dict[str, str]:
        """
        Applies canonical tag mapping to all image annotations and writes normalized copies
        into `data/processed/normalized_annotations/` without modifying `data/processed/annotations/`.
        """
        builtin = self.build_builtin_mapping()
        raw_to_canon: Dict[str, str] = {k: v["canonical_tag_id"] for k, v in builtin.items()}
        if custom_mapping:
            for k, v in custom_mapping.items():
                raw_to_canon[k.upper()] = v

        for fpath in sorted(self.annotations_dir.glob("NGO_IMG_*.json")):
            raw_record = json.loads(fpath.read_text(encoding="utf-8"))
            normalized_tags = []
            seen_canonical = set()

            for tag in raw_record.get("identified_tags", []):
                raw_id = tag["tag_id"].strip().upper()
                canon_id = raw_to_canon.get(raw_id, raw_id)
                canon_cat = builtin.get(raw_id, {}).get("parent_category", tag.get("category", "TECHNICAL"))
                norm_entry = {
                    **tag,
                    "raw_tag_id": raw_id,
                    "canonical_tag_id": canon_id,
                    "category": canon_cat,
                }
                normalized_tags.append(norm_entry)
                seen_canonical.add(canon_id)

            raw_record["normalized_tags"] = normalized_tags
            raw_record["canonical_tag_ids"] = sorted(seen_canonical)
            out_path = self.normalized_dir / fpath.name
            out_path.write_text(json.dumps(raw_record, indent=2), encoding="utf-8")

        return raw_to_canon
