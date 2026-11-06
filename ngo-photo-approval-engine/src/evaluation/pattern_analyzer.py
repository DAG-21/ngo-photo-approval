"""
Phase 3: Analyse the Enriched Dataset Against Historical Decisions
- Step 3.1: Programmatically calculates exact prevalence statistics in Python/Pandas:
  * `approved_prevalence = approved_images_with_tag / total_approved_images`
  * `rejected_prevalence = rejected_images_with_tag / total_rejected_images`
  * `prevalence_difference = rejected_prevalence - approved_prevalence`
  * Co-occurring tag combinations
  * Rule failure frequencies
  * Frequency of issues occurring despite all visually assessable rules passing
  * Frequency of photographs Gemini considers unsuitable despite no identified rule violation
- Step 3.2: Formats programmatic statistical tables into Prompt 3 for pattern discovery.
"""

import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


class HistoricalPatternAnalyzer:
    def __init__(self, metadata_csv: Path, normalized_annotations_dir: Path, output_reports_dir: Path):
        self.metadata_df = pd.read_csv(metadata_csv)
        self.normalized_dir = normalized_annotations_dir
        self.output_reports_dir = output_reports_dir
        self.output_reports_dir.mkdir(parents=True, exist_ok=True)

    def compute_programmatic_statistics(self) -> Dict[str, Any]:
        """
        Joins normalized model annotations with historical APPROVED/REJECTED labels
        and computes all required Phase 3 prevalence metrics programmatically.
        """
        records = []
        meta_lookup = self.metadata_df.set_index("image_id").to_dict(orient="index")

        for fpath in sorted(self.normalized_dir.glob("NGO_IMG_*.json")):
            ann = json.loads(fpath.read_text(encoding="utf-8"))
            img_id = ann["image_id"]
            if img_id not in meta_lookup:
                continue
            hist_decision = meta_lookup[img_id]["historical_decision"]
            event_id = meta_lookup[img_id].get("event_id")

            rule_statuses = {r["rule_id"]: r["status"] for r in ann.get("rule_assessments", [])}
            any_rule_failed = any(status == "FAIL" for status in rule_statuses.values())
            canon_tags = sorted(set(ann.get("canonical_tag_ids", [])))
            gem_rec = ann.get("visual_publication_assessment", {}).get("recommendation", "UNCERTAIN")

            records.append({
                "image_id": img_id,
                "historical_decision": hist_decision,
                "event_id": event_id,
                "rule_statuses": rule_statuses,
                "any_rule_failed": any_rule_failed,
                "canonical_tags": canon_tags,
                "gemini_recommendation": gem_rec,
            })

        total_annotated = len(records)
        approved_recs = [r for r in records if r["historical_decision"] == "APPROVED"]
        rejected_recs = [r for r in records if r["historical_decision"] == "REJECTED"]

        total_approved = len(approved_recs)
        total_rejected = len(rejected_recs)

        if total_approved == 0 or total_rejected == 0:
            raise ValueError(
                f"Cannot compare prevalence with empty class: approved={total_approved}, rejected={total_rejected}"
            )

        # 1. Tag prevalence calculation
        all_tags = sorted({t for r in records for t in r["canonical_tags"]})
        tag_stats = []
        for tag in all_tags:
            app_count = sum(1 for r in approved_recs if tag in r["canonical_tags"])
            rej_count = sum(1 for r in rejected_recs if tag in r["canonical_tags"])
            app_prev = app_count / total_approved
            rej_prev = rej_count / total_rejected
            prev_diff = rej_prev - app_prev

            tag_stats.append({
                "tag_id": tag,
                "total_count": app_count + rej_count,
                "approved_count": app_count,
                "rejected_count": rej_count,
                "approved_prevalence": round(app_prev, 4),
                "rejected_prevalence": round(rej_prev, 4),
                "prevalence_difference": round(prev_diff, 4),
            })

        tag_stats.sort(key=lambda x: x["prevalence_difference"], reverse=True)

        # 2. Rule failure frequencies
        rule_ids = sorted({rid for r in records for rid in r["rule_statuses"].keys()})
        rule_stats = []
        for rid in rule_ids:
            app_fail = sum(1 for r in approved_recs if r["rule_statuses"].get(rid) == "FAIL")
            rej_fail = sum(1 for r in rejected_recs if r["rule_statuses"].get(rid) == "FAIL")
            rule_stats.append({
                "rule_id": rid,
                "approved_fail_count": app_fail,
                "rejected_fail_count": rej_fail,
                "approved_fail_rate": round(app_fail / total_approved, 4),
                "rejected_fail_rate": round(rej_fail / total_rejected, 4),
                "fail_rate_difference": round((rej_fail / total_rejected) - (app_fail / total_approved), 4),
            })

        # 3. Most common tag combinations in rejected photos
        combo_counter: Counter = Counter()
        for r in rejected_recs:
            tags = r["canonical_tags"]
            if len(tags) >= 2:
                for pair in combinations(tags, 2):
                    combo_counter[f"{pair[0]} + {pair[1]}"] += 1

        top_combinations = [
            {"tag_pair": pair, "rejected_occurrences": cnt, "rejected_share": round(cnt / total_rejected, 4)}
            for pair, cnt in combo_counter.most_common(10)
        ]

        # 4. Issues occurring despite all visually assessable rules passing
        rejected_all_rules_pass = [r for r in rejected_recs if not r["any_rule_failed"]]
        rejected_all_rules_pass_with_tags = [
            r for r in rejected_all_rules_pass if len(r["canonical_tags"]) > 0
        ]
        unsuitable_without_rule_violation = [
            r for r in records
            if not r["any_rule_failed"] and r["gemini_recommendation"] == "NOT_SUITABLE"
        ]

        summary = {
            "sample_counts": {
                "total_expected_in_metadata": len(self.metadata_df),
                "total_annotated": total_annotated,
                "missing_annotations": len(self.metadata_df) - total_annotated,
                "total_approved_annotated": total_approved,
                "total_rejected_annotated": total_rejected,
            },
            "tag_prevalence_table": tag_stats,
            "rule_failure_table": rule_stats,
            "top_rejected_tag_combinations": top_combinations,
            "rulebook_gap_indicators": {
                "rejected_despite_all_rules_passing_count": len(rejected_all_rules_pass),
                "rejected_despite_all_rules_passing_rate": round(len(rejected_all_rules_pass) / total_rejected, 4),
                "issues_present_despite_all_rules_passing_count": len(rejected_all_rules_pass_with_tags),
                "gemini_unsuitable_despite_no_rule_violation_count": len(unsuitable_without_rule_violation),
                "gemini_unsuitable_despite_no_rule_violation_rate": round(
                    len(unsuitable_without_rule_violation) / total_annotated, 4
                ),
            },
        }

        report_path = self.output_reports_dir / "phase3_programmatic_statistics.json"
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        pd.DataFrame(tag_stats).to_csv(self.output_reports_dir / "tag_prevalence_comparison.csv", index=False)
        return summary
