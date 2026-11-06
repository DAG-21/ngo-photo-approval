"""
Phase 6: Validate the Production Approach (Ablation Experiments A, B, C, and D)
Measures across identical held-out TEST images:
- Approval precision (Zero-false-positive critical metric)
- Rejection recall & Rejection precision
- Macro / Binary F1-score
- Confusion matrix (including HUMAN_REVIEW escalations)
- Human-review escalation rate
- Performance across event groups
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from src.decision_engine.engine import PhotoApprovalDecisionEngine
from src.paths import resolve_image_path


class AblationExperimentRunner:
    def __init__(
        self,
        engine: PhotoApprovalDecisionEngine,
        test_ids_csv: Path,
        metadata_csv: Path,
        evaluation_reports_dir: Path,
    ):
        self.engine = engine
        self.test_df = pd.read_csv(test_ids_csv)
        self.meta_df = pd.read_csv(metadata_csv).set_index("image_id")
        self.evaluation_reports_dir = evaluation_reports_dir
        self.evaluation_reports_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(rows)
        if total == 0:
            return {}

        human_review_count = sum(1 for r in rows if r["predicted"] == "HUMAN_REVIEW")
        decided = [r for r in rows if r["predicted"] in ("APPROVE", "REJECT")]

        # Confusion counts on auto-decided vs overall
        tp_approve = sum(1 for r in rows if r["predicted"] == "APPROVE" and r["actual"] == "APPROVED")
        fp_approve = sum(1 for r in rows if r["predicted"] == "APPROVE" and r["actual"] == "REJECTED")
        tp_reject = sum(1 for r in rows if r["predicted"] == "REJECT" and r["actual"] == "REJECTED")
        fp_reject = sum(1 for r in rows if r["predicted"] == "REJECT" and r["actual"] == "APPROVED")

        total_actual_rejected = sum(1 for r in rows if r["actual"] == "REJECTED")
        # Safe rejection coverage: historically rejected images that were either REJECTED or caught by HUMAN_REVIEW
        safe_caught_rejected = sum(
            1 for r in rows if r["actual"] == "REJECTED" and r["predicted"] in ("REJECT", "HUMAN_REVIEW")
        )

        approval_precision = (
            tp_approve / (tp_approve + fp_approve) if (tp_approve + fp_approve) > 0 else 1.0
        )
        rejection_precision = (
            tp_reject / (tp_reject + fp_reject) if (tp_reject + fp_reject) > 0 else 1.0
        )
        rejection_recall_strict = (
            tp_reject / total_actual_rejected if total_actual_rejected > 0 else 0.0
        )
        rejection_safety_capture_rate = (
            safe_caught_rejected / total_actual_rejected if total_actual_rejected > 0 else 1.0
        )
        f1_rejection = (
            2 * rejection_precision * rejection_recall_strict / (rejection_precision + rejection_recall_strict)
            if (rejection_precision + rejection_recall_strict) > 0
            else 0.0
        )

        return {
            "total_test_images": total,
            "auto_decided_count": len(decided),
            "human_review_count": human_review_count,
            "human_review_rate": round(human_review_count / total, 4),
            "approval_precision": round(approval_precision, 4),
            "false_positive_approvals": fp_approve,
            "rejection_precision": round(rejection_precision, 4),
            "rejection_recall_auto": round(rejection_recall_strict, 4),
            "rejection_safety_capture_rate": round(rejection_safety_capture_rate, 4),
            "rejection_f1_score": round(f1_rejection, 4),
            "confusion_matrix": {
                "actual_APPROVED_pred_APPROVE": tp_approve,
                "actual_APPROVED_pred_REJECT": fp_reject,
                "actual_APPROVED_pred_HUMAN_REVIEW": sum(
                    1 for r in rows if r["actual"] == "APPROVED" and r["predicted"] == "HUMAN_REVIEW"
                ),
                "actual_REJECTED_pred_APPROVE_FALSE_POSITIVE": fp_approve,
                "actual_REJECTED_pred_REJECT": tp_reject,
                "actual_REJECTED_pred_HUMAN_REVIEW": sum(
                    1 for r in rows if r["actual"] == "REJECTED" and r["predicted"] == "HUMAN_REVIEW"
                ),
            },
        }

    def run_all_experiments(
        self,
        fixed_examples: List[Dict[str, Any]],
        external_metadata_by_id: Dict[str, Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        external_metadata_by_id = external_metadata_by_id or {}
        modes = ["EXPERIMENT_A", "EXPERIMENT_B", "EXPERIMENT_C", "EXPERIMENT_D"]
        results_by_mode: Dict[str, Any] = {}

        for mode in modes:
            rows = []
            for _, test_row in self.test_df.iterrows():
                img_id = test_row["image_id"]
                actual = test_row["historical_decision"]
                event_id = test_row["event_id"]
                img_path = resolve_image_path(self.meta_df.loc[img_id, "original_filepath"])
                ext_meta = {"event_id": event_id, **external_metadata_by_id.get(img_id, {})}

                output = self.engine.evaluate_photograph(
                    new_image_id=img_id,
                    image_path=img_path,
                    external_metadata=ext_meta,
                    mode=mode,
                    fixed_examples=fixed_examples if mode == "EXPERIMENT_C" else None,
                )
                rows.append({
                    "image_id": img_id,
                    "event_id": event_id,
                    "actual": actual,
                    "predicted": output.recommendation,
                    "confidence": output.confidence_level,
                })

            results_by_mode[mode] = self.compute_metrics(rows)

        out_path = self.evaluation_reports_dir / "phase6_ablation_comparison.json"
        out_path.write_text(json.dumps(results_by_mode, indent=2), encoding="utf-8")
        return results_by_mode
