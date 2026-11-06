"""
Phase 1: Historical Dataset Enrichment using Gemini Vision (Prompt 1)
- Step 1.1: Pilot runner (100 rejected + 100 approved, randomized IDs, strictly label-blind)
- Step 1.2: Full dataset batch runner (4,000 images, independent evaluation per photograph,
  records model version, prompt version, timestamp, and raw response).
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from src.gemini.client import GeminiConfigError, GeminiVisionClient
from src.paths import resolve_image_path
from src.schemas import HistoricalAnnotationRecord


class HistoricalDatasetAnnotator:
    def __init__(
        self,
        client: GeminiVisionClient,
        rules_path: Path,
        prompt_template_path: Path,
        annotations_dir: Path,
        prompt_version: str = "1.1.0",
    ):
        self.client = client
        self.rules = json.loads(rules_path.read_text(encoding="utf-8"))
        self.prompt_template = prompt_template_path.read_text(encoding="utf-8")
        self.annotations_dir = annotations_dir
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_version = prompt_version

    def build_label_blind_prompt(self, image_id: str) -> str:
        """
        Formats Prompt 1 with the 16 NGO rules and the anonymized `image_id`.
        NEVER includes historical approval/rejection status or original file path.
        """
        rules_formatted = json.dumps(self.rules, indent=2)
        return (
            self.prompt_template
            .replace("{NGO_16_RULES}", rules_formatted)
            .replace("{IMAGE_ID}", image_id)
        )

    def select_pilot_sample(
        self,
        metadata_df: pd.DataFrame,
        n_approved: int = 100,
        n_rejected: int = 100,
        seed: int = 42,
        exclude_duplicates: bool = True,
        exclude_label_conflicts: bool = True,
    ) -> pd.DataFrame:
        """
        Selects 100 approved and 100 rejected photographs from the training/validation pool,
        shuffles them, and strips the historical label column from the annotation queue.
        Exact duplicates are skipped (each photo is annotated once), as are photos whose identical
        file appears under both labels, since their historical decision is self-contradictory.
        """
        eligible = metadata_df[metadata_df["partition"] != "TEST"].copy()
        if exclude_label_conflicts:
            labels_per_file = metadata_df.groupby("sha256")["historical_decision"].nunique()
            conflicted = labels_per_file[labels_per_file > 1].index
            eligible = eligible[~eligible["sha256"].isin(conflicted)]
        if exclude_duplicates:
            eligible = eligible[~eligible["is_exact_duplicate"].astype(bool)]
        app_df = eligible[eligible["historical_decision"] == "APPROVED"]
        rej_df = eligible[eligible["historical_decision"] == "REJECTED"]

        app_sample = app_df.sample(n=min(n_approved, len(app_df)), random_state=seed)
        rej_sample = rej_df.sample(n=min(n_rejected, len(rej_df)), random_state=seed)

        combined = pd.concat([app_sample, rej_sample], ignore_index=True)
        # Shuffle so annotation order is completely randomized
        combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        # Drop historical_decision to guarantee label-blindness during pilot annotation
        return combined[["image_id", "original_filepath", "event_id"]]

    def annotate_single_image(self, image_id: str, image_path: Path) -> HistoricalAnnotationRecord:
        """Runs Prompt 1 independently on a single photograph and saves the validated JSON."""
        prompt_text = self.build_label_blind_prompt(image_id)
        record, envelope = self.client.generate_validated_json(
            record_id=image_id,
            prompt_text=prompt_text,
            prompt_version=self.prompt_version,
            schema_cls=HistoricalAnnotationRecord,
            image_paths=[image_path],
        )

        out_file = self.annotations_dir / f"{image_id}.json"
        payload = record.to_dict()
        payload["_audit_metadata"] = {
            "model_version": envelope["model_version"],
            "prompt_version": envelope["prompt_version"],
            "timestamp_utc": envelope["timestamp_utc"],
            "latency_ms": envelope["latency_ms"],
        }
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return record

    def run_batch(self, queue_df: pd.DataFrame) -> Dict[str, str]:
        """
        Processes a queue of images (either the 200-image pilot or full 4,000-image dataset).
        Explicitly tracks successes and failures—never silently interprets a failed call as clean.
        """
        statuses: Dict[str, str] = {}
        for _, row in queue_df.iterrows():
            img_id = row["image_id"]
            img_path = resolve_image_path(row["original_filepath"])
            try:
                self.annotate_single_image(img_id, img_path)
                statuses[img_id] = "COMPLETED"
            except GeminiConfigError:
                raise  # a bad key or missing SDK fails every image; abort instead of burning the whole queue
            except Exception as err:
                statuses[img_id] = f"FAILED: {err}"
        return statuses
