"""Enriched dataset export (JSONL + CSV).

Historical labels are joined here, AFTER annotation, from a separate file, and are stored in their own field. Model observations,
human-confirmed tags, confirmed rejection reasons and the historical decision are kept apart: a model's `potential_rejection_factors`
is never copied into `confirmed_rejection_reasons`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from .annotation import read_rows


def _load_csv(path: Optional[Path], key: str = "image_id") -> dict[str, dict[str, str]]:
    if not path:
        return {}
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return {r[key]: r for r in csv.DictReader(fh)}


def build_enriched(results_path: Path, out_prefix: Path, labels_csv: Optional[Path] = None, human_csv: Optional[Path] = None) -> dict[str, int]:
    """labels_csv: image_id,label (the immutable historical decision). human_csv: image_id,confirmed_tags,confirmed_rejection_reasons,reviewer_notes."""
    labels, human = _load_csv(labels_csv), _load_csv(human_csv)
    latest: dict[str, dict[str, Any]] = {}
    for r in read_rows(results_path):
        if r.get("ok"):
            latest[r["image_id"]] = r  # a later successful row supersedes an earlier one
    records, flat = [], []
    for iid, r in sorted(latest.items()):
        v, d = r["validated"], r["decision"]
        h = human.get(iid, {})
        rec = {
            "image_id": iid, "historical_decision": labels.get(iid, {}).get("label"),  # immutable; joined after annotation
            "model_observations": {
                "scene": v["layer_1_scene_understanding"], "criteria": v["layer_2_criteria"], "quality_dimensions": v["layer_3_quality_dimensions"],
                "visual_flags": v["layer_4_visual_flags"], "positive_attributes": v["positive_attributes"],
                "potential_rejection_factors": v["potential_rejection_factors"], "new_factor_proposals": v["new_factor_proposals"]},
            "engine_decision": {k: d[k] for k in ("recommendation", "photo_approval_status", "publication_readiness_status", "decision_reasons",
                                                 "failed_gates", "uncertain_gates", "not_applicable_gates", "warnings")} | {"quality": d["quality"], "versions": d["versions"]},
            "human_validation": {"reviewed": bool(h), "confirmed_tags": [t for t in h.get("confirmed_tags", "").split("|") if t],
                                 "confirmed_rejection_reasons": [t for t in h.get("confirmed_rejection_reasons", "").split("|") if t],
                                 "reviewer_notes": h.get("reviewer_notes") or None},
            "provenance": {k: r.get(k) for k in ("model_id", "prompt_version", "prompt_hash", "rulebook_version", "profile", "timestamp", "mode", "input_ref")}}
        records.append(rec)
        row = {"image_id": iid, "historical_decision": rec["historical_decision"], "recommendation": d["recommendation"], "photo_approval_status": d["photo_approval_status"],
               "publication_readiness_status": d["publication_readiness_status"], "failed_gates": "|".join(d["failed_gates"]), "uncertain_gates": "|".join(d["uncertain_gates"]),
               "quality_score": d["quality"]["overall_score"], "quality_band": d["quality"]["band"], "activity_type": v["layer_1_scene_understanding"]["scene_facts"]["f_activity_type"],
               "profile": r["profile"], "model_id": r["model_id"], "prompt_hash": r["prompt_hash"], "human_reviewed": rec["human_validation"]["reviewed"]}
        for g in d["gates"]:
            row[f"gate_{g['gate_id']}"] = g["status"]
        for dim, s in d["quality"]["dimension_scores"].items():
            row[f"dim_{dim}"] = s
        flat.append(row)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in records) + ("\n" if records else ""), encoding="utf-8")
    cols = list(dict.fromkeys(k for r in flat for k in r))
    with out_prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(flat)
    return {"photos": len(records), "with_label": sum(1 for r in records if r["historical_decision"]), "human_reviewed": sum(1 for r in records if r["human_validation"]["reviewed"])}
