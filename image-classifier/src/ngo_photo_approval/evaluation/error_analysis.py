"""Error analysis for false approvals / false rejections.

Only causes that can be derived mechanically are tagged automatically (failed analysis, uncertain
or failed rules, measured image-quality flags, program). Visual causes such as "banner too small"
or "rule ambiguous" need a human, so a blank ``manual_cause`` column is provided for reviewers.
"""

from __future__ import annotations

import pandas as pd

QUALITY_FLAGS = ["flag_blurry", "flag_dark", "flag_bright", "flag_low_contrast", "flag_small"]


def classify_error(label: str, decision: str) -> str | None:
    if label == "not_approved" and decision == "APPROVE":
        return "false_approval"
    if label == "approved" and decision == "REJECT":
        return "false_rejection"
    if decision in {"REVIEW", "ERROR"}:
        return f"unresolved_{decision.lower()}"
    return None


def build_error_analysis(predictions: pd.DataFrame, audit: pd.DataFrame | None = None) -> pd.DataFrame:
    """``predictions``: image_id, label, decision, plus optional hard_rule_failures (|-joined),
    uncertain_rules, reason_codes, error_type, probability, program."""
    p = predictions.copy()
    if audit is not None:
        cols = ["image_id"] + [c for c in QUALITY_FLAGS if c in audit]
        p = p.merge(audit[cols], on="image_id", how="left")
    p["error_type"] = [classify_error(lab, dec) for lab, dec in zip(p["label"], p["decision"], strict=True)]
    errors = p[p["error_type"].notna()].copy()

    def tags(r: pd.Series) -> str:
        t: list[str] = []
        for col, prefix in (("hard_rule_failures", "mandatory_failed"), ("uncertain_rules", "uncertain")):
            v = r.get(col)
            if isinstance(v, str) and v:
                t += [f"{prefix}:{x}" for x in v.split("|") if x]
        et = r.get("failure_kind")
        if isinstance(et, str) and et:
            t.append(f"analysis_{et}")
        for f in QUALITY_FLAGS:
            if r.get(f) is True or r.get(f) == True:  # noqa: E712 - NaN-safe on object columns
                t.append(f"quality:{f.removeprefix('flag_')}")
        if r["error_type"] == "false_approval" and not t:
            t.append("no_automatic_cause_found")
        return "|".join(t)

    errors["auto_tags"] = errors.apply(tags, axis=1)
    errors["manual_cause"] = ""  # to be filled by a human reviewer
    keep = [c for c in ["image_id", "label", "decision", "error_type", "probability", "program",
                        "hard_rule_failures", "uncertain_rules", "reason_codes", "auto_tags", "manual_cause"]
            if c in errors]
    return errors[keep].sort_values(["error_type", "image_id"]).reset_index(drop=True)


def summarize_errors(errors: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for et, g in errors.groupby("error_type"):
        counts: dict[str, int] = {}
        for tags in g["auto_tags"]:
            for t in (tags.split("|") if tags else ["(none)"]):
                counts[t] = counts.get(t, 0) + 1
        out[et] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    return out
