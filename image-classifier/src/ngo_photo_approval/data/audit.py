"""Dataset audit: per-image flags, eligibility for supervised training, and summary statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import DatasetConfig

EXCL_PENDING = "pending_label_not_final"
EXCL_UNSUPPORTED = "unsupported_media"
EXCL_UNREADABLE = "unreadable_or_corrupt"
EXCL_UNMAPPED = "unmapped_label"
EXCL_CONFLICT = "duplicate_label_conflict"


def build_audit(
    manifest: pd.DataFrame,
    dup_report: pd.DataFrame,
    cluster_of: dict[str, str],
    cfg: DatasetConfig,
) -> pd.DataFrame:
    """Per-image audit table including `eligible_for_training` and `exclusion_reason`."""
    q = cfg.quality
    df = manifest.copy()
    df["flag_blurry"] = df["blur_laplacian_var"] < q.blur_laplacian_var_low
    df["flag_dark"] = df["brightness"] < q.brightness_low
    df["flag_bright"] = df["brightness"] > q.brightness_high
    df["flag_low_contrast"] = df["contrast"] < q.contrast_low
    df["flag_small"] = df[["width", "height"]].astype("float").min(axis=1) < q.min_side_px_warn
    flag_cols = ["flag_blurry", "flag_dark", "flag_bright", "flag_low_contrast", "flag_small"]
    df[flag_cols] = df[flag_cols].fillna(False).astype(bool)

    df["duplicate_cluster_id"] = df["image_id"].map(cluster_of)
    drep = dup_report.set_index("image_id") if len(dup_report) else pd.DataFrame()
    for col in ("is_exact_duplicate", "is_near_duplicate", "label_conflict", "exact_label_conflict"):
        df[col] = df["image_id"].map(drep[col]) if col in drep else False
        df[col] = df[col].fillna(False).astype(bool)

    def reason(r: pd.Series) -> str | None:
        if r["label"] is None or pd.isna(r["label"]):
            return EXCL_UNMAPPED
        if r["label"] not in cfg.trainable_labels:
            return EXCL_PENDING if r["label"] == "pending" else EXCL_UNMAPPED
        if r["media_kind"] != "image":
            return EXCL_UNSUPPORTED
        if not r["readable"]:
            return EXCL_UNREADABLE
        if cfg.label_conflict_exclusion == "exact" and r["exact_label_conflict"]:
            return EXCL_CONFLICT
        if cfg.label_conflict_exclusion == "near" and r["label_conflict"]:
            return EXCL_CONFLICT
        return None

    df["exclusion_reason"] = df.apply(reason, axis=1)
    df["eligible_for_training"] = df["exclusion_reason"].isna()
    keep = ["image_id", "rel_path", "label", "original_label", "program", "media_kind", "readable",
            "error", *flag_cols, "duplicate_cluster_id", "is_exact_duplicate", "is_near_duplicate",
            "label_conflict", "exact_label_conflict", "eligible_for_training", "exclusion_reason"]
    return df[keep]


def _pct(s: pd.Series) -> dict[str, float]:
    s = s.dropna().astype(float)
    if s.empty:
        return {}
    return {f"p{p}": round(float(np.percentile(s, p)), 3) for p in (0, 5, 25, 50, 75, 95, 100)}


def summarize(manifest: pd.DataFrame, audit: pd.DataFrame, dup_report: pd.DataFrame) -> dict[str, Any]:
    """Aggregate statistics for the report (JSON-serializable)."""
    m = manifest.merge(audit[["image_id", "eligible_for_training", "exclusion_reason",
                              "flag_blurry", "flag_dark", "flag_bright", "flag_low_contrast",
                              "flag_small"]], on="image_id")
    out: dict[str, Any] = {"total_files": int(len(m))}
    out["files_per_label"] = m["label"].value_counts().to_dict()
    out["media_kind_per_label"] = m.groupby("label")["media_kind"].value_counts().unstack(fill_value=0).to_dict("index")
    out["unreadable"] = m.loc[(m["media_kind"] == "image") & ~m["readable"], ["rel_path", "error"]].to_dict("records")
    out["eligible_per_label"] = m[m["eligible_for_training"]]["label"].value_counts().to_dict()
    out["exclusions"] = m["exclusion_reason"].value_counts().to_dict()

    fin = m[m["label"].isin(["approved", "not_approved"])]
    out["program_by_label"] = (
        fin.groupby(["program", "label"]).size().unstack(fill_value=0).to_dict("index")
    )
    pg = fin.groupby("program")["label"].nunique()
    out["programs_in_single_label"] = sorted(pg[pg == 1].index.tolist())
    out["unparsed_filenames"] = int(fin["program"].isna().sum())

    rd = m[m["readable"]]
    out["dimensions"] = {
        lab: {"width": _pct(g["width"]), "height": _pct(g["height"]), "megapixels": _pct(g["megapixels"]),
              "file_size_mb": _pct(g["file_size"] / 1e6)}
        for lab, g in rd.groupby("label")
    }
    out["orientation_share"] = {
        lab: {
            "portrait": round(float((g["aspect_ratio"].astype(float) < 0.95).mean()), 3),
            "landscape": round(float((g["aspect_ratio"].astype(float) > 1.05).mean()), 3),
        }
        for lab, g in rd.groupby("label")
    }
    flags = ["flag_blurry", "flag_dark", "flag_bright", "flag_low_contrast", "flag_small"]
    out["quality_flag_rates"] = {lab: {f: round(float(g[f].mean()), 4) for f in flags} for lab, g in rd.groupby("label")}
    out["quality_metrics"] = {
        lab: {c: _pct(g[c]) for c in ("blur_laplacian_var", "brightness", "contrast")}
        for lab, g in rd.groupby("label")
    }
    out["exif_datetime_available"] = {
        lab: round(float(g["exif_datetime"].notna().mean()), 3) for lab, g in rd.groupby("label")
    }

    if len(dup_report):
        clusters = dup_report.drop_duplicates("cluster_id")
        out["duplicates"] = {
            "clusters": int(len(clusters)),
            "images_in_clusters": int(len(dup_report)),
            "exact_duplicate_images": int(dup_report["is_exact_duplicate"].sum()),
            "near_duplicate_images": int(dup_report["is_near_duplicate"].sum()),
            "clusters_with_label_conflict": int(clusters["label_conflict"].sum()),
            "clusters_with_exact_label_conflict": int(clusters["exact_label_conflict"].sum()),
            "largest_cluster": int(clusters["cluster_size"].max()),
        }
    else:
        out["duplicates"] = {"clusters": 0, "images_in_clusters": 0}

    img_fin = fin[fin["readable"] & (fin["media_kind"] == "image")]
    out["container_formats"] = {
        lab: g["pil_format"].fillna("unknown").value_counts().to_dict()
        for lab, g in m[m["media_kind"] == "image"].groupby("label")
    }
    out["extension_mismatch"] = {
        lab: int(g["extension_mismatch"].fillna(False).astype(bool).sum())
        for lab, g in m[m["media_kind"] == "image"].groupby("label")
    }
    # Provenance confound: how well do trivial, non-semantic properties separate the labels?
    from sklearn.metrics import roc_auc_score

    y = (img_fin["label"] == "approved").astype(int)
    confound: dict[str, Any] = {"share_ge_4mp": {
        lab: round(float((g["megapixels"].astype(float) >= 4).mean()), 3) for lab, g in img_fin.groupby("label")}}
    if y.nunique() == 2:
        confound["single_feature_auc_approved"] = {
            "file_size": round(float(roc_auc_score(y, img_fin["file_size"].astype(float))), 3),
            "megapixels": round(float(roc_auc_score(y, img_fin["megapixels"].astype(float))), 3),
        }
    out["provenance_confound"] = confound

    grp = fin[fin["filename_group"].notna()]
    out["filename_groups"] = {
        "groups": int(grp["filename_group"].nunique()),
        "images_with_group": int(len(grp)),
        "median_group_size": float(grp.groupby("filename_group").size().median()) if len(grp) else 0.0,
        "max_group_size": int(grp.groupby("filename_group").size().max()) if len(grp) else 0,
    }
    # Same `program-N` name appearing under both final labels (are they the same event?).
    names = fin.dropna(subset=["program", "submission_idx"]).assign(
        key=lambda d: d["program"].astype(str) + "-" + d["submission_idx"].astype(str)
    )
    both = names.groupby("key")["label"].nunique()
    out["program_submission_keys_in_both_labels"] = int((both > 1).sum())
    out["program_submission_keys_total"] = int(len(both))
    return out


def render_markdown(summary: dict[str, Any], generated_at: str, data_dir_name: str) -> str:
    """Human-readable DATASET_AUDIT.md body (auto-generated)."""
    s = summary
    lines: list[str] = []
    a = lines.append
    a("# Dataset Audit\n")
    a(f"_Auto-generated by `scripts/audit_dataset.py` on {generated_at}. Source folder: `{data_dir_name}` (read-only)._\n")

    a("## 1. Inventory and label interpretation\n")
    a("| Folder | Normalized label | Files | Used for supervised training? |")
    a("|---|---|---:|---|")
    meaning = {
        "approved": ("approved", "Yes (positive, final decision)"),
        "not_approved": ("not_approved", "Yes (negative, final decision; folder `disapproved`)"),
        "pending": ("pending", "**No** - unresolved; reserved for later manual validation"),
    }
    for lab, n in s["files_per_label"].items():
        _, use = meaning.get(lab, (lab, "?"))
        a(f"| {lab} | {lab} | {n} | {use} |")
    a(f"\nTotal files inventoried: **{s['total_files']}**.\n")
    a("Label meaning is taken from the folder names provided by the client. The `disapproved` folder is "
      "interpreted as a finalized rejection (`not_approved`). Whether it reflects rule violations versus "
      "other reasons (quality, duplicates, administrative) is **not documented** and is an open question.\n")

    a("### Media types\n")
    for lab, kinds in s["media_kind_per_label"].items():
        a(f"- {lab}: " + ", ".join(f"{k}={v}" for k, v in kinds.items()))
    a("")

    a("## 2. Readability and container formats\n")
    a("| Label | True container format (file header) | Extension disagrees with content |\n|---|---|---:|")
    for lab, fm in s["container_formats"].items():
        a(f"| {lab} | " + ", ".join(f"{k}={v}" for k, v in fm.items()) + f" | {s['extension_mismatch'].get(lab, 0)} |")
    a("\nHEIC/HEIF (iPhone) files saved as `.jpg` and MPO (a JPEG variant) are valid photos, not corruption; they are decoded with "
      "`pillow-heif`/Pillow and converted to JPEG for model input. **They occur only under `approved`** (plus a few PNG), so the raw "
      "container format must never reach a model; only decoded pixels are used.\n")
    if s["unreadable"]:
        a(f"**{len(s['unreadable'])} image file(s) failed to decode:**\n")
        for u in s["unreadable"][:50]:
            a(f"- `{u['rel_path']}`: {u['error']}")
    else:
        a("All image files decoded successfully.")
    a("")

    a("## 3. Exclusions from supervised training\n")
    a("| Reason | Files |\n|---|---:|")
    for k, v in s["exclusions"].items():
        a(f"| {k if k else '(eligible)'} | {v} |")
    a("\nEligible per label: " + ", ".join(f"{k}={v}" for k, v in s["eligible_per_label"].items()) + "\n")

    a("## 4. Programs (filename prefix) vs. label\n")
    a("The *original* filename prefix (`<program>-<n> (k).jpg`) identifies the NGO program/activity. It is **not** a label. "
      "Files have since been renamed to `<class>_<n>.<ext>`; the program and submission index are recovered from the original names "
      "recorded in `data/manifests/rename_map.csv`.\n")
    a("| Program | approved | not_approved |\n|---|---:|---:|")
    for prog, d in sorted(s["program_by_label"].items(), key=lambda kv: -sum(kv[1].values())):
        a(f"| {prog} | {d.get('approved', 0)} | {d.get('not_approved', 0)} |")
    if s["programs_in_single_label"]:
        a("\n**Confound warning:** these programs occur under only one final label: "
          + ", ".join(f"`{p}`" for p in s["programs_in_single_label"])
          + ". A model could learn the program rather than rule compliance.\n")
    a(f"Files whose names do not match the `<program>-<n>` pattern (final labels): {s['unparsed_filenames']}.\n")

    a("## 5. Image dimensions and file size (readable images)\n")
    for lab, d in s["dimensions"].items():
        a(f"**{lab}**: width p5/p50/p95 = {d['width'].get('p5')}/{d['width'].get('p50')}/{d['width'].get('p95')}; "
          f"height p5/p50/p95 = {d['height'].get('p5')}/{d['height'].get('p50')}/{d['height'].get('p95')}; "
          f"median {d['megapixels'].get('p50')} MP; file size MB p50/p95/max = "
          f"{d['file_size_mb'].get('p50')}/{d['file_size_mb'].get('p95')}/{d['file_size_mb'].get('p100')}\n")
    for lab, o in s["orientation_share"].items():
        a(f"- {lab}: portrait {o['portrait']:.0%}, landscape {o['landscape']:.0%}")
    a("")

    pc = s.get("provenance_confound", {})
    if pc:
        a("### Resolution / file-size confound (important)\n")
        a("Share of images >= 4 MP: " + ", ".join(f"{k}={v:.0%}" for k, v in pc["share_ge_4mp"].items()) + ".")
        if "single_feature_auc_approved" in pc:
            f = pc["single_feature_auc_approved"]
            a(f"Using **only** file size to rank images gives ROC-AUC {f['file_size']} (megapixels: {f['megapixels']}) for `approved` vs `not_approved`.")
        a("\nApproved images are mostly full-resolution camera originals while many not_approved images are small/compressed copies. "
          "Image resolution, file size and compression history are therefore **provenance proxies for the label**, not evidence of rule "
          "compliance. Do not use them as features, and treat any model (including a tuned VLM, which sees pixels) that scores well "
          "partly for this reason with suspicion. Ask the client whether production images will arrive as camera originals or compressed copies.\n")

    a("## 6. Image-quality indicators (diagnostic only)\n")
    a("Computed on a copy decoded at <=1024 px (draft mode) with OpenCV. Flags use thresholds from `configs/project.yaml`.\n")
    a("| Label | blurry | dark | bright | low contrast | small (<480 px) |\n|---|---:|---:|---:|---:|---:|")
    for lab, r in s["quality_flag_rates"].items():
        a(f"| {lab} | {r['flag_blurry']:.1%} | {r['flag_dark']:.1%} | {r['flag_bright']:.1%} | "
          f"{r['flag_low_contrast']:.1%} | {r['flag_small']:.1%} |")
    a("")

    a("## 7. Duplicates and potential label inconsistencies\n")
    d = s["duplicates"]
    if d.get("clusters"):
        a(f"- Duplicate clusters (exact + near, dHash): **{d['clusters']}** covering {d['images_in_clusters']} images "
          f"(exact: {d['exact_duplicate_images']}, near: {d['near_duplicate_images']}); largest cluster {d['largest_cluster']}.")
        a(f"- Clusters containing BOTH final labels (potential label inconsistency): **{d['clusters_with_label_conflict']}**; "
          f"of which byte-identical conflicts: **{d['clusters_with_exact_label_conflict']}**.")
    else:
        a("No duplicate clusters found.")
    a(f"- Names like `program-n` that occur under both final labels: {s['program_submission_keys_in_both_labels']} of "
      f"{s['program_submission_keys_total']}. These are treated as *different* submissions (the label is part of the group key).")
    a("- Near-duplicates are grouped (never deleted). Byte-identical conflicting copies are excluded from supervised data "
      "(`label_conflict_exclusion` in `configs/project.yaml`) and listed in `data/manifests/duplicate_report.csv` for human review.\n")

    a("## 8. Grouping information\n")
    g = s["filename_groups"]
    a(f"- No explicit event IDs exist. Filenames give a *submission index* per program: {g['groups']} distinct groups, "
      f"median size {g['median_group_size']}, max {g['max_group_size']}. These are **assumed** to correspond to one submission/event; "
      "this assumption is unverified.")
    a("- EXIF capture time availability: " + ", ".join(f"{k}={v:.0%}" for k, v in s["exif_datetime_available"].items())
      + " (most files appear to be stripped/re-encoded, so burst grouping by time is unreliable).")
    a("- Splits therefore group by: filename group + exact duplicates + near duplicates (union-find).\n")

    a("## 9. Observations from visual spot-checks (unverified, for the client to confirm)\n")
    a("- `not_approved` samples include handwritten bills/invoices, spreadsheet screenshots and multi-photo collages in addition to ordinary activity photos.")
    a("- `approved` samples include banners (e.g. program-name banners), product/kit photos, and scenes without people (e.g. bird feeding).")
    a("- Together this suggests rules may depend on program and that non-photo documents are rejected. **No rule has been inferred from this**; see `configs/rules.yaml`.\n")
    return "\n".join(lines)
