"""Build and validate Gemini supervised-tuning JSONL files from the saved split manifests.

JSONL schema (official docs, checked 2026-10-29):
  {"systemInstruction": {"role": ..., "parts": [{"text": ...}]},
   "contents": [{"role": "user",  "parts": [{"fileData": {"mimeType": ..., "fileUri": "gs://..."}},
                                             {"text": ...}]},
                {"role": "model", "parts": [{"text": ...}]}],
   "generationConfig": {"mediaResolution": "MEDIA_RESOLUTION_HIGH"}}

Two tracks (kept strictly separate):
  A  overall decision, trained from image-level human labels (approved / not_approved).
  B  rule-level checks, trained ONLY from human-verified rule annotations.

Nothing is uploaded here. Records reference planned gs:// URIs; ``upload_plan.csv`` lists the
local-file -> URI mapping for the (separately authorized) upload step.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ModelConfig, ProjectConfig
from ..data.preprocessing import prepare_for_tuned_model
from ..data.splitting import verify_no_leakage, verify_no_pending
from ..exceptions import DataError, InvalidImageError, TuningError
from ..schemas import RuleSet

logger = logging.getLogger(__name__)

TRACK_A_SYSTEM = (
    "You are the NGO's photo-approval model. Decide whether the photograph was approved or not "
    "approved for publication on the NGO's social media, following the NGO's approval policy. "
    "Answer only with the requested JSON object."
)
TRACK_A_PROMPT = (
    "Was this photograph approved for publication? "
    'Answer with JSON: {"decision": "approved"} or {"decision": "not_approved"}.'
)
TRACK_A_SCHEMA = {
    "type": "object",
    "properties": {"decision": {"type": "string", "enum": ["approved", "not_approved"]}},
    "required": ["decision"],
}

# label -> exact training target (documented in docs/TRAINING_GUIDE.md)
TRACK_A_LABEL_TO_TARGET = {
    "approved": '{"decision": "approved"}',
    "not_approved": '{"decision": "not_approved"}',
}

PLACEHOLDER_BUCKET = "<UNCONFIGURED-BUCKET>"


@dataclass
class ExportReport:
    track: str
    dry_run: bool
    uris_are_placeholders: bool
    bucket: str
    counts: dict[str, dict[str, int]] = field(default_factory=dict)  # split -> label -> n
    skipped: list[dict[str, str]] = field(default_factory=list)
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    derived_images: int = 0
    upload_plan_rows: int = 0
    balance: str = "none"
    image_normalization: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, default=str)


# --------------------------------------------------------------------------- record construction


def make_record(
    gcs_uri: str, mime: str, user_text: str, target_text: str, system_text: str, media_resolution: str
) -> dict[str, Any]:
    return {
        "systemInstruction": {"role": "system", "parts": [{"text": system_text}]},
        "contents": [
            {"role": "user", "parts": [{"fileData": {"mimeType": mime, "fileUri": gcs_uri}}, {"text": user_text}]},
            {"role": "model", "parts": [{"text": target_text}]},
        ],
        "generationConfig": {"mediaResolution": media_resolution},
    }


def validate_record(rec: dict[str, Any], model_cfg: ModelConfig, track: str,
                    rule_ids: list[str] | None = None, allow_placeholder: bool = True) -> list[str]:
    """Return a list of problems (empty == valid). Pure structural check, no I/O."""
    lim = model_cfg.tuning.limits
    problems: list[str] = []
    contents = rec.get("contents")
    if not isinstance(contents, list) or len(contents) < 2:
        return ["contents must be a list with a user turn and a model turn"]
    roles = [c.get("role") for c in contents]
    if roles[0] != "user" or roles[-1] != "model":
        problems.append(f"turn roles must start with user and end with model, got {roles}")
    n_images = 0
    for c in contents:
        parts = c.get("parts")
        if not isinstance(parts, list) or not parts:
            problems.append("turn without parts")
            continue
        for p in parts:
            keys = set(p)
            if keys == {"text"}:
                if not isinstance(p["text"], str) or not p["text"].strip():
                    problems.append("empty text part")
            elif keys == {"fileData"}:
                fd = p["fileData"]
                n_images += 1
                uri, mime = fd.get("fileUri", ""), fd.get("mimeType", "")
                if not uri.startswith("gs://"):
                    problems.append("fileUri must be a gs:// URI")
                elif not allow_placeholder and PLACEHOLDER_BUCKET in uri:
                    problems.append("fileUri uses a placeholder bucket")
                if mime not in lim.allowed_mime_types:
                    problems.append(f"mimeType {mime!r} not allowed")
            else:
                problems.append(f"part must contain exactly one of text/fileData, got {sorted(keys)}")
    if n_images > lim.max_images_per_example:
        problems.append(f"{n_images} images exceed the {lim.max_images_per_example} per-example limit")
    if n_images < 1:
        problems.append("no image in example")
    si = rec.get("systemInstruction")
    if si is not None and not (isinstance(si.get("parts"), list) and all("text" in p for p in si["parts"])):
        problems.append("systemInstruction must contain text parts only")
    # Target must be valid JSON of the right shape.
    try:
        target = json.loads(contents[-1]["parts"][0]["text"])
    except (KeyError, IndexError, json.JSONDecodeError, TypeError):
        problems.append("model turn is not valid JSON")
        return problems
    if track == "A":
        if target.get("decision") not in TRACK_A_LABEL_TO_TARGET:
            problems.append("track A target must be a decision label")
    else:
        got = [c.get("rule_id") for c in target.get("checks", [])]
        if rule_ids is not None and sorted(got) != sorted(rule_ids):
            problems.append("track B target must contain every active rule exactly once")
        if any(c.get("status") not in {"PASS", "FAIL", "UNCERTAIN"} for c in target.get("checks", [])):
            problems.append("track B target has an invalid status")
    approx_tokens = sum(len(p.get("text", "")) for c in contents for p in c["parts"]) // 3 + 1300 * max(n_images, 1)
    if approx_tokens > lim.max_tokens_per_example:
        problems.append("estimated tokens exceed the per-example limit")
    return problems


# --------------------------------------------------------------------------- image preparation


def _prepare(row: dict[str, Any], data_dir: Path, derived_dir: Path, model_cfg: ModelConfig) -> dict[str, Any]:
    """Validate the image and return upload info; writes a derived copy only if conversion is needed."""
    src = data_dir / row["rel_path"]
    try:
        prep = prepare_for_tuned_model(src, model_cfg.image_input, model_cfg.tuning.limits.allowed_mime_types)
    except InvalidImageError as exc:
        return {"image_id": row["image_id"], "error": str(exc)}
    if prep.original_bytes > model_cfg.tuning.limits.max_image_bytes and not prep.resized:
        return {"image_id": row["image_id"], "error": "image exceeds the provider per-image size limit"}
    ext = ".jpg" if prep.mime_type == "image/jpeg" else (".png" if prep.mime_type == "image/png" else ".webp")
    if prep.resized or prep.mime_type != _guess_orig_mime(src):
        derived_dir.mkdir(parents=True, exist_ok=True)
        local = derived_dir / f"{row['image_id']}{ext}"
        if not local.exists() or local.stat().st_size != len(prep.data):
            local.write_bytes(prep.data)
        derived = True
    else:
        local, derived = src, False
    return {"image_id": row["image_id"], "local_path": str(local), "mime": prep.mime_type, "ext": ext,
            "derived": derived, "bytes": len(prep.data), "width": prep.width, "height": prep.height}


def _guess_orig_mime(src: Path) -> str:
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(
        src.suffix.lower(), "application/octet-stream")


# --------------------------------------------------------------------------- main entry point


def load_verified_annotations(path: Path, rule_set: RuleSet) -> pd.DataFrame:
    """Rule annotations usable as Track B ground truth: only rows with verified == true."""
    if not path.exists():
        raise DataError(
            f"Track B needs human-verified rule annotations; {path} not found. "
            "Fill data/annotations/rule_annotation_template.csv (scripts/annotate_rules.py) first."
        )
    df = pd.read_csv(path)
    need = {"image_id", "rule_id", "status", "verified"}
    if not need <= set(df.columns):
        raise DataError(f"annotation file must contain columns {sorted(need)}")
    df["verified"] = df["verified"].astype(str).str.lower().isin({"true", "1", "yes"})
    df = df[df["verified"]].copy()
    df["status"] = df["status"].astype(str).str.upper()
    bad = df[~df["status"].isin({"PASS", "FAIL", "UNCERTAIN"})]
    if len(bad):
        raise DataError(f"{len(bad)} verified annotation rows have an invalid status")
    unknown = set(df["rule_id"]) - {r.rule_id for r in rule_set.rules}
    if unknown:
        raise DataError(f"annotations reference unknown rule ids: {sorted(unknown)}")
    if "evidence" not in df:
        df["evidence"] = ""
    df["evidence"] = df["evidence"].fillna("")
    return df


def downsample_majority(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Seeded downsampling of the majority class (train only) to a 1:1 ratio."""
    counts = df["label"].value_counts()
    n = int(counts.min())
    parts = [g.sample(n=n, random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sort_values("image_id").reset_index(drop=True)


def build_tuning_files(
    project: ProjectConfig,
    model_cfg: ModelConfig,
    rule_set: RuleSet,
    track: str = "A",
    out_dir: Path | None = None,
    derived_dir: Path | None = None,
    balance: str = "none",
    seed: int = 42,
    annotations_path: Path | None = None,
    workers: int = 8,
    dry_run: bool = True,
) -> ExportReport:
    """Create train/validation JSONL + upload plan + report. Never uploads, never reads test labels."""
    if track not in {"A", "B"}:
        raise TuningError("track must be 'A' or 'B'")
    split_dir = project.output_dir / "splits"
    try:
        frames = {n: pd.read_csv(split_dir / f"{n}.csv") for n in ("train", "validation", "test")}
    except FileNotFoundError as exc:
        raise DataError("split manifests missing; run scripts/prepare_dataset.py") from exc
    verify_no_leakage(frames)
    verify_no_pending(frames)
    train, val = frames["train"], frames["validation"]
    test_ids = set(frames["test"]["image_id"])  # used only to prove no test image is exported

    out_dir = out_dir or project.output_dir / "tuning"
    norm = model_cfg.image_input.tuning_normalize_long_side_px
    derived_dir = derived_dir or project.output_dir / "derived" / (f"tuning_images_norm{norm}" if norm else "tuning_images")
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket = project.cloud.gcs_bucket or PLACEHOLDER_BUCKET
    prefix = project.cloud.gcs_prefix.strip("/")
    report = ExportReport(track=track, dry_run=dry_run, uris_are_placeholders=bucket == PLACEHOLDER_BUCKET,
                          bucket=bucket if bucket != PLACEHOLDER_BUCKET else "(not configured)", balance=balance,
                          image_normalization=(f"long side {norm}px, JPEG q{model_cfg.image_input.tuning_jpeg_quality}" if norm else "none"))
    if report.uris_are_placeholders:
        report.warnings.append("NGO_GCS_BUCKET is not configured: JSONL uses placeholder URIs and cannot be submitted.")

    if balance == "downsample":
        train = downsample_majority(train, seed)
        report.warnings.append("train set majority class downsampled 1:1 (validation untouched)")
    elif balance != "none":
        raise TuningError(f"unknown balance mode {balance!r}")

    annotations = None
    rules = rule_set.active()
    rule_ids = [r.rule_id for r in rules]
    if track == "B":
        annotations = load_verified_annotations(
            annotations_path or project.output_dir / "annotations" / "rule_annotations.csv", rule_set)

    media_res = model_cfg.generation.media_resolution
    upload_rows: list[dict[str, Any]] = []
    seen_uploads: set[str] = set()

    for name, frame in (("train", train), ("validation", val)):
        if set(frame["image_id"]) & test_ids:
            raise DataError(f"test image present in {name}")
        records = frame.to_dict("records")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            prepared = list(ex.map(lambda r: _prepare(r, project.data_dir, derived_dir, model_cfg), records))
        lines: list[str] = []
        counts: dict[str, int] = {}
        for row, prep in zip(records, prepared, strict=True):
            iid = row["image_id"]
            if "error" in prep:
                report.skipped.append({"split": name, "image_id": iid, "reason": prep["error"]})
                continue
            if track == "A":
                target = TRACK_A_LABEL_TO_TARGET[row["label"]]
                user_text, system_text = TRACK_A_PROMPT, TRACK_A_SYSTEM
            else:
                from ..models.baseline_analyzer import SYSTEM_INSTRUCTION, build_prompt

                ann = annotations[annotations["image_id"] == iid]
                if sorted(ann["rule_id"]) != sorted(rule_ids):
                    report.skipped.append({"split": name, "image_id": iid, "reason":
                                           "missing verified annotation for one or more active rules"})
                    continue
                target = json.dumps({"checks": [
                    {"rule_id": a.rule_id, "status": a.status, "evidence": a.evidence}
                    for a in ann.sort_values("rule_id").itertuples()]})
                user_text, system_text = build_prompt(rules), SYSTEM_INSTRUCTION
            uri = f"gs://{bucket}/{prefix}/images/{iid}{prep['ext']}"
            rec = make_record(uri, prep["mime"], user_text, target, system_text, media_res)
            problems = validate_record(rec, model_cfg, track, rule_ids if track == "B" else None)
            if problems:
                report.skipped.append({"split": name, "image_id": iid, "reason": "; ".join(problems)})
                continue
            lines.append(json.dumps(rec, ensure_ascii=False))
            counts[row["label"]] = counts.get(row["label"], 0) + 1
            if iid not in seen_uploads:
                seen_uploads.add(iid)
                upload_rows.append({"image_id": iid, "local_path": prep["local_path"], "gcs_uri": uri,
                                    "mime_type": prep["mime"], "bytes": prep["bytes"], "derived_copy": prep["derived"]})
            report.derived_images += int(prep["derived"])
        path = out_dir / f"{name}.jsonl"
        payload = "\n".join(lines) + ("\n" if lines else "")
        path.write_text(payload, encoding="utf-8")
        size = path.stat().st_size
        report.files[name] = {"path": path.name, "examples": len(lines), "bytes": size}
        report.counts[name] = counts
        if size > model_cfg.tuning.limits.max_training_file_bytes:
            raise TuningError(f"{name}.jsonl exceeds the {model_cfg.tuning.limits.max_training_file_bytes} byte limit")
        if name == "train" and len(lines) < model_cfg.tuning.limits.recommended_min_examples:
            report.warnings.append(
                f"only {len(lines)} training examples; docs recommend starting with >= "
                f"{model_cfg.tuning.limits.recommended_min_examples}")

    report.upload_plan_rows = len(upload_rows)
    pd.DataFrame(upload_rows).to_csv(out_dir / "upload_plan.csv", index=False)
    pd.DataFrame(report.skipped, columns=["split", "image_id", "reason"]).to_csv(out_dir / "skipped_examples.csv", index=False)
    (out_dir / "export_report.json").write_text(report.to_json(), encoding="utf-8")
    return report


def validate_jsonl_file(path: Path, model_cfg: ModelConfig, track: str, rule_ids: list[str] | None = None,
                        allow_placeholder: bool = True) -> dict[str, Any]:
    """Re-validate an exported file line by line (used by preflight before any upload/submit)."""
    n, bad = 0, []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        n += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            bad.append((i, ["invalid JSON line"]))
            continue
        probs = validate_record(rec, model_cfg, track, rule_ids, allow_placeholder)
        if probs:
            bad.append((i, probs))
    return {"path": path.name, "examples": n, "invalid": len(bad), "first_problems": bad[:5]}


__all__ = [
    "TRACK_A_PROMPT", "TRACK_A_SCHEMA", "TRACK_A_SYSTEM", "TRACK_A_LABEL_TO_TARGET", "ExportReport",
    "build_tuning_files", "validate_record", "validate_jsonl_file", "make_record",
    "load_verified_annotations", "downsample_majority",
]
