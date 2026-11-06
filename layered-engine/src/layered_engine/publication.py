"""Layer 4: publication readiness.

Consent, rights, caption and event facts cannot be seen in a photograph, so they come from the metadata only. Gemini can raise
visual flags; it can never verify anything. Missing metadata is PENDING: never VERIFIED and never a violation.
"""

from __future__ import annotations

from typing import Any, Mapping

from .rulebook import ResolvedProfile
from .schemas import CONFIDENCE_RANK, Confidence, ModelAssessment, PublicationCheck, PublicationResult

VALID = {"VERIFIED", "PENDING", "BLOCKED", "NOT_REQUIRED"}


def _meta_status(metadata: Mapping[str, Any], key: str) -> str:
    v = metadata.get(key)
    if v is None:
        return "PENDING"
    v = str(v).upper()
    return v if v in VALID else "PENDING"


def evaluate_publication(assessment: ModelAssessment, metadata: Mapping[str, Any], profile: ResolvedProfile) -> PublicationResult:
    cfg = profile.publication
    flags = assessment.layer_4_visual_flags
    flag_types = {f.flag_type for f in flags}
    res = PublicationResult(status="READY")

    for chk in cfg["checks"]:
        if chk["approval_status"] not in profile.settings["enabled_approval_statuses"]:
            continue
        req = chk["required"]
        required = (req == "always" or (req.startswith("when_flag_") and req[len("when_flag_"):] in flag_types)
                    or (req == "when_metadata_event_claims_asserted" and bool(metadata.get("event_claims_asserted"))))
        if not required:
            status = _meta_status(metadata, chk["metadata_key"]) if metadata.get(chk["metadata_key"]) is not None else "NOT_REQUIRED"
            if status == "BLOCKED":
                res.blockers.append(f"{chk['id']} {chk['label']}: BLOCKED")
            res.checks.append(PublicationCheck(check_id=chk["id"], label=chk["label"], status=status))
            continue
        status = _meta_status(metadata, chk["metadata_key"])
        res.checks.append(PublicationCheck(check_id=chk["id"], label=chk["label"], status=status,
                                           detail="no verified information supplied" if metadata.get(chk["metadata_key"]) is None else ""))
        if status == "BLOCKED":
            res.blockers.append(f"{chk['id']} {chk['label']}: BLOCKED")
        elif status == "PENDING":
            res.pending.append(f"{chk['id']} {chk['label']}")

    min_conf = Confidence(cfg["min_confidence_to_block"])
    for f in flags:
        rule = cfg["visual_flags"].get(f.flag_type) or cfg["visual_flags"]["OTHER"]
        effect = rule["effect"]
        label = f"{rule.get('criterion') or 'flag'} {f.flag_type}: {f.evidence or 'no evidence given'}"
        if effect == "BLOCKED":
            if CONFIDENCE_RANK[f.confidence] >= CONFIDENCE_RANK[min_conf]:
                res.blockers.append(label)
            else:
                res.review_notes.append(f"low-confidence flag, needs a look: {label}")
                res.pending.append(f"{label} (low confidence)")
        elif effect == "REQUIRES_VERIFICATION":
            res.review_notes.append(f"{label}: needs verification (this is a flag, not a statement about age)")
        else:
            res.review_notes.append(label)

    # P005: platform dimensions (only when a platform AND the image size are known)
    platform = metadata.get("platform")
    if platform and platform in cfg.get("platforms", {}) and metadata.get("width") and metadata.get("height"):
        need = cfg["platforms"][platform]
        ok = metadata["width"] >= need["min_width"] and metadata["height"] >= need["min_height"]
        res.checks.append(PublicationCheck(check_id="P005", label="Meets platform dimensions", status="VERIFIED" if ok else "BLOCKED",
                                           detail=f"{metadata['width']}x{metadata['height']} vs minimum {need['min_width']}x{need['min_height']} ({platform})"))
        if not ok:
            res.blockers.append(f"P005 image is smaller than the {platform} minimum")
    else:
        res.checks.append(PublicationCheck(check_id="P005", label="Meets platform dimensions", status="NOT_ASSESSED", detail="no platform or image size supplied"))
    for nid in cfg.get("not_assessed", []):
        res.checks.append(PublicationCheck(check_id=nid, label="not assessed in this phase", status="NOT_ASSESSED"))

    res.status = "BLOCKED" if res.blockers else ("PENDING_VERIFICATION" if res.pending else "READY")
    return res
