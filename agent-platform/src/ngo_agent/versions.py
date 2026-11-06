"""Rulebook and prompt versions: registration, approval, activation. Every evaluation points at exact versions so it can be reproduced."""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from layered_engine.prompt_builder import build_prompt
from layered_engine.rulebook import RULEBOOK_DIR, Rulebook

from .enums import VersionApproval
from .errors import Conflict, Forbidden, NotFound
from .models import Organisation, PromptVersion, RulebookVersion, User, now

V1_PROFILE, V2_PROFILE = "layered_proposed_v1", "layered_proposed_v2"


def config_digest() -> str:
    """SHA-256 over the rulebook YAML files: changes whenever any rule, condition, exception or profile changes."""
    h = hashlib.sha256()
    for p in sorted(RULEBOOK_DIR.glob("*.yaml")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _model(kind: str):  # noqa: ANN202
    return RulebookVersion if kind == "rulebook" else PromptVersion


def create_version(s: Session, kind: str, organisation_id: str, version_number: str, configuration: dict[str, Any], change_description: str,
                   created_by: Optional[str], approval: str = VersionApproval.DRAFT.value) -> Any:
    M = _model(kind)
    if s.scalar(select(M).where(M.organisation_id == organisation_id, M.version_number == version_number)):
        raise Conflict(f"{kind} version {version_number!r} already exists")
    v = M(organisation_id=organisation_id, version_number=version_number, configuration=configuration, change_description=change_description,
          created_by=created_by, approval_status=approval)
    s.add(v)
    s.flush()
    return v


def get_version(s: Session, kind: str, organisation_id: str, version_id: str) -> Any:
    M = _model(kind)
    v = s.scalar(select(M).where(M.version_id == version_id, M.organisation_id == organisation_id))
    if v is None:
        raise NotFound("version not found")
    return v


def approve_version(s: Session, kind: str, version: Any, approver: User) -> Any:
    """A version needs a different person to approve it than the one who created it."""
    if approver.organisation_id != version.organisation_id:
        raise NotFound("version not found")
    if version.approval_status == VersionApproval.APPROVED.value:
        return version
    if version.created_by and version.created_by == approver.user_id:
        raise Forbidden("a version must be approved by someone other than its creator")
    version.approval_status = VersionApproval.APPROVED.value
    version.approved_by = approver.user_id
    version.effective_at = None
    s.flush()
    return version


def activate_version(s: Session, kind: str, version: Any) -> Organisation:
    if version.approval_status != VersionApproval.APPROVED.value:
        raise Forbidden(f"only an APPROVED {kind} version can be activated (this one is {version.approval_status})")
    org = s.get(Organisation, version.organisation_id)
    if kind == "rulebook":
        org.active_rulebook_version_id = version.version_id
    else:
        org.active_prompt_version_id = version.version_id
    org.updated_at = now()
    version.effective_at = now()
    s.flush()
    return org


def register_engine_versions(s: Session, organisation_id: str, created_by: Optional[str], rb: Optional[Rulebook] = None, activate_v1: bool = True) -> dict[str, Any]:
    """Register the layered engine's V1 (approved and active) and V2 (DRAFT, cannot be activated until an admin approves it) for an organisation.

    V1 is the evaluator the NGO has been shown. V2 stays a draft because five of its seven changes conflict with NGO decisions of 2026-11-02.
    The bootstrap approval is recorded as such in the configuration.
    """
    rb = rb or Rulebook()
    digest = config_digest()
    out: dict[str, Any] = {}
    for name, status in ((V1_PROFILE, VersionApproval.APPROVED.value), (V2_PROFILE, VersionApproval.DRAFT.value)):
        prof = rb.resolve(name)
        bp = build_prompt(rb, prof)
        existing = s.scalar(select(RulebookVersion).where(RulebookVersion.organisation_id == organisation_id, RulebookVersion.version_number == f"{prof.rulebook_version}:{name}"))
        if existing:
            out[name] = (existing, s.scalar(select(PromptVersion).where(PromptVersion.organisation_id == organisation_id, PromptVersion.version_number == f"{bp.template_version}:{bp.hash}")))
            continue
        rv = create_version(s, "rulebook", organisation_id, f"{prof.rulebook_version}:{name}",
                            {"profile": name, "without": [], "rulebook_version": prof.rulebook_version, "config_digest": digest, "bootstrap_approval": status == "APPROVED"},
                            "layered engine " + ("V1: the evaluator as piloted" if name == V1_PROFILE else "V2: proposed, pending NGO approval (see layered-engine/docs/V1_V2_COMPARISON.md)"),
                            created_by, status)
        pv = create_version(s, "prompt", organisation_id, f"{bp.template_version}:{bp.hash}",
                            {"profile": name, "template_version": bp.template_version, "prompt_hash": bp.hash, "criteria_asked": len(bp.criterion_ids)},
                            f"prompt of the {name} profile", created_by, status)
        if status == VersionApproval.APPROVED.value:
            rv.approved_by = pv.approved_by = created_by
        out[name] = (rv, pv)
    if activate_v1:
        org = s.get(Organisation, organisation_id)
        rv, pv = out[V1_PROFILE]
        if org.active_rulebook_version_id is None:
            activate_version(s, "rulebook", rv)
        if org.active_prompt_version_id is None:
            activate_version(s, "prompt", pv)
    return out
