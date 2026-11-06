"""Move an organisation to another policy profile (for example layered_tuned_v1, the owner's tuned banner rules) as new, approved, versioned rulebook and prompt versions.

The earlier versions are kept untouched and can be activated again, so the move can be undone. What the organisation had learned is carried over where it still applies:
rewritten rules and notes stay for every rule and gate the new profile still has; a rewrite of a rule the new profile no longer has is withdrawn, with the reason recorded (it could
no longer reach the prompt); learned settings (the overlay) are carried over only if the new profile can still apply them, otherwise the move stops before changing anything.
"""

from __future__ import annotations

from typing import Any

from layered_engine.rulebook import Rulebook
from sqlalchemy.orm import Session

from . import rule_service as rs
from . import versions as vs
from .enums import VersionApproval
from .errors import Conflict, Invalid
from .models import Organisation, PromptVersion, RulebookVersion, User
from .rule_drafter import rule_catalogue


def adopt_profile(s: Session, org: Organisation, profile: str, rb: Rulebook, admin: User, reason: str) -> tuple[RulebookVersion, PromptVersion, list[str]]:
    """Returns (the new rulebook version, the new prompt version, the targets whose rewrites were withdrawn). Raises Invalid or Conflict before changing anything."""
    if profile not in rb.profiles:
        raise Invalid(f"unknown profile {profile!r}")
    cur = s.get(RulebookVersion, org.active_rulebook_version_id)
    if cur.configuration.get("profile") == profile:
        raise Conflict(f"the organisation already uses {profile}")
    overlay = cur.configuration.get("overlay") or []
    try:
        rb.resolve(profile, without=cur.configuration.get("without", []), overlay=overlay)
    except ValueError as exc:
        raise Conflict(f"the learned settings of this organisation cannot be applied to {profile} ({exc}); roll them back first") from exc
    prof = rb.resolve(profile)
    conf: dict[str, Any] = {**cur.configuration, "profile": profile, "rulebook_version": prof.rulebook_version, "approval": {"by": admin.email, "reason": reason}}
    conf.pop("bootstrap_approval", None)
    number = f"{prof.rulebook_version}:{profile}"
    rv = vs.create_version(s, "rulebook", org.organisation_id, number, conf, f"moved from {cur.configuration.get('profile')} to {profile}: {reason}", admin.user_id, VersionApproval.APPROVED.value)
    rv.approved_by = admin.user_id
    vs.activate_version(s, "rulebook", rv)
    catalogue = rule_catalogue(rb, profile)
    dropped = []
    for target, d in list(rs.active_edits(s, org.organisation_id).items()):
        if target not in catalogue:
            rs.withdraw(s, org, d, f"withdrawn automatically: the rule {target} does not exist in the profile {profile}", rb, admin, {"by": "a profile change"})
            dropped.append(target)
    pv = rs._set_rules(s, org, {t: x.new_text for t, x in rs.active_edits(s, org.organisation_id).items()}, rb, f"prompt of the {profile} profile")   # keeps the notes of the active prompt
    return rv, pv, dropped
