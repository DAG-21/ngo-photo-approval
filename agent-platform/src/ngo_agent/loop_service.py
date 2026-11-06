"""L4, the learning loop: every few decided photographs the agent looks for patterns in what staff decided, drafts a change to the SETTINGS, tests it on the
stored decisions, applies it if it passes, watches the next photographs, and undoes it if it does worse. No approval is asked for each change; safety is automatic.

What it may change (the engine's overlay, layered_engine.rulebook.OVERLAY_KINDS): the quality thresholds, and whether a gate rejects by itself or only lowers
the quality score. It cannot touch the wording of a rule, the questions the model is asked, publication blockers, or which rules exist. No model is called:
every test is a replay of answers the model already gave, which is exact for settings (the answers do not depend on them).

The safety rules (each has a test):
  * a change that would approve even one photograph staff rejected is BLOCKED;
  * a change must reduce errors on the held-out photographs (a fixed 40% the drafting never looked at) by at least `min_gain`, or it is BLOCKED;
  * a change that loosens anything (more approvals) needs enough staff-rejected photographs among the held-out ones to show it is safe, or it is HELD for
    developers, not applied: too little data means "not proven safe";
  * one change is applied per cycle, so a bad one can be traced to its cycle;
  * after a change is applied, each new photograph is also decided as the previous version would have decided it; one new false approval, or clearly more
    errors, rolls the change back by itself, and a rolled-back change is never drafted again;
  * decisions made over a publication blocker, for a policy-change reason or called a mistake by their own author are not used (they have no learning example);
    reasons too generic to teach anything are not used for drafting, but still count as a staff decision when testing.
Everything is organisation-scoped. A paused or disabled loop does nothing. The numbers are starting assumptions (see DEFAULTS).
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from layered_engine.decision_engine import evaluate_final_decision
from layered_engine.rulebook import Rulebook
from layered_engine.schemas import ModelAssessment, Recommendation
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from . import versions as vs
from .enums import ReviewStatus, VersionApproval
from .errors import Conflict, Invalid, NotFound
from .evaluation_service import canonical
from .models import ChangeEvent, ChangeRecord, ChangeTest, EvaluationRun, HumanReview, ImprovementCycle, LearningExample, Organisation, Photo, RulebookVersion, User, now

log = logging.getLogger("ngo_agent.loop")

DEFAULTS: dict[str, Any] = {
    "enabled": False,                 # an administrator turns the loop on, once, knowing that settings then change automatically
    "paused": False,                  # the developers' switch: nothing runs while it is on
    "every_n": 5,                     # a cycle after this many newly decided photographs
    "min_rows": 10,                   # fewer usable decisions than this: nothing is attempted
    "min_pattern": 3,                 # the same gap in at least this many photographs before a gate change is drafted
    "pattern_share": 0.8,             # and staff must have gone the same way in at least this share of them
    "holdout_pct": 40,                # the share of photographs kept out of drafting and used to test
    "min_gain": 1,                    # errors a change must remove on the held-out photographs
    "min_rejected_for_loosening": 20, # staff-rejected photographs needed among the held-out ones before a loosening change may be applied
    "monitor_min": 5,                 # photographs seen under a new version before it can be rolled back
    "monitor_window": 10,             # photographs after which it is confirmed
    "error_margin": 2,                # more errors than the previous version over the window than this roll a change back (a new false approval always does)
}
BOUNDS = {"every_n": (1, 1000), "min_rows": (2, 100000), "min_pattern": (2, 1000), "pattern_share": (0.5, 1.0), "holdout_pct": (10, 90), "min_gain": (1, 1000),
          "min_rejected_for_loosening": (0, 100000), "monitor_min": (1, 1000), "monitor_window": (1, 1000), "error_margin": (1, 1000)}
TERMINAL = {"ROLLED_BACK"}
REVIEW_GRID = range(40, 86, 5)
RESELECT_GRID = range(0, 86, 5)
MAX_DRAFT_ROWS = 400


# ------------------------------------------------------------------------------------------------------------------------------------ settings
def settings(org: Organisation) -> dict[str, Any]:
    raw = {**DEFAULTS, **((org.configuration or {}).get("learning_loop") or {})}
    return {k: raw[k] for k in DEFAULTS}


def save_settings(org: Organisation, changes: dict[str, Any], confirm_automatic_changes: bool = False) -> dict[str, Any]:
    cur = settings(org)
    new = {**cur, **{k: v for k, v in changes.items() if v is not None}}
    for k in ("enabled", "paused"):
        if not isinstance(new[k], bool):
            raise Invalid(f"{k} must be true or false")
    for k, (lo, hi) in BOUNDS.items():
        v = new[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi or (k != "pattern_share" and v != int(v)):
            raise Invalid(f"{k} must be {'a number' if k == 'pattern_share' else 'a whole number'} from {lo} to {hi}")
    if new["monitor_window"] < new["monitor_min"]:
        raise Invalid("monitor_window cannot be smaller than monitor_min")
    if new["enabled"] and not cur["enabled"] and not confirm_automatic_changes:
        raise Invalid("turning the learning loop on means the quality thresholds and which gates reject will change automatically, tested first and rolled back if worse, "
                      "with no approval for each change; set confirm_automatic_changes to true to turn it on (and tell the NGO)")
    org.configuration = {**(org.configuration or {}), "learning_loop": {k: new[k] for k in DEFAULTS}}
    org.updated_at = now()
    return settings(org)


# --------------------------------------------------------------------------------------------------------------------------------------- data
@dataclass
class Row:
    photo_id: str
    evaluation_id: str
    review_id: str
    assessment: dict[str, Any]
    meta: dict[str, Any]
    human_approved: bool
    version_id: str
    low_information: bool
    held_out: bool
    created_at: Any = None


def is_held_out(photo_id: str, pct: int) -> bool:
    """A photograph is always on the same side: a stable hash, so the held-out photographs stay the same from cycle to cycle."""
    return int(hashlib.sha256(photo_id.encode()).hexdigest()[:8], 16) % 100 < pct


def build_dataset(s: Session, organisation_id: str, profile_name: str, pct: int) -> list[Row]:
    """The usable staff decisions of one organisation: the latest learning example of each photograph (so decisions made for a policy-change reason, called a
    mistake, or approved over a publication blocker are not here), with the stored answers of the evaluation it was a decision about."""
    latest: dict[str, LearningExample] = {}
    for ex in s.scalars(select(LearningExample).where(LearningExample.organisation_id == organisation_id).order_by(LearningExample.created_at.desc())):
        latest.setdefault(ex.photo_id, ex)
    rows: list[Row] = []
    for ex in latest.values():
        review = s.get(HumanReview, ex.human_review_id)
        run = s.get(EvaluationRun, review.evaluation_id) if review else None
        if run is None or run.processing_status != "AI_EVALUATED" or not run.validated_response or run.evaluation_profile != profile_name:
            continue
        photo = s.get(Photo, ex.photo_id)
        rows.append(Row(ex.photo_id, run.evaluation_id, review.review_id, run.validated_response, {"width": photo.image_width, "height": photo.image_height},
                        ex.verified_decision == "APPROVE", run.rulebook_version_id, bool(ex.low_information), is_held_out(ex.photo_id, pct), ex.created_at))
    rows.sort(key=lambda r: (str(r.created_at), r.photo_id))
    return rows


def replay(rows: list[Row], rb: Rulebook, profile_name: str, without: list[str], overlay: list[dict[str, Any]]) -> dict[str, Any]:
    """The decision each stored answer would have had under the profile with this overlay. Exact for settings: the model's answers do not depend on them.
    A row that cannot be replayed is left out. Raises ValueError if the overlay does not apply to the profile."""
    prof = rb.resolve(profile_name, without=without, overlay=overlay)
    out: dict[str, Any] = {}
    for r in rows:
        try:
            out[r.photo_id] = evaluate_final_decision(ModelAssessment.model_validate(r.assessment), prof, r.meta)
        except ValueError as exc:     # pydantic's ValidationError is a ValueError: an old answer that no longer fits is skipped, not fatal
            log.warning("photograph %s was not replayed: %s", r.photo_id, type(exc).__name__)
    return out


def approved(decision: Any) -> bool:
    return decision.recommendation in (Recommendation.APPROVE, Recommendation.APPROVED_PENDING_PUBLICATION_CHECKS)


def confusion(rows: list[Row], decisions: dict[str, Any]) -> dict[str, int]:
    """Positive = approved. FP: the system approves a photograph staff rejected. FN: it does not approve one staff approved."""
    c = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for r in rows:
        d = decisions.get(r.photo_id)
        if d is None:
            continue
        c[("T" if approved(d) == r.human_approved else "F") + ("P" if approved(d) else "N")] += 1
    c["n"] = sum(c.values())
    c["errors"] = c["FP"] + c["FN"]
    return c


# ---------------------------------------------------------------------------------------------------------------------------------- drafting
@dataclass
class Candidate:
    kind: str
    target: Optional[str]
    set: dict[str, Any]
    loosening: bool
    rationale: str
    evidence: dict[str, Any] = field(default_factory=dict)
    previous: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        return hashlib.sha256(canonical({"kind": self.kind, "target": self.target, "set": self.set}).encode()).hexdigest()

    def record(self, label: str) -> dict[str, Any]:
        r: dict[str, Any] = {"change_id": label, "kind": self.kind, "set": self.set, "previous": self.previous}
        if self.target:
            r["target"] = self.target
        return r


def _tally(items: list[tuple[str, bool]]) -> tuple[int, int, list[str]]:
    return sum(1 for _, a in items if a), sum(1 for _, a in items if not a), [p for p, _ in items][:10]


def mine(rows: list[Row], decisions: dict[str, Any], rb: Rulebook, profile_name: str, without: list[str], overlay: list[dict[str, Any]], cfg: dict[str, Any]) -> list[Candidate]:
    """Draft changes from the DISCOVERY rows only. Pure counting and search over replays; no model."""
    out: list[Candidate] = []
    share, need = cfg["pattern_share"], cfg["min_pattern"]
    only_gate: dict[str, list[tuple[str, bool]]] = {}
    soft_no: dict[str, list[tuple[str, bool]]] = {}
    for r in rows:
        d = decisions.get(r.photo_id)
        if d is None:
            continue
        if len(d.failed_gates) == 1:
            only_gate.setdefault(d.failed_gates[0], []).append((r.photo_id, r.human_approved))
        for g in d.gates:
            if not g.hard and getattr(g.status, "value", g.status) == "NO":
                soft_no.setdefault(g.gate_id, []).append((r.photo_id, r.human_approved))
    for gate, items in sorted(only_gate.items()):                       # LOOSENING: a gate that alone rejected photographs staff then approved
        a, rj, ids = _tally(items)
        if a >= need and a / (a + rj) >= share:
            out.append(Candidate("GATE_SOFTEN", gate, {}, True, f"{gate} alone rejected {a + rj} photographs; staff approved {a} of them and rejected {rj}",
                                 {"rejected_by_gate_alone": a + rj, "staff_approved": a, "staff_rejected": rj, "photos": ids}))
    for gate, items in sorted(soft_no.items()):                         # TIGHTENING: a soft gate that failed on photographs staff then rejected
        ap, rj, ids = _tally(items)
        if rj >= need and rj / (rj + ap) >= share:
            out.append(Candidate("GATE_HARDEN", gate, {}, False, f"{gate} failed on {ap + rj} photographs and only lowered the score; staff rejected {rj} of them and approved {ap}",
                                 {"soft_failures": ap + rj, "staff_rejected": rj, "staff_approved": ap, "photos": ids}))
    out += _threshold_candidates(rows[-MAX_DRAFT_ROWS:], rb, profile_name, without, overlay, cfg)
    return out


def _errors(rows: list[Row], rb: Rulebook, profile_name: str, without: list[str], overlay: list[dict[str, Any]]) -> Optional[dict[str, int]]:
    try:
        return confusion(rows, replay(rows, rb, profile_name, without, overlay))
    except ValueError:
        return None


def _threshold_candidates(rows: list[Row], rb: Rulebook, profile_name: str, without: list[str], overlay: list[dict[str, Any]], cfg: dict[str, Any]) -> list[Candidate]:
    """Search the review and reselection thresholds on the discovery rows: the change that removes the most errors without a new false approval."""
    if not rows:
        return []
    th = rb.resolve(profile_name, without=without, overlay=overlay).quality["thresholds"]
    cur_rv, cur_rs = th["review"], th["reselection"]
    base = _errors(rows, rb, profile_name, without, overlay)
    if base is None:
        return []

    def trial(rv: float, rs: float) -> Optional[dict[str, int]]:
        return _errors(rows, rb, profile_name, without, overlay + [{"change_id": "trial", "kind": "QUALITY_THRESHOLDS", "set": {"review": rv, "reselection": min(rs, rv)}}])

    best = (base["errors"], 0.0, cur_rv, cur_rs, base)       # (errors, distance from now, review, reselection, confusion)
    for rv in REVIEW_GRID:
        c = trial(rv, cur_rs)
        if c and c["FP"] <= base["FP"] and (c["errors"], abs(rv - cur_rv)) < best[:2]:
            best = (c["errors"], abs(rv - cur_rv), rv, min(cur_rs, rv), c)
    for rs in RESELECT_GRID:
        if rs > best[2]:
            break
        c = trial(best[2], rs)
        if c and c["FP"] <= base["FP"] and c["errors"] < best[0]:      # a different reselection threshold must strictly reduce the errors
            best = (c["errors"], best[1], best[2], rs, c)
    errors, _, rv, rs, c = best
    if base["errors"] - errors < cfg["min_gain"] or (rv, rs) == (cur_rv, cur_rs):
        return []
    s: dict[str, Any] = {}
    if rv != cur_rv:
        s["review"] = rv
    if rs != cur_rs:
        s["reselection"] = rs
    lower = rv < cur_rv or rs < cur_rs
    return [Candidate("QUALITY_THRESHOLDS", None, s, lower, f"on {base['n']} discovery photographs these thresholds remove {base['errors'] - errors} of {base['errors']} errors with no new false approval",
                      {"discovery_errors_before": base["errors"], "discovery_errors_after": errors, "rows": base["n"]}, {"review": cur_rv, "reselection": cur_rs})]


# ------------------------------------------------------------------------------------------------------------------------------------- testing
@dataclass
class Outcome:
    status: str           # APPLY, HELD or BLOCKED
    reason: str
    metrics: dict[str, Any]
    flips: dict[str, Any]
    guardrails: dict[str, Any]


def evaluate_candidate(rows: list[Row], base: dict[str, Any], rb: Rulebook, profile_name: str, without: list[str], overlay: list[dict[str, Any]], record: dict[str, Any],
                       structural_loosening: bool, cfg: dict[str, Any]) -> Outcome:
    """Test one change on ALL the stored decisions for safety and on the HELD-OUT ones for gain. `base` is the replay of the current settings."""
    try:
        cand = replay(rows, rb, profile_name, without, overlay + [record])
    except ValueError as exc:
        return Outcome("BLOCKED", f"the change cannot be applied: {exc}", {}, {}, {"applicable": False})
    human = {r.photo_id: r.human_approved for r in rows}
    ids = [r.photo_id for r in rows if r.photo_id in base and r.photo_id in cand]
    newly_approved = [p for p in ids if approved(cand[p]) and not approved(base[p])]
    newly_not = [p for p in ids if approved(base[p]) and not approved(cand[p])]
    new_fa = [p for p in newly_approved if not human[p]]
    held = [r for r in rows if r.held_out]
    metrics = {k: {"before": confusion(rs, base), "after": confusion(rs, cand)} for k, rs in (("all", rows), ("held_out", held))}
    loosening = structural_loosening or bool(newly_approved)
    held_rejected = sum(1 for r in held if not r.human_approved)
    gain = metrics["held_out"]["before"]["errors"] - metrics["held_out"]["after"]["errors"]
    guard = {"no_new_false_approval": not new_fa, "gain_on_held_out": gain, "min_gain": cfg["min_gain"], "loosening": loosening,
             "held_out_rejected": held_rejected, "needed_for_loosening": cfg["min_rejected_for_loosening"] if loosening else 0}
    flips = {"newly_approved": newly_approved[:50], "newly_not_approved": newly_not[:50], "new_false_approvals": new_fa[:50],
             "fixed_false_rejections": [p for p in newly_approved if human[p]][:50], "fixed_false_approvals": [p for p in newly_not if not human[p]][:50]}
    if new_fa:
        return Outcome("BLOCKED", f"it would approve {len(new_fa)} photograph(s) staff rejected", metrics, flips, guard)
    if gain < cfg["min_gain"]:
        return Outcome("BLOCKED", f"no measurable gain on the {len(held)} held-out photographs (errors {metrics['held_out']['before']['errors']} to {metrics['held_out']['after']['errors']})", metrics, flips, guard)
    if loosening and held_rejected < cfg["min_rejected_for_loosening"]:
        return Outcome("HELD", f"it loosens the rules, and the held-out photographs hold only {held_rejected} staff-rejected photographs, fewer than the {cfg['min_rejected_for_loosening']} "
                       "needed to show it is safe: kept for developers, not applied", metrics, flips, guard)
    return Outcome("APPLY", "it passed every guardrail", metrics, flips, guard)


# ---------------------------------------------------------------------------------------------------------------------------- records, events
def _next_label(s: Session, org_id: str) -> str:
    n = (s.scalar(select(func.count()).select_from(ChangeRecord).where(ChangeRecord.organisation_id == org_id)) or 0) + 1
    return f"L4-{n:04d}"


def _event(s: Session, org_id: str, change_id: str, status: str, reason: str = "", payload: Optional[dict] = None, version_id: Optional[str] = None,
           user: Optional[User] = None) -> ChangeEvent:
    seq = (s.scalar(select(func.max(ChangeEvent.seq)).where(ChangeEvent.change_id == change_id)) or 0) + 1
    e = ChangeEvent(organisation_id=org_id, change_id=change_id, seq=seq, status=status, reason=reason, payload=payload or {}, rulebook_version_id=version_id,
                    created_by=user.user_id if user else None)
    s.add(e)
    s.flush()
    return e


def latest_status(s: Session, change_id: str) -> Optional[str]:
    return s.scalar(select(ChangeEvent.status).where(ChangeEvent.change_id == change_id).order_by(ChangeEvent.seq.desc()).limit(1))


def _record(s: Session, org_id: str, cycle: ImprovementCycle, cand: Candidate) -> tuple[ChangeRecord, bool]:
    """The change record for a candidate, created once per fingerprint. Returns (record, created)."""
    fp = cand.fingerprint()
    existing = s.scalar(select(ChangeRecord).where(ChangeRecord.organisation_id == org_id, ChangeRecord.fingerprint == fp))
    if existing:
        return existing, False
    label = _next_label(s, org_id)
    rec = ChangeRecord(organisation_id=org_id, cycle_id=cycle.cycle_id, label=label, fingerprint=fp, kind=cand.kind, target=cand.target, loosening=cand.loosening,
                       record=cand.record(label), rationale=cand.rationale, evidence=cand.evidence)
    s.add(rec)
    s.flush()
    _event(s, org_id, rec.change_id, "DRAFTED", cand.rationale)
    return rec, True


def describe(rec: ChangeRecord) -> str:
    r, prev = rec.record, rec.record.get("previous") or {}
    if rec.kind == "QUALITY_THRESHOLDS":
        s, parts = r["set"], []
        if "review" in s:
            parts.append(f"photographs that pass every gate now go to a person when their quality score is below {s['review']:g} (it was {prev.get('review')})")
        if "reselection" in s:
            parts.append(f"{'and ' if parts else ''}are sent back for reselection below {s['reselection']:g} (it was {prev.get('reselection')})")
        return "Changed the quality thresholds: " + " ".join(parts) + "."
    if rec.kind == "GATE_SOFTEN":
        return f"Gate {rec.target} no longer rejects a photograph by itself; a failure now only lowers the quality score."
    if rec.kind == "GATE_HARDEN":
        return f"Gate {rec.target} now rejects a photograph when it fails (before, a failure only lowered the quality score)."
    return "Changed the quality weights."


def _test_summary(t: Optional[ChangeTest]) -> Optional[dict[str, Any]]:
    if t is None:
        return None
    m = t.metrics or {}
    h, a = m.get("held_out", {}), m.get("all", {})
    return {"test_id": t.test_id, "passed": t.passed, "photographs": a.get("before", {}).get("n"), "held_out": h.get("before", {}).get("n"),
            "false_approvals": [a.get("before", {}).get("FP"), a.get("after", {}).get("FP")], "false_rejections": [a.get("before", {}).get("FN"), a.get("after", {}).get("FN")],
            "held_out_errors": [h.get("before", {}).get("errors"), h.get("after", {}).get("errors")], "guardrails": t.guardrails, "created_at": t.created_at.isoformat() if t.created_at else None}


# ----------------------------------------------------------------------------------------------------------------------- apply and roll back
def _active(s: Session, org: Organisation) -> RulebookVersion:
    v = s.get(RulebookVersion, org.active_rulebook_version_id) if org.active_rulebook_version_id else None
    if v is None:
        raise Conflict("this organisation has no active rulebook version")
    return v


def _overlay_of(v: RulebookVersion) -> list[dict[str, Any]]:
    return list((v.configuration or {}).get("overlay") or [])


def _apply(s: Session, org: Organisation, rec: ChangeRecord, test: ChangeTest, rb: Rulebook) -> RulebookVersion:
    """A change that passed becomes a new rulebook version (the previous overlay plus this change), approved by the automatic guardrails, and is activated."""
    active = _active(s, org)
    cfg = dict(active.configuration)
    overlay = _overlay_of(active) + [rec.record]
    rb.resolve(cfg["profile"], without=cfg.get("without", []), overlay=overlay)          # raises ValueError if it cannot apply: nothing is created
    root = re.sub(r"\+L4-\d+$", "", active.version_number)
    top = max([0] + [int(m.group(1)) for v in s.scalars(select(RulebookVersion).where(RulebookVersion.organisation_id == org.organisation_id))
                     if (m := re.search(r"\+L4-(\d+)$", v.version_number))])
    cfg.update(overlay=overlay, overlay_digest=hashlib.sha256(canonical(overlay).encode()).hexdigest(),
               auto_approval={"by": "automatic guardrails", "change": rec.label, "test_id": test.test_id})
    v = vs.create_version(s, "rulebook", org.organisation_id, f"{root}+L4-{top + 1:04d}", cfg, f"{rec.label}: {describe(rec)} {rec.rationale}.", None, VersionApproval.APPROVED.value)
    vs.activate_version(s, "rulebook", v)
    _event(s, org.organisation_id, rec.change_id, "APPLIED", "it passed every guardrail", {"previous_version_id": active.version_id, "test_id": test.test_id}, v.version_id)
    return v


def _latest_applied(s: Session, org_id: str, statuses: tuple[str, ...] = ("APPLIED",)) -> Optional[ChangeEvent]:
    """The APPLIED event of the most recent change whose current status is one of `statuses`. Monitoring watches only changes still in APPLIED; a person may also undo
    one that monitoring has confirmed (MONITORED_OK), because it is still in force."""
    for ev in s.scalars(select(ChangeEvent).where(ChangeEvent.organisation_id == org_id, ChangeEvent.status == "APPLIED").order_by(ChangeEvent.created_at.desc(), ChangeEvent.seq.desc())):
        if latest_status(s, ev.change_id) in statuses:
            return ev
    return None


def _roll_back(s: Session, org: Organisation, ev: ChangeEvent, reason: str, payload: dict[str, Any], user: Optional[User] = None) -> None:
    prev = s.get(RulebookVersion, ev.payload["previous_version_id"])
    vs.activate_version(s, "rulebook", prev)
    _event(s, org.organisation_id, ev.change_id, "ROLLED_BACK", reason, {**payload, "restored_version_id": prev.version_id}, prev.version_id, user)


def rollback_change(s: Session, org: Organisation, user: User, change_id: str, reason: str) -> RulebookVersion:
    """A person undoes the latest applied change (older ones sit under later ones and cannot be removed on their own)."""
    if not (reason or "").strip():
        raise Invalid("say why the change is being rolled back")
    rec = s.scalar(select(ChangeRecord).where(ChangeRecord.change_id == change_id, ChangeRecord.organisation_id == org.organisation_id))
    if rec is None:
        raise NotFound("change not found")
    ev = _latest_applied(s, org.organisation_id, ("APPLIED", "MONITORED_OK"))
    if ev is None or ev.change_id != change_id:
        raise Conflict("only the latest change still in force can be rolled back; roll back the later changes first")
    if org.active_rulebook_version_id != ev.rulebook_version_id:
        raise Conflict("the active rulebook version is no longer the one this change created")
    _roll_back(s, org, ev, reason.strip(), {"by": "a person"}, user)
    return s.get(RulebookVersion, org.active_rulebook_version_id)


def monitor(s: Session, org: Organisation, rb: Rulebook, cfg: dict[str, Any]) -> Optional[str]:
    """Watch the latest applied change: decide the photographs that arrived since as the previous version would have, and compare. A new false approval,
    or clearly more errors, rolls it back. Returns 'ROLLED_BACK', 'MONITORED_OK' or None (nothing to do yet)."""
    ev = _latest_applied(s, org.organisation_id)
    if ev is None:
        return None
    if org.active_rulebook_version_id != ev.rulebook_version_id:
        _event(s, org.organisation_id, ev.change_id, "SUPERSEDED", "another rulebook version has been activated since")
        return None
    new_v, prev_v = s.get(RulebookVersion, ev.rulebook_version_id), s.get(RulebookVersion, ev.payload["previous_version_id"])
    profile, without = new_v.configuration["profile"], new_v.configuration.get("without", [])
    rows = [r for r in build_dataset(s, org.organisation_id, profile, cfg["holdout_pct"]) if r.version_id == new_v.version_id]
    if len(rows) < cfg["monitor_min"]:
        return None
    new = confusion(rows, replay(rows, rb, profile, without, _overlay_of(new_v)))
    old = confusion(rows, replay(rows, rb, profile, without, _overlay_of(prev_v)))
    metrics = {"photographs": len(rows), "with_the_change": new, "as_before": old}
    extra_fa, extra_errors = new["FP"] - old["FP"], new["errors"] - old["errors"]
    if extra_fa >= 1 or extra_errors >= cfg["error_margin"]:
        why = (f"on the {len(rows)} photographs decided since, it made {extra_fa} more false approval(s) than the previous settings would have" if extra_fa >= 1
               else f"on the {len(rows)} photographs decided since, it made {extra_errors} more errors than the previous settings would have")
        _roll_back(s, org, ev, why, metrics)
        return "ROLLED_BACK"
    if len(rows) >= cfg["monitor_window"]:
        _event(s, org.organisation_id, ev.change_id, "MONITORED_OK", f"{len(rows)} photographs decided since: no more errors than the previous settings would have made", metrics)
        return "MONITORED_OK"
    return None


# --------------------------------------------------------------------------------------------------------------------------------------- cycle
def decided_count(s: Session, org_id: str) -> int:
    return s.scalar(select(func.count()).select_from(HumanReview).where(HumanReview.organisation_id == org_id, HumanReview.review_status == ReviewStatus.FEEDBACK_SUBMITTED.value)) or 0


def maybe_run_cycle(factory: sessionmaker[Session], organisation_id: str, rb: Optional[Rulebook] = None, trigger: str = "AUTO") -> Optional[str]:
    """Run a cycle if one is due (AUTO) or now (MANUAL). Never raises: the loop must never disturb the people deciding photographs."""
    try:
        with factory() as s:
            org = s.get(Organisation, organisation_id)
            cfg = settings(org)
            if not cfg["enabled"] or cfg["paused"]:
                return "PAUSED" if cfg["enabled"] else "DISABLED"
            count = decided_count(s, organisation_id)
            last = s.scalar(select(ImprovementCycle).where(ImprovementCycle.organisation_id == organisation_id).order_by(ImprovementCycle.ordinal.desc()).limit(1))
            if trigger == "AUTO" and count - (last.decided_count if last else 0) < cfg["every_n"]:
                return None
            cycle = ImprovementCycle(organisation_id=organisation_id, ordinal=(last.ordinal if last else 0) + 1, decided_count=count, trigger=trigger, status="RUNNING")
            s.add(cycle)
            try:
                s.commit()                       # claimed: a second worker computing the same ordinal fails on the unique key and stops
            except IntegrityError:
                s.rollback()
                return None
            return _run(s, org, cycle, cfg, rb or Rulebook())
    except Exception as exc:  # noqa: BLE001
        log.warning("learning loop cycle failed for %s: %s", organisation_id, type(exc).__name__)
        return None


def _finish(s: Session, cycle: ImprovementCycle, status: str, summary: dict[str, Any]) -> str:
    cycle.status, cycle.summary, cycle.finished_at = status, summary, now()
    s.commit()
    return status


def _run(s: Session, org: Organisation, cycle: ImprovementCycle, cfg: dict[str, Any], rb: Rulebook) -> str:
    summary: dict[str, Any] = {"candidates": []}
    try:
        verdict = monitor(s, org, rb, cfg)
        summary["monitor"] = verdict
        if verdict == "ROLLED_BACK":
            return _finish(s, cycle, "ROLLED_BACK", summary)
        active = _active(s, org)
        acfg = active.configuration
        profile, without, overlay = acfg["profile"], acfg.get("without", []), _overlay_of(active)
        rows = build_dataset(s, org.organisation_id, profile, cfg["holdout_pct"])
        discovery = [r for r in rows if not r.held_out and not r.low_information]
        summary.update(rows=len(rows), discovery=len(discovery), held_out=sum(r.held_out for r in rows))
        if len(rows) < cfg["min_rows"]:
            return _finish(s, cycle, "NOT_ENOUGH_DATA", summary)
        base = replay(rows, rb, profile, without, overlay)
        passing: list[tuple[int, ChangeRecord, ChangeTest, Outcome]] = []
        for cand in mine(discovery, base, rb, profile, without, overlay, cfg):
            rec, created = _record(s, org.organisation_id, cycle, cand)
            status = latest_status(s, rec.change_id)
            if status in ("APPLIED", "ROLLED_BACK", "SUPERSEDED", "MONITORED_OK"):
                continue                            # already live, or undone and never to be drafted again
            out = evaluate_candidate(rows, base, rb, profile, without, overlay, rec.record, rec.loosening, cfg)
            test = ChangeTest(organisation_id=org.organisation_id, change_id=rec.change_id, cycle_id=cycle.cycle_id, metrics=out.metrics, flips=out.flips, guardrails=out.guardrails,
                              passed=out.status == "APPLY")
            s.add(test)
            s.flush()
            if out.status != "APPLY" and status != out.status:
                _event(s, org.organisation_id, rec.change_id, out.status, out.reason, {"test_id": test.test_id})
            summary["candidates"].append({"label": rec.label, "kind": rec.kind, "target": rec.target, "outcome": out.status, "reason": out.reason})
            if out.status == "APPLY":
                gain = out.guardrails["gain_on_held_out"]
                passing.append((gain, rec, test, out))
        if not summary["candidates"]:
            return _finish(s, cycle, "NOTHING_TO_CHANGE", summary)
        if not passing:
            return _finish(s, cycle, "NO_PASSING_CHANGE", summary)
        passing.sort(key=lambda t: (-t[0], t[1].label))
        _, rec, test, _ = passing[0]
        for _, other, otest, _ in passing[1:]:
            _event(s, org.organisation_id, other.change_id, "PASSED_NOT_APPLIED", "another change was applied in this cycle; it is tested again in the next one", {"test_id": otest.test_id})
        try:
            v = _apply(s, org, rec, test, rb)
        except ValueError as exc:
            _event(s, org.organisation_id, rec.change_id, "BLOCKED", f"the change cannot be applied: {exc}", {"test_id": test.test_id})
            return _finish(s, cycle, "NO_PASSING_CHANGE", summary)
        summary["applied"] = {"label": rec.label, "version_id": v.version_id}
        return _finish(s, cycle, "APPLIED", summary)
    except Exception as exc:  # noqa: BLE001
        s.rollback()
        summary["error"] = type(exc).__name__
        log.warning("learning loop cycle %s failed: %s", cycle.cycle_id, type(exc).__name__)
        return _finish(s, cycle, "FAILED", summary)


# --------------------------------------------------------------------------------------------------------------------------------------- views
def changelog(s: Session, organisation_id: str, status: Optional[str] = None, limit: int = 100) -> list[dict[str, Any]]:
    out = []
    for rec in s.scalars(select(ChangeRecord).where(ChangeRecord.organisation_id == organisation_id).order_by(ChangeRecord.created_at.desc()).limit(min(limit, 500))):
        events = list(s.scalars(select(ChangeEvent).where(ChangeEvent.change_id == rec.change_id).order_by(ChangeEvent.seq)))
        cur = events[-1].status if events else None
        if status and cur != status:
            continue
        test = s.scalar(select(ChangeTest).where(ChangeTest.change_id == rec.change_id).order_by(ChangeTest.created_at.desc()).limit(1))
        out.append({"change_id": rec.change_id, "label": rec.label, "kind": rec.kind, "target": rec.target, "loosening": rec.loosening, "status": cur, "description": describe(rec),
                    "rationale": rec.rationale, "evidence": rec.evidence, "latest_test": _test_summary(test),
                    "history": [{"status": e.status, "reason": e.reason, "at": e.created_at.isoformat() if e.created_at else None, "by_person": e.created_by is not None} for e in events]})
    return out


def overview(s: Session, org: Organisation) -> dict[str, Any]:
    cycles = list(s.scalars(select(ImprovementCycle).where(ImprovementCycle.organisation_id == org.organisation_id).order_by(ImprovementCycle.ordinal.desc()).limit(10)))
    active = s.get(RulebookVersion, org.active_rulebook_version_id) if org.active_rulebook_version_id else None
    return {**settings(org), "decided_photographs": decided_count(s, org.organisation_id), "active_rulebook_version": active.version_number if active else None,
            "active_overlay": [r["change_id"] for r in _overlay_of(active)] if active else [],
            "recent_cycles": [{"ordinal": c.ordinal, "decided_count": c.decided_count, "trigger": c.trigger, "status": c.status, "summary": c.summary,
                               "started_at": c.started_at.isoformat() if c.started_at else None} for c in cycles]}
