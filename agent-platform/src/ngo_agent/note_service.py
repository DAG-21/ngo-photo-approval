"""L5, prompt notes updated after every decision (docs/PHASE3_LEARNING_LOOP_DESIGN.md, section 19).

After a reviewer submits a decision, and once the gap analyzer (L3b) has mapped the reasons to gates, the drafter writes at most one note per disputed gate. What happens to
the draft depends on the organisation's mode:

* OFF (the default): nothing is drafted and no model is called.
* DRAFT_ONLY: drafts are stored and visible to developers; nothing reaches the prompt.
* APPLY: a draft that passes every check becomes a new prompt version (the base prompt plus the notes) and is used for the next evaluations. Each gate has one note: a newer
  one replaces the older, so notes do not pile up, and the number of notes is capped. A note that LOOSENS a gate taken from the client's own rules is applied and FLAGGED for
  developers. A note that contradicts the client's wording, a suggestion to change a setting, and an escalation are HELD for developers and never applied.

Every applied note is watched: for photographs where its gate mattered, the share of decisions in which the reviewer overrode the AI is compared before and after the note,
and a note is withdrawn automatically when that share rises by more than a margin over a window. A person can withdraw any note, with a reason.

Notes are append-only history (`note_drafts`, `note_events`); the live notes are the prompt version the organisation has active, and every evaluation records which prompt
version it used. A failure here never touches a decision.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from layered_engine.prompt_builder import build_prompt
from layered_engine.rulebook import Rulebook
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from . import versions as vs
from .enums import ReviewStatus, VersionApproval
from .errors import Conflict, Invalid, NotFound
from .models import EvaluationRun, GapAnalysis, HumanReview, LearningExample, NoteDraft, NoteEvent, Organisation, PromptVersion, RulebookVersion, User, now
from .note_drafter import DEFAULT_MAX_NOTE_CHARS, check_note, gate_catalogue
from .reason_quality import redact

log = logging.getLogger("ngo_agent.notes")

MODES = ("OFF", "DRAFT_ONLY", "APPLY")
DEFAULTS: dict[str, Any] = {
    "mode": "OFF",                          # OFF, DRAFT_ONLY (drafts are stored and visible, nothing is applied) or APPLY
    "paused": False,                        # the developers' switch: nothing runs while it is on
    "max_notes": 5,                         # at most this many notes in the prompt at once
    "max_note_chars": DEFAULT_MAX_NOTE_CHARS,
    "watch_window": 10,                     # decisions on photographs where the gate mattered, under a note, before it is judged
    "watch_min_before": 5,                  # decisions before the note needed to compare against
    "watch_margin": 0.25,                   # the override share may rise by less than this (0.25 = 25 percentage points) before the note is withdrawn
}
BOUNDS = {"max_notes": (1, 20), "max_note_chars": (50, 600), "watch_window": (1, 1000), "watch_min_before": (1, 1000), "watch_margin": (0.05, 1.0)}
ACTIVE = ("APPLIED", "CONFIRMED")
MAX_GATES_PER_REVIEW = 2                    # at most this many gates are drafted for from one decision (the most confident mappings first)
CONFIDENCE_RANK = {"HIGH": 2, "MEDIUM": 1}  # a LOW mapping is a guess and is never drafted from


# ------------------------------------------------------------------------------------------------------------------------------------ settings
def settings(org: Organisation) -> dict[str, Any]:
    raw = {**DEFAULTS, **((org.configuration or {}).get("prompt_notes") or {})}
    return {k: raw[k] for k in DEFAULTS}


def save_settings(org: Organisation, changes: dict[str, Any], confirm_ngo_informed: bool = False, drafter_available: bool = True) -> dict[str, Any]:
    cur = settings(org)
    new = {**cur, **{k: v for k, v in changes.items() if v is not None}}
    if new["mode"] not in MODES:
        raise Invalid(f"mode must be one of {', '.join(MODES)}")
    if not isinstance(new["paused"], bool):
        raise Invalid("paused must be true or false")
    for k, (lo, hi) in BOUNDS.items():
        v = new[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi or (k != "watch_margin" and v != int(v)):
            raise Invalid(f"{k} must be {'a number' if k == 'watch_margin' else 'a whole number'} from {lo} to {hi}")
    if new["watch_window"] < 1 or new["watch_min_before"] < 1:
        raise Invalid("the watch window and the minimum before must be at least 1")
    if MODES.index(new["mode"]) > MODES.index(cur["mode"]):
        if not drafter_available:
            raise Conflict("no model is configured to draft prompt notes on this deployment")
        if not confirm_ngo_informed:
            raise Invalid("drafting notes sends each reviewer's written reasons, as text, to a cloud model" + (
                "; applying them also changes the AI's instructions automatically after every decision, with no approval for each note (a note that loosens a gate from the client's "
                "own rules is flagged for developers)" if new["mode"] == "APPLY" else "") + "; set confirm_ngo_informed to true once the NGO has been told")
    org.configuration = {**(org.configuration or {}), "prompt_notes": {k: new[k] for k in DEFAULTS}}
    org.updated_at = now()
    return settings(org)


# ------------------------------------------------------------------------------------------------------------------------------------ history
def _event(s: Session, org_id: str, draft_id: str, status: str, reason: str = "", payload: Optional[dict] = None, version_id: Optional[str] = None,
           user: Optional[User] = None) -> NoteEvent:
    seq = (s.scalar(select(func.max(NoteEvent.seq)).where(NoteEvent.draft_id == draft_id)) or 0) + 1
    e = NoteEvent(organisation_id=org_id, draft_id=draft_id, seq=seq, status=status, reason=reason, payload=payload or {}, prompt_version_id=version_id,
                  created_by=user.user_id if user else None)
    s.add(e)
    s.flush()
    return e


def status_map(s: Session, org_id: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for e in s.scalars(select(NoteEvent).where(NoteEvent.organisation_id == org_id).order_by(NoteEvent.draft_id, NoteEvent.seq)):
        out[e.draft_id] = e.status
    return out


def active_notes(s: Session, org_id: str) -> dict[str, NoteDraft]:
    """The notes now in the prompt, by gate id."""
    st = status_map(s, org_id)
    out: dict[str, NoteDraft] = {}
    for d in s.scalars(select(NoteDraft).where(NoteDraft.organisation_id == org_id).order_by(NoteDraft.created_at)):
        if st.get(d.draft_id) in ACTIVE:
            out[d.gate_id] = d
    return out


def _set_notes(s: Session, org: Organisation, notes: dict[str, str], rb: Rulebook, why: str) -> PromptVersion:
    """Make the prompt version for exactly these notes the active one (created if it is new; with no notes it is the base prompt again)."""
    rv = s.get(RulebookVersion, org.active_rulebook_version_id)
    cfg = rv.configuration
    active = s.get(PromptVersion, org.active_prompt_version_id) if org.active_prompt_version_id else None
    rule_text = dict((active.configuration or {}).get("rule_text") or {}) if active else {}   # rewritten rules (rule_service) are kept as they are
    bp = build_prompt(rb, rb.resolve(cfg["profile"], without=cfg.get("without", [])), notes=notes or None, rule_text=rule_text or None)
    number = f"{bp.template_version}:{bp.hash}"
    pv = s.scalar(select(PromptVersion).where(PromptVersion.organisation_id == org.organisation_id, PromptVersion.version_number == number))
    if pv is None:
        conf = {"profile": cfg["profile"], "template_version": bp.template_version, "prompt_hash": bp.hash, "criteria_asked": len(bp.criterion_ids)}
        if notes:
            conf["notes"] = dict(notes)
        if rule_text:
            conf["rule_text"] = rule_text
        if notes or rule_text:
            conf["approval"] = {"by": "automatic guardrails", "reason": why}
        pv = vs.create_version(s, "prompt", org.organisation_id, number, conf, why, None, VersionApproval.APPROVED.value)
    vs.activate_version(s, "prompt", pv)
    return pv


# ---------------------------------------------------------------------------------------------------------------------------------- the pipeline
def _tasks(s: Session, review: HumanReview, drafter: Any, current: dict[str, str]) -> list[dict[str, Any]]:
    """The gates this decision disputed with enough confidence, with everything the drafter is allowed to see for each. Empty when there is nothing to draft from."""
    ex = s.scalar(select(LearningExample).where(LearningExample.human_review_id == review.review_id))
    if ex is None or ex.low_information:                  # excluded as for examples (a mistake, a policy change, over a blocker) or too generic to teach anything
        return []
    gap = s.scalar(select(GapAnalysis).where(GapAnalysis.review_id == review.review_id, GapAnalysis.status == "DONE", GapAnalysis.analyzer.like("llm-gap%"))
                   .order_by(GapAnalysis.created_at.desc()))
    if gap is None:
        return []
    done = set(s.scalars(select(NoteDraft.gate_id).where(NoteDraft.review_id == review.review_id)))
    run = s.get(EvaluationRun, review.evaluation_id)
    ai_gates = {g.get("gate_id"): g for g in (run.decision_output or {}).get("gates") or []}
    best: dict[str, dict[str, Any]] = {}
    for g in gap.gaps or []:
        rank = CONFIDENCE_RANK.get(g.get("confidence"), 0)
        if g.get("kind") != "CONTESTED_GATE" or not rank:
            continue
        for gate in g.get("related_gates") or []:
            if gate in drafter.gates and gate not in done and rank > best.get(gate, {}).get("rank", 0):
                best[gate] = {"rank": rank, "record": g}
    tasks = []
    for gate, b in sorted(best.items(), key=lambda kv: -kv[1]["rank"])[:MAX_GATES_PER_REVIEW]:
        a = ai_gates.get(gate) or {}
        evidence = f"{gate} was {a.get('status', 'not reported')}: {redact(a.get('evidence'))}" if a else f"{gate} was not reported as a problem"
        tasks.append({"gate_id": gate, "record": {k: b["record"].get(k) for k in ("summary", "rule_note", "confidence", "relation")},
                      "decision": {"decision": review.human_final_decision, "agreement": review.overall_agreement_with_ai}, "ai_evidence": evidence,
                      "current_note": current.get(gate, "")})
    return tasks


def process_review(factory: sessionmaker[Session], review_id: str, drafter: Any, rb: Rulebook) -> str:
    """Draft (and, in APPLY mode, apply) notes for one submitted decision. Never raises; returns a short status for logs and tests."""
    try:
        with factory() as s:
            review = s.get(HumanReview, review_id)
            if review is None or review.review_status != ReviewStatus.FEEDBACK_SUBMITTED.value:
                return "SKIPPED:not_submitted"
            org = s.get(Organisation, review.organisation_id)
            cfg = settings(org)
            if cfg["mode"] == "OFF" or cfg["paused"]:
                return "SKIPPED:off"
            current = {g: d.note_text for g, d in active_notes(s, org.organisation_id).items()}
            tasks = _tasks(s, review, drafter, current)
            org_id, evaluation_id, photo_id = org.organisation_id, review.evaluation_id, review.photo_id
        if not tasks:
            return "SKIPPED:nothing_to_draft"
        results = []
        for t in tasks:               # the model runs with no database transaction open: a slow call must not hold a lock
            try:
                results.append((t, drafter.draft(t, max_note_chars=cfg["max_note_chars"]), None))
            except Exception as exc:  # noqa: BLE001  (a bad or failed model call is recorded for developers, never raised)
                results.append((t, None, type(exc).__name__))
                log.warning("drafting for gate %s failed: %s", t["gate_id"], type(exc).__name__)
        with factory() as s:
            org = s.get(Organisation, org_id)
            for t, res, err in results:
                _store(s, org, cfg, rb, review_id, evaluation_id, photo_id, t, res, err, drafter)
            s.commit()
        return f"DONE:{len(results)}"
    except Exception as exc:  # noqa: BLE001
        log.warning("prompt notes for review %s could not be processed: %s", review_id, type(exc).__name__)
        return "FAILED"


def _store(s: Session, org: Organisation, cfg: dict[str, Any], rb: Rulebook, review_id: str, evaluation_id: str, photo_id: str, task: dict[str, Any], res: Any,
           err: Optional[str], drafter: Any) -> None:
    gate_id, info = task["gate_id"], drafter.gates[task["gate_id"]]
    if s.scalar(select(NoteDraft).where(NoteDraft.review_id == review_id, NoteDraft.gate_id == gate_id)):
        return                                                      # already drafted (a repeated call): neither stored nor paid for twice
    failed = [f"the drafter failed: {err}"] if err else ([] if res.outcome != "GATE_NOTE" else check_note(res.note, gate_id, cfg["max_note_chars"]))
    if res is not None and res.outcome == "GATE_NOTE" and res.direction not in ("LOOSEN", "TIGHTEN"):
        failed.append("the note must say whether it loosens or tightens the gate")
    loosens_client = res is not None and res.outcome == "GATE_NOTE" and res.direction == "LOOSEN" and info.client_gate
    d = NoteDraft(organisation_id=org.organisation_id, review_id=review_id, evaluation_id=evaluation_id, photo_id=photo_id, gate_id=gate_id,
                  outcome=res.outcome if res else "ERROR", direction=res.direction if res else "NONE", note_text=res.note if res else "", rationale=res.rationale if res else "",
                  contradicts_client_wording=bool(res and res.contradicts_client_wording), client_gate=info.client_gate, flags=["LOOSENS_CLIENT_GATE"] if loosens_client else [],
                  checks={"passed": not failed, "failed": failed}, source=task["record"], model_id=res.model_id if res else None, cost_usd=res.cost_usd if res else None)
    s.add(d)
    s.flush()
    ev = lambda status, reason="", payload=None, version=None: _event(s, org.organisation_id, d.draft_id, status, reason, payload, version)  # noqa: E731
    ev("DRAFTED", "written from the reviewer's reasons for this gate")
    if err:
        ev("REJECTED", failed[0])
    elif res.outcome == "NO_CHANGE":
        ev("NO_CHANGE", res.rationale or "the drafter proposed no change")
    elif res.outcome == "ESCALATE" or res.contradicts_client_wording:
        ev("HELD", "held for developers: the reviewer's view would contradict the client's wording of this gate. " + res.rationale)
    elif res.outcome == "SETTING_SUGGESTION":
        ev("HELD", "held for developers: the suggestion is about how strict the gate is, which the update cycle and developers own, not a note. " + res.rationale)
    elif failed:
        ev("REJECTED", "; ".join(failed))
    elif cfg["mode"] == "DRAFT_ONLY":
        pass                                                        # stays DRAFTED: visible to developers, not in the prompt (the apply switch is off)
    else:
        _apply(s, org, d, cfg, rb, loosens_client)


def _apply(s: Session, org: Organisation, d: NoteDraft, cfg: dict[str, Any], rb: Rulebook, loosens_client: bool) -> None:
    now_active = active_notes(s, org.organisation_id)
    current = {g: x.note_text for g, x in now_active.items()}
    prev = now_active.get(d.gate_id)
    if prev is not None and " ".join(prev.note_text.split()) == " ".join(d.note_text.split()):
        _event(s, org.organisation_id, d.draft_id, "NO_CHANGE", "the gate already has exactly this note")
        return
    if prev is None and len(current) >= cfg["max_notes"]:
        _event(s, org.organisation_id, d.draft_id, "HELD", f"held for developers: the limit of {cfg['max_notes']} notes in the prompt is reached; withdraw one first")
        return
    current[d.gate_id] = d.note_text
    pv = _set_notes(s, org, current, rb, f"reviewers' note for {d.gate_id}")
    if prev is not None:
        _event(s, org.organisation_id, prev.draft_id, "SUPERSEDED", "replaced by a newer note for the same gate", {"by_draft": d.draft_id}, pv.version_id)
    flags = list(d.flags) + (["FLAGGED_FOR_REVIEW"] if loosens_client else [])
    _event(s, org.organisation_id, d.draft_id, "APPLIED", "applied automatically (all checks passed)" + ("; flagged for developers: it loosens a gate from the client's own rules" if loosens_client else ""),
           {"flags": flags, "prompt_version": pv.version_number, "prompt_hash": pv.configuration.get("prompt_hash")}, pv.version_id)


# ----------------------------------------------------------------------------------------------------------------------------------- the watch
def override_stats(s: Session, org_id: str, gate_id: str, note_text: str) -> dict[str, int]:
    """Decisions on photographs where this gate mattered (the AI failed it or could not tell), split by whether the AI's prompt carried this note, and how many of them
    the reviewer overrode. Counted from stored evaluations and reviews: no model call."""
    notes_of = {pv.version_id: (pv.configuration or {}).get("notes") or {} for pv in s.scalars(select(PromptVersion).where(PromptVersion.organisation_id == org_id))}
    out = {"before_n": 0, "before_overrides": 0, "after_n": 0, "after_overrides": 0}
    rows = s.execute(select(HumanReview, EvaluationRun).join(EvaluationRun, EvaluationRun.evaluation_id == HumanReview.evaluation_id)
                     .where(HumanReview.organisation_id == org_id, HumanReview.review_status == ReviewStatus.FEEDBACK_SUBMITTED.value))
    for review, run in rows:
        g = next((x for x in (run.decision_output or {}).get("gates") or [] if x.get("gate_id") == gate_id), None)
        if not g or g.get("status") not in ("NO", "CANNOT_TELL"):
            continue
        side = "after" if notes_of.get(run.prompt_version_id, {}).get(gate_id) == note_text else "before"
        out[f"{side}_n"] += 1
        out[f"{side}_overrides"] += review.overall_agreement_with_ai == "DISAGREE"
    return out


def judge(stats: dict[str, int], cfg: dict[str, Any]) -> str:
    """WAIT (not enough decisions under the note yet), NO_BASELINE (too few before it to compare with, so no verdict: it simply stays applied), WITHDRAW (overrides rose by
    the margin or more) or CONFIRM."""
    if stats["after_n"] < cfg["watch_window"]:
        return "WAIT"
    if stats["before_n"] < cfg["watch_min_before"]:
        return "NO_BASELINE"
    rise = stats["after_overrides"] / stats["after_n"] - stats["before_overrides"] / stats["before_n"]
    return "WITHDRAW" if rise >= cfg["watch_margin"] - 1e-9 else "CONFIRM"


def watch(factory: sessionmaker[Session], organisation_id: str, rb: Rulebook) -> str:
    """Judge every applied note that has been under watch long enough. Never raises."""
    try:
        with factory() as s:
            org = s.get(Organisation, organisation_id)
            cfg = settings(org)
            if cfg["mode"] == "OFF" or cfg["paused"]:
                return "SKIPPED:off"
            st = status_map(s, organisation_id)
            changed = 0
            for gate, d in list(active_notes(s, organisation_id).items()):
                stats = override_stats(s, organisation_id, gate, d.note_text)
                verdict = judge(stats, cfg)
                if verdict == "WITHDRAW":
                    rate = lambda n, o: f"{round(100 * o / n)}%"  # noqa: E731
                    withdraw(s, org, d, f"withdrawn automatically: reviewers overrode the AI on {rate(stats['after_n'], stats['after_overrides'])} of {stats['after_n']} decisions under this note, "
                                        f"against {rate(stats['before_n'], stats['before_overrides'])} of {stats['before_n']} before it", rb, payload={"by": "the automatic watch", **stats})
                    changed += 1
                elif verdict == "CONFIRM" and st.get(d.draft_id) == "APPLIED":
                    _event(s, organisation_id, d.draft_id, "CONFIRMED", f"reviewers did not override the AI more often under this note ({stats['after_overrides']} of {stats['after_n']} "
                                                                      f"decisions, against {stats['before_overrides']} of {stats['before_n']} before it)", stats)
                    changed += 1
            s.commit()
            return f"DONE:{changed}"
    except Exception as exc:  # noqa: BLE001
        log.warning("the prompt-note watch for %s failed: %s", organisation_id, type(exc).__name__)
        return "FAILED"


def withdraw(s: Session, org: Organisation, d: NoteDraft, reason: str, rb: Rulebook, user: Optional[User] = None, payload: Optional[dict] = None) -> PromptVersion:
    """Take one note out of the prompt: the prompt version for the remaining notes becomes active again (the base prompt when none remain)."""
    remaining = {g: x.note_text for g, x in active_notes(s, org.organisation_id).items() if x.draft_id != d.draft_id}
    pv = _set_notes(s, org, remaining, rb, f"withdrew the note for {d.gate_id}")
    _event(s, org.organisation_id, d.draft_id, "WITHDRAWN", reason, {"by": "a person" if user else "the system", **(payload or {})}, pv.version_id, user)
    return pv


def withdraw_note(s: Session, org: Organisation, user: User, draft_id: str, reason: str, rb: Rulebook) -> PromptVersion:
    if not (reason or "").strip():
        raise Invalid("a reason is required to withdraw a note")
    d = s.scalar(select(NoteDraft).where(NoteDraft.draft_id == draft_id, NoteDraft.organisation_id == org.organisation_id))
    if d is None:
        raise NotFound("note not found")
    if status_map(s, org.organisation_id).get(d.draft_id) not in ACTIVE:
        raise Conflict("this note is not in the prompt, so there is nothing to withdraw")
    return withdraw(s, org, d, reason.strip(), rb, user)


def after_decision(factory: sessionmaker[Session], review_id: str, drafter: Any, rb: Rulebook) -> str:
    """The background step after a submitted decision: draft, then watch. Never raises."""
    status = process_review(factory, review_id, drafter, rb)
    try:
        with factory() as s:
            review = s.get(HumanReview, review_id)
            org_id = review.organisation_id if review else None
        if org_id:
            watch(factory, org_id, rb)
    except Exception as exc:  # noqa: BLE001
        log.warning("the prompt-note watch after review %s failed: %s", review_id, type(exc).__name__)
    return status


# ---------------------------------------------------------------------------------------------------------------------------------- the overview
def overview(s: Session, org: Organisation, drafter_available: bool, limit: int = 100) -> dict[str, Any]:
    st = status_map(s, org.organisation_id)
    events: dict[str, list[NoteEvent]] = {}
    for e in s.scalars(select(NoteEvent).where(NoteEvent.organisation_id == org.organisation_id).order_by(NoteEvent.draft_id, NoteEvent.seq)):
        events.setdefault(e.draft_id, []).append(e)
    drafts = list(s.scalars(select(NoteDraft).where(NoteDraft.organisation_id == org.organisation_id).order_by(NoteDraft.created_at.desc()).limit(limit)))

    def row(d: NoteDraft) -> dict[str, Any]:
        hist = [{"status": e.status, "reason": e.reason, "at": e.created_at.isoformat() if e.created_at else None, "by_person": e.created_by is not None, "payload": e.payload} for e in events.get(d.draft_id, [])]
        flags = sorted({*d.flags, *(f for e in events.get(d.draft_id, []) for f in (e.payload or {}).get("flags", []))})
        return {"draft_id": d.draft_id, "gate_id": d.gate_id, "outcome": d.outcome, "direction": d.direction, "note": d.note_text, "rationale": d.rationale, "status": st.get(d.draft_id, "DRAFTED"),
                "flags": flags, "client_gate": d.client_gate, "contradicts_client_wording": d.contradicts_client_wording, "checks": d.checks, "source": d.source, "review_id": d.review_id,
                "created_at": d.created_at.isoformat() if d.created_at else None, "history": hist}

    rows = [row(d) for d in drafts]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    pv = s.get(PromptVersion, org.active_prompt_version_id) if org.active_prompt_version_id else None
    return {"settings": settings(org), "drafter_available": drafter_available, "prompt_version": pv.version_number if pv else None,
            "active_notes": [r for r in rows if r["status"] in ACTIVE], "drafts": rows, "counts": counts}
