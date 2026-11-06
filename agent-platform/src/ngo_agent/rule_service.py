"""L5, rules rewritten after every decision (docs/PHASE3_LEARNING_LOOP_DESIGN.md, section 20).

The flow, for one submitted decision: the AI's verdict is matched with the person's verdict and reasons; the model-based gap analysis (L3b) ties each point of disagreement to the rules it is
about; for each such rule a model (the "open model", Groq) rewrites the rule's wording. Everything is open to change, the client's own rules included. What happens to a rewrite depends
on the organisation's mode:

* OFF (the default): nothing is drafted and no model is called.
* DRAFT_ONLY: rewrites are stored and visible to developers; nothing reaches the prompt.
* APPLY: a rewrite that passes every check becomes a new prompt version (the same prompt with that rule's wording replaced) and is used for the next evaluations. Each rule has one current
  wording: a newer rewrite replaces the older, and the number of rewritten rules is capped. A rewrite that loosens a rule behind a gate that can reject, or that changes a rule from the client's
  own document, is applied and FLAGGED for developers (nothing is held for contradicting the client's wording: that wording is what is being updated).

Every applied rewrite is watched: for photographs where the rule mattered, the share of decisions in which the reviewer overrode the AI is compared before and after the rewrite, and a rewrite is
withdrawn automatically when that share rises by more than a margin over a window. A person can withdraw any rewrite, with a reason; the rulebook files are never touched, so the original wording
is always what the rule goes back to.

A rule that has ALREADY been changed (a live rewrite exists, made by the loop or by an administrator) is never changed again silently: the new rewrite waits (`NEEDS_CONFIRM`) until a person
answers a confirmation card (`answer`): yes applies it, no keeps the wording in force. A rule still in its original wording is rewritten without any card.
An administrator can also change a rule by hand (`manual_edit`): it is applied at once whatever the mode, recorded like any other rewrite, and the automatic watch never withdraws it.

Rewrites are append-only history (`rule_edits`, `rule_edit_events`); the live wording is the prompt version the organisation has active, and every evaluation records which prompt version it used.
A failure here never touches a decision.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from layered_engine.prompt_builder import FACT_FIELDS, build_prompt
from layered_engine.rulebook import Rulebook
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from . import versions as vs
from .enums import ReviewStatus, VersionApproval
from .errors import Conflict, Invalid, NotFound
from .models import EvaluationRun, GapAnalysis, HumanReview, LearningExample, Organisation, Photo, PromptVersion, RuleEdit, RuleEditEvent, RulebookVersion, User, now
from .note_service import ACTIVE, CONFIDENCE_RANK, judge
from .reason_quality import redact
from .rule_drafter import DEFAULT_MAX_CHARS, check_wording, norm

log = logging.getLogger("ngo_agent.rules")

MODES = ("OFF", "DRAFT_ONLY", "APPLY")
DEFAULTS: dict[str, Any] = {
    "mode": "OFF",                          # OFF, DRAFT_ONLY (rewrites are stored and visible, nothing is applied) or APPLY
    "paused": False,                        # the developers' switch: nothing runs while it is on
    "max_edits": 20,                        # at most this many rules carry a rewritten wording at once
    "max_chars": DEFAULT_MAX_CHARS,
    "watch_window": 10,                     # decisions on photographs where the rule mattered, under a rewrite, before it is judged
    "watch_min_before": 5,                  # decisions before the rewrite needed to compare against
    "watch_margin": 0.25,                   # the override share may rise by less than this (0.25 = 25 percentage points) before the rewrite is withdrawn
}
BOUNDS = {"max_edits": (1, 100), "max_chars": (50, 1500), "watch_window": (1, 1000), "watch_min_before": (1, 1000), "watch_margin": (0.05, 1.0)}
MAX_TARGETS_PER_REVIEW = 2                  # at most this many rules are rewritten from one decision (the most confident mappings first)
FLAGGED = ("LOOSENS_GATE", "CLIENT_RULE")


# ------------------------------------------------------------------------------------------------------------------------------------ settings
def settings(org: Organisation) -> dict[str, Any]:
    raw = {**DEFAULTS, **((org.configuration or {}).get("rule_updates") or {})}
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
    if MODES.index(new["mode"]) > MODES.index(cur["mode"]):
        if not drafter_available:
            raise Conflict("no model is configured to rewrite rules on this deployment")
        if not confirm_ngo_informed:
            raise Invalid("rewriting rules sends each reviewer's written reasons, as text, to a cloud model" + (
                "; applying the rewrites also changes the AI's rules automatically after every decision, the client's own wording included, with no approval for each change (a rewrite "
                "that loosens a gate that can reject, or that changes a rule from the client's document, is flagged for developers)" if new["mode"] == "APPLY" else "")
                + "; set confirm_ngo_informed to true once the NGO has been told")
    org.configuration = {**(org.configuration or {}), "rule_updates": {k: new[k] for k in DEFAULTS}}
    org.updated_at = now()
    return settings(org)


# ------------------------------------------------------------------------------------------------------------------------------------ history
def _event(s: Session, org_id: str, edit_id: str, status: str, reason: str = "", payload: Optional[dict] = None, version_id: Optional[str] = None,
           user: Optional[User] = None) -> RuleEditEvent:
    seq = (s.scalar(select(func.max(RuleEditEvent.seq)).where(RuleEditEvent.edit_id == edit_id)) or 0) + 1
    e = RuleEditEvent(organisation_id=org_id, edit_id=edit_id, seq=seq, status=status, reason=reason, payload=payload or {}, prompt_version_id=version_id,
                      created_by=user.user_id if user else None)
    s.add(e)
    s.flush()
    return e


def status_map(s: Session, org_id: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for e in s.scalars(select(RuleEditEvent).where(RuleEditEvent.organisation_id == org_id).order_by(RuleEditEvent.edit_id, RuleEditEvent.seq)):
        out[e.edit_id] = e.status
    return out


def active_edits(s: Session, org_id: str) -> dict[str, RuleEdit]:
    """The rewrites now in the prompt, by rule id."""
    st = status_map(s, org_id)
    out: dict[str, RuleEdit] = {}
    for d in s.scalars(select(RuleEdit).where(RuleEdit.organisation_id == org_id).order_by(RuleEdit.created_at)):
        if st.get(d.edit_id) in ACTIVE:
            out[d.target_id] = d
    return out


def _set_rules(s: Session, org: Organisation, rule_text: dict[str, str], rb: Rulebook, why: str) -> PromptVersion:
    """Make the prompt version for exactly these rewritten rules the active one (created if it is new; with none it is the base prompt again). Reviewers' notes (note_service) are kept."""
    rv = s.get(RulebookVersion, org.active_rulebook_version_id)
    cfg = rv.configuration
    active = s.get(PromptVersion, org.active_prompt_version_id) if org.active_prompt_version_id else None
    notes = dict((active.configuration or {}).get("notes") or {}) if active else {}
    bp = build_prompt(rb, rb.resolve(cfg["profile"], without=cfg.get("without", [])), notes=notes or None, rule_text=rule_text or None)
    number = f"{bp.template_version}:{bp.hash}"
    pv = s.scalar(select(PromptVersion).where(PromptVersion.organisation_id == org.organisation_id, PromptVersion.version_number == number))
    if pv is None:
        conf = {"profile": cfg["profile"], "template_version": bp.template_version, "prompt_hash": bp.hash, "criteria_asked": len(bp.criterion_ids)}
        if notes:
            conf["notes"] = notes
        if rule_text:
            conf["rule_text"] = dict(rule_text)
        if notes or rule_text:
            conf["approval"] = {"by": "automatic guardrails", "reason": why}
        pv = vs.create_version(s, "prompt", org.organisation_id, number, conf, why, None, VersionApproval.APPROVED.value)
    vs.activate_version(s, "prompt", pv)
    return pv


# ---------------------------------------------------------------------------------------------------------------------------------- the pipeline
def ai_view(run: EvaluationRun, info: Any, rules: dict[str, Any]) -> str:
    """What the AI said about this rule on this photograph, as plain text for the drafter (redacted)."""
    v = run.validated_response or {}
    answers = {c.get("criterion_id"): c for c in v.get("layer_2_criteria") or []}
    if info.kind == "SCENE_FACT":
        field = FACT_FIELDS.get(info.target_id)
        return f"{info.target_id}: the AI reported {field} = {((v.get('layer_1_scene_understanding') or {}).get('scene_facts') or {}).get(field)}"
    if info.kind == "GUIDANCE":
        bad = [g for g in (run.decision_output or {}).get("gates") or [] if g.get("status") in ("NO", "CANNOT_TELL")]
        return f"{info.target_id} is a general instruction the AI was given with the rules. What the AI found wrong: " + ("; ".join(f"{g.get('gate_id')}: {redact(g.get('evidence'))}" for g in bad[:4]) or "no mandatory check failed")
    if info.kind == "CLIENT_RULE":
        bad = [c for cid, c in answers.items() if (rules.get(cid) and rules[cid].rule_id == info.target_id) and c.get("status") in ("NO", "CANNOT_TELL")]
        return "\n".join(f"{c.get('criterion_id')} was {c.get('status')}: {redact(c.get('evidence'))}" for c in bad[:4]) or f"no check under {info.target_id} was failed or doubted by the AI"
    a = answers.get(info.target_id)
    return f"{info.target_id} was {a.get('status')}: {redact(a.get('evidence'))}" if a else f"{info.target_id} was not reported"


def _tasks(s: Session, review: HumanReview, drafter: Any, current: dict[str, str]) -> list[dict[str, Any]]:
    """The rules this decision disputed with enough confidence, with everything the drafter is allowed to see for each. Empty when there is nothing to draft from."""
    ex = s.scalar(select(LearningExample).where(LearningExample.human_review_id == review.review_id))
    if ex is None or ex.low_information:                  # excluded as for examples (a mistake, a policy change, over a blocker) or too generic to teach anything
        return []
    gap = s.scalar(select(GapAnalysis).where(GapAnalysis.review_id == review.review_id, GapAnalysis.status == "DONE", GapAnalysis.analyzer.like("llm-gap%"))
                   .order_by(GapAnalysis.created_at.desc()))
    if gap is None:
        return []
    done = set(s.scalars(select(RuleEdit.target_id).where(RuleEdit.review_id == review.review_id)))
    run = s.get(EvaluationRun, review.evaluation_id)
    by_gate = {i.gate_id: t for t, i in drafter.rules.items() if i.gate_id}
    best: dict[str, dict[str, Any]] = {}
    for g in gap.gaps or []:
        rank = CONFIDENCE_RANK.get(g.get("confidence"), 0)
        if g.get("relation") != "CONTRADICTS_AI" or not rank:
            continue                                       # only a point where the reviewer disagreed with the AI, and not a guess
        for t in g.get("targets") or [by_gate[x] for x in g.get("related_gates") or [] if x in by_gate]:
            if t in drafter.rules and t not in done and rank > best.get(t, {}).get("rank", 0):
                best[t] = {"rank": rank, "record": g}
    tasks = []
    for t, b in sorted(best.items(), key=lambda kv: -kv[1]["rank"])[:MAX_TARGETS_PER_REVIEW]:
        info = drafter.rules[t]
        tasks.append({"target_id": t, "record": {k: b["record"].get(k) for k in ("summary", "rule_note", "confidence", "relation")},
                      "decision": {"decision": review.human_final_decision, "agreement": review.overall_agreement_with_ai}, "ai_evidence": ai_view(run, info, drafter.rules),
                      "current_wording": current.get(t) or info.base})
    return tasks


def process_review(factory: sessionmaker[Session], review_id: str, drafter: Any, rb: Rulebook) -> str:
    """Draft (and, in APPLY mode, apply) rewrites for one submitted decision. Never raises; returns a short status for logs and tests."""
    try:
        with factory() as s:
            review = s.get(HumanReview, review_id)
            if review is None or review.review_status != ReviewStatus.FEEDBACK_SUBMITTED.value:
                return "SKIPPED:not_submitted"
            org = s.get(Organisation, review.organisation_id)
            cfg = settings(org)
            if cfg["mode"] == "OFF" or cfg["paused"]:
                return "SKIPPED:off"
            current = {t: d.new_text for t, d in active_edits(s, org.organisation_id).items()}
            tasks = _tasks(s, review, drafter, current)
            org_id, evaluation_id, photo_id = org.organisation_id, review.evaluation_id, review.photo_id
        if not tasks:
            return "SKIPPED:nothing_to_draft"
        results = []
        for t in tasks:               # the model runs with no database transaction open: a slow call must not hold a lock
            try:
                results.append((t, drafter.draft(t, max_chars=cfg["max_chars"]), None))
            except Exception as exc:  # noqa: BLE001  (a bad or failed model call is recorded for developers, never raised)
                results.append((t, None, type(exc).__name__))
                log.warning("rewriting rule %s failed: %s", t["target_id"], type(exc).__name__)
        with factory() as s:
            org = s.get(Organisation, org_id)
            for t, res, err in results:
                _store(s, org, cfg, rb, review_id, evaluation_id, photo_id, t, res, err, drafter)
            s.commit()
        return f"DONE:{len(results)}"
    except Exception as exc:  # noqa: BLE001
        log.warning("rule rewrites for review %s could not be processed: %s", review_id, type(exc).__name__)
        return "FAILED"


def _store(s: Session, org: Organisation, cfg: dict[str, Any], rb: Rulebook, review_id: str, evaluation_id: str, photo_id: str, task: dict[str, Any], res: Any,
           err: Optional[str], drafter: Any) -> None:
    target, info = task["target_id"], drafter.rules[task["target_id"]]
    if s.scalar(select(RuleEdit).where(RuleEdit.review_id == review_id, RuleEdit.target_id == target)):
        return                                                      # already drafted (a repeated call): neither stored nor paid for twice
    update = res is not None and res.outcome == "UPDATE_RULE"
    failed = [f"the drafter failed: {err}"] if err else []
    if update:
        failed += check_wording(res.wording, task["current_wording"], target, cfg["max_chars"])
        if res.direction not in ("LOOSEN", "TIGHTEN", "NEUTRAL"):
            failed.append("the rewrite must say whether it loosens, tightens or only clarifies the rule")
        if not res.keeps_meaning:
            failed.append("the drafter says a YES answer would no longer mean that the requirement is met")
    flags = (["LOOSENS_GATE"] if update and res.direction == "LOOSEN" and info.hard else []) + (["CLIENT_RULE"] if update and info.client_sourced else [])
    d = RuleEdit(organisation_id=org.organisation_id, review_id=review_id, evaluation_id=evaluation_id, photo_id=photo_id, target_id=target, target_kind=info.kind,
                 outcome=res.outcome if res else "ERROR", direction=res.direction if res else "NONE", previous_text=task["current_wording"], new_text=res.wording if res else "",
                 rationale=res.rationale if res else "", keeps_meaning=bool(res and res.keeps_meaning), gate_id=info.gate_id, hard_gate=info.hard, client_sourced=info.client_sourced, flags=flags,
                 checks={"passed": not failed, "failed": failed}, source=task["record"], model_id=res.model_id if res else None, cost_usd=res.cost_usd if res else None)
    s.add(d)
    s.flush()
    ev = lambda status, reason="", payload=None, version=None: _event(s, org.organisation_id, d.edit_id, status, reason, payload, version)  # noqa: E731
    ev("DRAFTED", "written from the reviewer's reasons about this rule")
    if err:
        ev("REJECTED", failed[0])
    elif not update:
        ev("NO_CHANGE", res.rationale or "the drafter proposed no change")
    elif failed:
        ev("REJECTED", "; ".join(failed))
    elif cfg["mode"] == "DRAFT_ONLY":
        pass                                                        # stays DRAFTED: visible to developers, not in the prompt (the apply switch is off)
    else:
        _apply(s, org, d, cfg, rb, drafter)


def _apply(s: Session, org: Organisation, d: RuleEdit, cfg: dict[str, Any], rb: Rulebook, drafter: Any) -> None:
    now_active = active_edits(s, org.organisation_id)
    current = {t: x.new_text for t, x in now_active.items()}
    prev = now_active.get(d.target_id)
    if norm(d.new_text) == norm(current.get(d.target_id) or drafter.rules[d.target_id].base):
        _event(s, org.organisation_id, d.edit_id, "NO_CHANGE", "the rule already has exactly this wording")
        return
    if prev is not None:
        # this rule was changed after the rulebook was set (by the loop or by an administrator): a person must confirm before it is changed again
        _event(s, org.organisation_id, d.edit_id, "NEEDS_CONFIRM", "this rule was changed earlier: waiting for a person to confirm that it should be changed again",
               {"current_edit": prev.edit_id, "current_set_by_administrator": bool((prev.source or {}).get("administrator"))})
        return
    if len(current) >= cfg["max_edits"]:
        _event(s, org.organisation_id, d.edit_id, "HELD", f"held for developers: the limit of {cfg['max_edits']} rewritten rules is reached; withdraw one first")
        return
    current[d.target_id] = d.new_text
    pv = _set_rules(s, org, current, rb, f"rule {d.target_id} rewritten from reviewers' feedback")
    flags = list(d.flags) + (["FLAGGED_FOR_REVIEW"] if d.flags else [])
    why = {"LOOSENS_GATE": "it loosens a gate that can reject", "CLIENT_RULE": "it changes a rule from the client's own document"}
    _event(s, org.organisation_id, d.edit_id, "APPLIED", "applied automatically (all checks passed)" + ("; flagged for developers: " + " and ".join(why[f] for f in d.flags) if d.flags else ""),
           {"flags": flags, "prompt_version": pv.version_number, "prompt_hash": pv.configuration.get("prompt_hash")}, pv.version_id)


def pending(s: Session, org_id: str) -> list[dict[str, Any]]:
    """The rewrites waiting for a person to confirm that a rule changed earlier may be changed again, oldest first, with what the person needs to decide: the rule, its wording in force,
    when and why it was changed before, and the wording now proposed."""
    st = status_map(s, org_id)
    live = active_edits(s, org_id)
    applied_at: dict[str, Any] = {}
    for e in s.scalars(select(RuleEditEvent).where(RuleEditEvent.organisation_id == org_id, RuleEditEvent.status == "APPLIED").order_by(RuleEditEvent.created_at)):
        applied_at[e.edit_id] = e.created_at
    out = []
    for d in s.scalars(select(RuleEdit).where(RuleEdit.organisation_id == org_id).order_by(RuleEdit.created_at)):
        if st.get(d.edit_id) != "NEEDS_CONFIRM":
            continue
        cur = live.get(d.target_id)
        photo = s.get(Photo, d.photo_id) if d.photo_id else None
        admin = ((cur.source or {}).get("administrator") or {}).get("by") if cur else None
        out.append({"edit_id": d.edit_id, "rule_id": d.target_id, "kind": d.target_kind, "gate_id": d.gate_id, "can_reject": d.hard_gate, "client_rule": d.client_sourced, "direction": d.direction,
                    "current": cur.new_text if cur else d.previous_text, "proposed": d.new_text, "why": d.rationale, "reviewer_point": (d.source or {}).get("summary") or "", "photo": photo.original_filename if photo else None,
                    "earlier_change": {"when": applied_at[cur.edit_id].isoformat() if cur and cur.edit_id in applied_at else None, "by_administrator": admin, "because": cur.rationale if cur else ""} if cur else None,
                    "created_at": d.created_at.isoformat() if d.created_at else None})
    return out


def answer(s: Session, org: Organisation, user: User, edit_id: str, accept: bool, rb: Rulebook) -> Optional[PromptVersion]:
    """A person answers the question "this rule was changed earlier: change it again?". Yes applies the waiting wording (the wording in force is superseded); no keeps the wording in force and
    records that. Raises NotFound or Conflict. Returns the new prompt version, or None when nothing changed."""
    d = s.scalar(select(RuleEdit).where(RuleEdit.edit_id == edit_id, RuleEdit.organisation_id == org.organisation_id))
    if d is None:
        raise NotFound("change not found")
    if status_map(s, org.organisation_id).get(d.edit_id) != "NEEDS_CONFIRM":
        raise Conflict("this change is not waiting for a confirmation")
    if not accept:
        _event(s, org.organisation_id, d.edit_id, "DECLINED", f"{user.email} chose to keep the wording in force", {"by": "a person"}, None, user)
        return None
    now_active = active_edits(s, org.organisation_id)
    prev = now_active.get(d.target_id)
    if prev is not None and norm(prev.new_text) == norm(d.new_text):
        _event(s, org.organisation_id, d.edit_id, "NO_CHANGE", "the rule already has exactly this wording", {"by": "a person"}, None, user)
        return None
    if prev is None and len(now_active) >= settings(org)["max_edits"]:
        raise Conflict(f"the limit of {settings(org)['max_edits']} rewritten rules is reached; withdraw one first")
    new_set = {t: x.new_text for t, x in now_active.items()}
    new_set[d.target_id] = d.new_text
    pv = _set_rules(s, org, new_set, rb, f"rule {d.target_id} changed again after {user.email} confirmed it")
    if prev is not None:
        _event(s, org.organisation_id, prev.edit_id, "SUPERSEDED", "replaced by a newer wording that a person confirmed", {"by_edit": d.edit_id}, pv.version_id)
    flags = list(d.flags) + (["FLAGGED_FOR_REVIEW"] if d.flags else [])
    _event(s, org.organisation_id, d.edit_id, "APPLIED", f"applied after {user.email} confirmed that the rule may be changed again", {"flags": flags, "by": "a person", "confirmed_by": user.email,
           "prompt_version": pv.version_number, "prompt_hash": pv.configuration.get("prompt_hash")}, pv.version_id, user)
    return pv


# ----------------------------------------------------------------------------------------------------------------------------------- the watch
def mattered(run: EvaluationRun, info: Any, rules: dict[str, Any]) -> bool:
    """Whether this rule played a part in the AI's answer on this photograph: the AI failed or doubted it (or, for a client rule, a check under it). A scene fact always counts."""
    if info.kind in ("SCENE_FACT", "GUIDANCE"):
        return True
    bad = {c.get("criterion_id") for c in (run.validated_response or {}).get("layer_2_criteria") or [] if c.get("status") in ("NO", "CANNOT_TELL")}
    if info.kind == "CLIENT_RULE":
        return any(rules.get(cid) and rules[cid].rule_id == info.target_id for cid in bad)
    return info.target_id in bad


def override_stats(s: Session, org_id: str, info: Any, rules: dict[str, Any], new_text: str) -> dict[str, int]:
    """Decisions on photographs where this rule mattered, split by whether the AI's prompt carried this wording, and how many of them the reviewer overrode.
    Counted from stored evaluations and reviews: no model call."""
    texts = {pv.version_id: (pv.configuration or {}).get("rule_text") or {} for pv in s.scalars(select(PromptVersion).where(PromptVersion.organisation_id == org_id))}
    out = {"before_n": 0, "before_overrides": 0, "after_n": 0, "after_overrides": 0}
    rows = s.execute(select(HumanReview, EvaluationRun).join(EvaluationRun, EvaluationRun.evaluation_id == HumanReview.evaluation_id)
                     .where(HumanReview.organisation_id == org_id, HumanReview.review_status == ReviewStatus.FEEDBACK_SUBMITTED.value))
    for review, run in rows:
        if not mattered(run, info, rules):
            continue
        side = "after" if texts.get(run.prompt_version_id, {}).get(info.target_id) == new_text else "before"
        out[f"{side}_n"] += 1
        out[f"{side}_overrides"] += review.overall_agreement_with_ai == "DISAGREE"
    return out


def watch(factory: sessionmaker[Session], organisation_id: str, rb: Rulebook, rules: dict[str, Any]) -> str:
    """Judge every applied rewrite that has been under watch long enough. Never raises."""
    try:
        with factory() as s:
            org = s.get(Organisation, organisation_id)
            cfg = settings(org)
            if cfg["mode"] == "OFF" or cfg["paused"]:
                return "SKIPPED:off"
            st = status_map(s, organisation_id)
            changed = 0
            for target, d in list(active_edits(s, organisation_id).items()):
                if target not in rules or (d.source or {}).get("administrator"):
                    continue                                        # an administrator's own wording is not judged or withdrawn by the automatic watch
                stats = override_stats(s, organisation_id, rules[target], rules, d.new_text)
                verdict = judge(stats, cfg)
                if verdict == "WITHDRAW":
                    rate = lambda n, o: f"{round(100 * o / n)}%"  # noqa: E731
                    withdraw(s, org, d, f"withdrawn automatically: reviewers overrode the AI on {rate(stats['after_n'], stats['after_overrides'])} of {stats['after_n']} decisions under this wording, "
                                        f"against {rate(stats['before_n'], stats['before_overrides'])} of {stats['before_n']} before it", rb, payload={"by": "the automatic watch", **stats})
                    changed += 1
                elif verdict == "CONFIRM" and st.get(d.edit_id) == "APPLIED":
                    _event(s, organisation_id, d.edit_id, "CONFIRMED", f"reviewers did not override the AI more often under this wording ({stats['after_overrides']} of {stats['after_n']} "
                                                                     f"decisions, against {stats['before_overrides']} of {stats['before_n']} before it)", stats)
                    changed += 1
            s.commit()
            return f"DONE:{changed}"
    except Exception as exc:  # noqa: BLE001
        log.warning("the rule-rewrite watch for %s failed: %s", organisation_id, type(exc).__name__)
        return "FAILED"


def withdraw(s: Session, org: Organisation, d: RuleEdit, reason: str, rb: Rulebook, user: Optional[User] = None, payload: Optional[dict] = None) -> PromptVersion:
    """Take one rewrite out of the prompt: the rule goes back to its wording in the rulebook, and the prompt version for the remaining rewrites becomes active."""
    remaining = {t: x.new_text for t, x in active_edits(s, org.organisation_id).items() if x.edit_id != d.edit_id}
    pv = _set_rules(s, org, remaining, rb, f"withdrew the rewrite of rule {d.target_id}")
    _event(s, org.organisation_id, d.edit_id, "WITHDRAWN", reason, {"by": "a person" if user else "the system", **(payload or {})}, pv.version_id, user)
    return pv


def withdraw_edit(s: Session, org: Organisation, user: User, edit_id: str, reason: str, rb: Rulebook) -> PromptVersion:
    if not (reason or "").strip():
        raise Invalid("a reason is required to withdraw a rewrite")
    d = s.scalar(select(RuleEdit).where(RuleEdit.edit_id == edit_id, RuleEdit.organisation_id == org.organisation_id))
    if d is None:
        raise NotFound("rewrite not found")
    if status_map(s, org.organisation_id).get(d.edit_id) not in ACTIVE:
        raise Conflict("this rewrite is not in the prompt, so there is nothing to withdraw")
    return withdraw(s, org, d, reason.strip(), rb, user)


def current_wording(s: Session, org_id: str, info: Any) -> str:
    """The wording of a rule in force now: the live rewrite if there is one, else its wording in the rulebook files."""
    live = active_edits(s, org_id).get(info.target_id)
    return live.new_text if live else info.base


def manual_edit(s: Session, org: Organisation, user: User, target_id: str, wording: str, reason: str, direction: str, rb: Rulebook, rules: dict[str, Any]) -> PromptVersion:
    """An administrator sets the wording of one rule by hand. The same checks as for a drafted rewrite apply (form, length, no other rule's id, no instruction to the evaluator); it takes effect at
    once whatever the mode (it is a person's decision, not an automatic one), replaces any live wording of that rule, and is kept in the same history. Raises Invalid, Conflict or NotFound."""
    info = rules.get(target_id)
    if info is None:
        raise NotFound(f"there is no rule {target_id!r} in the prompt")
    if not (reason or "").strip():
        raise Invalid("a reason is required")
    if direction not in ("LOOSEN", "TIGHTEN", "NEUTRAL"):
        raise Invalid("direction must be LOOSEN, TIGHTEN or NEUTRAL")
    cfg = settings(org)
    now_active = active_edits(s, org.organisation_id)
    prev = now_active.get(target_id)
    before = prev.new_text if prev else info.base
    failed = check_wording(wording, before, target_id, cfg["max_chars"])
    if failed:
        raise Invalid("the new wording cannot be used: " + "; ".join(failed))
    if norm(wording) == norm(before):
        raise Conflict("the rule already has exactly this wording")
    if prev is None and len(now_active) >= cfg["max_edits"]:
        raise Conflict(f"the limit of {cfg['max_edits']} rewritten rules is reached; withdraw one first")
    d = RuleEdit(organisation_id=org.organisation_id, review_id=None, evaluation_id=None, photo_id=None, target_id=target_id, target_kind=info.kind, outcome="UPDATE_RULE", direction=direction,
                 previous_text=before, new_text=norm(wording), rationale=reason.strip(), keeps_meaning=True, gate_id=info.gate_id, hard_gate=info.hard, client_sourced=info.client_sourced,
                 flags=[], checks={"passed": True, "failed": []}, source={"administrator": {"by": user.email, "user_id": user.user_id}, "summary": reason.strip(), "confidence": "ADMINISTRATOR"})
    s.add(d)
    s.flush()
    _event(s, org.organisation_id, d.edit_id, "DRAFTED", f"entered by an administrator ({user.email}): {reason.strip()}", user=user)
    new_set = {t: x.new_text for t, x in now_active.items()}
    new_set[target_id] = d.new_text
    pv = _set_rules(s, org, new_set, rb, f"rule {target_id} set by an administrator")
    if prev is not None:
        _event(s, org.organisation_id, prev.edit_id, "SUPERSEDED", "replaced by a wording an administrator entered", {"by_edit": d.edit_id}, pv.version_id)
    _event(s, org.organisation_id, d.edit_id, "APPLIED", "applied by an administrator", {"flags": [], "by": "a person", "prompt_version": pv.version_number, "prompt_hash": pv.configuration.get("prompt_hash")},
           pv.version_id, user)
    return pv


def after_decision(factory: sessionmaker[Session], review_id: str, drafter: Any, rb: Rulebook) -> str:
    """The background step after a submitted decision: rewrite, then watch. Never raises."""
    status = process_review(factory, review_id, drafter, rb)
    try:
        with factory() as s:
            review = s.get(HumanReview, review_id)
            org_id = review.organisation_id if review else None
        if org_id:
            watch(factory, org_id, rb, drafter.rules)
    except Exception as exc:  # noqa: BLE001
        log.warning("the rule-rewrite watch after review %s failed: %s", review_id, type(exc).__name__)
    return status


# ---------------------------------------------------------------------------------------------------------------------------------- the overview
def overview(s: Session, org: Organisation, drafter_available: bool, limit: int = 100, rules: dict[str, Any] | None = None) -> dict[str, Any]:
    st = status_map(s, org.organisation_id)
    events: dict[str, list[RuleEditEvent]] = {}
    for e in s.scalars(select(RuleEditEvent).where(RuleEditEvent.organisation_id == org.organisation_id).order_by(RuleEditEvent.edit_id, RuleEditEvent.seq)):
        events.setdefault(e.edit_id, []).append(e)
    drafts = list(s.scalars(select(RuleEdit).where(RuleEdit.organisation_id == org.organisation_id).order_by(RuleEdit.created_at.desc()).limit(limit)))

    def row(d: RuleEdit) -> dict[str, Any]:
        hist = [{"status": e.status, "reason": e.reason, "at": e.created_at.isoformat() if e.created_at else None, "by_person": e.created_by is not None, "payload": e.payload} for e in events.get(d.edit_id, [])]
        flags = sorted({*d.flags, *(f for e in events.get(d.edit_id, []) for f in (e.payload or {}).get("flags", []))})
        return {"edit_id": d.edit_id, "target_id": d.target_id, "target_kind": d.target_kind, "gate_id": d.gate_id, "hard_gate": d.hard_gate, "client_sourced": d.client_sourced, "outcome": d.outcome,
                "direction": d.direction, "previous": d.previous_text, "wording": d.new_text, "rationale": d.rationale, "status": st.get(d.edit_id, "DRAFTED"), "flags": flags, "checks": d.checks,
                "administrator": ((d.source or {}).get("administrator") or {}).get("by"), "source": d.source, "review_id": d.review_id, "created_at": d.created_at.isoformat() if d.created_at else None, "history": hist}

    rows = [row(d) for d in drafts]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    pv = s.get(PromptVersion, org.active_prompt_version_id) if org.active_prompt_version_id else None
    live = active_edits(s, org.organisation_id)
    return {"settings": settings(org), "drafter_available": drafter_available, "prompt_version": pv.version_number if pv else None,
            "active_rewrites": [r for r in rows if r["status"] in ACTIVE], "drafts": rows, "counts": counts,
            "rules": [{"rule_id": t, "kind": i.kind, "gate_id": i.gate_id, "can_reject": i.hard, "original": i.base, "now": live[t].new_text if t in live else i.base} for t, i in sorted((rules or {}).items())]}
