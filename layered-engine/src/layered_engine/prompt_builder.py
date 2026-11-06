"""Builds the effective prompt from the rulebook and the active profile. Nothing about a rule is typed into this file:
the criteria, their wording, which ones are mandatory gates, the exceptions, the scoring dimensions and the activity list all come
from config/rulebook/*.yaml, so changing the configuration changes the prompt (and its hash).

Static prose lives in config/prompt_sections/*.md. The prompt never contains a historical decision.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rulebook import CONFIG_DIR, Criterion, ResolvedProfile, Rulebook

PROMPT_TEMPLATE_VERSION = "1.0"
SECTIONS_DIR = CONFIG_DIR / "prompt_sections"

FACT_FIELDS = {  # which scene-fact criterion feeds which response field
    "F_PHOTO": "f_photo", "F_SINGLE": "f_single", "F_VOLUNTEERS": "f_volunteers", "F_BANNER": "f_banner", "F_BANNER_PROMINENT": "f_banner_prominent",
    "F_BENEFICIARIES": "f_beneficiaries", "F_ANIMALS": "f_animals", "F_ACTIVITY_TYPE": "f_activity_type", "F_ITEM_GIVEN": "f_item_given",
    "F_HANDOVER": "f_handover", "F_VOLUNTEER_GIVER": "f_volunteer_giver", "F_ORIENTATION": "f_orientation", "F_LOGO": "f_logo",
    "F_MEDICAL": "f_medical", "F_GROUP": "f_group", "F_GAZE_TAG": "f_gaze_tag",
}


def asked_fact_fields(profile: ResolvedProfile) -> set[str]:
    """The response fields of the scene facts this profile asks the model for (a fact the profile drops is neither in the prompt nor in the answer schema)."""
    return {FACT_FIELDS[cid] for cid, c in profile.criteria.items() if c.decision_effect == "FACT" and cid in FACT_FIELDS}


@dataclass
class BuiltPrompt:
    system_text: str
    user_text: str
    template_version: str
    profile: str
    rulebook_version: str
    criterion_ids: list[str]
    hash: str


def _read(name: str) -> str:
    return (SECTIONS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def _tag(c: Criterion, gate_id: str | None) -> str:
    src = {"CLIENT": "client rule broken down", "TEAM": "team rule, not in the client's document", "PROPOSED": "proposed, awaiting NGO approval"}[c.source]
    gate = f"MANDATORY GATE {gate_id}; " if gate_id else ""
    return f"[{gate}{src}]"


def _norm(text: object) -> str:
    return " ".join(str(text or "").split())


def guidance_of(profile: ResolvedProfile) -> dict[str, str]:
    """The team guidance statements of the profile (section E of the prompt), each with a stable id (GUIDE_01, GUIDE_02, ... in the order they are listed), so that each can be rewritten like any other rule."""
    return {f"GUIDE_{i + 1:02d}": str(line) for i, line in enumerate(profile.settings.get("guidance", []))}


def build_prompt(rb: Rulebook, profile: ResolvedProfile, notes: dict[str, str] | None = None, rule_text: dict[str, str] | None = None,
                 response_format: str = "full") -> BuiltPrompt:
    """`notes` (optional): short clarifications written from the NGO's reviewers' feedback, one per gate id (platform L5). With no notes the prompt, and so its hash,
    are exactly what they have always been; with notes they are added as section E2 and the hash changes. A note for a gate the profile does not have is ignored.

    `rule_text` (optional): new wording for individual rules, written from the reviewers' feedback (platform L5, rule rewrites). The key is a criterion id (its question, in
    sections B and D), a client rule id such as R004 (its text in section C) or a team guidance id such as GUIDE_02 (section E); an id the prompt does not show is ignored. The rulebook files are never changed: the rewrite lives
    only in this call, so the original wording is always still there. With no rewrites the prompt and its hash are exactly what they have always been.

    `response_format`: "full" (the default: exactly the prompt it has always been) or "compact" (the short answer format of compact.py: the same questions, but no subject list,
    no prose for YES answers or quality grades, no observation lists; the hash then records the format)."""
    if response_format not in ("full", "compact"):
        raise ValueError(f"unknown response format {response_format!r}")
    compact = response_format == "compact"
    wording = {k: _norm(v) for k, v in (rule_text or {}).items()}
    client_ids = {r["rule_id"] for r in rb.sources["client_rules"]}
    guidance = guidance_of(profile)
    used_text = {k: v for k, v in wording.items() if v and (k in profile.criteria or k in client_ids or k in guidance)}
    gate_of = {g.criterion_id: g.gate_id for g in profile.hard_gates()}
    for g in profile.hard_gates():
        for comp in g.components:
            gate_of.setdefault(comp, g.gate_id)
    intro = _read("role_and_philosophy")
    for rep in profile.settings.get("prompt_text_replace") or []:   # a profile may replace a sentence of the fixed introduction (and fails loudly if the sentence is not there)
        if rep["find"] not in intro:
            raise ValueError(f"prompt_text_replace: {rep['find']!r} is not in the introduction")
        intro = intro.replace(rep["find"], rep["replace"])
    parts: list[str] = [intro]

    # ---- B. Layer 1 -------------------------------------------------------------------------------------------------
    lines = ["## B. Layer 1: understand the scene first (fill `scene_facts` and `activity_description`)" if compact else
             "## B. Layer 1: understand the scene first (fill `scene_facts`, `subjects`, `activity_description`)", ""]
    facts = [c for c in profile.criteria.values() if c.decision_effect == "FACT"]
    for c in sorted(facts, key=lambda c: c.criterion_id):
        if c.criterion_id in FACT_FIELDS:
            lines.append(f"- `{FACT_FIELDS[c.criterion_id]}` ({c.criterion_id}): {used_text.get(c.criterion_id, c.description_operational)}")
    lines += ["", "Activity types (use exactly one id for `f_activity_type`):"]
    lines += [f"- {a['id']}: {a['description']}" for a in profile.taxonomy]
    if compact:
        lines += ["", "The scope of the face requirements is stated in each criterion. Do not infer age or identity."]
    else:
        lines += ["", "Subjects: list each distinct person or animal group with a `ref`, a `category` (VOLUNTEER, BENEFICIARY, ANIMAL, OTHER) and a `role`:",
                  "PRIMARY = visually central or featured; SECONDARY = supports the activity; INCIDENTAL = in the background. "
                  "The scope of the face requirements is stated in each criterion. Do not infer age or identity.",
                  "`candidate_applicable_rule_groups` is advisory: the rule engine decides what applies."]
    parts.append("\n".join(lines))

    # ---- C. the client's rules, verbatim --------------------------------------------------------------------------------
    # a profile may reword some of the client's lines (`client_rule_text`) and leave others out (`client_rules_hidden`): the owner decided on 2026-11-04 that the client's rules are
    # mutable. rule_sources.yaml keeps the client's own words untouched; a rewrite stored for an organisation (`rule_text`) still wins over the profile's wording.
    profile_wording = profile.settings.get("client_rule_text") or {}
    hidden = set(profile.settings.get("client_rules_hidden") or [])
    cr = ["## C. The client's rules (the NGO's own wording, for reference)", ""]
    cr += [f"- {r['rule_id']} - {r['source_ref']}: "
           f"{used_text.get(r['rule_id'], _norm(profile_wording[r['rule_id']]) if r['rule_id'] in profile_wording else r['description_original'])}"
           for r in rb.sources["client_rules"] if r["rule_id"] not in hidden]
    cr += ["", "The client's document does not mark any rule as mandatory. The mandatory gates below are the NGO team's policy applied on top of it."]
    parts.append("\n".join(cr))

    # ---- D. Layer 2 criteria ---------------------------------------------------------------------------------------------
    asked = [c for c in profile.asked_criteria() if c.layer == 2 or c.decision_effect in ("GATE", "DIAGNOSTIC", "SOFT_SCORE")]
    by_parent: dict[str, list[Criterion]] = {}
    for c in asked:
        by_parent.setdefault(c.parent_rule_id or "OTHER", []).append(c)
    d = ["## D. Layer 2 and 3 criteria: answer EVERY one of these (see the response rules)", "",
         "Each line: `criterion_id` [what it is]: the question. YES = the requirement is met. "
         + ("Give evidence for every NO and CANNOT_TELL." if compact else "Mandatory gates must carry evidence."), ""]
    for parent in sorted(by_parent, key=lambda p: (p == "OTHER", p)):
        d.append(f"### {parent}")
        for c in sorted(by_parent[parent], key=lambda c: (c.priority, c.criterion_id)):
            gid = gate_of.get(c.criterion_id)
            text = used_text.get(c.criterion_id, c.description_operational)
            if c.na_condition_text:
                text += f" (NOT_APPLICABLE only if {c.na_condition_text})"
            d.append(f"- `{c.criterion_id}` {_tag(c, gid)}: {text}")
        d.append("")
    parts.append("\n".join(d).rstrip())

    # ---- E. exceptions and precedence ------------------------------------------------------------------------------------
    e = ["## E. Team guidance, exceptions and precedence", ""]
    e += [f"- {used_text.get(gid, line)}" for gid, line in guidance.items()]
    e += ["- Mandatory gates outrank everything else. Quality grades never excuse a failed mandatory requirement.",
          "- Answer the criteria as they are written. Exceptions, if any are active, are applied afterwards by the rule engine, not by you:"]
    if profile.exceptions:
        for ex in profile.exceptions:
            e.append(f"  - {ex['id']}: {' '.join(ex['description'].split())}")
    else:
        e.append("  - (none are active)")
    parts.append("\n".join(e))

    # ---- E2. reviewers' notes (only when there are any) -----------------------------------------------------------------------
    valid_gates = {g.gate_id for g in profile.gates}
    used_notes = {gid: " ".join(str(txt).split()) for gid, txt in (notes or {}).items() if gid in valid_gates and " ".join(str(txt).split())}
    if used_notes:
        parts.append("\n".join(["## E2. Notes from this NGO's reviewers (how they read specific mandatory gates)", "",
                                "A note adds to how a gate's question is read. It never changes the question, never overrides the client's rules and never changes how you must "
                                "answer. If a note seems to conflict with the gate's own question, follow the question.", ""]
                               + [f"- Gate {gid}: {used_notes[gid]}" for gid in sorted(used_notes)]))

    # ---- F. Layer 3 ------------------------------------------------------------------------------------------------------
    q = profile.quality
    f = ["## F. Layer 3: grade each quality dimension 0-4", "",
         "Scale: " + ", ".join(f"{k} = {v}" for k, v in q["scale"].items()) + ".",
         "Mark a dimension `applicable: false` only when it genuinely does not apply (for example brand presentation when no banner, logo or uniform is visible).",
         ("Give only the score, a whole number (-1 when the dimension does not apply). " if compact else
          "Give a score, a short explanation, positive and negative observations, and the evidence you saw. ")
         + "Do not compensate a weak dimension with a strong one, and do not count one defect in two dimensions. You grade; you do not compute any total.", ""]
    rubric = profile.settings.get("quality_rubric") or {}   # optional per-dimension guidance on what the low scores mean (layered_final_v1 uses it for the soft gates' degree)
    for name, cfg in q["dimensions"].items():
        f.append(f"- `{name}`: {cfg['label']}" + (f". {_norm(rubric[name])}" if name in rubric else ""))
    parts.append("\n".join(f))

    # ---- G. Layer 4 visual flags -----------------------------------------------------------------------------------------
    g = ["## G. Layer 4: visual flags (report only what is visible; you cannot verify anything)", ""]
    for name, rule in profile.publication["visual_flags"].items():
        g.append(f"- `{name}`")
    g.append("")
    g.append("`POSSIBLE_MINOR` means a child might be identifiable and needs verification: it is NOT a statement about anyone's age.")
    parts.append("\n".join(g))

    # ---- H. observations ---------------------------------------------------------------------------------------------------
    if not compact:   # the short format has no observation lists
        parts.append("## H. Additional observations\n\nUse `positive_attributes`, `potential_rejection_factors` (only issues you can point to in the photograph, "
                     "never a guessed historical reason) and `new_factor_proposals` (a concern that none of the listed criteria covers) sparingly.")
    parts.append(_read("response_rules_compact" if compact else "response_rules"))

    system = "\n\n".join(parts)
    user = "Assess this photograph following the instructions. Return only the JSON object."
    ids = [c.criterion_id for c in asked]
    version = str(profile.settings.get("prompt_version", PROMPT_TEMPLATE_VERSION))
    hashed: dict[str, Any] = {"v": version, "profile": profile.name, "rb": profile.rulebook_version, "system": system, "user": user}
    if used_notes:
        hashed["notes"] = used_notes   # only when present, so the hash of every prompt without notes is unchanged
    if used_text:
        hashed["rule_text"] = used_text   # likewise for rewritten rules
    if compact:
        hashed["format"] = "compact"      # and for the short answer format
    digest = hashlib.sha256(json.dumps(hashed, sort_keys=True).encode()).hexdigest()[:16]
    return BuiltPrompt(system_text=system, user_text=user, template_version=version, profile=profile.name,
                       rulebook_version=profile.rulebook_version, criterion_ids=ids, hash=digest)
