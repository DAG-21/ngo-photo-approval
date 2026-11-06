"""Assembling the judge prompt from files.

Every piece comes from a file (the client's rules from the main project's rules.yaml, which is tied to the client's
document; everything else from prompts/<version>/). Nothing is typed in by hand at run time, and the final text is
hashed so each result can be traced to the exact prompt that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import PROMPTS_DIR, TrialConfig, ensure_main_importable
from .data import image_path
from .checklist import check_ids, checklist_schema, checks_block, load_checks
from .schema import response_schema

USER_TEXT = "Judge this photograph using the rules and the output format above."
ACK_TEXT = "Understood. Send the photograph to judge."
CHECKLIST_USER_TEXT = "Answer every question above for this photograph."


@dataclass(frozen=True)
class ExampleSpec:
    image_id: str
    label: str
    reason: str | None
    path: Path

    @property
    def decision_text(self) -> str:
        verdict = "APPROVED" if self.label == "approved" else "DISAPPROVED"
        base = f"Reviewer decision: {verdict}."
        return (f"{base} Reason (tagged afterwards by the NGO team; the original reviewers' own reason was not recorded): {self.reason}"
                if self.reason else base)


@dataclass
class BuiltPrompt:
    variant: str
    version: str
    system_text: str
    examples_intro: str
    examples: list[ExampleSpec]
    user_text: str
    schema: dict[str, Any]
    rule_ids: list[str]
    hash: str = ""
    flags: dict[str, bool] = field(default_factory=dict)
    summary_reason: bool = False
    check_ids: list[str] | None = None  # set in checklist mode: the yes/no questions the model must answer


def _read(version_dir: Path, name: str) -> str:
    return (version_dir / name).read_text(encoding="utf-8").strip()


def load_variants(version: str = "v1") -> dict[str, Any]:
    return yaml.safe_load((PROMPTS_DIR / version / "variants.yaml").read_text(encoding="utf-8"))


def verbatim_block(rules) -> str:
    lines = ["## The client's rules (copied word for word from the client's document)", ""]
    for r in rules:
        lines.append(f"- {r.rule_id} - {r.source_ref}: {r.original_text}")
    return "\n".join(lines)


def restated_block(rules, groups: dict[str, list[str]]) -> str:
    by_id = {r.rule_id: r for r in rules}
    listed = [i for ids in groups.values() for i in ids]
    if sorted(listed) != sorted(by_id) or len(set(listed)) != len(listed):
        missing, extra = sorted(set(by_id) - set(listed)), sorted(set(listed) - set(by_id))
        raise ValueError(
            f"variants.yaml restated_groups does not match the enabled rules (not grouped: {missing}; unknown: {extra}). "
            "Update prompts/<version>/variants.yaml - a rule must never be silently left out.")
    lines = ["## The same rules, grouped and restated (nothing here adds a requirement)", ""]
    for title, ids in groups.items():
        lines.append(f"### {title}")
        for i in ids:
            r = by_id[i]
            cond = f" (applies when {r.applies_when})" if r.applies_when else ""
            lines.append(f"- {r.rule_id}: {r.description}{cond}")
        lines.append("")
    return "\n".join(lines).rstrip()


def guidance_block(rules, replacement: tuple[list[str], list[str]] | None = None) -> str | None:
    """Team guidance, stated once. ``replacement`` = (the rule guidance being replaced, the new lines): used by a prompt
    version that amends the team's guidance. It must match the guidance the rules carry exactly, so nothing else is hidden."""
    seen: list[str] = []
    for r in rules:
        g = (r.additional_guidance or "").strip()
        if g and g not in seen:
            seen.append(g)
    if replacement is not None:
        replaced, new_lines = replacement
        if sorted(seen) != sorted(s.strip() for s in replaced):
            raise ValueError("the rules' additional_guidance no longer matches team_guidance.replaces_rule_guidance in "
                             f"variants.yaml (rules carry {len(seen)} distinct guidance text(s)); review and update the prompt version.")
        seen = new_lines
    if not seen:
        return None
    lines = ["## Team guidance (added by the NGO team; it is NOT in the client's document)", ""]
    lines += [f"- {g}" for g in seen]
    return "\n".join(lines)


def build_prompt(cfg: TrialConfig, variant: str, rule_set, examples: list[dict], version: str = "v1") -> BuiltPrompt:
    spec = load_variants(version)
    if variant not in spec["variants"]:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(spec['variants'])}")
    flags = spec["variants"][variant]
    vdir = PROMPTS_DIR / version
    rules = rule_set.active()
    if spec.get("mode") == "checklist":
        return _build_checklist_prompt(vdir, version, variant, flags, rules)
    parts = [_read(vdir, "persona.md"), verbatim_block(rules)]
    if flags["restated"]:
        parts.append(restated_block(rules, spec["restated_groups"]))
    if flags["guidance"]:
        replacement = None
        tg = spec.get("team_guidance")
        if tg:
            lines = [ln.strip() for ln in _read(vdir, tg["file"]).splitlines() if ln.strip()]
            replacement = ([" ".join(str(s).split()) for s in tg["replaces_rule_guidance"]], lines)
        g = guidance_block(rules, replacement)
        if g:
            parts.append(g)
    parts += [_read(vdir, "decision_guide.md"), _read(vdir, "output_instructions.md")]
    system_text = "\n\n".join(parts)

    chosen: list[ExampleSpec] = []
    if flags["examples"]:
        if not examples:
            raise ValueError("variant needs examples but none are saved; run scripts/select_examples.py")
        chosen = [ExampleSpec(e["image_id"], e["label"], e.get("reason"), image_path(cfg, e["image_id"])) for e in examples]
    rule_ids = [r.rule_id for r in rules]
    summary_reason = bool(spec.get("summary_reason", False))
    schema = response_schema(rule_ids, summary_reason)
    intro = " ".join(str(spec["examples_intro"]).split())
    digest = hashlib.sha256(json.dumps({
        "version": version, "variant": variant, "system": system_text, "intro": intro if chosen else "",
        "examples": [(e.image_id, e.label, e.reason) for e in chosen], "user": USER_TEXT, "schema": schema,
    }, sort_keys=True).encode()).hexdigest()[:16]
    return BuiltPrompt(variant, version, system_text, intro, chosen, USER_TEXT, schema, rule_ids, digest, dict(flags), summary_reason)


def _build_checklist_prompt(vdir: Path, version: str, variant: str, flags: dict, rules) -> BuiltPrompt:
    cspec = load_checks(version)
    ids = check_ids(cspec)
    system_text = "\n\n".join([_read(vdir, "persona.md"), verbatim_block(rules), _read(vdir, "definitions.md"),
                               checks_block(cspec, rules), _read(vdir, "output_instructions.md")])
    schema = checklist_schema(ids)
    digest = hashlib.sha256(json.dumps({"version": version, "variant": variant, "system": system_text,
                                        "user": CHECKLIST_USER_TEXT, "schema": schema}, sort_keys=True).encode()).hexdigest()[:16]
    return BuiltPrompt(variant, version, system_text, "", [], CHECKLIST_USER_TEXT, schema, [r.rule_id for r in rules], digest,
                       dict(flags), False, ids)


def build_for_variant(cfg: TrialConfig, variant: str, version: str = "v1") -> BuiltPrompt:
    from .examples import load_examples

    ensure_main_importable(cfg)
    from ngo_photo_approval.rules.loader import load_rule_set

    return build_prompt(cfg, variant, load_rule_set(), load_examples(cfg), version)


def render_text(p: BuiltPrompt) -> str:
    """Human-readable view of everything sent before the photo being judged (images shown as placeholders)."""
    out = ["=== SYSTEM INSTRUCTION ===", p.system_text]
    if p.examples:
        out += ["", "=== WORKED EXAMPLES (sent once, then cached when caching is on) ===", p.examples_intro]
        for n, e in enumerate(p.examples, 1):
            out += [f"[Example {n}]", f"[image: {e.image_id}]", e.decision_text]
        out += [f"[model]: {ACK_TEXT}"]
    out += ["", "=== PER-PHOTO MESSAGE ===", "[image: the photograph to judge]", p.user_text]
    return "\n".join(out)
