"""Parse the client's rules document and verify the machine-readable rule set against it.

The text document is the single source of truth. A rule is *mandatory* only if the document itself says
so (configurable marker words such as "mandatory"); nothing else may make a rule mandatory. If the
document changes, ``verify_against_source`` reports every difference so the rule set cannot silently
go stale.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..schemas import RequirementType, RuleSet

_INVISIBLE = {"⁠": " ", " ": " ", "​": ""}
_NEGATION = re.compile(r"\b(?:not|non)[\s-]+mandatory\b|\boptional\b", re.IGNORECASE)


def normalize(text: str) -> str:
    """Collapse whitespace and drop invisible characters (the source contains U+2060 word joiners)."""
    for bad, good in _INVISIBLE.items():
        text = text.replace(bad, good)
    return " ".join(text.split())


@dataclass(frozen=True)
class SourceRule:
    ref: str        # stable identifier: the section title, or "Final QC Test Q<n>"
    section: str    # the all-caps section heading the item sits under
    text: str       # the client's wording (normalized whitespace)


def _is_heading(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 4 and line == line.upper()


def parse_rules_document(document: str) -> list[SourceRule]:
    """Split the document into rule items.

    Layout understood: ALL-CAPS lines are section headings; inside a section, blank-line-separated
    blocks are ``title`` + ``body`` lines; numbered lines ("1. ...") are questions. Text before the first
    heading (the document title) is ignored.
    """
    rules: list[SourceRule] = []
    section: str | None = None
    block: list[str] = []
    seen: dict[str, int] = {}

    def add(ref: str, text: str) -> None:
        n = seen.get(ref, 0) + 1
        seen[ref] = n
        rules.append(SourceRule(ref if n == 1 else f"{ref} ({n})", section or "", normalize(text)))

    def flush() -> None:
        nonlocal block
        if block and section is not None:
            if len(block) == 1:
                add(block[0], block[0])
            else:
                add(block[0], " ".join(block[1:]))
        block = []

    for raw in document.splitlines():
        line = normalize(raw)
        if not line:
            flush()
            continue
        if _is_heading(line):
            flush()
            section = line
            continue
        if section is None:
            continue
        m = re.match(r"^(\d+)[.)]\s*(.+)$", line)
        if m:
            flush()
            add(f"Final QC Test Q{m.group(1)}", m.group(2))
        else:
            block.append(line)
    flush()
    # An intro line such as "Before submitting a photograph, ask:" is not a rule.
    return [r for r in rules if not r.text.rstrip().endswith(":")]


def is_mandatory_by_text(rule: SourceRule, markers: Iterable[str]) -> bool:
    """True only if the document's own words (section heading, title, body) mark the rule mandatory."""
    haystack = f"{rule.section} {rule.ref} {rule.text}"
    if _NEGATION.search(haystack):
        return False
    return any(re.search(rf"\b{re.escape(m)}\b", haystack, re.IGNORECASE) for m in markers)


def slugify(ref: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", ref.lower()).strip("_")
    return slug if slug and slug[0].isalpha() else f"rule_{slug}"


def verify_against_source(
    rule_set: RuleSet, source: Sequence[SourceRule], markers: Iterable[str]
) -> list[str]:
    """Return human-readable problems; an empty list means the rule set matches the document exactly."""
    markers = list(markers)
    problems: list[str] = []
    by_ref = {r.source_ref: r for r in rule_set.rules if r.source_ref}
    src_refs = {s.ref for s in source}

    for r in rule_set.rules:
        if r.enabled and not r.source_ref:
            problems.append(f"{r.rule_id} ({r.name}) is enabled but not linked to the rules document (source_ref empty)")
        elif r.enabled and r.source_ref not in src_refs:
            problems.append(f"{r.rule_id} ({r.name}): '{r.source_ref}' no longer exists in the rules document")

    for s in source:
        rule = by_ref.get(s.ref)
        if rule is None:
            problems.append(f"NEW in the rules document, not in rules.yaml: '{s.ref}': {s.text}")
            continue
        if normalize(rule.original_text) != s.text:
            problems.append(f"{rule.rule_id}: wording changed in the rules document for '{s.ref}'")
        wants = RequirementType.MANDATORY if is_mandatory_by_text(s, markers) else RequirementType.SCORED
        if rule.requirement_type != wants:
            problems.append(
                f"{rule.rule_id}: requirement_type is '{rule.requirement_type.value}' but the document "
                f"{'says' if wants == RequirementType.MANDATORY else 'does not say'} it is mandatory"
            )
    return problems


def sync_rule_dict(
    raw: dict, source: Sequence[SourceRule], markers: Iterable[str], today: str
) -> tuple[dict, list[str]]:
    """Return (updated rules.yaml mapping, change log) bringing the rule set in line with the document.

    * NEW items get a stub rule (wording verbatim, scored unless the text says mandatory).
    * CHANGED wording replaces ``original_text`` and ``description`` (re-review visual_evidence /
      applies_when afterwards - the note says so).
    * REMOVED items are disabled, never deleted (rule ids and history stay stable).
    * ``requirement_type`` is re-derived from the document's own words.
    """
    import copy

    markers = list(markers)
    out = copy.deepcopy(raw)
    rules: list[dict] = out["rules"]
    by_ref = {r.get("source_ref"): r for r in rules if r.get("source_ref")}
    src_refs = {s.ref for s in source}
    log: list[str] = []

    def apply_requirement(rule: dict, s: SourceRule) -> None:
        mandatory = is_mandatory_by_text(s, markers)
        want = "mandatory" if mandatory else "scored"
        if rule.get("requirement_type") != want:
            log.append(f"{rule['rule_id']}: requirement_type -> {want} (per the document's wording)")
            rule["requirement_type"] = want
        rule["uncertain_policy"] = "reject" if mandatory else "score_only"
        rule["mandatory_status_confirmed"] = True

    next_id = max((int(r["rule_id"][1:]) for r in rules), default=0) + 1
    for s in source:
        rule = by_ref.get(s.ref)
        if rule is None:
            rule = {
                "rule_id": f"R{next_id:03d}", "name": slugify(s.ref), "original_text": s.text, "source_ref": s.ref,
                "description": s.text, "visual_evidence": s.text, "applies_when": "", "requirement_type": "scored",
                "mandatory_status_confirmed": True, "check_method": "vlm", "enabled": True,
                "uncertain_policy": "score_only", "applies_to_programs": [], "confirmed": True,
                "needs_manual_annotation": True, "limitations": "",
                "notes": f"ADDED by sync on {today}: review visual_evidence / applies_when.",
            }
            next_id += 1
            rules.append(rule)
            log.append(f"{rule['rule_id']}: ADDED for new item '{s.ref}'")
        elif normalize(rule["original_text"]) != s.text:
            rule["original_text"], rule["description"] = s.text, s.text
            rule["notes"] = (rule.get("notes", "") + f" CHANGED by sync on {today}: re-review visual_evidence / applies_when.").strip()
            rule["enabled"] = True
            log.append(f"{rule['rule_id']}: wording UPDATED for '{s.ref}'")
        apply_requirement(rule, s)

    for rule in rules:
        ref = rule.get("source_ref")
        if ref and ref not in src_refs and rule.get("enabled", True):
            rule["enabled"] = False
            rule["notes"] = (rule.get("notes", "") + f" REMOVED from the document; disabled by sync on {today}.").strip()
            log.append(f"{rule['rule_id']}: DISABLED ('{ref}' no longer in the document)")

    if log:
        import re as _re

        base = _re.sub(r"\+\d{8}$", "", str(out.get("rule_set_version", "1")))
        out["rule_set_version"] = f"{base}+{today.replace('-', '')}"
        out["rule_set_confirmed"] = False
    return out, log
