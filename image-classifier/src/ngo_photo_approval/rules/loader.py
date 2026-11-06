"""Load and validate the machine-readable rule set (configs/rules.yaml).

The client's text document is the source of truth: by default the rule set is verified against it and
loading FAILS if they disagree (new/changed/removed rules, or a rule made mandatory when the document
does not say so), so a stale rules.yaml can never be used silently.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from pydantic import ValidationError

from ..config import CONFIG_DIR, load_project_config, load_yaml
from ..exceptions import ConfigError, RuleConfigError
from ..schemas import RuleSet
from .source import parse_rules_document, verify_against_source

logger = logging.getLogger(__name__)

SYNC_HINT = "Run `python scripts/sync_rules.py` to see the differences and `--write` to apply them."


def source_problems(rs: RuleSet) -> list[str]:
    """Differences between ``rs`` and the client's rules document (empty list = in sync)."""
    cfg = load_project_config()
    doc = cfg.resolve(cfg.paths.rules_document) if cfg.paths.rules_document else None
    if doc is None or not doc.is_file():
        return [f"rules document not found ({cfg.paths.rules_document}); it is the source of truth for the rules"]
    source = parse_rules_document(doc.read_text(encoding="utf-8"))
    if not source:
        return ["no rules could be parsed from the rules document"]
    return verify_against_source(rs, source, cfg.rules.mandatory_markers)


def load_rule_set(path: Path | None = None, check_source: bool | None = None) -> RuleSet:
    """Parse + validate. Raises RuleConfigError on any problem.

    ``check_source`` defaults to True for the shipped ``configs/rules.yaml`` (and honours
    ``rules.enforce_source_sync`` in project.yaml) and to False when an explicit path is given.
    """
    explicit = path is not None
    path = path or CONFIG_DIR / "rules.yaml"
    try:
        raw = load_yaml(path)
    except ConfigError as exc:
        raise RuleConfigError(str(exc)) from exc
    try:
        rs = RuleSet.model_validate(raw)
    except ValidationError as exc:
        raise RuleConfigError(f"invalid rule set {path.name}: {exc}") from exc
    if not rs.active():
        raise RuleConfigError("rule set has no enabled rules")
    if check_source is None:
        check_source = (not explicit) and load_project_config().rules.enforce_source_sync
    if check_source:
        problems = source_problems(rs)
        if problems:
            raise RuleConfigError(
                "rules.yaml is out of sync with the rules document:\n  - " + "\n  - ".join(problems) + "\n" + SYNC_HINT
            )
    return rs


def rule_set_fingerprint(rs: RuleSet) -> str:
    """Content hash so results can be tied to the exact rule definitions used."""
    payload = json.dumps(rs.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def config_warnings(rs: RuleSet) -> list[str]:
    """Human-readable warnings about unresolved/unconfirmed configuration."""
    out: list[str] = []
    if not rs.rule_set_confirmed:
        out.append(f"rule set {rs.rule_set_version!r} is NOT confirmed by the client")
    for r in rs.rules:
        if not r.enabled:
            continue
        if not r.confirmed:
            out.append(f"{r.rule_id} ({r.name}): wording not confirmed by the client")
        if r.requirement_type.value == "mandatory" and not r.mandatory_status_confirmed:
            out.append(f"{r.rule_id} ({r.name}): mandatory status not confirmed")
    return out
