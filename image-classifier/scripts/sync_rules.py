"""Keep configs/rules.yaml in sync with the client's rules document (the single source of truth).

    python scripts/sync_rules.py            # report differences; exit code 1 if out of sync
    python scripts/sync_rules.py --write    # apply: add new rules, update changed wording, disable removed rules,
                                            # and re-derive mandatory/scored from the document's own words

A rule is mandatory ONLY if the document says so (marker words in configs/project.yaml -> rules.mandatory_markers).
After `--write`, review the `notes` of added/changed rules: visual_evidence / applies_when are restatements
written for the model and must not add requirements the text does not contain.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import yaml

from ngo_photo_approval.config import CONFIG_DIR, load_project_config, load_yaml
from ngo_photo_approval.exceptions import NgoPhotoError
from ngo_photo_approval.rules.loader import load_rule_set
from ngo_photo_approval.rules.source import (
    is_mandatory_by_text,
    parse_rules_document,
    sync_rule_dict,
    verify_against_source,
)
from ngo_photo_approval.schemas import RuleSet

HEADER = """\
# Machine-readable form of the client's rules document (paths.rules_document in configs/project.yaml).
# THE TEXT DOCUMENT IS THE SINGLE SOURCE OF TRUTH; this file only adds structure. Managed with scripts/sync_rules.py.
#   * original_text is verbatim; source_ref ties each rule to its item in the document.
#   * A rule is `mandatory` ONLY if the document itself says so (rules.mandatory_markers in project.yaml).
#   * If the document changes, loading refuses to proceed until: python scripts/sync_rules.py --write
#   * description / visual_evidence / applies_when restate the wording for the model; they must never add requirements.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="apply the changes to configs/rules.yaml")
    args = ap.parse_args()

    cfg = load_project_config()
    doc = cfg.resolve(cfg.paths.rules_document)
    if not doc.is_file():
        print(f"rules document not found: {cfg.paths.rules_document}", file=sys.stderr)
        return 2
    source = parse_rules_document(doc.read_text(encoding="utf-8"))
    markers = cfg.rules.mandatory_markers
    print(f"document: {cfg.paths.rules_document} -> {len(source)} items")
    for s in source:
        print(f"  [{'MANDATORY' if is_mandatory_by_text(s, markers) else 'scored   '}] {s.ref}")

    path = CONFIG_DIR / "rules.yaml"
    raw = load_yaml(path)
    try:
        problems = verify_against_source(RuleSet.model_validate(raw), source, markers)
    except Exception as exc:  # noqa: BLE001 - report invalid yaml clearly
        print(f"rules.yaml is invalid: {exc}", file=sys.stderr)
        return 2
    if not problems:
        print("rules.yaml is IN SYNC with the document.")
        return 0

    print(f"\n{len(problems)} difference(s):")
    for p in problems:
        print("  -", p)
    if not args.write:
        print("\nRe-run with --write to apply, or edit configs/rules.yaml by hand.")
        return 1

    new_raw, log = sync_rule_dict(raw, source, markers, date.today().isoformat())
    RuleSet.model_validate(new_raw)  # never write an invalid file
    path.write_text(HEADER + yaml.safe_dump(new_raw, sort_keys=False, allow_unicode=True, width=10_000), encoding="utf-8")
    print("\napplied:")
    for line in log:
        print("  +", line)
    try:
        load_rule_set()
        print("rules.yaml now loads cleanly and matches the document. Review the notes of added/changed rules.")
    except NgoPhotoError as exc:
        print(f"still not in sync: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
