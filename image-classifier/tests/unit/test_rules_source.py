from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from ngo_photo_approval.config import CONFIG_DIR, PROJECT_ROOT, load_project_config
from ngo_photo_approval.exceptions import RuleConfigError
from ngo_photo_approval.rules import loader
from ngo_photo_approval.rules.loader import load_rule_set
from ngo_photo_approval.rules.source import (
    is_mandatory_by_text,
    parse_rules_document,
    sync_rule_dict,
    verify_against_source,
)
from ngo_photo_approval.schemas import RuleSet

MARKERS = ["mandatory", "compulsory", "must", "non-negotiable", "without fail"]
DOC = (PROJECT_ROOT / load_project_config().paths.rules_document).read_text(encoding="utf-8")


def real_raw() -> dict:
    return yaml.safe_load((CONFIG_DIR / "rules.yaml").read_text(encoding="utf-8"))


def verify(raw: dict, doc: str = DOC) -> list[str]:
    return verify_against_source(RuleSet.model_validate(raw), parse_rules_document(doc), MARKERS)


def test_parses_the_real_document():
    items = parse_rules_document(DOC)
    assert len(items) == 16
    assert items[0].ref == "Banner Visibility & Presentation" and items[0].text.startswith("If a banner is present")
    assert [i.ref for i in items[-5:]] == [f"Final QC Test Q{n}" for n in range(1, 6)]
    assert items[-1].text == "Does it reflect the spirit of humanity and compassion?"  # numbering + invisible chars stripped
    assert not any(is_mandatory_by_text(i, MARKERS) for i in items)  # the document never says mandatory


def test_real_rule_set_matches_document():
    assert verify(real_raw()) == []


MANDATORY_DOC = """Title

CHECKS

Banner
A banner is mandatory in every photo.

Lighting
Ensure photos are well lit.

MANDATORY CHECKS

Faces
Show faces.

Optional Extra
This is optional and not mandatory.
"""


def test_mandatory_only_when_the_text_says_so():
    by_ref = {i.ref: i for i in parse_rules_document(MANDATORY_DOC)}
    assert is_mandatory_by_text(by_ref["Banner"], MARKERS)  # body says "mandatory"
    assert not is_mandatory_by_text(by_ref["Lighting"], MARKERS)  # "Ensure ..." is not a marker
    assert is_mandatory_by_text(by_ref["Faces"], MARKERS)  # the section heading says MANDATORY
    assert not is_mandatory_by_text(by_ref["Optional Extra"], MARKERS)  # negation / optional wins


@pytest.mark.parametrize("sentence", ["The banner must be visible.", "This is non-negotiable.", "Do it without fail.",
                                      "This is compulsory."])
def test_other_marker_words(sentence):
    item = parse_rules_document(f"CHECKS\n\nBanner\n{sentence}\n")[0]
    assert is_mandatory_by_text(item, MARKERS)


def test_words_that_merely_resemble_markers_do_not_count():
    item = parse_rules_document("CHECKS\n\nBanner\nThe photograph requires a muster of volunteers; mustard is fine.\n")[0]
    assert not is_mandatory_by_text(item, MARKERS)


# ------------------------------------------------------------------ drift detection


def test_detects_changed_wording_new_and_removed_items():
    changed = DOC.replace("Ensure photographs are sharp and in focus.", "Ensure photographs are razor sharp.")
    assert any("R010: wording changed" in p for p in verify(real_raw(), changed))

    added = DOC.replace("Clean Background\n", "Framing Rule\nKeep the subject centred.\n\nClean Background\n")
    assert any("NEW in the rules document" in p and "Framing Rule" in p for p in verify(real_raw(), added))

    removed = DOC.replace("Clean Background", "Tidy Background")
    assert any("'Clean Background' no longer exists" in p for p in verify(real_raw(), removed))


def test_detects_rule_made_mandatory_without_text_and_text_mandatory_not_applied():
    raw = real_raw()
    raw["rules"][0]["requirement_type"] = "mandatory"
    raw["rules"][0]["uncertain_policy"] = "reject"
    assert any("R001" in p and "does not say it is mandatory" in p for p in verify(raw))

    doc = DOC.replace("Proper Lighting\nEnsure subjects", "Proper Lighting\nThis is mandatory. Ensure subjects")
    assert any("R009" in p and "says it is mandatory" in p for p in verify(real_raw(), doc))


def test_enabled_rule_without_source_link_is_rejected():
    raw = real_raw()
    extra = copy.deepcopy(raw["rules"][0])
    extra.update(rule_id="R099", name="made_up_rule", source_ref="", original_text="Invented.")
    raw["rules"].append(extra)
    assert any("R099" in p and "not linked" in p for p in verify(raw))
    extra["enabled"] = False
    assert not any("R099" in p for p in verify(raw))


# ------------------------------------------------------------------ sync


def test_sync_applies_every_kind_of_change_and_is_idempotent():
    doc = DOC.replace("Ensure photographs are sharp and in focus.", "Ensure photographs are razor sharp.")
    doc = doc.replace("Clean Background", "Tidy Background")
    doc = doc.replace("Proper Lighting\nEnsure", "Proper Lighting\nThis is mandatory. Ensure")
    doc += "\nPunctuality\nVolunteers must arrive on time.\n"
    source = parse_rules_document(doc)

    new, log = sync_rule_dict(real_raw(), source, MARKERS, "2030-01-02")
    RuleSet.model_validate(new)  # still a valid rule set
    assert verify_against_source(RuleSet.model_validate(new), source, MARKERS) == []
    by_id = {r["rule_id"]: r for r in new["rules"]}
    assert by_id["R010"]["original_text"].startswith("Ensure photographs are razor sharp.") and "CHANGED by sync" in by_id["R010"]["notes"]
    assert by_id["R009"]["requirement_type"] == "mandatory" and by_id["R009"]["uncertain_policy"] == "reject"
    assert by_id["R011"]["enabled"] is False and "REMOVED" in by_id["R011"]["notes"]
    added = [r for r in new["rules"] if r["rule_id"] > "R016"]
    assert {r["source_ref"] for r in added} == {"Tidy Background", "Punctuality"}
    punct = next(r for r in added if r["source_ref"] == "Punctuality")
    assert punct["requirement_type"] == "mandatory" and punct["original_text"] == "Volunteers must arrive on time."
    assert new["rule_set_version"] == "1.2-sop+20300102" and new["rule_set_confirmed"] is False
    assert len(log) >= 5

    again, log2 = sync_rule_dict(new, source, MARKERS, "2030-01-03")
    assert log2 == [] and again["rule_set_version"] == new["rule_set_version"]


# ------------------------------------------------------------------ fail-closed loading


def _loader_with_doc(monkeypatch, tmp_path: Path, doc: str) -> None:
    cfg = load_project_config()
    path = tmp_path / "rules.txt"
    path.write_text(doc, encoding="utf-8")
    cfg.paths.rules_document = path
    monkeypatch.setattr(loader, "load_project_config", lambda: cfg)


def test_loader_refuses_stale_rules_when_the_document_changes(monkeypatch, tmp_path):
    _loader_with_doc(monkeypatch, tmp_path, DOC.replace("Ensure photographs are sharp and in focus.", "Be sharp."))
    with pytest.raises(RuleConfigError, match="out of sync"):
        load_rule_set()
    assert load_rule_set(check_source=False).rule_set_version  # explicit opt-out for tooling only


def test_loader_refuses_when_document_missing(monkeypatch, tmp_path):
    cfg = load_project_config()
    cfg.paths.rules_document = tmp_path / "nope.txt"
    monkeypatch.setattr(loader, "load_project_config", lambda: cfg)
    with pytest.raises(RuleConfigError, match="rules document not found"):
        load_rule_set()


def test_explicit_rule_file_skips_the_source_check(tmp_path):
    f = tmp_path / "rules.yaml"
    f.write_text(yaml.safe_dump(real_raw(), allow_unicode=True), encoding="utf-8")
    assert len(load_rule_set(f).rules) == 16
