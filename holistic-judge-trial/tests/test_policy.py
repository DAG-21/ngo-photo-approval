from __future__ import annotations

import re
from pathlib import Path

import pytest

from judge_trial.checklist import check_ids, load_checks
from judge_trial.config import TRIAL_ROOT
from judge_trial.policy import PolicyError, decide, load_policy, render

V = "checklist5"


@pytest.fixture(scope="module")
def spec():
    return load_checks(V)


@pytest.fixture(scope="module")
def policy(spec):
    return load_policy(V, spec)


def answers(spec, **overrides):
    out = {i: {"answer": "YES", "evidence": ""} for i in check_ids(spec)}
    for k, v in overrides.items():
        out[k] = {"answer": v, "evidence": "seen in the photo"}
    return out


def test_a_clean_photo_is_approved_and_not_applicable_never_blocks(spec, policy):
    assert decide(answers(spec), policy)["decision"] == "APPROVED"
    na = {i: {"answer": "NOT_APPLICABLE", "evidence": ""} for i in check_ids(spec)}
    na["R015_PERSON_PRESENT"] = {"answer": "YES", "evidence": ""}
    assert decide(na, policy)["decision"] == "APPROVED"


# One test per instruction: if someone loosens a gate, the matching instruction fails here by name.
INSTRUCTION_CHECKS = {
    "1": ["R007_ANGLE", "R007_FACES_SEEN"],
    "2": ["R006_ITEM_SHARP"],
    "3": ["R003_FACE", "R003_CLEAR"],
    "4": ["R001_FLAT"],
    "5": ["R003_FACE_SHARP", "R003_FACE_SIZE", "R009_FACE_LIT"],
    "6": ["R006_COVERAGE", "R006_ZOOMED_OUT"],
    "7": ["R015_BEN_RECEIVING", "R002_GIVER_WHITE"],
    "8": ["R015_PERSON_PRESENT"],
    "9": ["R004_BEN_GAZE"],
}


@pytest.mark.parametrize("instruction,check", [(i, c) for i, cs in INSTRUCTION_CHECKS.items() for c in cs])
def test_each_instruction_forces_a_disapproval(spec, policy, instruction, check):
    r = decide(answers(spec, **{check: "NO"}), policy)
    assert r["decision"] == "DISAPPROVED"
    assert [t["id"] for t in r["triggered"]] == [check] and r["triggered"][0]["instruction"] == instruction
    assert check in render(r)


def test_cannot_tell_on_a_gate_fails_it_when_strict_and_not_otherwise(spec, policy):
    a = answers(spec, R003_FACE="CANNOT_TELL")
    assert decide(a, policy)["decision"] == "DISAPPROVED"
    assert decide(a, {**policy, "unknown_is_fail": False})["decision"] == "APPROVED"


def test_unreadable_photos_go_to_review_not_to_a_guess(spec, policy):
    assert decide(answers(spec), policy, cannot_judge=True)["decision"] == "NEEDS_REVIEW"


def test_several_failures_are_all_reported(spec, policy):
    r = decide(answers(spec, R001_FLAT="NO", R006_ITEM_SHARP="NO", R003_FACE="NO"), policy)
    assert {t["id"] for t in r["triggered"]} == {"R001_FLAT", "R006_ITEM_SHARP", "R003_FACE"}


def test_a_gate_for_a_check_that_does_not_exist_stops_the_load(spec, tmp_path, monkeypatch):
    import judge_trial.policy as P

    monkeypatch.setattr(P, "PROMPTS_DIR", tmp_path)
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "policy.yaml").write_text("version: 1\ngates:\n  - {id: NOPE, instruction: '1', why: 'w'}\n", encoding="utf-8")
    with pytest.raises(PolicyError, match="not in the checklist"):
        P.load_policy("x", spec)
    (tmp_path / "x" / "policy.yaml").write_text("version: 1\ngates:\n  - {id: R001_FLAT}\n", encoding="utf-8")
    with pytest.raises(PolicyError, match="instruction"):
        P.load_policy("x", spec)


def test_the_corrections_register_and_the_policy_agree(spec, policy):
    doc = (TRIAL_ROOT / "CORRECTIONS.md").read_text(encoding="utf-8")
    rows = [r for r in doc.splitlines() if re.match(r"\| (\d|E\d) \|", r)]
    in_doc: dict[str, set[str]] = {}
    for r in rows:
        cells = [c.strip() for c in r.strip("|").split("|")]
        gates = set(re.findall(r"[A-Z]\d{3}_[A-Z_]+", cells[3]))
        if gates:
            in_doc[cells[0]] = gates
    in_policy: dict[str, set[str]] = {}
    for g in policy["gates"]:
        in_policy.setdefault(g["instruction"], set()).add(g["id"])
    assert in_doc == in_policy == {k: set(v) for k, v in INSTRUCTION_CHECKS.items()}
    # every question the register names exists in the checklist
    names = set(re.findall(r"[A-Z]\d{3}_[A-Z_]+", doc))
    assert names <= set(check_ids(spec)), sorted(names - set(check_ids(spec)))


def test_a_photo_needs_a_beneficiary_or_a_volunteer_not_specifically_a_beneficiary(spec, policy):
    # no beneficiary at all, but a volunteer is there: the person-present gate is satisfied and nothing else fires
    a = answers(spec, R015_BEN_PRESENT="NO")
    for c in ("R003_FACE", "R003_CLEAR", "R003_FACE_SHARP", "R003_FACE_SIZE", "R009_FACE_LIT", "R015_BEN_RECEIVING", "R004_BEN_GAZE", "R007_FACES_SEEN"):
        a[c] = {"answer": "NOT_APPLICABLE", "evidence": ""}
    assert decide(a, policy)["decision"] == "APPROVED"
    assert decide(answers(spec, R015_PERSON_PRESENT="NO"), policy)["decision"] == "DISAPPROVED"


def test_a_volunteer_looking_into_the_camera_is_no_longer_asked_about(spec):
    assert "R004_VOL_GAZE" not in check_ids(spec) and "R004_BEN_GAZE" in check_ids(spec)
    giver = next(c for c in spec["checks"] if c["id"] == "R002_GIVER_WHITE")
    assert "giving" in giver["na"] and "white top" in giver["q"]
