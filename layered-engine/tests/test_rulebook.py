from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from layered_engine.annotation import build_engine
from layered_engine.prompt_builder import build_prompt
from layered_engine.rulebook import RULEBOOK_DIR, Rulebook, validate_rulebook
from layered_engine.schema_export import build_response_schema, check_provider_safe

CLIENT_RULES = Path(__file__).resolve().parents[2] / "image-classifier" / "configs" / "rules.yaml"
CHECKLIST5 = Path(__file__).resolve().parents[2] / "holistic-judge-trial" / "prompts" / "checklist5" / "checks.yaml"


def test_validation_report_is_clean(rb):
    r = validate_rulebook(rb)
    assert r.ok, r.errors
    assert r.counts["legacy checklist5 questions preserved"] == 65 and r.counts["gates (all profiles)"] == 21


def test_every_client_rule_is_kept_verbatim(rb):
    rules = {r["rule_id"]: r for r in rb.sources["client_rules"]}
    assert sorted(rules) == [f"R{i:03d}" for i in range(1, 17)]
    assert all(r["source"] == "CLIENT" and r["approval_status"] == "CLIENT_APPROVED" for r in rules.values())
    if CLIENT_RULES.exists():  # compare with the main project's copy of the client's document
        main = {r["rule_id"]: r["original_text"] for r in yaml.safe_load(CLIENT_RULES.read_text(encoding="utf-8"))["rules"]}
        assert {k: v["description_original"] for k, v in rules.items()} == main


@pytest.mark.skipif(not CHECKLIST5.exists(), reason="checklist5 inventory not present")
def test_all_65_checklist5_ids_are_preserved_with_their_wording(rb):
    old = yaml.safe_load(CHECKLIST5.read_text(encoding="utf-8"))["checks"]
    assert len(old) == 65
    for c in old:
        assert c["id"] in rb.criteria, c["id"]
        assert rb.criteria[c["id"]].description_operational == c["q"]
        assert rb.criteria[c["id"]].legacy_id == c["id"]


def test_fifteen_legacy_gates_plus_three_layered_gates_all_team_source(rb, legacy, proposed):
    assert [g.gate_id for g in legacy.gates] == [f"G{i:03d}" for i in range(1, 16)]  # the legacy profile keeps exactly the original fifteen
    assert [g.gate_id for g in rb.gates] == [f"G{i:03d}" for i in range(1, 22)]   # G019 belongs to layered_tuned_v1 only; G020 and G021 (soft) to layered_final_v1 only
    assert all(g.source == "TEAM" and g.approval_status == "TEAM_APPROVED" and g.instruction for g in rb.gates)
    old = [g for g in rb.gates if g.active_in is None]
    assert all(rb.criteria[g.criterion_id].decision_effect == "GATE" and rb.criteria[g.criterion_id].gate_id == g.gate_id for g in old)
    new = [g for g in rb.gates if g.active_in is not None and g.gate_id not in ("G019", "G020", "G021")]      # G019 is the owner's in-frame banner gate, for layered_tuned_v1 only (test_tuned_profile.py)
    assert [g.gate_id for g in new] == ["G016", "G017", "G018"] and all(g.forced_hard and g.active_in == ["layered_proposed_v1", "layered_proposed_v2"] and g.policy_change for g in new)
    g19 = next(g for g in rb.gates if g.gate_id == "G019")
    assert g19.forced_hard and g19.active_in == ["layered_tuned_v1"] and g19.policy_change and g19.criterion_id == "R001_POSITION"
    g20, g21 = (next(g for g in rb.gates if g.gate_id == x) for x in ("G020", "G021"))
    assert g20.active_in == g21.active_in == ["layered_final_v1"] and not g20.forced_hard and not g21.forced_hard and (g20.criterion_id, g21.criterion_id) == ("R006_SIZE", "R010_TECHNICAL")
    assert {g.gate_id for g in proposed.gates} - {g.gate_id for g in legacy.gates} == {"G016", "G017", "G018"}


def test_every_criterion_has_the_required_fields_and_a_source(rb):
    for c in rb.criteria.values():
        assert c.source in ("CLIENT", "TEAM", "PROPOSED")
        assert c.approval_status in ("CLIENT_APPROVED", "TEAM_APPROVED", "PROPOSED", "PENDING_CONFIRMATION", "DEPRECATED")
        assert c.layer in (1, 2, 3, 4) and c.applicability in rb.conditions and c.failure_condition and c.uncertainty_policy
        if c.source == "PROPOSED":
            assert c.approval_status in ("PROPOSED", "PENDING_CONFIRMATION"), f"{c.criterion_id}: a proposed rule must not look approved"


def test_proposed_rules_are_configured_but_not_active_in_the_legacy_profile(legacy, proposed):
    new = [f"NEW-{i:03d}" for i in range(1, 25)]
    assert all(n in proposed.criteria for n in new)
    assert not any(n in legacy.criteria for n in new)
    assert not any(c.approval_status == "PROPOSED" and c.decision_effect != "FACT" for c in legacy.criteria.values())
    assert len(legacy.hard_gates()) == 15 and len(proposed.hard_gates()) == 16  # 15 - G010 - G011 (soft) + G016, G017, G018


def test_every_override_is_recorded_as_a_policy_change(rb, proposed):
    for cid, ov in rb.profiles["layered_proposed_v1"]["overrides"].items():
        assert "policy_change" in ov, cid
    kinds = {(c.target, c.kind) for c in proposed.policy_changes}
    assert ("R006_COVERAGE", "GATE_TO_SOFT") in kinds and ("G016", "GATE_ADDED") in kinds and ("G001", "SPLIT_INTO_COMPONENTS") in kinds
    assert all(c.status in ("PROPOSED", "PENDING_CONFIRMATION", "TEAM_APPROVED") for c in proposed.policy_changes)
    # the NGO's decisions are team-approved; what is still only a proposal stays marked as one
    approved = {c.target for c in proposed.policy_changes if c.status == "TEAM_APPROVED"}
    assert approved == {"G016", "G017", "G018", "R015_PERSON_PRESENT"}


def test_the_withdrawn_proposals_are_really_gone(rb, proposed, legacy):
    assert not any(c.kind == "SCOPE_NARROWED" for c in proposed.policy_changes)  # face scope stays strict until the CV layer exists
    assert proposed.settings["beneficiary_scope"] == legacy.settings["beneficiary_scope"] == "ALL_VISIBLE_BENEFICIARIES"
    assert proposed.criteria["R003_FACE"].description_operational == legacy.criteria["R003_FACE"].description_operational
    assert proposed.criteria["R001_FLAT"].applicability == "banner_present"  # not narrowed to prominent banners
    assert proposed.exceptions == [e for e in rb.exceptions if e["id"] == "EXC_MEDICAL_PPE"]  # back-view and direct-gaze exceptions are dormant
    assert {e["id"] for e in rb.exceptions if not e["active_in"]} == {"EXC_BACK_VIEW", "EXC_DIRECT_GAZE"}
    assert not any(c in proposed.criteria for c in ("T_BACK_EXC_DRESS", "T_BACK_EXC_LOGO", "T_BACK_EXC_BENEFICIARIES"))


def test_a_broken_rulebook_is_reported_not_silently_accepted(tmp_path):
    d = tmp_path / "rb"
    shutil.copytree(RULEBOOK_DIR, d)
    book = yaml.safe_load((d / "rulebook_v1.yaml").read_text(encoding="utf-8"))
    book["criteria"][5]["applicability"] = "no_such_condition"
    book["criteria"][6]["source"] = "SOMEONE"
    (d / "rulebook_v1.yaml").write_text(yaml.safe_dump(book), encoding="utf-8")
    errors = validate_rulebook(Rulebook(d)).errors
    assert any("no_such_condition" in e for e in errors) and any("SOMEONE" in e for e in errors)


def test_changed_client_wording_is_caught(tmp_path):
    d = tmp_path / "rb"
    shutil.copytree(RULEBOOK_DIR, d)
    book = yaml.safe_load((d / "rulebook_v1.yaml").read_text(encoding="utf-8"))
    for c in book["criteria"]:
        if c["criterion_id"] == "R001_FLAT":
            c["description_original"] = "Banners should be nice."
    (d / "rulebook_v1.yaml").write_text(yaml.safe_dump(book), encoding="utf-8")
    assert any("client wording differs" in e for e in validate_rulebook(Rulebook(d)).errors)


def test_the_activity_taxonomy_is_extendable_without_code(tmp_path, cfg):
    d = tmp_path / "rb"
    shutil.copytree(RULEBOOK_DIR, d)
    tax = yaml.safe_load((d / "activity_taxonomy.yaml").read_text(encoding="utf-8"))
    tax["activities"].append({"id": "BLOOD_DONATION_CAMP", "description": "Blood donation", "distribution": False})
    (d / "activity_taxonomy.yaml").write_text(yaml.safe_dump(tax), encoding="utf-8")
    e = build_engine(cfg, None, "layered_proposed_v1", rulebook_dir=d)
    enum = e.schema["properties"]["layer_1_scene_understanding"]["properties"]["scene_facts"]["properties"]["f_activity_type"]["enum"]
    assert "BLOOD_DONATION_CAMP" in enum and "BLOOD_DONATION_CAMP" in e.prompt.system_text


def test_changing_the_configuration_changes_the_prompt_and_its_hash(rb, proposed, tmp_path):
    base = build_prompt(rb, proposed)
    d = tmp_path / "rb"
    shutil.copytree(RULEBOOK_DIR, d)
    book = yaml.safe_load((d / "rulebook_v1.yaml").read_text(encoding="utf-8"))
    for c in book["criteria"]:
        if c["criterion_id"] == "R010_SHARP":
            c["description_operational"] = "Is the picture crisp?"
    (d / "rulebook_v1.yaml").write_text(yaml.safe_dump(book), encoding="utf-8")
    rb2 = Rulebook(d)
    p2 = build_prompt(rb2, rb2.resolve("layered_proposed_v1"))
    assert "Is the picture crisp?" in p2.system_text and p2.hash != base.hash


def test_the_response_schema_is_provider_safe_and_follows_the_profile(legacy, proposed):
    for prof in (legacy, proposed):
        ids = [c.criterion_id for c in prof.asked_criteria()]
        s = build_response_schema(prof, ids)
        assert check_provider_safe(s) == [], check_provider_safe(s)
        items = s["properties"]["layer_2_criteria"]["items"]["properties"]["criterion_id"]["enum"]
        assert items == ids
        assert list(s["properties"]["layer_3_quality_dimensions"]["properties"]) == list(prof.quality["dimensions"])
        assert "model_recommendation" not in s["properties"] and "model_overall_score" not in s["properties"] and "image_id" not in s["properties"]
