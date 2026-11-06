"""ONE-TIME BOOTSTRAP: write config/rulebook/*.yaml from the client's rules and the existing checklist5 inventory.

    python scripts/build_rulebook.py [--force]

Reads (read-only):
  ../image-classifier/configs/rules.yaml                 the 16 client rules, verbatim
  ../holistic-judge-trial/prompts/checklist5/checks.yaml the 65 checklist5 questions (IDs, wording, source)
  ../holistic-judge-trial/prompts/checklist5/policy.yaml the 15 gates and unknown_is_fail
After the first run the YAML files are the source of truth and are edited by hand (see README 'How to add a rule');
this script refuses to overwrite them without --force so that edits are never lost.
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
INTERNSHIP = ROOT.parent
CLIENT_RULES = INTERNSHIP / "image-classifier" / "configs" / "rules.yaml"
C5 = INTERNSHIP / "holistic-judge-trial" / "prompts" / "checklist5"
OUT = ROOT / "config" / "rulebook"

# ---- gates: (gate_id, checklist5 id, instruction it came from) --------------------------------------------------------------
GATES = [
    ("G001", "R007_ANGLE", "1"), ("G002", "R007_FACES_SEEN", "1"), ("G003", "R006_ITEM_SHARP", "2"), ("G004", "R003_FACE", "3"),
    ("G005", "R003_CLEAR", "3"), ("G006", "R003_FACE_SHARP", "5"), ("G007", "R003_FACE_SIZE", "5"), ("G008", "R009_FACE_LIT", "5"),
    ("G009", "R001_FLAT", "4"), ("G010", "R006_COVERAGE", "6"), ("G011", "R006_ZOOMED_OUT", "6"), ("G012", "R015_BEN_RECEIVING", "7"),
    ("G013", "R002_GIVER_WHITE", "7"), ("G014", "R015_PERSON_PRESENT", "8"), ("G015", "R004_BEN_GAZE", "9"),
]
GATE_OF = {c: g for g, c, _ in GATES}

# ---- how each existing question is classified ------------------------------------------------------------------------------
# applicability condition name (see gate_applicability.yaml) and quality dimension for the checklist5 questions that are not facts
APPLIC = {
    "R003_FACE": "beneficiaries_visible", "R003_CLEAR": "beneficiaries_visible", "R003_CROP": "beneficiaries_visible",
    "R003_FACE_SHARP": "beneficiaries_visible", "R003_FACE_SIZE": "beneficiaries_visible",
    "R001_VISIBLE": "banner_present", "R001_POSITION": "banner_present", "R001_STRAIGHT": "banner_present", "R001_FLAT": "banner_present",
    "R002_WHITE": "volunteers_visible", "R002_NEAT": "volunteers_visible", "R002_GIVER_WHITE": "volunteer_giving",
    "R004_BEN_GAZE": "beneficiaries_visible", "R004_GENUINE": "people_visible", "R005_CANDID": "people_visible", "R005_INTERACT": "people_visible",
    "R006_ITEM_SHARP": "item_given", "R006_HANDOVER": "item_given", "R006_VOLUNTEER_DOING": "always",
    "R007_FACES_SEEN": "beneficiaries_visible", "R007_INTERACTION_VIEW": "volunteers_and_beneficiaries",
    "R008_BACK": "volunteers_visible", "T_FRONT": "people_visible", "T_BACK_EXCEPTION": "volunteer_from_behind",
    "R009_PEOPLE_LIT": "people_visible", "R009_FACE_LIT": "beneficiaries_visible",
    "R015_BEN_RECEIVING": "beneficiaries_visible", "R015_BEN_FRONT": "beneficiaries_visible", "R015_BEN_POSE": "beneficiaries_visible",
    "R015_BEN_PROMINENT": "beneficiaries_visible", "R015_BEN_INTERACT": "beneficiaries_visible",
}
DIM = {"R001": "brand_presentation", "R002": "brand_presentation", "R003": "activity_communication", "R004": "naturalness_storytelling",
       "R005": "naturalness_storytelling", "R006": "activity_communication", "R007": "composition", "R008": "composition",
       "R009": "technical_quality", "R010": "technical_quality", "R011": "visual_polish", "R012": "naturalness_storytelling",
       "R013": "naturalness_storytelling", "R015": "activity_communication", "T": "composition"}
GATE_PRIORITY = {"1": 10, "3": 20, "5": 21, "4": 30, "6": 40, "2": 50, "7": 60, "8": 5, "9": 70}
APPROVAL = {"client": "CLIENT_APPROVED", "team": "TEAM_APPROVED", "tagger": "TEAM_APPROVED", "fact": "TEAM_APPROVED"}

# ---- new scene facts (Layer 1) ----------------------------------------------------------------------------------------------
NEW_FACTS = [
    ("F_BENEFICIARIES", "Are human beneficiaries visible? (decide from role in the activity only, never from appearance)"),
    ("F_BANNER_PROMINENT", "If a banner is present, is it a prominent or required branding element of the photograph?"),
    ("F_ANIMALS", "Are animals receiving care or assistance?"),
    ("F_ACTIVITY_TYPE", "What type of seva is depicted? (one value from the activity taxonomy)"),
    ("F_ITEM_GIVEN", "Is a tangible item (food, packet, book, medicine...) being handed over or distributed?"),
    ("F_HANDOVER", "Is a handover or serving interaction visible?"),
    ("F_VOLUNTEER_GIVER", "Is a volunteer actively giving or providing assistance?"),
    ("F_ORIENTATION", "Are the primary subjects seen from the FRONT, SIDE or BACK (or MIXED)?"),
    ("F_LOGO", "Is the NGO logo or name visible?"),
    ("F_MEDICAL", "Does the scene depict healthcare or medical activity?"),
    ("F_GROUP", "Is this a single-subject, group or crowd photograph?"),
    ("F_PRIMARY_SUBJECTS", "Which people, animals or activities are visually central?"),
    ("F_SECONDARY_SUBJECTS", "Which subjects support the primary activity?"),
    ("F_INCIDENTAL_SUBJECTS", "Which subjects appear incidentally in the background?"),
]

# ---- additional operational diagnostics needed by the layered design ---------------------------------------------------------
EXTRA_DIAGNOSTICS = [
    # id, parent, question, applicability, dimension, note
    ("G001_ACTIVITY_VISIBLE", "R007", "From the camera position, is the activity itself visible?", "always", "composition", "G001 component (proposed profile)"),
    ("G001_BENEFICIARIES_VISIBLE", "R007", "From the camera position, are the relevant primary beneficiaries visible?", "beneficiaries_visible", "composition", "G001 component (proposed profile)"),
    ("G001_INTERACTION_VISIBLE", "R007", "From the camera position, is the volunteer-beneficiary interaction visible where it is part of the activity?", "volunteers_and_beneficiaries", "composition", "G001 component (proposed profile)"),
    ("G001_NOTHING_HIDDEN", "R007", "Is every important scene element free from being hidden by the camera angle?", "always", "composition", "G001 component (proposed profile)"),
    ("T_BACK_EXC_DRESS", "R008", "Where a volunteer is seen from behind: is that volunteer wearing the required dress code?", "volunteer_from_behind", "composition", "T2 condition 1"),
    ("T_BACK_EXC_LOGO", "R008", "Where a volunteer is seen from behind: are the NGO logo AND name visible on the back of the clothing?", "volunteer_from_behind", "composition", "T2 condition 2"),
    ("T_BACK_EXC_BENEFICIARIES", "R008", "Where a volunteer is seen from behind: are the relevant beneficiaries clearly visible?", "volunteer_from_behind", "composition", "T2 condition 3"),
    ("R003_OBSTRUCTION_PPE_ONLY", "R003", "If any beneficiary's face is obstructed, is the ONLY obstruction necessary medical protective equipment (mask, face shield)?", "medical_scene", "activity_communication", "medical PPE exception"),
]

# ---- new criteria proposed during this restructuring -------------------------------------------------------------------------
# id, layer, effect, asked, dimension, description, note
NEW_CRITERIA = [
    ("NEW-001", 4, "EXTERNAL", False, None, "Every identifiable subject has the appropriate permission for the intended public use where NGO policy requires it. (needs the consent record: cannot be seen in a photograph)", "metadata: consent_status"),
    ("NEW-002", 4, "FLAG", True, None, "Is a possible minor identifiable? (record as a flag needing verification; never assert age)", "visual flag POSSIBLE_MINOR; metadata: child_safeguarding_status"),
    ("NEW-003", 4, "FLAG", True, None, "Does the photograph unintentionally expose confidential information (medical reports, personal ID, phone numbers, addresses, sensitive records)?", "visual flag CONFIDENTIAL_INFORMATION"),
    ("NEW-004", 4, "FLAG", True, None, "Does the photograph humiliate, degrade, exploit or unnecessarily expose a vulnerable individual?", "visual flag; overlaps R012 (not double counted)"),
    ("NEW-005", 4, "FLAG", True, None, "Is there visible evidence of material manipulation that misrepresents people, activities or outcomes? (routine colour/exposure correction and permitted cropping are fine)", "visual flag MANIPULATION_SUSPECTED"),
    ("NEW-006", 4, "EXTERNAL", False, None, "The NGO holds the rights or permissions required to publish the photograph. (needs rights metadata)", "metadata: rights_status"),
    ("NEW-007", 4, "FLAG", True, None, "Does the photograph create a material foreseeable risk of harm to beneficiaries, volunteers or communities?", "visual flag UNSAFE_ACTIVITY"),
    ("NEW-008", 3, "SOFT_SCORE", True, "composition", "Are critical visual elements free from being accidentally cut off?", "overlaps R007_CUTOFF (not double counted)"),
    ("NEW-009", 3, "SOFT_SCORE", True, "composition", "Is the primary subject visually distinguishable from the background?", ""),
    ("NEW-010", 3, "SOFT_SCORE", True, "technical_quality", "Is important content free from being obscured by glare or reflection?", ""),
    ("NEW-011", 3, "SOFT_SCORE", True, "visual_polish", "Are the colours reasonably natural?", ""),
    ("NEW-012", 3, "SOFT_SCORE", True, "technical_quality", "Is the photograph free from severe compression artefacts or pixelation?", ""),
    ("NEW-013", 3, "SOFT_SCORE", True, "technical_quality", "Would important details remain visible at the intended publishing size?", ""),
    ("NEW-014", 3, "SOFT_SCORE", True, "visual_polish", "Is the photograph free from unintended watermark, editing marks or camera overlays?", ""),
    ("NEW-015", 3, "SOFT_SCORE", True, "technical_quality", "Is the photograph free from an accidental finger, lens or equipment obstruction?", ""),
    ("NEW-016", 3, "SOFT_SCORE", True, "naturalness_storytelling", "Where appropriate, does the photograph communicate the purpose or outcome of the activity? (documentary photographs need not)", ""),
    ("NEW-017", 3, "SOFT_SCORE", True, "naturalness_storytelling", "Does it show meaningful participation rather than merely assembled people?", ""),
    ("NEW-018", 3, "SOFT_SCORE", True, "naturalness_storytelling", "Does it represent beneficiaries as participants with agency rather than passive recipients?", ""),
    ("NEW-019", 3, "SOFT_SCORE", True, "naturalness_storytelling", "Is the representation respectful and inclusive?", ""),
    ("NEW-020", 3, "SOFT_SCORE", True, "composition", "Does the photograph have a clear visual focal point?", ""),
    ("NEW-021", 3, "SOFT_SCORE", True, "brand_presentation", "Is the photograph free from unnecessary dominance of unrelated branding?", ""),
    ("NEW-022", 3, "SOFT_SCORE", True, "naturalness_storytelling", "Can the story be understood without an elaborate caption?", ""),
    ("NEW-023", 3, "BATCH", False, None, "Adds a distinct perspective to the batch of photographs. (needs the batch: not implemented)", "future: batch selection"),
    ("NEW-024", 3, "EXTERNAL", False, None, "Matches its intended communication purpose. (needs the campaign brief)", "metadata: campaign_objective"),
]

# P-criteria (Layer 4): id, effect, kind, asked, metadata_key, description
P_CRITERIA = [
    ("P001", "EXTERNAL", "SOFT", False, "campaign_objective", "Matches the intended campaign or communication objective"),
    ("P002", "BATCH", "BATCH", False, None, "Not unnecessarily repetitive compared with already selected photographs"),
    ("P003", "BATCH", "BATCH", False, None, "Is one of the stronger photographs from the event or sequence"),
    ("P004", "SOFT_SCORE", "SOFT", True, None, "Has a clear subject suitable for a social media post"),
    ("P005", "TECHNICAL", "TECHNICAL", False, "platform", "Meets the intended platform resolution and dimensions"),
    ("P006", "TECHNICAL", "PUBLICATION", False, "platform", "Platform cropping does not remove essential content"),
    ("P007", "EXTERNAL", "MANDATORY_BEFORE_PUBLISHING", False, "caption_status", "Accompanying caption accurately describes the photograph"),
    ("P008", "EXTERNAL", "MANDATORY_BEFORE_PUBLISHING_IF_ASSERTED", False, "event_details_status", "Event details, dates, locations and claims are verified"),
    ("P009", "FLAG", "CONDITIONAL", True, None, "Logos and third-party brand elements are used appropriately"),
    ("P010", "EXTERNAL", "SOFT", False, "channel", "Appropriate for the intended audience and channel"),
    ("P011", "EXTERNAL", "MANDATORY_BEFORE_PUBLISHING", False, "credits_rights_status", "Necessary credits and usage rights are documented"),
]


def load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump(path: Path, obj, header: str = "") -> None:
    text = yaml.safe_dump(obj, sort_keys=False, allow_unicode=True, width=140)
    path.write_text(header + text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "rulebook_v1.yaml").exists() and not args.force:
        print("config/rulebook/rulebook_v1.yaml already exists; edit it by hand or pass --force to regenerate (this discards edits).")
        return 1

    client = load(CLIENT_RULES)["rules"]
    c5 = load(C5 / "checks.yaml")["checks"]
    pol = load(C5 / "policy.yaml")
    assert len(c5) == 65, len(c5)
    gate_ids = {g["id"] for g in pol["gates"]}
    assert gate_ids == {c for _, c, _ in GATES} and len(gate_ids) == 15
    assert pol["unknown_is_fail"] is True
    client_by = {r["rule_id"]: r for r in client}

    # ---------------- rule_sources.yaml ---------------------------------------------------------------------------------
    sources = {
        "version": 1,
        "sources": {
            "CLIENT": "Original requirements supplied by the NGO (wording kept verbatim below).",
            "TEAM": "Rules and mandatory gates added internally (including questions suggested by the disapproval taggers).",
            "PROPOSED": "Recommendations introduced during this restructuring. NOT policy until the NGO approves them.",
        },
        "approval_statuses": ["CLIENT_APPROVED", "TEAM_APPROVED", "PROPOSED", "PENDING_CONFIRMATION", "DEPRECATED"],
        "note": "The client's document marks NO rule as mandatory. Gates are team policy applied on top of it.",
        "client_rules": [
            {"rule_id": r["rule_id"], "source_ref": r["source_ref"], "description_original": r["original_text"], "source": "CLIENT",
             "approval_status": "CLIENT_APPROVED",
             "summary_of": {"R014": ["R006"], "R015": ["R002", "R003", "R008"], "R016": ["R004", "R007", "R009", "R010"]}.get(r["rule_id"])}
            for r in client
        ],
    }
    dump(OUT / "rule_sources.yaml", sources, "# The three rule sources and the 16 client rules, verbatim. Do not edit client wording.\n")

    # ---------------- criteria ------------------------------------------------------------------------------------------
    crit: list[dict] = []

    def add(**kw):
        kw.setdefault("exceptions", [])
        kw.setdefault("evidence_required", False)
        kw.setdefault("uncertainty_policy", "HUMAN_REVIEW")
        kw.setdefault("version", 1)
        kw.setdefault("failure_condition", "answer == NO")
        kw.setdefault("asked", True)
        kw.setdefault("model_na_allowed", False)
        crit.append(kw)

    for c in c5:
        cid, rule, src = c["id"], c["rule"], c["source"]
        is_fact = rule == "FACT"
        parent = rule if not is_fact else None
        if cid.startswith("T_"):
            parent = "R008"
        gate = GATE_OF.get(cid)
        add(
            criterion_id=cid, source="CLIENT" if src == "client" else "TEAM", origin={"tagger": "TAGGERS", "fact": "SCENE_FACT"}.get(src, src.upper()),
            parent_rule_id=parent, gate_id=gate,
            description_original=(client_by[rule]["original_text"] if src == "client" and rule in client_by else None),
            description_operational=c["q"], na_condition_text=c.get("na") or None,
            layer=1 if is_fact else 2,
            decision_effect="FACT" if is_fact else ("GATE" if gate else "DIAGNOSTIC"),
            priority=0 if is_fact else (GATE_PRIORITY.get(next((i for g, k, i in GATES if k == cid), ""), 100) if gate else 100),
            applicability="always" if (is_fact or cid not in APPLIC) else APPLIC[cid],
            evidence_required=bool(gate), approval_status=APPROVAL[src] if not gate else "TEAM_APPROVED",
            dimension=None if is_fact else DIM.get(parent or "", "composition"), legacy_id=cid,
            model_na_allowed=(cid in ("R006_VOLUNTEER_DOING", "R007_PLACEMENT", "R007_CUTOFF")),
        )
    for cid, parent, q, appl, dim, note in EXTRA_DIAGNOSTICS:
        add(criterion_id=cid, source="TEAM", origin="LAYERED_DESIGN", parent_rule_id=parent, gate_id=None, description_original=None,
            description_operational=q, na_condition_text=None, layer=2, decision_effect="DIAGNOSTIC", priority=100, applicability=appl,
            approval_status="PROPOSED", dimension=dim, legacy_id=None, note=note)
    for cid, q in NEW_FACTS:
        add(criterion_id=cid, source="PROPOSED", origin="SCENE_FACT", parent_rule_id=None, gate_id=None, description_original=None,
            description_operational=q, na_condition_text=None, layer=1, decision_effect="FACT", priority=0, applicability="always",
            approval_status="PROPOSED", dimension=None, legacy_id=None)
    for cid, layer, eff, asked, dim, desc, note in NEW_CRITERIA:
        add(criterion_id=cid, source="PROPOSED", origin="LAYERED_DESIGN", parent_rule_id=None, gate_id=None, description_original=None,
            description_operational=desc, na_condition_text=None, layer=layer, decision_effect=eff, priority=100, applicability="always",
            approval_status="PROPOSED", dimension=dim, legacy_id=None, asked=asked, note=note or None, model_na_allowed=eff == "SOFT_SCORE")
    for cid, eff, kind, asked, mkey, desc in P_CRITERIA:
        add(criterion_id=cid, source="PROPOSED", origin="PUBLICATION_READINESS", parent_rule_id=None, gate_id=None, description_original=None,
            description_operational=desc, na_condition_text=None, layer=4, decision_effect=eff, priority=100, applicability="always",
            approval_status="PROPOSED", dimension=None, legacy_id=None, asked=asked, kind=kind, metadata_key=mkey)
    for rid, covers in (("R014", ["R006"]), ("R015", ["R002", "R003", "R008"]), ("R016", ["R004", "R007", "R009", "R010"])):
        add(criterion_id=f"{rid}_SUMMARY", source="CLIENT", origin="CLIENT_SUMMARY", parent_rule_id=rid, gate_id=None,
            description_original=client_by[rid]["original_text"], description_operational="Summary of other criteria. Never scored separately.",
            na_condition_text=None, layer=3, decision_effect="SUMMARY", priority=999, applicability="always", approval_status="CLIENT_APPROVED",
            dimension=None, legacy_id=None, asked=False, summarises=covers)

    gates = []
    byid = {c["criterion_id"]: c for c in crit}
    for gid, cid, instr in GATES:
        c = byid[cid]
        gates.append({"gate_id": gid, "criterion_id": cid, "source": "TEAM", "parent_rule_id": c["parent_rule_id"], "instruction": instr,
                      "approval_status": "TEAM_APPROVED", "components": []})

    book = {
        "rulebook_version": "1.0.0", "client_rules_file": "rule_sources.yaml",
        "legacy_question_count": 65, "legacy_gate_count": 15, "unknown_is_fail": True,
        "criteria": crit, "gates": gates,
    }
    dump(OUT / "rulebook_v1.yaml", book, "# Master rulebook. Every criterion has a source, a parent rule, a layer, a decision effect and an approval status.\n"
                                       "# Bootstrapped by scripts/build_rulebook.py; edit by hand afterwards and record changes in RULEBOOK_CHANGELOG.md.\n")
    print(f"criteria written: {len(crit)} (legacy 65 + {len(crit) - 65} new); gates: {len(gates)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
