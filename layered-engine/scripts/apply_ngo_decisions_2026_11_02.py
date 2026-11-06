"""One-off record of the rulebook edit made for the NGO's five decisions of 2026-11-02 (rulebook 1.0.0 -> 1.1.0).

Kept in the repository so the change is reproducible and reviewable; it refuses to run twice. After this, edit the YAML by hand.
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "config" / "rulebook" / "rulebook_v1.yaml"
LAYERED = "layered_proposed_v1"


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    header = "".join(line for line in text.splitlines(keepends=True) if line.startswith("#"))
    book = yaml.safe_load(text)
    if book["rulebook_version"] != "1.0.0":
        print("already applied (rulebook_version is", book["rulebook_version"] + ")")
        return 1
    crit = {c["criterion_id"]: c for c in book["criteria"]}

    # 2. banners: when present they must be clearly visible AND distinguishable (new criterion, layered profile only)
    base = crit["R001_VISIBLE"]
    book["criteria"].append({
        **{k: v for k, v in base.items() if k not in ("criterion_id", "description_operational", "legacy_id", "origin", "source", "approval_status", "description_original")},
        "criterion_id": "R001_DISTINCT", "source": "TEAM", "origin": "NGO_DECISION_2026_11_02", "description_original": None, "legacy_id": None,
        "description_operational": "Is the banner clearly distinguishable from its surroundings: does it stand out from the background, with its text or logo readable and not blending in?",
        "approval_status": "TEAM_APPROVED", "active_in": [LAYERED], "evidence_required": True})
    # 5. the back-view exception (T2) is withdrawn: its three sub-questions are no longer asked
    for cid in ("T_BACK_EXC_DRESS", "T_BACK_EXC_LOGO", "T_BACK_EXC_BENEFICIARIES"):
        crit[cid]["active_in"] = []

    pc = lambda kind, reason: {"kind": kind, "affects_mandatory": True, "status": "TEAM_APPROVED", "reason": reason}  # noqa: E731
    book["gates"] += [
        {"gate_id": "G016", "criterion_id": "R008_BACK", "source": "TEAM", "parent_rule_id": "R008", "instruction": "NGO-2026-11-02-5",
         "approval_status": "TEAM_APPROVED", "components": ["R008_BACK", "T_FRONT", "R015_BEN_FRONT"], "forced_hard": True, "active_in": [LAYERED],
         "policy_change": pc("GATE_ADDED", "Back views are disallowed: the photograph must be taken from the front (or side) of the people; no volunteer or beneficiary may be shown mainly from behind. The logo exception (T2) is withdrawn.")},
        {"gate_id": "G017", "criterion_id": "R001_VISIBLE", "source": "TEAM", "parent_rule_id": "R001", "instruction": "NGO-2026-11-02-2",
         "approval_status": "TEAM_APPROVED", "components": [], "forced_hard": True, "active_in": [LAYERED],
         "policy_change": pc("GATE_ADDED", "When a banner is present it must be clearly visible. Banners are not expected in every photograph.")},
        {"gate_id": "G018", "criterion_id": "R001_DISTINCT", "source": "TEAM", "parent_rule_id": "R001", "instruction": "NGO-2026-11-02-2",
         "approval_status": "TEAM_APPROVED", "components": [], "forced_hard": True, "active_in": [LAYERED],
         "policy_change": pc("GATE_ADDED", "When a banner is present it must be clearly distinguishable from its surroundings.")},
    ]
    book["rulebook_version"] = "1.1.0"
    PATH.write_text(header + yaml.safe_dump(book, sort_keys=False, allow_unicode=True, width=140), encoding="utf-8")
    print("applied: rulebook 1.1.0 with", len(book["criteria"]), "criteria and", len(book["gates"]), "gates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
