"""What would the learning loop have done with the real pilot? A local, free replay of the 50 pilot photographs and the staff's decisions (no model call, no network).

    python scripts/replay_pilot_changes.py

Part 1 tests the kind of loosening the V2 proposal made (softening gates G015, G017, G018) with the loop's guardrails, first with the evidence rule the loop ships with,
then with that rule switched off, to show what the safety checks alone would say. Part 2 lets the loop draft its own changes from the pilot and tests them.
Only counts are printed; the staff decisions come from the pilot's private key file and stay on this machine.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE / "src"), str(HERE.parent / "layered-engine" / "src")]

from layered_engine.annotation import read_rows  # noqa: E402
from layered_engine.rulebook import Rulebook  # noqa: E402

from ngo_agent import loop_service as loop  # noqa: E402

ENGINE = HERE.parent / "layered-engine"
PROFILE = "layered_proposed_v1"


def load_rows() -> list[loop.Row]:
    """The pilot as the loop sees it: the stored answers and the staff decision for each photograph. Empty if the pilot files are not on this machine."""
    key_file, results = ENGINE / "pilot" / "key3_PRIVATE.csv", ENGINE / "runs" / "pilot3_layered" / "results.jsonl"
    if not (key_file.exists() and results.exists()):
        return []
    key = {r["opaque_id"]: r["label"] for r in csv.DictReader(key_file.open(encoding="utf-8"))}
    rows = []
    for r in read_rows(results):
        if r.get("ok") and r["image_id"] in key:
            rows.append(loop.Row(r["image_id"], r["image_id"], r["image_id"], r["validated"], r.get("metadata_used") or {}, key[r["image_id"]] == "approved", "pilot", False,
                                 loop.is_held_out(r["image_id"], 40)))
    return rows


def main() -> int:
    rows = load_rows()
    if not rows:
        print("the pilot files are not on this machine")
        return 0
    rb = Rulebook()
    base = loop.replay(rows, rb, PROFILE, [], [])
    b = loop.confusion(rows, base)
    held = [r for r in rows if r.held_out]
    print(f"{len(rows)} photographs ({sum(r.human_approved for r in rows)} approved, {sum(not r.human_approved for r in rows)} rejected by staff); {len(held)} held out, of which "
          f"{sum(not r.human_approved for r in held)} staff-rejected")
    print(f"V1 settings: staff-rejected photographs the system approves {b['FP']}, staff-approved photographs it does not approve {b['FN']}\n")
    shipped = dict(loop.DEFAULTS)
    relaxed = {**loop.DEFAULTS, "min_rejected_for_loosening": 0}
    print("PART 1: V2-style loosenings tested by the loop's guardrails")
    for gate in ("G015", "G017", "G018", "G016"):
        rec = {"change_id": "T", "kind": "GATE_SOFTEN", "target": gate, "set": {}}
        for name, cfg in (("shipped evidence rule", shipped), ("evidence rule off", relaxed)):
            o = loop.evaluate_candidate(rows, base, rb, PROFILE, [], [], rec, True, cfg)
            g = o.guardrails
            print(f"  soften {gate}, {name}: {o.status}; new false approvals on all {len(o.flips.get('new_false_approvals', []))}"
                  + (f", held-out gain {g.get('gain_on_held_out')}" if g.get('gain_on_held_out') is not None else "") + f"; {o.reason[:110]}")
    print("\nPART 2: what the loop drafts by itself from the pilot (discovery photographs only)")
    discovery = [r for r in rows if not r.held_out]
    for label, cfg in (("shipped settings", {**shipped, "min_pattern": 3}), ("relaxed evidence rule", {**relaxed, "min_pattern": 3})):
        cands = loop.mine(discovery, base, rb, PROFILE, [], [], cfg)
        if not cands:
            print(f"  {label}: no pattern strong enough to draft a change")
        for c in cands:
            o = loop.evaluate_candidate(rows, base, rb, PROFILE, [], [], c.record("T"), c.loosening, cfg)
            print(f"  {label}: {c.kind} {c.target or c.set}: {o.status}; {o.reason[:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
