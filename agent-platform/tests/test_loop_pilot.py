"""The learning loop against the real pilot (50 photographs and the staff's decisions): the V2 proposal raised false approvals from 2 to 6, so the loop must not let that kind of
loosening through. Skipped when the pilot files (which include private labels) are not on this machine. No model call; a replay of stored answers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import replay_pilot_changes as pilot  # noqa: E402

from ngo_agent import loop_service as loop  # noqa: E402

ROWS = pilot.load_rows()
pytestmark = pytest.mark.skipif(not ROWS, reason="the pilot files are not on this machine")


@pytest.fixture(scope="module")
def base(rb):
    return loop.replay(ROWS, rb, pilot.PROFILE, [], [])


def test_the_pilot_is_the_one_the_v1_v2_comparison_studied(base):
    c = loop.confusion(ROWS, base)
    assert c["n"] == 50 and c["FP"] == 2 and sum(r.human_approved for r in ROWS) == 25         # V1 approved 2 photographs staff rejected


def test_a_loosening_that_would_approve_a_photograph_staff_rejected_is_blocked_even_when_it_helps_elsewhere(rb, base):
    o = loop.evaluate_candidate(ROWS, base, rb, pilot.PROFILE, [], [], {"change_id": "T", "kind": "GATE_SOFTEN", "target": "G017", "set": {}}, True, {**loop.DEFAULTS, "min_rejected_for_loosening": 0})
    assert o.status == "BLOCKED" and len(o.flips["new_false_approvals"]) >= 1 and o.guardrails["no_new_false_approval"] is False


@pytest.mark.parametrize("gate", ["G015", "G016", "G017", "G018"])
def test_no_v2_style_gate_loosening_is_applied_on_the_pilot(rb, base, gate):
    for cfg in (loop.DEFAULTS, {**loop.DEFAULTS, "min_rejected_for_loosening": 0}):                # with the shipped evidence rule, and with it switched off
        o = loop.evaluate_candidate(ROWS, base, rb, pilot.PROFILE, [], [], {"change_id": "T", "kind": "GATE_SOFTEN", "target": gate, "set": {}}, True, cfg)
        assert o.status != "APPLY", (gate, o.reason)


def test_whatever_the_loop_drafts_from_the_pilot_it_does_not_apply(rb, base):
    discovery = [r for r in ROWS if not r.held_out]
    for cfg in (loop.DEFAULTS, {**loop.DEFAULTS, "min_rejected_for_loosening": 0}):
        for cand in loop.mine(discovery, base, rb, pilot.PROFILE, [], [], cfg):
            assert loop.evaluate_candidate(ROWS, base, rb, pilot.PROFILE, [], [], cand.record("T"), cand.loosening, cfg).status != "APPLY", cand
