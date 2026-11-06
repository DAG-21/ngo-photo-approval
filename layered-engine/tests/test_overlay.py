"""The overlay: stored change records applied on top of a profile. It changes only quality thresholds, weights and gate hardness; it never edits the shared
configuration; with no overlay nothing is different; and the prompt does not depend on it."""

from __future__ import annotations

import copy

import pytest

from conftest import make_assessment
from layered_engine.decision_engine import evaluate_final_decision
from layered_engine.prompt_builder import build_prompt
from layered_engine.rulebook import OVERLAY_KINDS, validate_overlay
from layered_engine.schemas import Recommendation

P = "layered_proposed_v1"
THR = lambda **s: {"change_id": "T1", "kind": "QUALITY_THRESHOLDS", "set": s}   # noqa: E731


def decide(rb, assessment, overlay=None, meta=None):
    prof = rb.resolve(P, overlay=overlay)
    return evaluate_final_decision(assessment, prof, meta or {})


def borderline(rb):
    """Every gate passes and quality is about 62: with the shipped thresholds that needs an editorial look."""
    prof = rb.resolve(P)
    return make_assessment(prof, dims={"activity_communication": 2, "technical_quality": 2, "composition": 3, "naturalness_storytelling": 2, "brand_presentation": 3, "visual_polish": 2})


def test_no_overlay_changes_nothing_and_the_shared_configuration_is_never_edited(rb):
    a = borderline(rb)
    before = copy.deepcopy(rb.quality)
    base = decide(rb, a)
    assert base.model_dump() == decide(rb, a, overlay=[]).model_dump() == decide(rb, a, overlay=None).model_dump()
    out = decide(rb, a, overlay=[THR(review=40, reselection=30), {"change_id": "G1", "kind": "GATE_SOFTEN", "target": "G004"}])
    assert out.versions["overlay"] == ["T1", "G1"] and "overlay" not in base.versions
    assert rb.quality == before                                   # the shared quality configuration still has its shipped values
    assert rb.resolve(P).quality["thresholds"]["review"] == 70    # and a later plain resolve is untouched by an earlier overlay


def test_lowering_the_review_threshold_turns_a_borderline_photo_into_an_approval_and_raising_it_does_the_opposite(rb):
    a = borderline(rb)
    score = decide(rb, a).quality.overall_score
    assert 55 <= score < 70 and decide(rb, a).recommendation is Recommendation.HUMAN_REVIEW
    assert decide(rb, a, [THR(review=int(score) - 1)]).recommendation in (Recommendation.APPROVE, Recommendation.APPROVED_PENDING_PUBLICATION_CHECKS)
    assert decide(rb, a, [THR(reselection=int(score) + 1, review=int(score) + 2)]).recommendation is Recommendation.RESELECT


def test_new_weights_change_the_score_and_must_add_up_to_100(rb):
    prof = rb.resolve(P)
    a = make_assessment(prof, dims={"activity_communication": 4, "technical_quality": 0, "composition": 4, "naturalness_storytelling": 4, "brand_presentation": 4, "visual_polish": 4})
    w = {"activity_communication": 25, "technical_quality": 20, "composition": 20, "naturalness_storytelling": 15, "brand_presentation": 10, "visual_polish": 10}
    heavier = {**w, "technical_quality": 40, "activity_communication": 5}
    s0 = decide(rb, a, [{"change_id": "W0", "kind": "QUALITY_WEIGHTS", "set": w}]).quality.overall_score
    s1 = decide(rb, a, [{"change_id": "W1", "kind": "QUALITY_WEIGHTS", "set": heavier}]).quality.overall_score
    assert s1 < s0                                                 # a weak technical quality counts for more
    with pytest.raises(ValueError, match="add up to 100"):
        validate_overlay([{"change_id": "W", "kind": "QUALITY_WEIGHTS", "set": {**w, "technical_quality": 21}}])


def test_softening_a_gate_stops_it_rejecting_and_hardening_a_soft_gate_makes_it_reject(rb):
    prof = rb.resolve(P)
    soft_ids = [g.gate_id for g in prof.gates if not prof.is_hard(g)]
    hard_ids = [g.gate_id for g in prof.gates if prof.is_hard(g)]
    assert soft_ids and hard_ids
    hard = next(g for g in prof.gates if prof.is_hard(g) and g.criterion_id in {c.criterion_id for c in prof.asked_criteria()} and not g.components)
    a = make_assessment(prof, answers={hard.criterion_id: "NO"})
    assert decide(rb, a).recommendation is Recommendation.REJECT and hard.gate_id in decide(rb, a).failed_gates
    softened = decide(rb, a, [{"change_id": "S1", "kind": "GATE_SOFTEN", "target": hard.gate_id}])
    assert hard.gate_id not in softened.failed_gates and softened.recommendation is not Recommendation.REJECT
    soft = next(g for g in prof.gates if not prof.is_hard(g) and not g.components and g.criterion_id in {c.criterion_id for c in prof.asked_criteria()})
    b = make_assessment(prof, answers={soft.criterion_id: "NO"})
    assert decide(rb, b).recommendation is not Recommendation.REJECT
    hardened = decide(rb, b, [{"change_id": "H1", "kind": "GATE_HARDEN", "target": soft.gate_id}])
    assert hardened.recommendation is Recommendation.REJECT and soft.gate_id in hardened.failed_gates


def test_a_softened_gate_lowers_its_quality_dimension_once_when_the_criterion_has_one(rb):
    prof = rb.resolve(P)
    g = next(g for g in prof.gates if prof.is_hard(g) and getattr(prof.criteria[g.criterion_id], "dimension", None) and not g.components
             and g.criterion_id in {c.criterion_id for c in prof.asked_criteria()})
    dim = prof.criteria[g.criterion_id].dimension
    a = make_assessment(prof, answers={g.criterion_id: "NO"})
    out = decide(rb, a, [{"change_id": "S1", "kind": "GATE_SOFTEN", "target": g.gate_id, "set": {"cap_max_score": 1}}])
    assert any(dim in c for c in out.quality.caps_applied)         # one defect, one cap
    assert out.quality.dimension_scores[dim] == 1


def test_the_prompt_never_depends_on_an_overlay_so_its_hash_is_stable(rb):
    base = rb.resolve(P)
    assert build_prompt(rb, base).hash == build_prompt(rb, rb.resolve(P)).hash
    # an overlay-resolved profile has a different hardness, so a prompt built from IT would differ: callers must build the prompt from the base profile
    softened = rb.resolve(P, overlay=[{"change_id": "S1", "kind": "GATE_SOFTEN", "target": "G004"}])
    assert build_prompt(rb, softened).hash != build_prompt(rb, base).hash


@pytest.mark.parametrize("bad, why", [
    ([{"change_id": "X", "kind": "WIPE_RULES", "target": "G001"}], "unknown overlay kind"),
    ([{"kind": "GATE_SOFTEN", "target": "G001"}], "unique change_id"),
    ([{"change_id": "A", "kind": "GATE_SOFTEN", "target": "G001"}, {"change_id": "A", "kind": "GATE_HARDEN", "target": "G002"}], "unique change_id"),
    ([{"change_id": "A", "kind": "GATE_SOFTEN"}], "needs a target"),
    ([{"change_id": "A", "kind": "QUALITY_THRESHOLDS", "set": {}}], "sets 'review'"),
    ([{"change_id": "A", "kind": "QUALITY_THRESHOLDS", "set": {"strong_candidate": 99}}], "sets 'review'"),
    ([{"change_id": "A", "kind": "QUALITY_THRESHOLDS", "set": {"review": 140}}], "from 0 to 100"),
    ([{"change_id": "A", "kind": "QUALITY_THRESHOLDS", "set": {"review": True}}], "from 0 to 100"),
    ([{"change_id": "A", "kind": "GATE_SOFTEN", "target": "G001", "set": {"max": 1}}], "may only set"),
    ([{"change_id": f"C{i}", "kind": "GATE_HARDEN", "target": "G001"} for i in range(51)], "at most 50"),
])
def test_malformed_overlays_are_refused(bad, why):
    with pytest.raises(ValueError, match=why):
        validate_overlay(bad)


def test_an_overlay_that_cannot_apply_to_the_profile_is_refused_by_resolve(rb):
    prof = rb.resolve(P)
    soft = next(g.gate_id for g in prof.gates if not prof.is_hard(g))
    hard = next(g.gate_id for g in prof.gates if prof.is_hard(g))
    with pytest.raises(ValueError, match="no gate"):
        rb.resolve(P, overlay=[{"change_id": "A", "kind": "GATE_SOFTEN", "target": "G999"}])
    with pytest.raises(ValueError, match="already soft"):
        rb.resolve(P, overlay=[{"change_id": "A", "kind": "GATE_SOFTEN", "target": soft}])
    with pytest.raises(ValueError, match="already hard"):
        rb.resolve(P, overlay=[{"change_id": "A", "kind": "GATE_HARDEN", "target": hard}])
    with pytest.raises(ValueError, match="thresholds must satisfy"):
        rb.resolve(P, overlay=[THR(review=40, reselection=60)])
    assert set(OVERLAY_KINDS) == {"QUALITY_THRESHOLDS", "QUALITY_WEIGHTS", "GATE_SOFTEN", "GATE_HARDEN"}
