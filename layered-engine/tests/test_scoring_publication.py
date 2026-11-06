from __future__ import annotations

import pytest

from layered_engine.decision_engine import evaluate_final_decision
from layered_engine.schemas import Recommendation as R

from conftest import make_assessment, verified_metadata


def decide(profile, assessment, meta=None):
    return evaluate_final_decision(assessment, profile, verified_metadata() if meta is None else meta)


# ------------------------------------------------------------------------------------------------ the scoring formula
@pytest.mark.parametrize("dims,expected", [
    ({}, 100.0),                                                         # all 4
    ({k: 3 for k in ("activity_communication", "technical_quality", "composition", "naturalness_storytelling", "brand_presentation", "visual_polish")}, 75.0),
    ({"activity_communication": 2}, 87.5),                               # (2/4)*25 + 75
    ({"brand_presentation": None}, 100.0),                               # not applicable: excluded and renormalised
    ({"brand_presentation": None, "activity_communication": 2}, 86.11),  # (12.5+20+20+15+10)/90*100
])
def test_the_weighted_score_and_renormalisation(proposed, dims, expected):
    d = decide(proposed, make_assessment(proposed, dims=dims))
    assert d.quality.overall_score == pytest.approx(expected, abs=0.01)


def test_the_weights_add_up_to_100(proposed):
    assert sum(c["weight"] for c in proposed.quality["dimensions"].values()) == 100


def test_the_model_cannot_set_its_own_score(proposed):
    a = make_assessment(proposed, dims={"activity_communication": 2}).model_copy(update={"model_overall_score": 99.0})
    d = decide(proposed, a)
    assert d.quality.overall_score == 87.5 and any("recomputed" in w for w in d.warnings)


def test_a_missing_dimension_makes_the_assessment_incomplete_not_good(proposed):
    a = make_assessment(proposed)
    a.layer_3_quality_dimensions.pop("composition")
    d = decide(proposed, a)
    assert d.quality.incomplete and d.recommendation is R.HUMAN_REVIEW and d.quality.overall_score is None


# ------------------------------------------------------------------------------------------------ thresholds
def test_threshold_bands_in_the_proposed_profile(proposed):
    review = {"technical_quality": 2, "composition": 2, "activity_communication": 3, "naturalness_storytelling": 3, "brand_presentation": 3, "visual_polish": 3}
    d = decide(proposed, make_assessment(proposed, dims=review))
    assert d.quality.overall_score == 65.0 and d.quality.band == "EDITORIAL_REVIEW" and d.recommendation is R.HUMAN_REVIEW
    low = decide(proposed, make_assessment(proposed, default_dim=2))  # 50 < 55
    assert low.quality.overall_score == 50.0 and low.recommendation is R.RESELECT
    good = decide(proposed, make_assessment(proposed, default_dim=3))
    assert good.quality.band == "ACCEPTABLE_CANDIDATE" and good.recommendation is R.APPROVE
    assert decide(proposed, make_assessment(proposed)).quality.band == "STRONG_CANDIDATE"


def test_a_strong_dimension_cannot_hide_a_severe_weakness(proposed):
    unusable = decide(proposed, make_assessment(proposed, dims={"activity_communication": 0}))
    assert unusable.quality.overall_score == 75.0 and unusable.recommendation is R.RESELECT  # 75 alone would be acceptable
    poor = decide(proposed, make_assessment(proposed, dims={"activity_communication": 1}))
    assert poor.quality.overall_score == 81.25 and poor.recommendation is R.HUMAN_REVIEW


def test_a_soft_score_never_overrides_a_failed_gate(legacy, proposed):
    for prof in (legacy, proposed):
        d = decide(prof, make_assessment(prof, answers={"R003_FACE": "NO"}, default_dim=4))
        assert d.recommendation is R.REJECT and d.quality.overall_score == 100
    # and a terrible score does not decide in the legacy profile, where only gates decide
    assert decide(legacy, make_assessment(legacy, default_dim=0)).recommendation is R.APPROVE


def test_overlapping_defects_are_not_counted_twice(proposed):
    d = decide(proposed, make_assessment(proposed, answers={"R006_COVERAGE": "NO", "R006_ZOOMED_OUT": "NO"}))
    assert len(d.quality.caps_applied) == 1 and d.quality.overall_score == 87.5


def test_an_inconsistent_dimension_is_flagged(proposed):
    d = decide(proposed, make_assessment(proposed, answers={"R007_LEVEL": "NO", "R007_PERSPECTIVE": "NO"}))
    assert any("composition" in w and "R007_LEVEL" in w for w in d.warnings)


# ------------------------------------------------------------------------------------------------ publication readiness
def test_missing_metadata_is_pending_never_verified(proposed, legacy):
    d = decide(proposed, make_assessment(proposed), {})
    assert d.recommendation is R.APPROVED_PENDING_PUBLICATION_CHECKS and d.photo_approval_status == "APPROVED"
    assert d.publication_readiness_status == "PENDING_VERIFICATION"
    assert {c.check_id for c in d.publication.checks if c.status == "PENDING"} >= {"NEW-001", "NEW-006", "P007", "P011"}
    leg = decide(legacy, make_assessment(legacy), {})
    assert leg.recommendation is R.APPROVE and leg.publication_readiness_status == "NOT_EVALUATED"  # legacy: publication does not decide


def test_everything_verified_is_ready(proposed):
    d = decide(proposed, make_assessment(proposed))
    assert d.recommendation is R.APPROVE and d.publication_readiness_status == "READY"


def test_a_blocked_verification_goes_to_a_human(proposed):
    d = decide(proposed, make_assessment(proposed), verified_metadata(consent_status="BLOCKED"))
    assert d.recommendation is R.HUMAN_REVIEW and any("NEW-001" in r for r in d.decision_reasons)


def test_an_identifiable_minor_needs_verification_and_is_never_stated_as_a_fact(proposed):
    flag = [{"flag_type": "POSSIBLE_MINOR", "evidence": "a small child is in the foreground", "confidence": "MEDIUM"}]
    d = decide(proposed, make_assessment(proposed, flags=flag), verified_metadata())
    assert d.recommendation is R.APPROVED_PENDING_PUBLICATION_CHECKS and any("NEW-002" in p for p in d.publication.pending)
    assert any("not a statement about age" in n for n in d.publication.review_notes)
    ok = decide(proposed, make_assessment(proposed, flags=flag), verified_metadata(child_safeguarding_status="VERIFIED"))
    assert ok.recommendation is R.APPROVE
    bad = decide(proposed, make_assessment(proposed, flags=flag), verified_metadata(child_safeguarding_status="BLOCKED"))
    assert bad.recommendation is R.HUMAN_REVIEW


def test_visual_blockers_and_low_confidence_flags(proposed):
    conf = [{"flag_type": "CONFIDENTIAL_INFORMATION", "evidence": "a medical report is readable", "confidence": "HIGH"}]
    assert decide(proposed, make_assessment(proposed, flags=conf)).recommendation is R.HUMAN_REVIEW
    low = [{"flag_type": "CONFIDENTIAL_INFORMATION", "evidence": "maybe a document", "confidence": "LOW"}]
    d = decide(proposed, make_assessment(proposed, flags=low))
    assert d.recommendation is R.APPROVED_PENDING_PUBLICATION_CHECKS and d.publication.review_notes  # never silently ignored
    note = [{"flag_type": "THIRD_PARTY_BRANDING", "evidence": "a sponsor's logo", "confidence": "HIGH"}]
    assert decide(proposed, make_assessment(proposed, flags=note)).recommendation is R.APPROVE  # a review note, not a blocker


def test_platform_dimensions_and_event_claims(proposed):
    small = decide(proposed, make_assessment(proposed), verified_metadata(platform="instagram_feed", width=800, height=800))
    assert small.recommendation is R.HUMAN_REVIEW and any("P005" in b for b in small.publication.blockers)
    big = decide(proposed, make_assessment(proposed), verified_metadata(platform="instagram_feed", width=2000, height=2000))
    assert big.recommendation is R.APPROVE
    assert next(c for c in decide(proposed, make_assessment(proposed), verified_metadata()).publication.checks if c.check_id == "P005").status == "NOT_ASSESSED"
    claims = decide(proposed, make_assessment(proposed), verified_metadata(event_claims_asserted=True))
    assert claims.recommendation is R.APPROVED_PENDING_PUBLICATION_CHECKS and any("P008" in p for p in claims.publication.pending)


def test_the_historical_label_cannot_enter_through_metadata(proposed):
    for key in ("label", "historical_decision", "approved"):
        with pytest.raises(ValueError, match="historical labels"):
            evaluate_final_decision(make_assessment(proposed), proposed, {key: "approved"})
