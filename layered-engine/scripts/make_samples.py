"""Write SYNTHETIC sample assessments with their authoritative decisions to samples/ (offline; no API call).

These are what a correct model would report for the scenarios in tests/test_decision_fixtures.py, run through the real decision engine.
They are NOT evaluations of real photographs and are not evidence that Gemini produces these assessments.
"""

import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import make_assessment, verified_metadata  # noqa: E402

from layered_engine.decision_engine import evaluate_final_decision  # noqa: E402
from layered_engine.rulebook import ROOT, Rulebook  # noqa: E402

SCENARIOS = {
    "synthetic_01_clear_food_distribution": dict(),
    "synthetic_02_crumpled_prominent_banner": dict(answers={"R001_FLAT": "NO"}, evidence={"R001_FLAT": "deep creases run across the left half of the banner"}),
    "synthetic_03_minor_needs_verification": dict(flags=[{"flag_type": "POSSIBLE_MINOR", "evidence": "a small child is identifiable in the foreground", "confidence": "HIGH"}]),
}


def main() -> int:
    profile = Rulebook().resolve("layered_proposed_v1")
    out = ROOT / "samples"
    out.mkdir(exist_ok=True)
    for name, kw in SCENARIOS.items():
        meta = verified_metadata(child_safeguarding_status=None) if "minor" in name else verified_metadata()
        a = make_assessment(profile, **kw)
        d = evaluate_final_decision(a, profile, meta)
        (out / f"{name}.json").write_text(json.dumps({"NOTE": "SYNTHETIC: not a real photograph and not a Gemini response", "metadata": meta,
                                                      "model_assessment": a.model_dump(mode="json"), "decision": d.model_dump(mode="json")}, indent=2), encoding="utf-8")
        print(f"{name}: {d.recommendation.value}  {d.decision_reasons[:1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
