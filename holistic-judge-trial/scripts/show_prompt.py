"""Print exactly what a variant sends (offline; no API call).

    python scripts/show_prompt.py --variant D
"""

import argparse

import _bootstrap  # noqa: F401

from judge_trial.backend_cost import estimate_prefix_tokens
from judge_trial.config import load_trial_config
from judge_trial.prompt import build_for_variant, render_text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="D", help="A-D (holistic) or K (checklist)")
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()
    cfg = load_trial_config()
    p = build_for_variant(cfg, args.variant, args.version)
    print(render_text(p))
    print(f"\n--- variant {p.variant}, version {p.version}, prompt hash {p.hash}, {len(p.examples)} example photos, "
          f"about {estimate_prefix_tokens(p, cfg.cost.tokens_per_image)} tokens in the fixed part (estimate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
