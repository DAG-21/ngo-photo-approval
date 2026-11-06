"""Choose the worked examples for the prompt (offline; no API call) and save a contact sheet to review them.

    python scripts/select_examples.py            # writes examples/examples.json + examples/contact_sheet.png
    python scripts/select_examples.py --force    # overwrite an existing selection (then re-run build_pools.py)

Changing the examples changes the prompt hash and changes which photos may be evaluated, so pools must be rebuilt.
"""

import argparse

import _bootstrap  # noqa: F401

from judge_trial.config import TRIAL_ROOT, load_trial_config
from judge_trial.examples import contact_sheet, select_examples, save_examples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_trial_config()
    if cfg.examples_path.exists() and not args.force:
        print(f"{cfg.examples_path} already exists; use --force to replace it (the pools must then be rebuilt).")
        return 1
    picks = select_examples(cfg)
    path = save_examples(cfg, picks)
    sheet = contact_sheet(cfg, picks, TRIAL_ROOT / "examples" / "contact_sheet.png")
    print(f"saved {len(picks)} examples -> {path}")
    for n, p in enumerate(picks, 1):
        print(f"  #{n} {p['label']:<13} {p['program']:<20} {p['image_id']}")
    print(f"contact sheet -> {sheet}\nNext: python scripts/build_pools.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
