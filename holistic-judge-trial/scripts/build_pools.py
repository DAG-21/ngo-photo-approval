"""Build the dev / confirm / validation photo pools (offline; no API call).

    python scripts/build_pools.py

Pools exclude every photo that is, or is related to (same submission group, duplicate cluster, identical content),
a worked example. The test split is never read.
"""

import _bootstrap  # noqa: F401

from judge_trial.config import load_trial_config
from judge_trial.data import build_pools, save_pools
from judge_trial.examples import load_examples


def main() -> int:
    cfg = load_trial_config()
    examples = load_examples(cfg)
    if not examples:
        print("WARNING: no examples saved yet (run scripts/select_examples.py first); pools are built without exclusions.")
    pools = build_pools(cfg, {e["image_id"] for e in examples})
    save_pools(cfg, pools)
    for name, df in pools.items():
        c = df["label"].value_counts().to_dict()
        print(f"{name:<11} {len(df):>4} photos  approved={c.get('approved', 0)}  not_approved={c.get('not_approved', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
