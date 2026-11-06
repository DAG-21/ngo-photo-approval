"""Build the 'tagged and paired' photo list (offline; no API call).

    python scripts/make_paired_ids.py --pool dev --seed 7 --out ids/paired_tagged_dev.csv

Every disapproved photo in the pool that a person tagged with a reason, plus the same number of approved photos from the
same programmes (taken at random, seeded), so a difference between the two groups cannot just mean 'different programme'.
If a programme has fewer approved photos than tagged disapproved ones, all of them are taken and the shortfall is printed.
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from judge_trial.config import TRIAL_ROOT, load_trial_config
from judge_trial.data import load_pool
from judge_trial.examples import load_examples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="dev", choices=["dev", "confirm"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="ids/paired_tagged_dev.csv")
    args = ap.parse_args()
    cfg = load_trial_config()
    pool = load_pool(cfg, args.pool, {e["image_id"] for e in load_examples(cfg)})
    tags = pd.read_csv(cfg.main_root / "data" / "annotations" / "disapproval_reasons.csv").set_index("image_id")
    dis = pool[(pool.label == "not_approved") & pool.image_id.isin(tags.index)].copy()
    dis["tagged_primary"] = dis.image_id.map(tags["primary"])
    app = pool[pool.label == "approved"]
    picks = [dis]
    short = {}
    for prog, g in dis.groupby("program"):
        cand = app[app.program == prog]
        take = min(len(g), len(cand))
        if take < len(g):
            short[prog] = len(g) - take
        picks.append(cand.sample(n=take, random_state=args.seed).assign(tagged_primary=None))
    out = pd.concat(picks)[["image_id", "label", "program", "group_id", "tagged_primary"]].sort_values(["label", "program", "image_id"])
    path = Path(args.out) if Path(args.out).is_absolute() else TRIAL_ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"wrote {path}  ({len(out)} photos)")
    print(pd.crosstab(out.program, out.label).to_string())
    print("tagged primary reasons:", dis.tagged_primary.value_counts().to_dict())
    if short:
        print("approved photos short by programme (all available taken):", short)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
