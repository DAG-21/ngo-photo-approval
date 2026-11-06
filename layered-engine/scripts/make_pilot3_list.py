"""Draw the pilot-3 photographs: 25 approved + 25 disapproved, random (seed 50), from the unique train+validation photos, none used in pilots 1-2 (offline).

The test split is never read. Opaque ids NGO_IMG_000201.. ; the private key (label, tags, programme) is written separately and never sent to the model.
"""

import csv
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

INTERN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(INTERN / "holistic-judge-trial" / "src"))
from judge_trial import data  # noqa: E402
from judge_trial.config import load_trial_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 50


def main() -> int:
    cfg = load_trial_config()
    _, _, unique = data.load_main(cfg)
    u = unique[unique.split.isin(["train", "validation"]) & unique.label.isin(["approved", "not_approved"])]
    used = {r["image_id"] for f in ("key_PRIVATE.csv", "key2_PRIVATE.csv") for r in csv.DictReader((ROOT / "pilot" / f).open(encoding="utf-8"))}
    u = u[~u.image_id.isin(used) & u.image_id.map(lambda i: data.image_path(cfg, i).exists())]
    sel = pd.concat([u[u.label == "approved"].sample(n=25, random_state=SEED), u[u.label == "not_approved"].sample(n=25, random_state=SEED)])
    sel = sel.sample(frac=1, random_state=SEED).reset_index(drop=True)
    tags = pd.read_csv(INTERN / "image-classifier" / "data" / "annotations" / "disapproval_reasons.csv").set_index("image_id")
    sel["opaque_id"] = [f"NGO_IMG_{i + 201:06d}" for i in range(len(sel))]
    sel["path"] = [str(data.image_path(cfg, i).resolve()) for i in sel.image_id]
    sel["tags"] = [",".join(str(x) for x in (tags.loc[i].primary, tags.loc[i].second, tags.loc[i].third) if isinstance(x, str)) if i in tags.index else "" for i in sel.image_id]
    sel[["opaque_id", "path"]].rename(columns={"opaque_id": "image_id"}).to_csv(ROOT / "pilot" / "list3.csv", index=False)
    sel[["opaque_id", "image_id", "label", "program", "tags", "split"]].to_csv(ROOT / "pilot" / "key3_PRIVATE.csv", index=False)
    print(sel.label.value_counts().to_dict(), "| programmes:", sel.groupby("label").program.value_counts().to_dict(), "| tagged disapproved:", int((sel.tags != "").sum()),
          "| split:", sel.split.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
