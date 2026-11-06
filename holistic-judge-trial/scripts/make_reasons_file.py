"""Write examples/reasons.csv: the human-tagged reason for each example that has one (offline; no API call).

    python scripts/make_reasons_file.py

Reads the main project's imported tags (data/annotations/disapproval_reasons.csv) and the saved examples. A reason is the
tagger's own choice(s) from the reason list, worded as the sheet worded them; nothing is written for an untagged photo.
"""

import _bootstrap  # noqa: F401
import pandas as pd

from judge_trial.config import ensure_main_importable, load_trial_config


def main() -> int:
    import json

    cfg = load_trial_config()
    ensure_main_importable(cfg)
    from ngo_photo_approval.data.tagging import reason_options
    from ngo_photo_approval.rules.loader import load_rule_set

    text = {o["code"]: o["text"] for o in reason_options(load_rule_set())}
    tags = pd.read_csv(cfg.main_root / "data" / "annotations" / "disapproval_reasons.csv").set_index("image_id")
    picks = json.loads(cfg.examples_path.read_text(encoding="utf-8"))
    rows = []
    for p in picks:
        if p["label"] != "not_approved" or p["image_id"] not in tags.index:
            continue
        t = tags.loc[p["image_id"]]
        codes = [c for c in (t["primary"], t["second"], t["third"]) if isinstance(c, str) and c]
        rows.append({"image_id": p["image_id"], "reason": "; ".join(text.get(c, c) for c in codes)})
    out = cfg.examples_path.parent / "reasons.csv"
    pd.DataFrame(rows, columns=["image_id", "reason"]).to_csv(out, index=False, encoding="utf-8")
    print(f"wrote {len(rows)} reasons -> {out}")
    for r in rows:
        print(f"  {r['image_id']}: {r['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
