"""Judge ONE photograph of the review desk again with the current rules (one Claude Haiku 5.5 call, about Rs 0.15; the photograph goes to Anthropic) and replace only its entry in review-desk/data/data.js.

    python scripts/rejudge_one.py p09        # the id the review desk uses (data/data.js); the original is found in the labelled folders by its file name
    python scripts/rejudge_one.py p09 --dry  # shows what would be judged, no call

Nothing is written to the demo's database. The previous verdict is printed next to the new one."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys

import review_desk_data as rdd   # describe(), the paths and the prompt helpers (its main() does not run)

csf = rdd.csf

from layered_engine.providers import load_vision_config, make_provider, read_env_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("photo_id")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    path = rdd.OUT / "data" / "data.js"
    text = path.read_text(encoding="utf-8")
    data = json.loads(re.sub(r";\s*$", "", text.replace("window.REVIEW_DATA = ", "", 1)))
    item = next((p for p in data["photos"] if p["id"] == args.photo_id), None)
    if item is None:
        print(f"no photograph {args.photo_id} in {path}")
        return 1
    src = csf.RAW / item["name"] if "/" in item["name"] else None
    if src is None or not src.exists():
        print(f"{item['name']}: only photographs from the labelled folders can be judged again here")
        return 1
    print(f"{args.photo_id} = {item['name']}; before: {item['verdict']}, {item['headline']}")
    if args.dry:
        print("dry run: nothing was sent")
        return 0

    from PIL import Image

    from ngo_agent.evaluators import LayeredEngineEvaluator

    db = sqlite3.connect(f"file:{csf.DB.as_posix()}?mode=ro", uri=True)
    live = csf.live_prompt(db)
    live["profile"] = rdd.FINAL
    cfg = load_vision_config("haiku")
    env = read_env_file(csf.HERE.parent / "image-classifier" / ".env", (cfg["auth"]["api_key_env"], cfg["auth"]["authorization_env"]))
    ev = LayeredEngineEvaluator(make_provider(cfg, env=env), cfg, provider_name="anthropic-claude")
    data_bytes = src.read_bytes()
    with Image.open(src) as im:
        meta = {"width": im.size[0], "height": im.size[1]}
    r = ev.evaluate(data_bytes, "image/jpeg", live, meta)
    info = rdd.describe(r.decision, r.validated)
    print(f"after: {info['verdict']}, {info['headline']}  (quality {info['quality']}; ${r.cost_usd or 0:.4f}, about Rs {(r.cost_usd or 0) * csf.INR_PER_USD:.2f}; prompt {r.prompt_hash})")
    for x in info["reasons"]:
        print("   reason:", x["title"], "-", x["why"])
    for a in info["lowAreas"]:
        print(f"   {a['area']}: {a['score']} of 4 - {a['why']}")
    for c in info["checks"]:
        if c["status"] != "YES":
            print(f"   check not YES: {c['name']}: {c['status']} - {c['why']}")
    item.update(info)
    path.write_text("window.REVIEW_DATA = " + json.dumps(data, indent=1, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"updated {args.photo_id} in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
