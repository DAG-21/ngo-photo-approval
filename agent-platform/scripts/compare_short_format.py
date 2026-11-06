"""Compare the short answer format against the staff's decisions on 20 photographs, to see whether it is cheap enough AND good enough.

    python scripts/compare_short_format.py          # dry run: lists the photographs and configurations and estimates the cost; NO model call
    python scripts/compare_short_format.py --go     # makes the calls (needs the cloud switch and key in image-classifier/.env)

The 20 photographs: every photograph decided on the local demo (demo_data/fresh; the staff's verdict is known and today's full answer from gemini-3.8-flash is stored, so it is
NOT paid for again), topped up to 20 with labelled photographs from image-classifier/data/raw (approved / disapproved folders; real JPEGs only). The prompt is the demo's current
prompt (its rewritten rules included). Nothing is written to the demo's database: results go to reports/short_format_<timestamp>.jsonl and a summary is printed.

For each configuration it reports: verdicts that agree with the staff, FALSE APPROVALS (the AI approves a photograph the staff rejected; the number that matters most), agreement
with today's stored answer on the gates, tokens (input, output, thinking, cached), the real cost per photograph in dollars and rupees, and the time taken. A spending cap
(--max-usd) stops the run.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE / "src"), str(HERE.parent / "layered-engine" / "src")]

from layered_engine.costs import cost_usd  # noqa: E402
from layered_engine.providers import GeminiProvider, load_gemini_config, read_env_file  # noqa: E402

INR_PER_USD = 95.0          # an assumption (no confirmed rate on 2026-11-04); only used to print rupees
TOTAL = 20
EXTRA_PER_LABEL = 3         # labelled photographs taken from each folder when the demo has fewer than 20 decided ones
RAW = HERE.parent / "image-classifier" / "data" / "raw"
DB = HERE / "demo_data" / "fresh" / "fresh.db"

# name, model, answer format, image setting, thinking. A failed call (for example a setting a model does not accept) is recorded and the run goes on; it is not billed.
CONFIGS = [
    ("3.5 Flash-Lite, short, high image", "gemini-3.5-flash-lite", "compact", "MEDIA_RESOLUTION_HIGH", {"level": "MINIMAL"}),
    ("3.5 Flash-Lite, short, low image", "gemini-3.5-flash-lite", "compact", "MEDIA_RESOLUTION_LOW", {"level": "MINIMAL"}),
    ("2.5 Flash-Lite, short, high image", "gemini-2.5-flash-lite", "compact", "MEDIA_RESOLUTION_HIGH", {"budget": 0}),
]
# --set thinking: the same model, prompt and photographs with thinking off, with a budget of 1024 tokens, and with the model's own dynamic budget (-1), to see what thinking
# changes in accuracy and cost (2026-11-05, owner's request). Everything else (short format, high image) is as the live setting.
THINKING_CONFIGS = [
    ("2.5 Flash-Lite, short, thinking OFF", "gemini-2.5-flash-lite", "compact", "MEDIA_RESOLUTION_HIGH", {"budget": 0}),
    ("2.5 Flash-Lite, short, thinking 1024", "gemini-2.5-flash-lite", "compact", "MEDIA_RESOLUTION_HIGH", {"budget": 1024}),
    ("2.5 Flash-Lite, short, thinking DYNAMIC", "gemini-2.5-flash-lite", "compact", "MEDIA_RESOLUTION_HIGH", {"budget": -1}),
]
# rough token use per photograph of the short format, for the dry-run estimate only (input: instructions + image; output: the short answer)
EST_TOKENS = {"MEDIA_RESOLUTION_HIGH": (7400, 600), "MEDIA_RESOLUTION_LOW": (6600, 600)}


def approves(rec: str | None) -> bool:
    return rec in ("APPROVE", "APPROVED_PENDING_PUBLICATION_CHECKS")


def demo_photos(db: sqlite3.Connection) -> list[dict]:
    rows = db.execute("""select p.photo_id, p.original_filename, p.image_storage_path, p.file_format, p.image_width, p.image_height, p.file_hash, h.human_final_decision,
                                e.ai_recommendation, e.decision_output, e.input_tokens, e.output_tokens
                         from human_reviews h join photos p on p.photo_id = h.photo_id join evaluation_runs e on e.evaluation_id = h.evaluation_id
                         where h.review_status = 'FEEDBACK_SUBMITTED' order by h.reviewed_at""").fetchall()
    out = []
    for r in rows:
        d = json.loads(r[9]) if r[9] else {}
        out.append({"name": r[1], "source": "demo", "key": r[2], "mime": {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(r[3], "image/jpeg"),
                    "meta": {"width": r[4], "height": r[5]}, "hash": r[6], "staff": r[7], "baseline": r[8],
                    "baseline_gates": {g["gate_id"]: g["status"] for g in d.get("gates") or []}, "baseline_tokens": (r[10], r[11])})
    return out


def labelled_photos(need: int, skip_hashes: set[str]) -> list[dict]:
    from PIL import Image

    out: list[dict] = []
    for folder, staff in (("approved", "APPROVE"), ("disapproved", "REJECT")):
        taken, quota = 0, min(EXTRA_PER_LABEL, need - len(out))     # the quota is fixed per folder, so each label gets its share
        for p in sorted((RAW / folder).glob("*")):
            if taken >= quota:
                break
            data = p.read_bytes()
            try:
                with Image.open(io.BytesIO(data)) as im:
                    if im.format != "JPEG":
                        continue                 # the folder holds HEIC and MPO files under .jpg names: skip them
                    w, h = im.size
            except Exception:  # noqa: BLE001
                continue
            if hashlib.sha256(data).hexdigest() in skip_hashes:
                continue
            out.append({"name": f"{folder}/{p.name}", "source": "labelled folder", "file": p, "mime": "image/jpeg", "meta": {"width": w, "height": h}, "staff": staff,
                        "baseline": None, "baseline_gates": {}, "baseline_tokens": (None, None)})
            taken += 1
    return out


def live_prompt(db: sqlite3.Connection) -> dict:
    org = db.execute("select active_rulebook_version_id, active_prompt_version_id from organisations").fetchone()
    rb_cfg = json.loads(db.execute("select configuration from rulebook_versions where version_id=?", (org[0],)).fetchone()[0])
    pv = json.loads(db.execute("select configuration from prompt_versions where version_id=?", (org[1],)).fetchone()[0])
    out = dict(rb_cfg)
    if pv.get("notes"):
        out["prompt_notes"] = pv["notes"]
    if pv.get("rule_text"):
        out["rule_text"] = pv["rule_text"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="make the model calls (without it: a dry run, nothing is sent)")
    ap.add_argument("--max-usd", type=float, default=0.30, help="stop once the measured spend reaches this (default 0.30 USD, about Rs 28)")
    ap.add_argument("--set", choices=["formats", "thinking"], default="formats", help="which configurations to compare: the original three (formats) or thinking off / 1024 / dynamic on 2.5 Flash-Lite")
    args = ap.parse_args()
    if args.set == "thinking":
        CONFIGS[:] = THINKING_CONFIGS

    db =sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    photos = demo_photos(db)[:TOTAL]
    photos += labelled_photos(TOTAL - len(photos), {p["hash"] for p in photos})
    cfg0 = load_gemini_config()
    print(f"{len(photos)} photographs ({sum(p['source'] == 'demo' for p in photos)} decided on the demo, {sum(p['source'] != 'demo' for p in photos)} from the labelled folders):")
    for p in photos:
        print(f"  {p['name']:40} staff: {p['staff']:8} today's stored AI verdict: {p['baseline'] or '-'}")
    est = 0.0
    for name, model, fmt, media, thinking in CONFIGS:
        i, o = EST_TOKENS[media]
        one = cost_usd({"prompt_tokens": i, "output_tokens": o}, cfg0, model) or 0.0
        est += one * len(photos)
        print(f"  {name:38} about ${one:.4f} (Rs {one * INR_PER_USD:.2f}) a photograph, ${one * len(photos):.3f} for all {len(photos)}")
    print(f"Estimated total: ${est:.2f} (about Rs {est * INR_PER_USD:.0f}). Today's full answers are not paid for again: they are stored. Spending cap: ${args.max_usd:.2f}.")
    if not args.go:
        print("Dry run: nothing was sent. Add --go to make the calls.")
        return 0

    from ngo_agent.evaluators import LayeredEngineEvaluator
    from ngo_agent.storage import LocalStorage

    store = LocalStorage(HERE / "demo_data" / "fresh" / "storage")

    env = read_env_file(HERE.parent / "image-classifier" / ".env", (cfg0["auth"]["api_key_env"], cfg0["auth"]["authorization_env"]))
    rulebook_config = live_prompt(db)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = HERE / "reports" / f"short_format_{stamp}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    spent, results = 0.0, []
    with out_path.open("w", encoding="utf-8") as fh:
        for name, model, fmt, media, thinking in CONFIGS:
            cfg = json.loads(json.dumps(cfg0))
            cfg["model_id"] = model
            cfg["generation"]["media_resolution"] = media
            cfg["generation"]["thinking"] = thinking
            cfg.setdefault("evaluation", {})["response_format"] = fmt
            ev = LayeredEngineEvaluator(GeminiProvider(cfg, env=env), cfg)
            for p in photos:
                if spent >= args.max_usd:
                    print(f"spending cap reached (${spent:.3f}): stopping")
                    break
                started = time.monotonic()
                row = {"config": name, "photo": p["name"], "staff": p["staff"], "baseline": p["baseline"]}
                try:
                    image = store.get(p["key"]) if p.get("key") else Path(p["file"]).read_bytes()
                    r = ev.evaluate(image, p["mime"], rulebook_config, p["meta"])
                    usd = cost_usd(r.usage, cfg, model) or 0.0
                    spent += usd
                    gates = {g["gate_id"]: g["status"] for g in r.decision["gates"]}
                    row.update({"ok": True, "verdict": r.decision["recommendation"], "usd": usd, "seconds": round(time.monotonic() - started, 1), "usage": r.usage,
                                "gates_same_as_today": (sum(gates.get(k) == v for k, v in p["baseline_gates"].items()), len(p["baseline_gates"])),
                                "failed_gates": [k for k, v in gates.items() if v in ("NO", "CANNOT_TELL")]})
                except Exception as exc:  # noqa: BLE001  (recorded; a rejected request is not billed)
                    row.update({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                fh.write(json.dumps(row) + "\n")
                results.append(row)
                print(f"  {name:36} {p['name']:36} " + (f"{row['verdict']:36} ${row['usd']:.4f}" if row["ok"] else "ERROR " + row["error"]), flush=True)
    print(f"\nSUMMARY (spent ${spent:.3f}, about Rs {spent * INR_PER_USD:.1f}; details in {out_path})")
    base = [p for p in photos if p["baseline"]]
    if base:
        print(f"  today (3.8 Flash, full; stored): agrees with staff {sum(approves(p['baseline']) == (p['staff'] == 'APPROVE') for p in base)}/{len(base)}, "
              f"false approvals {sum(approves(p['baseline']) and p['staff'] == 'REJECT' for p in base)}")
    for name, *_ in CONFIGS:
        rows = [r for r in results if r["config"] == name and r["ok"]]
        if not rows:
            print(f"  {name}: no usable answer")
            continue
        agree = sum(approves(r["verdict"]) == (r["staff"] == "APPROVE") for r in rows)
        false_app = sum(approves(r["verdict"]) and r["staff"] == "REJECT" for r in rows)
        same_g = [r["gates_same_as_today"] for r in rows if r["gates_same_as_today"][1]]
        per = sum(r["usd"] for r in rows) / len(rows)
        print(f"  {name}: agrees with staff {agree}/{len(rows)}, FALSE APPROVALS {false_app}, gates same as today {sum(a for a, _ in same_g)}/{sum(b for _, b in same_g)}, "
              f"${per:.4f} (Rs {per * INR_PER_USD:.2f}) a photograph, {sum(r['seconds'] for r in rows) / len(rows):.0f} s each, {sum(1 for r in results if r['config'] == name and not r['ok'])} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
