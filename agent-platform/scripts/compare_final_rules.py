"""The final rules (layered_final_v1) on the same 20 photographs with Claude Haiku 5.5 and with Gemini 2.5 Flash-Lite, against the staff's decisions.

    python scripts/compare_final_rules.py                     # dry run: lists the photographs and the estimated cost; NO model call
    python scripts/compare_final_rules.py --go --max-usd 0.15 # makes the calls (needs both providers' switches and keys in image-classifier/.env)

The photographs are the ones of compare_short_format.py (every photograph decided on the local demo, topped up with labelled ones from data/raw). The prompt is the PROFILE's own
(layered_final_v1, short answer format, nothing the demo's learning loop rewrote), so both models answer exactly the same questions. Each model's own configuration is used (Haiku:
anthropic_config.yaml, Gemini: gemini_config.yaml). Nothing is written to the demo's database: results go to reports/final_rules_<timestamp>.jsonl and a summary is printed.
Calls run one after the other, so Haiku's prompt cache is warm from the second photograph (a real queue of photographs arriving minutes apart would pay more cache writes).
A spending cap (--max-usd) stops the run; a failed call is recorded and the run goes on."""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import compare_short_format as csf   # the photograph selection and the helpers (its main() does not run)

from layered_engine.providers import load_vision_config, make_provider, read_env_file  # noqa: E402

HERE, FINAL = csf.HERE, "layered_final_v1"
RUNS = [("Claude Haiku 5.5", "haiku", "anthropic-claude"), ("Gemini 2.5 Flash-Lite", "gemini", "google-gemini")]
EST_USD = {"haiku": 0.0020, "gemini": 0.0013}      # per photograph, from the earlier estimates and the 20-photograph test; only for the dry run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="make the model calls (without it: a dry run, nothing is sent)")
    ap.add_argument("--max-usd", type=float, default=0.15, help="stop once the measured spend reaches this (default 0.15 USD, about Rs 14)")
    ap.add_argument("--only", default="", help="comma-separated photograph names (as listed by the dry run) to run instead of all 20, for example approved_59.jpg,approved/approved_10.jpg")
    ap.add_argument("--models", default="haiku,gemini", help="which models to run: haiku, gemini or both (default)")
    args = ap.parse_args()
    runs = [r for r in RUNS if r[1] in {m.strip() for m in args.models.split(",")}]

    db = sqlite3.connect(f"file:{csf.DB.as_posix()}?mode=ro", uri=True)
    photos = csf.demo_photos(db)[:csf.TOTAL]
    photos += csf.labelled_photos(csf.TOTAL - len(photos), {p["hash"] for p in photos})
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        photos = [p for p in photos if p["name"] in wanted]
        if {p["name"] for p in photos} != wanted:
            ap.error(f"photographs not found among the 20: {sorted(wanted - {p['name'] for p in photos})}")
    print(f"{len(photos)} photographs ({sum(p['source'] == 'demo' for p in photos)} decided on the demo, {sum(p['source'] != 'demo' for p in photos)} from the labelled folders); "
          f"staff approved {sum(p['staff'] == 'APPROVE' for p in photos)}, rejected {sum(p['staff'] == 'REJECT' for p in photos)}")
    est = 0.0
    for name, which, _ in runs:
        est += EST_USD[which] * len(photos)
        print(f"  {name:24} about ${EST_USD[which]:.4f} (Rs {EST_USD[which] * csf.INR_PER_USD:.2f}) a photograph, ${EST_USD[which] * len(photos):.3f} for all {len(photos)}")
    print(f"Estimated total ${est:.2f} (about Rs {est * csf.INR_PER_USD:.0f}). Spending cap ${args.max_usd:.2f}. Prompt: profile {FINAL}, short answer format, no rewrites.")
    if not args.go:
        print("Dry run: nothing was sent. Add --go to make the calls.")
        return 0

    from ngo_agent.evaluators import LayeredEngineEvaluator
    from ngo_agent.storage import LocalStorage

    store = LocalStorage(HERE / "demo_data" / "fresh" / "storage")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = HERE / "reports" / f"final_rules_{stamp}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    spent, results = 0.0, []
    with out_path.open("w", encoding="utf-8") as fh:
        for name, which, provider_name in runs:
            cfg = load_vision_config(which)
            env = read_env_file(HERE.parent / "image-classifier" / ".env", (cfg["auth"]["api_key_env"], cfg["auth"]["authorization_env"]))
            ev = LayeredEngineEvaluator(make_provider(cfg, env=env), cfg, provider_name=provider_name)   # fails closed without that provider's switch and key
            for p in photos:
                if spent >= args.max_usd:
                    print(f"spending cap reached (${spent:.3f}): stopping")
                    break
                started = time.monotonic()
                row = {"config": name, "photo": p["name"], "staff": p["staff"]}
                try:
                    image = store.get(p["key"]) if p.get("key") else Path(p["file"]).read_bytes()
                    r = ev.evaluate(image, p["mime"], {"profile": FINAL}, p["meta"])
                    usd = r.cost_usd or 0.0
                    spent += usd
                    gates = {g["gate_id"]: g["status"] for g in r.decision["gates"]}
                    row.update({"ok": True, "verdict": r.decision["recommendation"], "usd": usd, "seconds": round(time.monotonic() - started, 1), "usage": r.usage,
                                "failed_gates": [k for k, v in gates.items() if v in ("NO", "CANNOT_TELL")], "reasons": r.decision.get("decision_reasons", [])[:4]})
                except Exception as exc:  # noqa: BLE001  (recorded; a rejected request is not billed)
                    row.update({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                fh.write(json.dumps(row) + "\n")
                results.append(row)
                print(f"  {name:22} {p['name']:34} " + (f"{row['verdict']:36} ${row['usd']:.4f}" if row["ok"] else "ERROR " + row["error"]), flush=True)
    print(f"\nSUMMARY (spent ${spent:.3f}, about Rs {spent * csf.INR_PER_USD:.1f}; details in {out_path})")
    for name, *_ in runs:
        rows = [r for r in results if r["config"] == name and r["ok"]]
        failed = sum(1 for r in results if r["config"] == name and not r["ok"])
        if not rows:
            print(f"  {name}: no usable answer ({failed} failed)")
            continue
        agree = sum(csf.approves(r["verdict"]) == (r["staff"] == "APPROVE") for r in rows)
        false_app = sum(csf.approves(r["verdict"]) and r["staff"] == "REJECT" for r in rows)
        missed = sum((not csf.approves(r["verdict"])) and r["staff"] == "APPROVE" for r in rows)
        per = sum(r["usd"] for r in rows) / len(rows)
        u = lambda k: sum((r["usage"].get(k) or 0) for r in rows) / len(rows)  # noqa: E731
        print(f"  {name}: agrees with staff {agree}/{len(rows)}, FALSE APPROVALS {false_app}, approved photos not approved {missed}, ${per:.4f} (Rs {per * csf.INR_PER_USD:.2f}) a photograph, "
              f"{sum(r['seconds'] for r in rows) / len(rows):.0f} s each, {failed} failed")
        print(f"     tokens a photograph: input {u('prompt_tokens'):.0f} (cache read {u('cached_tokens'):.0f}, cache write {u('cache_write_tokens'):.0f}), output {u('output_tokens'):.0f}; verdicts {dict(collections.Counter(r['verdict'] for r in rows))}")
        wrong = [r for r in rows if csf.approves(r["verdict"]) != (r["staff"] == "APPROVE")]
        if wrong:
            print("     disagreements: " + "; ".join(f"{r['photo']} (staff {r['staff'][:3]}, AI {r['verdict'][:6]}, gates {','.join(r['failed_gates']) or '-'})" for r in wrong))
    return 0


if __name__ == "__main__":
    sys.exit(main())
