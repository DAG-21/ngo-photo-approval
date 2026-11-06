"""Report on a pilot: two profiles run on the same photos (offline; reads runs/ and pilot/key_PRIVATE.csv, which holds the human tags).

    python scripts/pilot_report.py pilot_layered_proposed_v1 pilot_checklist5_legacy
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from layered_engine.annotation import read_rows
from layered_engine.rulebook import ROOT

# which gates correspond to a tagged rule (for reading the result; a tag is a tagger's reading, not ground truth)
TAG_GATES = {"R001": ["G009", "G017", "G018"], "R002": ["G013"], "R003": ["G004", "G005", "G006", "G007", "G008"], "R004": ["G015"],
             "R006": ["G003", "G010", "G011"], "R007": ["G001", "G002", "G016"], "R008": ["G016"], "R009": ["G008"], "R005": [], "R010": [], "R011": [], "R012": []}


def main(runs: list[str]) -> int:
    key = {r["opaque_id"]: r for r in csv.DictReader((ROOT / "pilot" / "key_PRIVATE.csv").open(encoding="utf-8"))}
    data = {}
    for name in runs:
        rows = read_rows(ROOT / "runs" / name / "results.jsonl")
        latest = {}
        for r in rows:
            if r.get("ok"):
                latest[r["image_id"]] = r
        data[name] = (rows, latest)
    for name, (rows, latest) in data.items():
        print(f"\n===== {name} =====")
        attempts = [r for r in rows if r.get("model_id")]
        print(f"calls made: {len(rows)}; valid answers: {sum(1 for r in rows if r['ok'])}; failed: {sum(1 for r in rows if not r['ok'])}")
        for r in rows:
            if not r["ok"]:
                print("  failure:", r["image_id"], r["error"][:120])
        use = [r["usage"] for r in latest.values() if r.get("usage")]
        pt = sum(u["prompt_tokens"] for u in use)
        ot = sum(u["output_tokens"] for u in use)
        th = sum(u.get("thinking_tokens") or 0 for u in use)
        cost = sum(r.get("cost_usd") or 0 for r in rows)
        print(f"tokens per photo: prompt {pt // len(use)}, output {ot // len(use)}, thinking {th // len(use)}; latency {sum(u['latency_s'] for u in use) / len(use):.0f}s; "
              f"cost of ALL calls in this run folder at the placeholder prices: ${cost:.3f} (the price for this model is unknown)")
        cover = Counter()
        for r in latest.values():
            d = r["decision"]
            syn = [w for w in d["warnings"] if "synthesized" in w]
            nac = [w for w in d["warnings"] if "na_conflict" in w]
            cover["missing_answers"] += len(syn)
            cover["na_conflicts"] += len(nac)
            cover["photos_with_missing"] += bool(syn)
            cover["gates_cannot_tell"] += len(d["uncertain_gates"])
        print("coverage:", dict(cover), "| criteria asked per photo:", len(next(iter(latest.values()))["validated"]["layer_2_criteria"]))
        print("\nphoto            decision                               failed gates              uncertain       tags (human)")
        for iid in sorted(latest):
            r = latest[iid]
            d = r["decision"]
            print(f"{iid}  {d['recommendation']:<37} {','.join(d['failed_gates']) or '-':<25} {','.join(d['uncertain_gates']) or '-':<15} {key[iid]['tags']}")
        # do the gates that fired match the tagged reasons?
        hit = total = 0
        for iid, r in latest.items():
            tags = [t for t in key[iid]["tags"].split(",") if t]
            fired = set(r["decision"]["failed_gates"]) | set(r["decision"]["uncertain_gates"])
            exp = {g for t in tags for g in TAG_GATES.get(t, [])}
            total += 1
            hit += bool(exp & fired)
        print(f"\nphotos where at least one gate matching a tagged reason fired (failed or unsure): {hit} of {total}")
        print("decisions:", dict(Counter(r["decision"]["recommendation"] for r in latest.values())))
        gcount = Counter(g for r in latest.values() for g in r["decision"]["failed_gates"])
        print("failed gates across photos:", dict(sorted(gcount.items())))
        q = [r["decision"]["quality"]["overall_score"] for r in latest.values() if r["decision"]["quality"]["overall_score"] is not None]
        print("quality scores:", q)
    if len(runs) == 2:
        a, b = data[runs[0]][1], data[runs[1]][1]
        shared = sorted(set(a) & set(b))
        print("\n===== layered vs legacy, same photos: shared legacy gates G001-G015 (excluding the ones the layered profile changes) =====")
        agree = diff = 0
        for iid in shared:
            ga = {g["gate_id"]: g["status"] for g in a[iid]["decision"]["gates"]}
            gb = {g["gate_id"]: g["status"] for g in b[iid]["decision"]["gates"]}
            for gid in gb:
                if gid in ga and gid not in ("G001", "G010", "G011", "G012", "G014"):
                    agree += ga[gid] == gb[gid]
                    diff += ga[gid] != gb[gid]
        print(f"gate statuses equal: {agree}, different: {diff}  (same photo, two separate calls with different prompts: the difference is a mix of prompt change and run-to-run variation)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
