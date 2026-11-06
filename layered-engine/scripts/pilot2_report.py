"""Pilot 2 report: gate and decision rates by class (offline; reads runs/pilot2_layered, pilot/key2_PRIVATE.csv and, for comparison, pilot 1 and my own audit).

    python scripts/pilot2_report.py
"""

import csv
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from layered_engine.annotation import read_rows
from layered_engine.rulebook import ROOT

INTERN = ROOT.parent


def latest(run: str) -> dict:
    out = {}
    for r in read_rows(ROOT / "runs" / run / "results.jsonl"):
        if r.get("ok"):
            out[r["image_id"]] = r
    return out


def auc(pos: list[float], neg: list[float]) -> float:
    """P(a random approved photo gets a higher value than a random disapproved one); ties count half."""
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main() -> int:
    key = {r["opaque_id"]: r for r in csv.DictReader((ROOT / "pilot" / "key2_PRIVATE.csv").open(encoding="utf-8"))}
    p2 = latest("pilot2_layered")
    cls = {iid: key[iid]["label"] for iid in p2}
    app = [i for i in p2 if cls[i] == "approved"]
    dis = [i for i in p2 if cls[i] == "not_approved"]
    print(f"valid answers: {len(p2)} of 20 ({len(app)} approved, {len(dis)} disapproved)")
    use = [r["usage"] for r in p2.values()]
    print(f"per photo: prompt {sum(u['prompt_tokens'] for u in use) // len(use)}, output {sum(u['output_tokens'] for u in use) // len(use)}, "
          f"thinking {sum(u.get('thinking_tokens') or 0 for u in use) // len(use)}, latency {sum(u['latency_s'] for u in use) / len(use):.0f}s, cost ${sum(r['cost_usd'] for r in p2.values()):.3f} (placeholder prices)")
    cov = sum(len([w for w in r["decision"]["warnings"] if "synthesized" in w or "na_conflict" in w]) for r in p2.values())
    print("criteria missing or N/A-conflicting across all photos:", cov)

    print("\nDECISIONS BY CLASS")
    for name, grp in (("approved by the NGO", app), ("disapproved by the NGO", dis)):
        c = Counter(p2[i]["decision"]["recommendation"] for i in grp)
        print(f"  {name:<24} n={len(grp):>2}: {dict(c)}")
    rej_app = sum(p2[i]["decision"]["recommendation"] == "REJECT" for i in app)
    notapp = sum(p2[i]["decision"]["recommendation"] in ("REJECT", "HUMAN_REVIEW", "RESELECT") for i in app)
    caught = sum(p2[i]["decision"]["recommendation"] in ("REJECT", "HUMAN_REVIEW", "RESELECT") for i in dis)
    approved_ok = sum(p2[i]["decision"]["recommendation"] in ("APPROVE", "APPROVED_PENDING_PUBLICATION_CHECKS") for i in dis)
    print(f"  approved photos REJECTED outright: {rej_app}/{len(app)}; not auto-approved (reject/review/reselect): {notapp}/{len(app)}")
    print(f"  disapproved photos stopped: {caught}/{len(dis)}; WRONGLY auto-approved (the costly error): {approved_ok}/{len(dis)}")

    print("\nFAILED-GATE RATE BY CLASS (share of photos with that gate = NO)")
    gates = sorted({g["gate_id"] for r in p2.values() for g in r["decision"]["gates"]})
    fa = {g: sum(1 for i in app if g in p2[i]["decision"]["failed_gates"]) for g in gates}
    fd = {g: sum(1 for i in dis if g in p2[i]["decision"]["failed_gates"]) for g in gates}
    print("  gate   approved  disapproved")
    for g in gates:
        if fa[g] or fd[g]:
            print(f"  {g}   {fa[g]:>2}/{len(app)}     {fd[g]:>2}/{len(dis)}")
    nf = {i: len(p2[i]["decision"]["failed_gates"]) for i in p2}
    qs = {i: p2[i]["decision"]["quality"]["overall_score"] for i in p2}
    print(f"\nmean failed gates: approved {np.mean([nf[i] for i in app]):.1f}, disapproved {np.mean([nf[i] for i in dis]):.1f}")
    print(f"mean quality score: approved {np.mean([qs[i] for i in app]):.1f}, disapproved {np.mean([qs[i] for i in dis]):.1f}")
    print(f"AUC (n=10 vs 10, very imprecise): fewer failed gates => approved: {auc([-nf[i] for i in app], [-nf[i] for i in dis]):.2f}; higher quality score => approved: {auc([qs[i] for i in app], [qs[i] for i in dis]):.2f}")

    # pool with pilot 1 (10 disapproved photos, same profile) for a larger disapproved sample
    p1 = latest("pilot_layered_proposed_v1")
    nf1 = [len(r["decision"]["failed_gates"]) for r in p1.values()]
    q1 = [r["decision"]["quality"]["overall_score"] for r in p1.values()]
    print(f"\nPOOLED with pilot 1 (disapproved n={len(dis) + len(p1)}): mean failed gates {np.mean([nf[i] for i in dis] + nf1):.1f}; AUC for failed-gate count: "
          f"{auc([-nf[i] for i in app], [-nf[i] for i in dis] + [-x for x in nf1]):.2f}; for quality score: {auc([qs[i] for i in app], [qs[i] for i in dis] + q1):.2f}")

    print("\nAPPROVED PHOTOS: gates that failed vs the rules I flagged when I looked at the photo myself (my audit)")
    spec = importlib.util.spec_from_file_location("claude_audit", INTERN / "holistic-judge-trial" / "scripts" / "claude_audit_excel.py")
    sys.path.insert(0, str(INTERN / "holistic-judge-trial" / "scripts"))
    sys.path.insert(0, str(INTERN / "holistic-judge-trial" / "src"))
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        orig_ids = sorted(r["image_id"] for r in csv.DictReader((INTERN / "holistic-judge-trial" / "ids" / "paired_tagged_dev.csv").open(encoding="utf-8")) if r["label"] == "approved")
        mine = {iid: sorted({rule for rule, _ in mod.DATA[n]}) for n, iid in enumerate(orig_ids, 1)}
        for iid in app:
            print(f"  {iid}  {p2[iid]['decision']['recommendation']:<37} failed {','.join(p2[iid]['decision']['failed_gates']) or '-':<22} | my audit of this photo: {','.join(mine.get(key[iid]['image_id'], [])) or '(none)'}")
    except Exception as exc:  # noqa: BLE001
        print("  (could not load my audit:", type(exc).__name__, exc, ")")
    print("\nDISAPPROVED PHOTOS")
    for iid in dis:
        print(f"  {iid}  {p2[iid]['decision']['recommendation']:<20} failed {','.join(p2[iid]['decision']['failed_gates']) or '-':<24} | tagged: {key[iid]['tags']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
