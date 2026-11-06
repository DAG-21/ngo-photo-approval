"""Pilot 3 report (50 random photos, 25 + 25): decisions and gate rates by class, and a test of the hypothesis found in pilot 2 (offline).

    python scripts/pilot3_report.py
"""

import csv
import sys
from collections import Counter

import _bootstrap  # noqa: F401
import numpy as np

from layered_engine.annotation import read_rows
from layered_engine.rulebook import ROOT

FACE_BACK = {"G002", "G004", "G005", "G006", "G007", "G008", "G016"}  # the gates that separated the classes in pilot 2
AUTO = {"APPROVE", "APPROVED_PENDING_PUBLICATION_CHECKS"}


def latest(run):
    return {r["image_id"]: r for r in read_rows(ROOT / "runs" / run / "results.jsonl") if r.get("ok")}


def auc(pos, neg):
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def boot(pos, neg, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.array(pos), np.array(neg)
    v = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)]
    return np.percentile(v, [2.5, 97.5])


def main() -> int:
    key = {r["opaque_id"]: r for r in csv.DictReader((ROOT / "pilot" / "key3_PRIVATE.csv").open(encoding="utf-8"))}
    rows = read_rows(ROOT / "runs" / "pilot3_layered" / "results.jsonl")
    p3 = latest("pilot3_layered")
    app = [i for i in p3 if key[i]["label"] == "approved"]
    dis = [i for i in p3 if key[i]["label"] != "approved"]
    print(f"calls {len(rows)}, valid answers {len(p3)} of 50 ({len(app)} approved, {len(dis)} disapproved), failures {sum(1 for r in rows if not r['ok'])}")
    for r in rows:
        if not r["ok"]:
            print("  failure:", r["image_id"], r["error"][:140])
    use = [r["usage"] for r in p3.values()]
    print(f"per photo: prompt {sum(u['prompt_tokens'] for u in use) // len(use)}, output {sum(u['output_tokens'] for u in use) // len(use)}, thinking {sum(u.get('thinking_tokens') or 0 for u in use) // len(use)}, "
          f"latency {sum(u['latency_s'] for u in use) / len(use):.0f}s; cost ${sum(r['cost_usd'] for r in p3.values()):.2f} (placeholder prices)")
    print("criteria missing / N-A conflicts:", sum(len([w for w in r["decision"]["warnings"] if "synthesized" in w or "na_conflict" in w]) for r in p3.values()))

    print("\nDECISIONS BY CLASS")
    for name, grp in (("approved", app), ("disapproved", dis)):
        print(f"  {name:<12} n={len(grp):>2}: {dict(Counter(p3[i]['decision']['recommendation'] for i in grp))}")
    fr = sum(p3[i]["decision"]["recommendation"] not in AUTO for i in app)
    fa = sum(p3[i]["decision"]["recommendation"] in AUTO for i in dis)
    print(f"  approved photos NOT auto-approved (false rejection): {fr}/{len(app)} = {fr / len(app):.0%}")
    print(f"  disapproved photos auto-approved (false approval, the costly error): {fa}/{len(dis)} = {fa / len(dis):.0%}")
    print(f"  agreement with the humans: {(len(app) - fr) + (len(dis) - fa)}/{len(p3)}")

    nf = {i: len(p3[i]["decision"]["failed_gates"]) for i in p3}
    q = {i: p3[i]["decision"]["quality"]["overall_score"] for i in p3}
    a_ = auc([-nf[i] for i in app], [-nf[i] for i in dis])
    lo, hi = boot([-nf[i] for i in app], [-nf[i] for i in dis])
    qa = auc([q[i] for i in app], [q[i] for i in dis])
    qlo, qhi = boot([q[i] for i in app], [q[i] for i in dis])
    print(f"\nmean failed gates: approved {np.mean([nf[i] for i in app]):.1f}, disapproved {np.mean([nf[i] for i in dis]):.1f}")
    print(f"AUC fewer failed gates => approved: {a_:.2f} (95% range {lo:.2f}-{hi:.2f}); quality score: {qa:.2f} ({qlo:.2f}-{qhi:.2f}); 0.5 = no separation")

    print("\nGATE RATES (failed = NO)   approved(25)  disapproved(25)")
    gates = sorted({g["gate_id"] for r in p3.values() for g in r["decision"]["gates"]})
    for g in gates:
        a = sum(g in p3[i]["decision"]["failed_gates"] for i in app)
        d = sum(g in p3[i]["decision"]["failed_gates"] for i in dis)
        if a or d:
            print(f"  {g}  {a:>3} ({a / len(app):>4.0%})   {d:>3} ({d / len(dis):>4.0%})" + ("   <- face/back gate" if g in FACE_BACK else ""))
    ha = sum(bool(FACE_BACK & set(p3[i]["decision"]["failed_gates"])) for i in app)
    hd = sum(bool(FACE_BACK & set(p3[i]["decision"]["failed_gates"])) for i in dis)
    print(f"\nHYPOTHESIS FROM PILOT 2 (face/back gates fire on disapproved photos only): approved {ha}/25, disapproved {hd}/25  (pilot 2: 0/10 and 7/10; pilot 1: 0 approved, 4/10)")
    rule_dis = [i for i in dis if FACE_BACK & set(p3[i]["decision"]["failed_gates"])]
    print(f"  a rule 'reject only if a face/back gate fails' would stop {len(rule_dis)}/25 disapproved and reject {ha}/25 approved photos")
    unsure = sum(1 for r in p3.values() if r["decision"]["uncertain_gates"])
    print(f"  photos with a gate the model could not establish: {unsure}")

    print("\nTAGGED DISAPPROVED PHOTOS (taggers' reason vs failed gates)")
    for i in dis:
        if key[i]["tags"]:
            print(f"  {i} tags {key[i]['tags']:<12} -> {p3[i]['decision']['recommendation']:<12} failed {','.join(p3[i]['decision']['failed_gates']) or '-'}")
    print("\nAPPROVED PHOTOS THAT WERE AUTO-APPROVED:", [i for i in app if p3[i]["decision"]["recommendation"] in AUTO])
    print("DISAPPROVED PHOTOS THAT WERE AUTO-APPROVED:", [(i, key[i]["program"]) for i in dis if p3[i]["decision"]["recommendation"] in AUTO])
    return 0


if __name__ == "__main__":
    sys.exit(main())
