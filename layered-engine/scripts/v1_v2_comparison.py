"""V1 vs V2 on the same 50 photographs, and which V2 change does what. Offline: reads stored runs, calls no model.

    python scripts/v1_v2_comparison.py                       # tables on screen + reports/v1_vs_v2_comparison.json
    python scripts/v1_v2_comparison.py --excel reports/v1_vs_v2_comparison.xlsx

Inputs: runs/pilot3_layered (V1, profile layered_proposed_v1) and runs/pilot3_v2 (V2, profile layered_proposed_v2), the private key
pilot/key3_PRIVATE.csv (labels and tags, joined only here) and the photo numbering of reports/pilot3_50_photos.xlsx.

What each table means (see also the Notes sheet):
  * "as run": the decision stored when the model answered, V1 under its prompt, V2 under its prompt. Differences between the two mix the
    changed policy, the changed wording and ordinary run-to-run variation of the model.
  * "replay": the same stored answers re-decided by Python under a different policy. Exact for changes that only alter what Python does
    (hard or soft, applicability, aggregation). It says nothing about a change in what the model is ASKED.
  * Development comparison, not validation: these 50 photos motivated the V2 changes.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

from layered_engine.comparison import (AUTO, confusion_from, gate_table, headline, load_answers, redecide, transitions, without_gates)
from layered_engine.rulebook import ROOT, Rulebook

V1, V2 = "layered_proposed_v1", "layered_proposed_v2"
ALL = ["V2-01", "V2-02", "V2-03", "V2-04", "V2-05", "V2-06", "V2-07"]
NAMES = {"V2-01": "G015 gaze -> soft tag", "V2-02": "face gates -> primary/featured beneficiaries", "V2-03": "G009 folds only for a prominent banner",
         "V2-04": "G017/G018 banner visibility -> soft", "V2-05": "G016 -> volunteer back view only", "V2-06": "G001 component wording (D = activity only)",
         "V2-07": "G012 handover-based applicability"}
# the specification's ablation order (section 15): each step adds to the previous ones
STAGES = [("E1  V1 as run (original gates)", []), ("E2  + G015 soft", ["V2-01"]), ("E3  + beneficiary scope (primary/featured)", ["V2-02"]),
          ("E4  + conditional G012", ["V2-07"]), ("E5  + G001 wording and G016 restructure", ["V2-06", "V2-05"]), ("E6  + banner: G009 prominent-only, G017/G018 soft = full V2", ["V2-03", "V2-04"])]
FACE = {"G002", "G004", "G005", "G006", "G007", "G008"}
REMOVALS = [("G015 (gaze)", {"G015"}), ("G001 (angle)", {"G001"}), ("face gates G002, G004-G008", FACE), ("face gates + G001", FACE | {"G001"}),
            ("face + G001 + G015 + G016", FACE | {"G001", "G015", "G016"}), ("every gate removed", {f"G{n:03d}" for n in range(1, 19)})]


def load_key():
    key = {r["opaque_id"]: r for r in csv.DictReader((ROOT / "pilot" / "key3_PRIVATE.csv").open(encoding="utf-8"))}
    paths = {r["image_id"]: r["path"] for r in csv.DictReader((ROOT / "pilot" / "list3.csv").open(encoding="utf-8"))}
    num: dict[str, int] = {}
    wb = ROOT / "reports" / "pilot3_50_photos.xlsx"
    if wb.exists():
        import openpyxl
        ws = openpyxl.load_workbook(wb, read_only=True)["Photos"]
        for r in ws.iter_rows(min_row=2, values_only=True):
            if r[17]:
                num[r[17]] = int(r[0])
    if len(num) != 50:  # fall back to a stable order rather than guess the workbook's
        num = {i: n for n, i in enumerate(sorted(key), 1)}
    return key, paths, num


def pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def fmt_rate(r):
    return f"{r['k']}/{r['n']} = {pct(r['rate'])} ({pct(r['ci_low'])}-{pct(r['ci_high'])})"


def stage_rows(rows, rb, approved, num):
    """Cumulative replay of one set of stored answers through the E1..E6 policies."""
    out, prev = [], None
    for name, on in STAGES:
        cum = [c for _, ch in STAGES[: STAGES.index((name, on)) + 1] for c in ch]
        dec = redecide(rows, rb, V2, without=[c for c in ALL if c not in cum])
        h = headline(dec, approved)
        t = transitions(prev, dec, approved) if prev else None
        out.append({"stage": name, "changes_on": cum, "headline": h, "decisions": {i: d.recommendation.value for i, d in dec.items()},
                    "rescued_approved": sorted(num[i] for i in t["rescued_approved"]) if t else [], "new_false_approvals": sorted(num[i] for i in t["new_false_approvals"]) if t else [],
                    "newly_rejected_approved": sorted(num[i] for i in t["newly_rejected_approved"]) if t else [], "_dec": dec})
        prev = dec
    return out


def leave_one_out(rows, rb, approved, num):
    full = redecide(rows, rb, V2)
    out = []
    for ch in ALL:
        dec = redecide(rows, rb, V2, without=[ch])
        t = transitions(dec, full, approved)  # from "without this change" to "with it"
        out.append({"change": ch, "name": NAMES[ch], "rescued_approved": sorted(num[i] for i in t["rescued_approved"]), "new_false_approvals": sorted(num[i] for i in t["new_false_approvals"]),
                    "newly_rejected_approved": sorted(num[i] for i in t["newly_rejected_approved"]), "newly_stopped_disapproved": sorted(num[i] for i in t["newly_stopped_disapproved"])})
    return out


def noise(v1, v2, rb):
    """Run-to-run agreement on questions whose wording and applicability did not change between V1 and V2 (an estimate of how much two runs of the
    same model differ on the same photo). It also moves with the changed prompt, so it is a floor for the variation, not a clean measurement."""
    p1, p2 = rb.resolve(V1), rb.resolve(V2)
    same = [c for c in p1.asked_criteria() if c.criterion_id in p2.criteria and p2.criteria[c.criterion_id].description_operational == c.description_operational
            and p2.criteria[c.criterion_id].applicability == c.applicability]
    per = []
    for c in same:
        agree = tot = 0
        for i in v1:
            if i not in v2:
                continue
            a = {x["criterion_id"]: x["status"] for x in v1[i]["validated"]["layer_2_criteria"]}.get(c.criterion_id)
            b = {x["criterion_id"]: x["status"] for x in v2[i]["validated"]["layer_2_criteria"]}.get(c.criterion_id)
            if a is None or b is None:
                continue
            tot += 1
            agree += a == b
        per.append((c.criterion_id, agree, tot))
    facts = defaultdict(lambda: [0, 0])
    for i in v1:
        if i in v2:
            f1 = v1[i]["validated"]["layer_1_scene_understanding"]["scene_facts"]
            f2 = v2[i]["validated"]["layer_1_scene_understanding"]["scene_facts"]
            for k in f1:
                facts[k][1] += 1
                facts[k][0] += f1[k] == f2.get(k)
    ag = sum(a for _, a, _ in per)
    tt = sum(t for _, _, t in per)
    return {"unchanged_questions": len(same), "answers_compared": tt, "answers_identical": ag, "identical_share": ag / tt if tt else None,
            "per_question": sorted(per, key=lambda x: (x[1] / x[2]) if x[2] else 1)[:12], "facts": {k: v for k, v in facts.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", default="pilot3_layered")
    ap.add_argument("--v2", default="pilot3_v2")
    ap.add_argument("--json", default=str(ROOT / "reports" / "v1_vs_v2_comparison.json"))
    ap.add_argument("--excel", default=None)
    args = ap.parse_args()
    rb = Rulebook()
    key, paths, num = load_key()
    r1, r2 = load_answers(ROOT / "runs" / args.v1), load_answers(ROOT / "runs" / args.v2)
    ids = sorted(set(r1) & set(r2))
    print(f"V1 valid answers {len(r1)}, V2 valid answers {len(r2)}, on both {len(ids)}")
    r1, r2 = {i: r1[i] for i in ids}, {i: r2[i] for i in ids}
    approved = {i: key[i]["label"] == "approved" for i in ids}
    inv = {n: i for i, n in num.items()}

    # ---- sanity: replaying V1 answers under V1's own profile must reproduce what was stored
    rep = redecide(r1, rb, V1)
    same = sum(rep[i].recommendation.value == r1[i]["decision"]["recommendation"] and rep[i].failed_gates == r1[i]["decision"]["failed_gates"] for i in ids)
    rep2 = redecide(r2, rb, V2)
    same2 = sum(rep2[i].recommendation.value == r2[i]["decision"]["recommendation"] and rep2[i].failed_gates == r2[i]["decision"]["failed_gates"] for i in ids)
    print(f"replay reproduces the stored decisions: V1 {same}/{len(ids)}, V2 {same2}/{len(ids)}")

    # ---- as-run decisions (re-validated through the engine so both sides use the identical code path)
    d1, d2 = redecide(r1, rb, V1), redecide(r2, rb, V2)
    h1, h2 = headline(d1, approved), headline(d2, approved)
    trans = transitions(d1, d2, approved)
    out = {"n": len(ids), "replay_reproduces_stored": {"v1": same, "v2": same2}, "v1": h1, "v2": h2,
           "transitions_v1_to_v2": {k: sorted(num[i] for i in v) for k, v in trans.items()}}

    print("\n=== V1 vs V2, AS RUN (same 50 photos, 25 approved + 25 disapproved) ===")
    rows_ = [("approved photos auto-approved (TP)", lambda h: h["confusion"]["TP"]), ("approved photos NOT auto-approved (FN)", lambda h: h["confusion"]["FN"]),
             ("  of which REJECT", lambda h: h["decisions_by_class"]["approved"].get("REJECT", 0)), ("  of which HUMAN_REVIEW", lambda h: h["decisions_by_class"]["approved"].get("HUMAN_REVIEW", 0)),
             ("  of which RESELECT", lambda h: h["decisions_by_class"]["approved"].get("RESELECT", 0)),
             ("disapproved photos stopped (TN)", lambda h: h["confusion"]["TN"]), ("disapproved photos auto-approved (FP)", lambda h: h["confusion"]["FP"]),
             ("false-rejection rate", lambda h: fmt_rate(h["false_rejection_rate"])), ("  hard (explicit REJECT) only", lambda h: fmt_rate(h["hard_false_rejection_rate"])),
             ("false-approval rate", lambda h: fmt_rate(h["false_approval_rate"])), ("approval precision", lambda h: fmt_rate(h["approval_precision"])),
             ("rejection precision (of explicit REJECTs)", lambda h: fmt_rate(h["rejection_precision"])), ("rejection recall (disapproved explicitly REJECTed)", lambda h: fmt_rate(h["rejection_recall"])),
             ("agreement with historical decision", lambda h: fmt_rate(h["agreement"])), ("human reviews", lambda h: h["ai_human_review"]),
             ("mean failed gates (approved / disapproved)", lambda h: f"{h['mean_failed_gates']['approved']:.1f} / {h['mean_failed_gates']['disapproved']:.1f}"),
             ("AUC fewer failed gates => approved", lambda h: f"{h['auc_failed_gates']['auc']:.2f} ({h['auc_failed_gates']['ci_low']:.2f}-{h['auc_failed_gates']['ci_high']:.2f})"),
             ("AUC quality score", lambda h: f"{h['auc_quality_score']:.2f}"),
             ("mean quality score (approved / disapproved)", lambda h: f"{h['mean_quality']['approved']:.1f} / {h['mean_quality']['disapproved']:.1f}")]
    print(f"{'':54}{'V1':>26}{'V2':>26}")
    for label, fn in rows_:
        print(f"{label:54}{str(fn(h1)):>26}{str(fn(h2)):>26}")
    print("false approvals V1:", sorted(num[i] for i in h1["false_approval_ids"]), "| V2:", sorted(num[i] for i in h2["false_approval_ids"]))
    print("photo-level changes V1 -> V2:", {k: len(v) for k, v in trans.items()})
    for k in ("rescued_approved", "new_false_approvals", "newly_rejected_approved", "newly_stopped_disapproved"):
        print(f"  {k}: {sorted(num[i] for i in trans[k])}")

    # ---- replays
    print("\n=== OFFLINE REPLAY of V1's stored answers through the spec's ablation order (policy-only changes show; wording changes cannot) ===")
    st1 = stage_rows(r1, rb, approved, num)
    for s in st1:
        c = s["headline"]["confusion"]
        print(f"{s['stage']:<62} TP{c['TP']:>3} FN{c['FN']:>3} TN{c['TN']:>3} FP{c['FP']:>3}  rescued {s['rescued_approved']}  new FA {s['new_false_approvals']}")
    print("\n=== OFFLINE REPLAY of V2's stored answers through the same stages (E1 = V1's gate policy applied to V2's answers) ===")
    st2 = stage_rows(r2, rb, approved, num)
    for s in st2:
        c = s["headline"]["confusion"]
        print(f"{s['stage']:<62} TP{c['TP']:>3} FN{c['FN']:>3} TN{c['TN']:>3} FP{c['FP']:>3}  rescued {s['rescued_approved']}  new FA {s['new_false_approvals']}")
    print("\n=== LEAVE-ONE-OUT on V2's stored answers: what each change does to the full V2 (same answers, so no run-to-run noise) ===")
    loo = leave_one_out(r2, rb, approved, num)
    for r in loo:
        print(f"{r['change']} {r['name']:<48} rescues approved {r['rescued_approved']}  new false approvals {r['new_false_approvals']}  newly rejected approved {r['newly_rejected_approved']}  newly stopped disapproved {r['newly_stopped_disapproved']}")
    out["stages_v1_answers"] = [{k: v for k, v in s.items() if k != "_dec"} for s in st1]
    out["stages_v2_answers"] = [{k: v for k, v in s.items() if k != "_dec"} for s in st2]
    out["leave_one_out_v2_answers"] = loo

    # ---- gate level
    g1, g2 = gate_table(d1, approved), gate_table(d2, approved)
    gm1, gm2 = {g["gate"]: g for g in g1}, {g["gate"]: g for g in g2}
    print("\n=== GATE LEVEL (denominator = photos on which the gate applies). fail = status NO. 'rej' = photos REJECTed with this gate among their failures; 'sole' = rejected for this gate alone ===")
    print(f"{'gate':<5}{'hard':<6}| {'V1 app fail':>12}{'V1 dis fail':>12}{'V1 rej A/D':>11}{'V1 sole A/D':>12} || {'V2 app fail':>12}{'V2 dis fail':>12}{'V2 rej A/D':>11}{'V2 sole A/D':>12}")
    for gid in sorted(set(gm1) | set(gm2)):
        def cell(g):
            if g is None:
                return ["-"] * 4
            a, d = g["approved"], g["disapproved"]
            f = lambda x: f"{x['no']}/{x['applicable']}"  # noqa: E731
            return [f(a), f(d), f"{g['false_rejections_with_gate']}/{g['correct_rejections_with_gate']}", f"{g['sole_cause_false_rejections']}/{g['sole_cause_correct_rejections']}"]
        a, b = cell(gm1.get(gid)), cell(gm2.get(gid))
        hard1, hard2 = ("H" if gm1.get(gid, {}).get("hard") else "s"), ("H" if gm2.get(gid, {}).get("hard") else "s")
        print(f"{gid:<5}{hard1 + '->' + hard2:<6}| {a[0]:>12}{a[1]:>12}{a[2]:>11}{a[3]:>12} || {b[0]:>12}{b[1]:>12}{b[2]:>11}{b[3]:>12}")
    out["gate_table_v1"], out["gate_table_v2"] = g1, g2

    # ---- upper bounds
    print("\n=== UPPER BOUND: what if these gates were simply deleted (not a proposal) ===")
    ub = []
    for label, rem in REMOVALS:
        for tag, dec, prof in (("V1 answers", d1, rb.resolve(V1)), ("V2 answers", d2, rb.resolve(V2))):
            v = without_gates(dec, rem, prof)
            c = confusion_from(v, approved)
            ub.append({"removed": label, "answers": tag, **c})
            print(f"  remove {label:<30} {tag}: approved auto-approved {c['TP']:>2}/25, disapproved auto-approved (false approvals) {c['FP']:>2}/25")
    out["upper_bounds"] = ub

    # ---- conditional applicability, unsure, missing
    print("\n=== APPLICABILITY AND UNCERTAINTY ===")
    for tag, rows, dec in (("V1", r1, d1), ("V2", r2, d2)):
        na = Counter(g.gate_id for d in dec.values() for g in d.gates if g.hard and g.status.value == "NOT_APPLICABLE")
        ct = sum(1 for d in dec.values() for g in d.gates if g.hard and g.status.value == "CANNOT_TELL")
        miss = sum(1 for d in dec.values() for w in d.warnings if "synthesized" in w or "na_conflict" in w)
        print(f"  {tag}: hard-gate NOT_APPLICABLE {sum(na.values())} {dict(sorted(na.items()))}; CANNOT_TELL on a hard gate {ct}; criteria missing or in conflict {miss}")
        out.setdefault("applicability", {})[tag] = {"not_applicable": dict(na), "cannot_tell": ct, "missing_or_conflict": miss}
    gz = defaultdict(Counter)
    for i in ids:
        gz["approved" if approved[i] else "disapproved"][r2[i]["validated"]["layer_1_scene_understanding"]["scene_facts"].get("f_gaze_tag")] += 1
    print("  V2 gaze tag by class:", {k: dict(v) for k, v in gz.items()})
    out["gaze_tag_by_class"] = {k: dict(v) for k, v in gz.items()}
    act = Counter((r1[i]["validated"]["layer_1_scene_understanding"]["scene_facts"]["f_activity_type"], r2[i]["validated"]["layer_1_scene_understanding"]["scene_facts"]["f_activity_type"]) for i in ids)
    print("  activity type changed between runs on", sum(v for (a, b), v in act.items() if a != b), "of", len(ids), "photos")

    nz = noise(r1, r2, rb)
    print(f"\n=== RUN-TO-RUN: {nz['unchanged_questions']} questions whose wording did not change; identical answers on {nz['answers_identical']}/{nz['answers_compared']} = {nz['identical_share']:.0%} ===")
    for cid, a, t in nz["per_question"][:8]:
        print(f"  least stable: {cid:<30} {a}/{t} identical")
    print("  scene facts identical between runs:", {k: f"{v[0]}/{v[1]}" for k, v in nz["facts"].items()})
    out["run_to_run"] = nz

    # ---- per photo
    per = []
    for i in sorted(ids, key=lambda x: num[x]):
        per.append({"n": num[i], "id": i, "label": "approved" if approved[i] else "disapproved", "tags": key[i]["tags"], "v1": d1[i].recommendation.value, "v1_failed": d1[i].failed_gates,
                    "v2": d2[i].recommendation.value, "v2_failed": d2[i].failed_gates, "v2_uncertain": d2[i].uncertain_gates, "v1_quality": d1[i].quality.overall_score, "v2_quality": d2[i].quality.overall_score})
    out["photos"] = per
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("\nwrote", args.json)

    if args.excel:
        from excel_v1_v2 import build  # noqa: E402  (kept in its own file: it is only presentation)
        build(Path(args.excel), out, key, paths, num, r1, r2, d1, d2, approved, STAGES, NAMES, ALL)
        print("wrote", args.excel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
