"""Does showing the model similar past staff decisions change what it answers, and does that agree better with staff? A small, real-model measurement on pilot photographs.

    python scripts/measure_examples_effect.py --photos 20                 # DRY RUN: the plan and the cost estimate, no call
    python scripts/measure_examples_effect.py --photos 20 --run           # the real run: sends photographs to Gemini (needs the cloud switch and key)
    python scripts/measure_examples_effect.py --analyze                   # the report from the saved results (no call)

Design
* The same photographs are evaluated twice, now, with the same prompt, rules and model: WITHOUT examples (the control) and WITH the 3 most similar other pilot photographs and their
  staff decisions as examples (balanced approved/rejected). A photograph is never among its own examples (leave-one-out). Comparing with a re-run, not with the stored pilot answers,
  because the model is not deterministic.
* Only the staff DECISION of each example is sent (approved / rejected), with no tags and no staff reasons (the pilot recorded none, and its tags are the benchmark's labels).
* A hard spend cap, a call cap, and a results file that is appended to after every call, so a crash or a second run never pays twice for the same call.
* Only aggregates are printed; photograph ids are the pilot's opaque ids and the staff decisions come from its private key file.
With 20 photographs this can show a large effect or a harm; it cannot show a small one. The report says so.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE / "src"), str(HERE.parent / "layered-engine" / "src")]

from layered_engine.annotation import read_rows  # noqa: E402
from layered_engine.providers import GeminiProvider, load_gemini_config, read_env_file  # noqa: E402
from layered_engine.rulebook import Rulebook  # noqa: E402

from ngo_agent.embeddings import make_embedder  # noqa: E402
from ngo_agent.evaluators import LayeredEngineEvaluator  # noqa: E402
from ngo_agent.example_service import Candidate, select_examples  # noqa: E402

ENGINE = HERE.parent / "layered-engine"
PROJECT_ENV = HERE.parent / "image-classifier" / ".env"
OUT = HERE / "reports"
RESULTS = OUT / "l2_examples_results.jsonl"
PROFILE = "layered_proposed_v1"
SEED = 20261103
NO_REASON = "No written reason was recorded for this pilot decision."
IMAGE_TOKENS = 1120          # assumed tokens per image at high media resolution; the run measures the real figure


def load_pilot():
    key = {r["opaque_id"]: r["label"] for r in csv.DictReader((ENGINE / "pilot" / "key3_PRIVATE.csv").open(encoding="utf-8"))}
    paths = {r["image_id"]: Path(r["path"]) for r in csv.DictReader((ENGINE / "pilot" / "list3.csv").open(encoding="utf-8"))}
    ids = sorted(i for i in paths if i in key and paths[i].exists())
    return ids, {i: ("APPROVE" if key[i] == "approved" else "REJECT") for i in ids}, paths


def choose_test_photos(ids, decision, n):
    rnd = random.Random(SEED)
    half = n // 2
    a, r = [i for i in ids if decision[i] == "APPROVE"], [i for i in ids if decision[i] == "REJECT"]
    return sorted(rnd.sample(a, half) + rnd.sample(r, n - half))


def build_examples(test_id, ids, decision, paths, vectors, k):
    """The k examples for one test photograph: the most similar OTHER pilot photographs, balanced, with their staff decision only."""
    cands = [Candidate(j, float(vectors[test_id] @ vectors[j]), decision[j]) for j in ids if j != test_id]
    chosen = select_examples(cands, k, 0.0)
    return [{"image_bytes": paths[c.photo_id].read_bytes(), "mime_type": "image/jpeg", "decision": c.decision, "tags": [], "reasons": NO_REASON,
             "similarity": round(c.similarity, 4), "photo_id": c.photo_id} for c in chosen], max(c.similarity for c in cands)


def stored_baseline():
    rows = [r for r in read_rows(ENGINE / "runs" / "pilot3_layered" / "results.jsonl") if r.get("ok")]
    pt = [r["usage"]["prompt_tokens"] for r in rows if r.get("usage", {}).get("prompt_tokens")]
    ot = [r["usage"]["output_tokens"] for r in rows if r.get("usage", {}).get("output_tokens")]
    return {r["image_id"]: r for r in rows}, float(np.mean(pt)), float(np.mean(ot))


def done_keys():
    if not RESULTS.exists():
        return set()
    return {(r["photo"], r["condition"]) for r in map(json.loads, RESULTS.read_text(encoding="utf-8").splitlines()) if r.get("ok")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=int, default=20)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--run", action="store_true", help="really call the model (without it: a dry run)")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--max-cost", type=float, default=3.0, help="hard cap in USD at the placeholder price")
    ap.add_argument("--max-calls", type=int, default=45)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.analyze:
        return analyze()
    ids, decision, paths = load_pilot()
    test = choose_test_photos(ids, decision, args.photos)
    print(f"pilot: {len(ids)} photographs; test set: {len(test)} ({sum(decision[i] == 'APPROVE' for i in test)} approved, {sum(decision[i] == 'REJECT' for i in test)} rejected by staff), chosen with seed {SEED}")
    emb = make_embedder("clip")
    print("embedding the pilot photographs locally (CLIP) ...")
    vectors = {i: np.asarray(emb.embed(paths[i].read_bytes()), dtype=np.float32) for i in ids}
    plan = {}
    for t in test:
        ex, top = build_examples(t, ids, decision, paths, vectors, args.k)
        plan[t] = (ex, top)
    near = sum(top > 0.9 for _, top in plan.values())
    mix = [sum(e["decision"] == "APPROVE" for e in ex) for ex, _ in plan.values()]
    print(f"examples per photograph: {args.k}; approved among them: min {min(mix)}, max {max(mix)}; photographs whose closest example is above 0.9 similarity (possible near-duplicate): {near}")
    stored, p_tok, o_tok = stored_baseline()
    cfg = load_gemini_config()
    inp, outp = cfg["cost"]["input_per_million_usd"], cfg["cost"]["output_per_million_usd"]
    base = (p_tok * inp + o_tok * outp) / 1e6
    withx = base + args.k * (IMAGE_TOKENS + 120) * inp / 1e6
    todo = [(t, c) for t in test for c in ("without", "with") if (t, c) not in done_keys()]
    est = sum(base if c == "without" else withx for _, c in todo)
    print(f"stored pilot calls averaged {p_tok:.0f} input and {o_tok:.0f} output tokens: about ${base:.4f} per call at the placeholder price {inp}/{outp} USD per million tokens")
    print(f"estimate: {len(todo)} calls to make ({len(test) * 2 - len(todo)} already done); about ${est:.2f} (without examples ${base:.4f} a call, with {args.k} examples ${withx:.4f}); cap ${args.max_cost:.2f} and {args.max_calls} calls")
    if est > args.max_cost or len(todo) > args.max_calls:
        print("REFUSED: the estimate is above the cap")
        return 2
    if not args.run:
        print("DRY RUN: nothing was sent. Add --run to make the calls.")
        return 0
    env = read_env_file(PROJECT_ENV, (cfg["auth"]["api_key_env"], cfg["auth"]["authorization_env"]))
    ev = LayeredEngineEvaluator(GeminiProvider(cfg, env=env), cfg, Rulebook())        # fails closed without the cloud switch and the key
    lock, spent, calls = threading.Lock(), [0.0], [0]

    def work(task):
        t, cond = task
        with lock:
            if spent[0] + withx > args.max_cost or calls[0] >= args.max_calls:
                return None
            calls[0] += 1
        img = paths[t].read_bytes()
        with Image.open(io.BytesIO(img)) as im:
            meta = {"width": im.size[0], "height": im.size[1]}
        row = {"photo": t, "condition": cond, "ok": False, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            ex = plan[t][0] if cond == "with" else None
            o = ev.evaluate(img, "image/jpeg", {"profile": PROFILE}, meta, examples=ex)
            gates = {g["gate_id"]: g["status"] for g in o.decision.get("gates", [])}
            row.update(ok=True, recommendation=o.decision["recommendation"], failed_gates=o.decision.get("failed_gates", []), gates=gates, model_id=o.model_id, usage=o.usage, cost_usd=o.cost_usd,
                       criteria={c["criterion_id"]: c["status"] for c in o.validated.get("layer_2_criteria", [])},
                       examples=[{"decision": e["decision"], "similarity": e["similarity"]} for e in ex] if ex else [])
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        with lock:
            spent[0] += row.get("cost_usd") or 0.0
            with RESULTS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"  {t} {cond}: {row.get('recommendation') or row.get('error')}  (spent so far ${spent[0]:.3f})", flush=True)
        return row

    print(f"RUNNING {len(todo)} calls with {args.workers} at a time ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    print(f"finished: {calls[0]} calls made, ${spent[0]:.3f} spent at the placeholder price")
    return analyze()


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact p for the discordant pairs (b: only the first condition right, c: only the second)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def auc(pos, neg):
    """P(a staff-approved photograph has fewer failed gates than a staff-rejected one), ties counting half."""
    if not pos or not neg:
        return float("nan")
    return float(np.mean([(1.0 if n > p else 0.5 if n == p else 0.0) for p in pos for n in neg]))


def finer_analysis(photos, by, stored, human, summary):
    """Evidence finer than the yes/no outcome, from the saved results (no model call): did the examples move the model's answers more than running it again does,
    and does the number of failed gates separate staff-approved from staff-rejected photographs any better?"""
    print("\nFINER EVIDENCE (the yes/no outcome is stuck at 'not approved' for almost every photograph, so it cannot show an effect)")
    recs = lambda cond: [by[(p, cond)]["recommendation"] for p in photos]  # noqa: E731
    moved = [(a, b) for a, b in zip(recs("without"), recs("with")) if a != b]
    print(f"recommendation: changed by the examples on {len(moved)} of {len(photos)} photographs "
          + (", ".join(f"{a} to {b}" for a, b in moved) if moved else "") + "; recommendations in the control: " + ", ".join(f"{k} {recs('without').count(k)}" for k in sorted(set(recs('without')))))
    st_recs = [stored[p]["decision"]["recommendation"] for p in photos if p in stored]
    print(f"recommendation: the control differs from the stored pilot answer on {sum(a != b for a, b in zip(recs('without'), st_recs))} photographs; the examples run differs from it on {sum(a != b for a, b in zip(recs('with'), st_recs))}")
    stc = {p: {c["criterion_id"]: c["status"] for c in stored[p]["validated"]["layer_2_criteria"]} for p in photos if p in stored}

    def diff(a, b):
        keys = set(a) & set(b)
        return sum(a[k] != b[k] for k in keys) / len(keys)

    noise = [diff(stc[p], by[(p, "without")]["criteria"]) for p in photos if p in stc]
    effect = [diff(by[(p, "without")]["criteria"], by[(p, "with")]["criteria"]) for p in photos]
    both = [diff(stc[p], by[(p, "with")]["criteria"]) for p in photos if p in stc]
    print(f"criteria answers that differ: running the same prompt again moves {np.mean(noise):.1%} of them (stored pilot vs control: the noise); adding examples moves {np.mean(effect):.1%} "
          f"(control vs examples); stored pilot vs examples run {np.mean(both):.1%}. Photographs where the examples moved more than the noise did: {sum(e > n for e, n in zip(effect, noise))} of {len(noise)}")
    fg = {"stored": {p: len(stored[p]["decision"]["failed_gates"]) for p in photos if p in stored}, "without": {p: len(by[(p, 'without')]['failed_gates']) for p in photos},
          "with": {p: len(by[(p, 'with')]['failed_gates']) for p in photos}}
    rnd = np.random.default_rng(0)
    for name, d in fg.items():
        pos, neg = [d[p] for p in d if human[p]], [d[p] for p in d if not human[p]]
        boots = []
        for _ in range(2000):
            bp, bn = rnd.choice(pos, len(pos)), rnd.choice(neg, len(neg))
            boots.append(auc(list(bp), list(bn)))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        print(f"failed gates, {name}: staff-approved photographs {np.mean(pos):.1f} on average, staff-rejected {np.mean(neg):.1f}; chance that an approved photograph has fewer failed gates than a rejected one "
              f"{auc(pos, neg):.2f} (95% interval {lo:.2f}-{hi:.2f}; 0.50 is chance)")
        summary[f"auc_failed_gates_{name}"] = auc(pos, neg)
    summary.update(criteria_moved_by_noise=float(np.mean(noise)), criteria_moved_by_examples=float(np.mean(effect)), recommendation_changes=len(moved))


def analyze() -> int:
    ids, decision, paths = load_pilot()
    rows = [r for r in map(json.loads, RESULTS.read_text(encoding="utf-8").splitlines()) if r.get("ok")] if RESULTS.exists() else []
    by = {(r["photo"], r["condition"]): r for r in rows}
    stored, _, _ = stored_baseline()
    photos = sorted({p for p, _ in by if (p, "without") in by and (p, "with") in by})
    if not photos:
        print("no complete pair of results yet")
        return 1
    appr = lambda rec: rec in ("APPROVE", "APPROVED_PENDING_PUBLICATION_CHECKS")  # noqa: E731
    human = {p: decision[p] == "APPROVE" for p in photos}

    def conf(cond):
        c = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for p in photos:
            a = appr(by[(p, cond)]["recommendation"])
            c[("T" if a == human[p] else "F") + ("P" if a else "N")] += 1
        return c

    print(f"\nRESULTS on {len(photos)} photographs ({sum(human.values())} approved, {sum(not v for v in human.values())} rejected by staff); positive = the system approves\n")
    summary = {}
    for cond, label in (("without", "WITHOUT examples (control, run now)"), ("with", "WITH 3 similar past decisions")):
        c = conf(cond)
        agree = c["TP"] + c["TN"]
        lo, hi = wilson(agree, len(photos))
        summary[cond] = c
        print(f"{label}: agrees with staff on {agree}/{len(photos)} ({agree / len(photos):.0%}; 95% interval {lo:.0%}-{hi:.0%}); system approved a photograph staff rejected: {c['FP']}; "
              f"did not approve one staff approved: {c['FN']}")
    stored_ok = [p for p in photos if p in stored]
    if stored_ok:
        flips = sum(appr(by[(p, 'without')]['recommendation']) != appr(stored[p]["decision"]["recommendation"]) for p in stored_ok)
        print(f"run-to-run noise: the same prompt, run again, changed the approved / not approved outcome on {flips} of {len(stored_ok)} photographs compared with the stored pilot answer")
    right = lambda cond, p: appr(by[(p, cond)]["recommendation"]) == human[p]  # noqa: E731
    b = sum(right("without", p) and not right("with", p) for p in photos)       # right only without examples
    c_ = sum(right("with", p) and not right("without", p) for p in photos)      # right only with examples
    print(f"\npaired: photographs only the control got right {b}; only the examples run got right {c_}; same result {len(photos) - b - c_}. Exact test p = {exact_mcnemar(b, c_):.2f}")
    fp_new = [p for p in photos if appr(by[(p, 'with')]['recommendation']) and not human[p] and not appr(by[(p, 'without')]['recommendation'])]
    print(f"photographs staff rejected that the examples run approved and the control did not: {len(fp_new)}")
    changed = [sum(1 for k, v in by[(p, "with")]["criteria"].items() if by[(p, "without")]["criteria"].get(k) != v) for p in photos]
    print(f"criteria answers that differ between the two runs: on average {np.mean(changed):.1f} per photograph (of {len(by[(photos[0], 'with')]['criteria'])}); photographs with any difference {sum(x > 0 for x in changed)}")
    # copying: does the verdict follow the majority of the examples more than the control does?
    follow = lambda cond: sum(appr(by[(p, cond)]["recommendation"]) == (sum(e["decision"] == "APPROVE" for e in by[(p, "with")]["examples"]) >= 2) for p in photos)  # noqa: E731
    print(f"copying check: the system's verdict equals the majority of the 3 examples on {follow('with')}/{len(photos)} photographs with examples, {follow('without')}/{len(photos)} without them")
    finer_analysis(photos, by, stored, human, summary)
    for cond in ("without", "with"):
        u = [by[(p, cond)]["usage"] for p in photos]
        pt, ot = np.mean([x["prompt_tokens"] for x in u]), np.mean([x["output_tokens"] for x in u])
        cost = np.mean([by[(p, cond)]["cost_usd"] or 0 for p in photos])
        print(f"{cond}: {pt:.0f} input tokens and {ot:.0f} output tokens per call on average; ${cost:.4f} per call at the placeholder price")
        summary[cond + "_usage"] = {"input_tokens": float(pt), "output_tokens": float(ot), "cost_usd": float(cost)}
    summary.update(photographs=len(photos), only_control_right=b, only_examples_right=c_, p=exact_mcnemar(b, c_))
    (OUT / "l2_examples_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nWith this few photographs only a large effect could show; a small gain or loss cannot be told from chance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
