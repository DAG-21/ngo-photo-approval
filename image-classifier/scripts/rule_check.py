"""Run every approved / not_approved image against all active rules, then evaluate how the results line up
with the human labels.

    python scripts/rule_check.py run --limit 20            # DRY RUN (prints the plan, sends nothing)
    python scripts/rule_check.py run --limit 20 --execute  # pilot: 20 unique images (half per class)
    python scripts/rule_check.py run --execute             # everything (resumable: re-run to continue)
    python scripts/rule_check.py eval                      # evaluation + docs/RULE_CHECK_REPORT.md

"Follows a rule" = the rule applied to the photo and the model said PASS. A PASS whose evidence starts with
"N/A:" (rule not applicable, e.g. no banner in the photo) is NOT counted as following the rule.
Byte-identical images are analyzed once (by SHA-256) and the result is reused for every copy, so each unique
photo is sent to Gemini at most once. This is an exploratory rule check, not model selection: nothing here is used
to tune thresholds or choose models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ngo_photo_approval.config import load_model_config, load_project_config, setup_logging
from ngo_photo_approval.exceptions import NgoPhotoError
from ngo_photo_approval.models.baseline_analyzer import PROMPT_VERSION, RuleAnalyzer
from ngo_photo_approval.models.batch_runner import run_batch
from ngo_photo_approval.models.cloud_client import GeminiBackend
from ngo_photo_approval.rules.loader import load_rule_set
from ngo_photo_approval.scoring.offline import load_analyses, load_manifest

LABELS = ("approved", "not_approved")


def slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def run_dir(project, model_id: str, rule_set, version: str | None = None) -> Path:
    """Results are kept per (model, prompt version, rule-set version) so analyses made with different wording or
    guidance can never be mixed or silently resumed into each other."""
    name = version or f"{PROMPT_VERSION}__{rule_set.rule_set_version}"
    return project.output_dir / "results" / "rule_check" / slug(model_id) / name


def unique_items(project, limit: int | None, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(all labelled readable images, representatives: one image per distinct content hash)."""
    m = load_manifest(project)
    m = m[m["label"].isin(LABELS) & (m["media_kind"] == "image") & m["readable"].astype(bool) & m["sha256"].notna()]
    reps = m.sort_values("image_id").drop_duplicates("sha256", keep="first")
    if limit:
        half = max(1, limit // 2)
        reps = pd.concat([g.sample(n=min(half, len(g)), random_state=seed) for _, g in reps.groupby("label")])
    return m, reps


def is_applicable_pass(check: dict[str, Any]) -> tuple[bool, bool]:
    """(applies, passes). A PASS starting with 'N/A' means the rule did not apply."""
    na = check["status"] == "PASS" and str(check.get("evidence", "")).strip().upper().startswith("N/A")
    return (not na), (check["status"] == "PASS" and not na)


def cmd_run(args, project, model_cfg, rule_set) -> int:
    model_id = args.model_id or model_cfg.models[args.model].id
    all_imgs, reps = unique_items(project, args.limit, args.seed)
    out = run_dir(project, model_id, rule_set, args.version) / "analyses.jsonl"
    done = {i for i, r in load_analyses(out).items() if r.get("ok")}
    todo = reps[~reps.image_id.isin(done)]
    k = len(rule_set.active())
    print(f"model={model_id} prompt={PROMPT_VERSION} rules={k} (rule set {rule_set.rule_set_version})")
    print(f"labelled images={len(all_imgs)}  unique contents={all_imgs.sha256.nunique()}  "
          f"selected this run={len(reps)}  already done={len(reps) - len(todo)}  to send now={len(todo)}")
    print(f"rough tokens for this run: ~{len(todo) * (300 + 120 * k):,} in / ~{len(todo) * (40 + 50 * k):,} out")
    if not args.execute:
        print("\nDRY RUN: nothing sent. Add --execute to analyze (photos are sent to Gemini).")
        return 0
    if todo.empty:
        print("nothing to do")
        return 0

    analyzer = RuleAnalyzer(GeminiBackend(project.cloud), rule_set, model_cfg, model_id)
    items = [(r.image_id, project.data_dir / r.rel_path, getattr(r, "program", None)) for r in todo.itertuples()]
    stats = run_batch(items, lambda i, p, g: analyzer.analyze(p, i, g), out, workers=args.workers)
    print(f"done: {stats} -> {out}")
    return 0


def per_image_table(project, model_id: str, rule_set, version: str | None = None) -> pd.DataFrame:
    path = run_dir(project, model_id, rule_set, version) / "analyses.jsonl"
    analyses = load_analyses(path)
    all_imgs, _ = unique_items(project, None, 0)
    rep_of = all_imgs.sort_values("image_id").drop_duplicates("sha256").set_index("sha256").image_id
    rule_ids = [r.rule_id for r in rule_set.active()]
    rows = []
    for r in all_imgs.itertuples():
        rec = analyses.get(rep_of[r.sha256])
        base = {"image_id": r.image_id, "label": r.label, "sha256": r.sha256, "program": r.program, "analyzed": False}
        if rec is None or not rec.get("ok"):
            rows.append(base)
            continue
        checks = {c["rule_id"]: c for c in rec["analysis"]["checks"]}
        n_pass = n_fail = n_unc = n_na = 0
        for rid in rule_ids:
            applies, passes = is_applicable_pass(checks[rid])
            st = checks[rid]["status"]
            base[f"{rid}"] = "NA" if not applies else st
            n_na += not applies
            n_pass += passes
            n_fail += applies and st == "FAIL"
            n_unc += applies and st == "UNCERTAIN"
        usage = rec.get("usage") or {}
        rows.append({**base, "analyzed": True, "n_pass": n_pass, "n_fail": n_fail, "n_uncertain": n_unc,
                     "n_not_applicable": n_na, "n_applicable": len(rule_ids) - n_na,
                     "pass_rate": n_pass / (len(rule_ids) - n_na) if len(rule_ids) > n_na else np.nan,
                     "pass_rate_decided": n_pass / (n_pass + n_fail) if (n_pass + n_fail) else np.nan,
                     "tokens": usage.get("total_tokens"), "latency_s": usage.get("latency_s")})
    return pd.DataFrame(rows)


def cmd_eval(args, project, model_cfg, rule_set) -> int:
    model_id = args.model_id or model_cfg.models[args.model].id
    t = per_image_table(project, model_id, rule_set, args.version)
    rule_ids = [r.rule_id for r in rule_set.active()]
    names = {r.rule_id: r.name for r in rule_set.active()}
    ok = t[t.analyzed]
    if ok.empty:
        print("no analyses found: run `rule_check.py run --execute` first", file=sys.stderr)
        return 2
    app, dis = ok[ok.label == "approved"], ok[ok.label == "not_approved"]

    def share(df: pd.DataFrame, cond) -> float:
        return float(cond(df).mean()) if len(df) else float("nan")

    res: dict[str, Any] = {
        "model": model_id, "prompt_version": PROMPT_VERSION, "rule_set_version": rule_set.rule_set_version,
        "images": {"labelled": int(len(t)), "analyzed": int(len(ok)), "not_analyzed": int(len(t) - len(ok)),
                   "approved": int(len(app)), "not_approved": int(len(dis)),
                   "unique_contents_analyzed": int(ok.sha256.nunique())},
        "your_criteria": {
            "approved_follow_at_least_one_rule": share(app, lambda d: d.n_pass >= 1),
            "not_approved_follow_no_rule": share(dis, lambda d: d.n_pass == 0),
        },
        "stricter_views": {},
        "mean": {"n_pass": {"approved": float(app.n_pass.mean()), "not_approved": float(dis.n_pass.mean())},
                 "pass_rate": {"approved": float(app.pass_rate.mean()), "not_approved": float(dis.pass_rate.mean())},
                 "n_fail": {"approved": float(app.n_fail.mean()), "not_approved": float(dis.n_fail.mean())},
                 "n_not_applicable": {"approved": float(app.n_not_applicable.mean()),
                                      "not_approved": float(dis.n_not_applicable.mean())}},
    }
    for label, cond in (("pass_rate_ge_50pct", lambda d: d.pass_rate >= 0.5), ("pass_rate_ge_75pct", lambda d: d.pass_rate >= 0.75),
                        ("n_pass_ge_5", lambda d: d.n_pass >= 5), ("no_failed_rule", lambda d: d.n_fail == 0)):
        res["stricter_views"][label] = {"approved": share(app, cond), "not_approved": share(dis, cond)}

    y = (ok.label == "approved").astype(int)
    auc = {}
    for col in ("n_pass", "pass_rate", "pass_rate_decided"):
        v = ok[col].fillna(0)
        auc[col] = float(roc_auc_score(y, v)) if y.nunique() == 2 else None
    res["separation_roc_auc_approved_vs_not"] = auc

    # Counting every COPY of a photo over-weights heavily duplicated items (e.g. one bill repeated 30+ times), so also
    # report one row per unique photo, restricted to train+validation images (the test split is not used here).
    sp = pd.concat([pd.read_csv(project.output_dir / "splits" / f"{k}.csv") for k in ("train", "validation")])
    u = ok.merge(sp[["image_id"]], on="image_id").sort_values("image_id").drop_duplicates("sha256")
    ua, ud = u[u.label == "approved"], u[u.label == "not_approved"]
    res["deduplicated_train_val"] = {
        "unique_photos": int(len(u)), "approved": int(len(ua)), "not_approved": int(len(ud)),
        "approved_follow_at_least_one_rule": share(ua, lambda d: d.n_pass >= 1),
        "not_approved_follow_no_rule": share(ud, lambda d: d.n_pass == 0),
        "mean_rules_followed": {"approved": float(ua.n_pass.mean()), "not_approved": float(ud.n_pass.mean())},
        "mean_rules_failed": {"approved": float(ua.n_fail.mean()), "not_approved": float(ud.n_fail.mean())},
        "roc_auc_n_pass": float(roc_auc_score((u.label == "approved").astype(int), u.n_pass))
        if u.label.nunique() == 2 else None,
    }

    per_rule = []
    for rid in rule_ids:
        row: dict[str, Any] = {"rule_id": rid, "name": names[rid]}
        for lab, d in (("approved", app), ("not_approved", dis)):
            ap = d[d[rid] != "NA"]
            row[f"applicable_{lab}"] = float(len(ap) / len(d))
            row[f"pass_{lab}"] = float((ap[rid] == "PASS").mean()) if len(ap) else None
            row[f"fail_{lab}"] = float((ap[rid] == "FAIL").mean()) if len(ap) else None
            row[f"uncertain_{lab}"] = float((ap[rid] == "UNCERTAIN").mean()) if len(ap) else None
        row["pass_gap"] = (None if row["pass_approved"] is None or row["pass_not_approved"] is None
                           else row["pass_approved"] - row["pass_not_approved"])
        per_rule.append(row)
    res["per_rule"] = per_rule
    res["usage"] = {"total_tokens": int(ok.tokens.fillna(0).sum()), "mean_latency_s": float(ok.latency_s.mean()),
                    "p95_latency_s": float(ok.latency_s.quantile(0.95))}

    if args.version and "__" in args.version:
        res["prompt_version"], res["rule_set_version"] = args.version.split("__", 1)
    outdir = run_dir(project, model_id, rule_set, args.version)
    t.to_csv(outdir / "per_image.csv", index=False)
    (outdir / "eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    report = render_report(res)
    (project.resolve(project.paths.docs_dir) / "RULE_CHECK_REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


def pct(v: float | None) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1%}"


def _dedup_lines(d: dict[str, Any] | None) -> list[str]:
    if not d:
        return []
    m, f = d["mean_rules_followed"], d["mean_rules_failed"]
    return [f"Unique photos in train + validation: {d['unique_photos']} ({d['approved']} approved, {d['not_approved']} not approved).\n",
            "| Measure | Approved | Not approved |\n|---|---:|---:|",
            f"| rules followed (average) | {m['approved']:.1f} | {m['not_approved']:.1f} |",
            f"| rules failed (average) | {f['approved']:.1f} | {f['not_approved']:.1f} |",
            f"| follow at least one rule | {pct(d['approved_follow_at_least_one_rule'])} | - |",
            f"| follow no rule | - | {pct(d['not_approved_follow_no_rule'])} |\n",
            f"ROC-AUC of the number of rules followed (0.5 = no separation): **{d['roc_auc_n_pass']:.3f}**.\n"]


def render_report(r: dict[str, Any]) -> str:
    im, yc, sv = r["images"], r["your_criteria"], r["stricter_views"]
    L = [f"# Rule check: all labelled images vs the {len(r['per_rule'])} rules\n",
         f"_Model `{r['model']}`, prompt `{r['prompt_version']}`, rule set `{r['rule_set_version']}`. Auto-generated by `scripts/rule_check.py eval`._\n",
         "A rule is **followed** only if it applied to the photo and the model said PASS (`N/A:` passes do not count). "
         "Exploratory check, not model selection.\n",
         f"Images: {im['analyzed']} analyzed of {im['labelled']} labelled ({im['approved']} approved, {im['not_approved']} not approved; "
         f"{im['unique_contents_analyzed']} unique contents; exact copies reuse one analysis; {im['not_analyzed']} not analyzed).\n",
         "## Read this first: each unique photo counted once\n",
         *_dedup_lines(r.get("deduplicated_train_val")),
         "## Your two criteria (every image counted, including copies)\n",
         "_Heavily duplicated items (the same bill repeated dozens of times) are counted once per copy below, which "
         "inflates the apparent separation between the groups. Prefer the table above._\n",
         "| Criterion | Result |\n|---|---:|",
         f"| Approved images that follow **at least one** rule | **{pct(yc['approved_follow_at_least_one_rule'])}** |",
         f"| Not-approved images that follow **no** rule | **{pct(yc['not_approved_follow_no_rule'])}** |\n",
         "## Stricter views (same data)\n",
         "| Measure | Approved | Not approved |\n|---|---:|---:|"]
    for k, v in sv.items():
        L.append(f"| {k} | {pct(v['approved'])} | {pct(v['not_approved'])} |")
    m = r["mean"]
    L += ["", "| Average per image | Approved | Not approved |\n|---|---:|---:|",
          f"| rules followed | {m['n_pass']['approved']:.2f} | {m['n_pass']['not_approved']:.2f} |",
          f"| pass rate over applicable rules | {pct(m['pass_rate']['approved'])} | {pct(m['pass_rate']['not_approved'])} |",
          f"| rules failed | {m['n_fail']['approved']:.2f} | {m['n_fail']['not_approved']:.2f} |",
          f"| rules not applicable | {m['n_not_applicable']['approved']:.2f} | {m['n_not_applicable']['not_approved']:.2f} |", "",
          "ROC-AUC for separating approved from not approved: " + ", ".join(
              f"`{k}` = {v:.3f}" for k, v in r["separation_roc_auc_approved_vs_not"].items()) + " (0.5 = no separation).\n",
          "## Per rule (pass rate among images where the rule applied)\n",
          "| Rule | Applies (appr / not) | Pass appr | Pass not | Gap | Fail appr | Fail not | Uncertain appr / not |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in sorted(r["per_rule"], key=lambda x: -(x["pass_gap"] if x["pass_gap"] is not None else -9)):
        gap = "n/a" if p["pass_gap"] is None else format(p["pass_gap"], "+.1%")
        L.append(f"| {p['rule_id']} {p['name']} | {pct(p['applicable_approved'])} / {pct(p['applicable_not_approved'])} | "
                 f"{pct(p['pass_approved'])} | {pct(p['pass_not_approved'])} | "
                 f"{gap} | {pct(p['fail_approved'])} | {pct(p['fail_not_approved'])} | "
                 f"{pct(p['uncertain_approved'])} / {pct(p['uncertain_not_approved'])} |")
    u = r["usage"]
    L += ["", f"Usage: {u['total_tokens']:,} tokens total; latency mean {u['mean_latency_s']:.1f}s, p95 {u['p95_latency_s']:.1f}s. Cost: unknown (no prices configured)."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "eval"])
    ap.add_argument("--limit", type=int, default=None, help="pilot: number of unique images (half per class)")
    ap.add_argument("--execute", action="store_true", help="actually call the API (default: dry run)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", choices=["primary", "comparison"], default="primary")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--version", default=None, help="results folder to use, e.g. rules-v3__1.1-sop (default: current rules)")
    args = ap.parse_args()

    project = load_project_config()
    setup_logging(project.logging["level"])
    try:
        fn = cmd_run if args.command == "run" else cmd_eval
        return fn(args, project, load_model_config(), load_rule_set())
    except NgoPhotoError as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
