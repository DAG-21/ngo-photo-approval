"""Excel review sheet for a checklist run: photo, human label and tags, Gemini's answers and whether it caught the tagged reason.

    python scripts/export_checklist_excel.py --pool dev --tag paired-tagged --out reports/paired_tagged_checklist_review.xlsx

Offline: reads results.jsonl, the pool CSV and the imported human tags; calls nothing. Tagged disapproved photos come first.
"""

import argparse
import io
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from PIL import Image

from judge_trial.checklist import load_checks, rule_summary
from judge_trial.config import ensure_main_importable, load_trial_config
from judge_trial.data import image_path, load_pool
from judge_trial.examples import load_examples
from judge_trial.runner import read_results, run_dir_for

FONT, THIN = "Arial", Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
HEAD = PatternFill("solid", start_color="1F3864")
TAG_TO_CHECKS = {"X1": ["F_PHOTO"], "X2": ["F_SINGLE"]}


def thumb(path: Path, size=(250, 190)) -> io.BytesIO:
    im = Image.open(path).convert("RGB")
    im.thumbnail(size)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    buf.seek(0)
    return buf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, choices=["dev", "confirm", "validation"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--version", default="checklist1")
    ap.add_argument("--variant", default="K")
    ap.add_argument("--judge-tag", default=None, help="tag of a holistic-judge run (variant D) on the same photos: adds its decision, score and reason")
    ap.add_argument("--judge-version", default="v3")
    ap.add_argument("--policy", default=None, help="checklist version whose policy.yaml decides approve/disapprove from the checklist answers (replaces --judge-tag)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_trial_config()
    ensure_main_importable(cfg)
    from ngo_photo_approval.data.tagging import reason_options
    from ngo_photo_approval.rules.loader import load_rule_set

    wording = {o["code"]: o["text"] for o in reason_options(load_rule_set())}
    spec = load_checks(args.version)
    by_rule: dict[str, list[str]] = {}
    for c in spec["checks"]:
        by_rule.setdefault(c["rule"], []).append(c["id"])
    by_rule.update(TAG_TO_CHECKS)
    latest = {r["image_id"]: r for r in read_results(run_dir_for(cfg, args.pool, args.variant, args.tag)) if r.get("ok")}
    pool = load_pool(cfg, args.pool, {e["image_id"] for e in load_examples(cfg)}).set_index("image_id").loc[list(latest)]
    tags = pd.read_csv(cfg.main_root / "data" / "annotations" / "disapproval_reasons.csv").set_index("image_id")

    judge = {}
    if args.policy:
        from judge_trial.policy import decide, load_policy, render as render_gates

        policy = load_policy(args.policy, spec)
        for i, r in latest.items():
            res = decide(r["checks"], policy, r["cannot_judge"])
            judge[i] = {"decision": res["decision"].replace("NEEDS_REVIEW", "DISAPPROVED"), "score": len(res["triggered"]),
                        "reason": render_gates(res) or "(no gate fired)"}
    elif args.judge_tag:
        judge = {r["image_id"]: r for r in read_results(run_dir_for(cfg, args.pool, "D", args.judge_tag)) if r.get("ok")}
    recs = []
    for i, r in latest.items():
        codes = []
        if i in tags.index:
            t = tags.loc[i]
            codes = [c for c in (t["primary"], t["second"], t["third"]) if isinstance(c, str) and c]
        answers = r["checks"]
        no = [f"{c}: {a['evidence'] or '(no evidence given)'}" for c, a in answers.items() if a["answer"] == "NO" and not c.startswith("F_")]
        facts_no = [f"{c}" for c, a in answers.items() if a["answer"] == "NO" and c.startswith("F_")]
        unsure = [f"{c}: {a['evidence']}" for c, a in answers.items() if a["answer"] == "CANNOT_TELL"]
        rs = rule_summary(answers, spec)
        failed = ", ".join(f"{k} ({s['fraction_met']:.0%} met)" for k, s in rs.items() if s["status"] == "FAIL")
        caught = None
        if codes:
            hits = [c for c in codes if any(answers[x]["answer"] == "NO" for x in by_rule.get(c, []))]
            caught = ("Yes: " + ", ".join(hits)) if hits else "No"
        recs.append({"id": i, "label": pool.loc[i, "label"], "program": pool.loc[i, "program"],
                     "human": "\n".join(wording.get(c, c) for c in codes) if codes else ("(approved: no tags)" if pool.loc[i, "label"] == "approved" else "(not tagged)"),
                     "scene": r["scene"], "no": "\n".join(no), "unsure": "\n".join(unsure), "failed": failed,
                     "facts": ", ".join(facts_no), "caught": caught, "tagged": bool(codes), "judge": judge.get(i)})
    recs.sort(key=lambda d: (d["label"] == "approved", not d["tagged"], d["program"], d["id"]))

    wb = Workbook()
    ws = wb.active
    ws.title = "Photos"
    j = bool(judge)
    heads = ["#", "Image", "Human decision", "Human tags (why disapproved)"]
    widths = [5, 36, 15, 38]
    if j:
        heads += (["Policy decision", "Gates fired (count)", "Why: gates that fired [instruction] check (answer): reason - Gemini's evidence", "Match", "Outcome"]
                  if args.policy else ["Gemini decision", "Gemini score (0-100)", "Gemini reasoning", "Match", "Outcome"])
        widths += [15, 11, 55, 11, 16]
    heads += ["Gemini checklist: what it sees", "Gemini checklist: said NO to (with its evidence)", "Gemini checklist: could not tell",
              "Rules the checklist fails (share of checks met)", "Checklist caught the human's reason?", "Program", "Image ID"]
    widths += [40, 70, 34, 34, 18, 18, 18]
    off = 5 if j else 0  # extra columns before the checklist block
    for c, (h, w) in enumerate(zip(heads, widths), 1):
        cell = ws.cell(1, c, h)
        cell.font, cell.fill, cell.border = Font(name=FONT, bold=True, color="FFFFFF"), HEAD, BOX
        cell.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
        ws.column_dimensions[cell.column_letter].width = w
    ws.row_dimensions[1].height = 34
    ws.freeze_panes = "C2"
    for n, d in enumerate(recs, 1):
        row = n + 1
        vals = [n, None, "DISAPPROVED" if d["label"] == "not_approved" else "APPROVED", d["human"]]
        if j:
            jr = d["judge"]
            vals += [jr["decision"], jr["score"], jr.get("reason") or "",
                     f'=IF(C{row}=E{row},"Match","No match")',
                     f'=IF(AND(C{row}="APPROVED",E{row}="APPROVED"),"True positive",IF(AND(C{row}="DISAPPROVED",E{row}="DISAPPROVED"),'
                     f'"True negative",IF(AND(C{row}="DISAPPROVED",E{row}="APPROVED"),"False positive","False negative")))']
        vals += [d["scene"], d["no"] or "(nothing)", d["unsure"], d["failed"] or "(none)", d["caught"] or "n/a", d["program"], d["id"]]
        for c, v in enumerate(vals, 1):
            cell = ws.cell(row, c, v)
            cell.font, cell.border = Font(name=FONT, size=10), BOX
            wrap_cols = {4, 5 + off, 6 + off, 7 + off, 8 + off} | ({7} if j else set())
            cell.alignment = Alignment(wrap_text=True, vertical="top" if c in wrap_cols else "center",
                                       horizontal="center" if c in (1, 3, 9 + off) or (j and c in (5, 6, 8, 9)) else None)
        ws.row_dimensions[row].height = max(145, 15 * max(len(d["no"].splitlines()), 1) + 20)
        if d["caught"] and d["caught"] != "No":
            ws.cell(row, 9 + off).fill = PatternFill("solid", start_color="C6E0B4")
        elif d["caught"] == "No":
            ws.cell(row, 9 + off).fill = PatternFill("solid", start_color="F8CBAD")
        img = XLImage(thumb(image_path(cfg, d["id"])))
        img.anchor = f"B{row}"
        ws.add_image(img)

    if j:
        from openpyxl.formatting.rule import CellIsRule

        last = len(recs) + 1
        for text, color in (("No match", "F8CBAD"), ("Match", "C6E0B4")):
            ws.conditional_formatting.add(f"H2:H{last}", CellIsRule(operator="equal", formula=[f'"{text}"'], fill=PatternFill("solid", bgColor=color)))
        for text, color in (("False positive", "F4B183"), ("False negative", "FFE699"), ("True positive", "C6E0B4"), ("True negative", "C6E0B4")):
            ws.conditional_formatting.add(f"I2:I{last}", CellIsRule(operator="equal", formula=[f'"{text}"'], fill=PatternFill("solid", bgColor=color)))
        sm = wb.create_sheet("Summary", 1)
        sm.column_dimensions["A"].width = 60
        sm.column_dimensions["B"].width = 12
        rng = f"Photos!I2:I{last}"
        rows_ = [("Measure", "Value"), ("Photos", f"=COUNTA(Photos!A2:A{last})"),
                 ("True positive (humans approved, Gemini approved)", f'=COUNTIF({rng},"True positive")'),
                 ("True negative (humans disapproved, Gemini disapproved)", f'=COUNTIF({rng},"True negative")'),
                 ("False positive (humans disapproved, Gemini APPROVED)", f'=COUNTIF({rng},"False positive")'),
                 ("False negative (humans approved, Gemini disapproved)", f'=COUNTIF({rng},"False negative")'),
                 ("Share matching the humans", f'=COUNTIF(Photos!H2:H{last},"Match")/B2')]
        for r_, (a, b) in enumerate(rows_, 1):
            sm.cell(r_, 1, a).font = Font(name=FONT, size=10, bold=r_ == 1)
            sm.cell(r_, 2, b).font = Font(name=FONT, size=10, bold=r_ == 1)
        sm.cell(7, 2).number_format = "0%"
    n_t = sum(d["tagged"] for d in recs)
    n_c = sum(1 for d in recs if d["caught"] and d["caught"] != "No")
    nt = wb.create_sheet("Notes")
    nt.column_dimensions["A"].width = 120
    for i, t in enumerate([
        f"Checklist run '{args.tag}': {len(recs)} photos ({sum(d['label'] == 'approved' for d in recs)} approved, {sum(d['label'] != 'approved' for d in recs)} disapproved).",
        f"{n_t} disapproved photos have human tags; Gemini said NO to at least one check of a tagged rule on {n_c} of them.",
        "Human tags come from the first tagging round, when taggers only saw disapproved photos; they are a guide, not proof of why the NGO decided.",
        "'Gemini said NO to' lists every question it answered NO, with the evidence it gave. Questions about the scene (is a banner present, is a volunteer visible) are not listed.",
        "'Caught the human's reason?' is Yes when any check belonging to a tagged rule was answered NO. It does not say Gemini is right or wrong.",
        "Photos are the resolution-normalized copies Gemini saw, shown as thumbnails. This file contains photos of real people: share only with authorised people.",
    ], 1):
        c = nt.cell(i, 1, t)
        c.font = Font(name=FONT, size=10)
        c.alignment = Alignment(wrap_text=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"wrote {out.resolve()}  ({len(recs)} photos; tagged {n_t}; reason caught on {n_c})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
