"""Export one finished run to an Excel review sheet (photo, human decision, Gemini decision/score/reasoning, match, outcome).

    python scripts/export_excel.py --variant D --pool dev --tag random50-v3 --out reports/random50_v3.xlsx

Reads only that run's results.jsonl and the pool CSV; calls nothing. Positive = approved, so a *false positive* is a
photo the humans disapproved that Gemini approved (the costly error) and a *false negative* is the reverse.
The match and outcome columns are Excel formulas over the two decision columns, so they stay correct if a cell is edited.
"""

import argparse
import io
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from PIL import Image

from judge_trial.config import load_trial_config
from judge_trial.data import image_path, load_pool
from judge_trial.examples import load_examples
from judge_trial.runner import read_results, run_dir_for

THUMB = (230, 172)  # px; row height below is chosen to fit it
FONT = "Arial"
HEAD_FILL = PatternFill("solid", start_color="1F3864")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def thumbnail_png(path: Path) -> io.BytesIO:
    im = Image.open(path).convert("RGB")
    im.thumbnail(THUMB)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    buf.seek(0)
    return buf


def rule_notes(reasons: list[dict]) -> str:
    return "\n".join(f"{r['rule_id']} ({'for' if r['effect'] == 'supports' else 'against'}): {r['note']}" for r in reasons or [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=list("ABCD"))
    ap.add_argument("--pool", required=True, choices=["dev", "confirm", "validation"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None, help="text for the top of the Notes sheet")
    args = ap.parse_args()

    cfg = load_trial_config()
    run_dir = run_dir_for(cfg, args.pool, args.variant, args.tag)
    rows = [r for r in read_results(run_dir) if r.get("ok")]
    latest = {r["image_id"]: r for r in rows}  # later rows supersede earlier ones
    if not latest:
        print(f"no successful results in {run_dir}")
        return 2
    hashes = {r["prompt_hash"] for r in latest.values()}
    pool = load_pool(cfg, args.pool, {e["image_id"] for e in load_examples(cfg)}).set_index("image_id")

    wb = Workbook()
    ws = wb.active
    ws.title = "Results"
    heads = ["Image number", "Image", "Human response", "Gemini response", "Gemini score (0-100)", "Gemini reasoning",
             "Match", "Outcome", "Image ID", "Gemini rule-by-rule detail"]
    widths = [9, 34, 16, 16, 12, 60, 11, 16, 20, 70]
    for c, (h, w) in enumerate(zip(heads, widths), 1):
        cell = ws.cell(1, c, h)
        cell.font, cell.fill, cell.border = Font(name=FONT, bold=True, color="FFFFFF"), HEAD_FILL, BOX
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[cell.column_letter].width = w
    ws.row_dimensions[1].height = 34
    ws.freeze_panes = "A2"

    ids = sorted(latest)
    for n, image_id in enumerate(ids, 1):
        r, row = latest[image_id], n + 1
        human = "APPROVED" if pool.loc[image_id, "label"] == "approved" else "DISAPPROVED"
        values = [n, None, human, r["decision"], r["score"], r.get("reason") or "",
                  f'=IF(C{row}=D{row},"Match","No match")',
                  f'=IF(AND(C{row}="APPROVED",D{row}="APPROVED"),"True positive",'
                  f'IF(AND(C{row}="DISAPPROVED",D{row}="DISAPPROVED"),"True negative",'
                  f'IF(AND(C{row}="DISAPPROVED",D{row}="APPROVED"),"False positive","False negative")))',
                  image_id, rule_notes(r.get("reasons"))]
        for c, v in enumerate(values, 1):
            cell = ws.cell(row, c, v)
            cell.font, cell.border = Font(name=FONT, size=10), BOX
            cell.alignment = Alignment(vertical="center", wrap_text=c in (6, 10), horizontal=None if c in (6, 10) else "center")
        ws.row_dimensions[row].height = 135
        img = XLImage(thumbnail_png(image_path(cfg, image_id)))
        img.anchor = f"B{row}"
        ws.add_image(img)

    last = len(ids) + 1
    for text, color in (("No match", "F8CBAD"), ("Match", "C6E0B4")):
        ws.conditional_formatting.add(f"G2:G{last}", CellIsRule(operator="equal", formula=[f'"{text}"'], fill=PatternFill("solid", bgColor=color)))
    for text, color in (("False positive", "F4B183"), ("False negative", "FFE699"), ("True positive", "C6E0B4"), ("True negative", "C6E0B4")):
        ws.conditional_formatting.add(f"H2:H{last}", CellIsRule(operator="equal", formula=[f'"{text}"'], fill=PatternFill("solid", bgColor=color)))

    sm = wb.create_sheet("Summary")
    sm.column_dimensions["A"].width = 44
    sm.column_dimensions["B"].width = 14
    sm.column_dimensions["C"].width = 70
    rng = f"Results!H2:H{last}"
    lines = [
        ("Photos in this sheet", f"=COUNTA(Results!A2:A{last})", ""),
        ("True positive (approved by humans, approved by Gemini)", f'=COUNTIF({rng},"True positive")', ""),
        ("True negative (disapproved by humans, disapproved by Gemini)", f'=COUNTIF({rng},"True negative")', ""),
        ("False positive (disapproved by humans, APPROVED by Gemini)", f'=COUNTIF({rng},"False positive")', "The costly error: a bad photo would get through."),
        ("False negative (approved by humans, DISAPPROVED by Gemini)", f'=COUNTIF({rng},"False negative")', "A good photo would be held back."),
        ("Matches", f'=COUNTIF(Results!G2:G{last},"Match")', ""),
        ("Share of decisions matching the humans", "=B7/B2", ""),
        ("Disapproved photos Gemini wrongly approved", "=B5/(B5+B4)", "False positives out of all human-disapproved photos."),
        ("Approved photos Gemini wrongly disapproved", "=B6/(B6+B3)", "False negatives out of all human-approved photos."),
    ]
    sm.append(["Measure", "Value", "Note"])
    for l in lines:
        sm.append(list(l))
    for r_ in sm.iter_rows(min_row=1, max_row=len(lines) + 1):
        for cell in r_:
            cell.font = Font(name=FONT, size=10, bold=cell.row == 1, color="FFFFFF" if cell.row == 1 else "000000")
            cell.border = BOX
            cell.alignment = Alignment(wrap_text=True, vertical="center")
            if cell.row == 1:
                cell.fill = HEAD_FILL
    for r_ in (8, 9, 10):
        sm.cell(r_, 2).number_format = "0.0%"

    nt = wb.create_sheet("Notes")
    nt.column_dimensions["A"].width = 120
    notes = [
        args.title or f"Gemini judge, variant {args.variant}, pool {args.pool}, run '{args.tag}'",
        f"Model: {next(iter(latest.values()))['model_id']}.  Prompt hash: {', '.join(sorted(hashes))}.  Photos: {len(ids)}.",
        "Human response = the NGO reviewers' decision recorded in the data. Gemini response = the judge's APPROVED/DISAPPROVED (APPROVED when its score is 50 or more).",
        "Positive = approved. True positive: both approved. True negative: both disapproved. False positive: humans disapproved, Gemini approved. False negative: humans approved, Gemini disapproved.",
        "Match and Outcome are Excel formulas over the two decision columns.",
        "The photos are the resolution-normalized copies Gemini saw (long side 1280 px), shown here as thumbnails.",
        "Read with care: with this few photos the percentages are very imprecise, and Gemini's scores can move between identical runs. This is not an accuracy figure for the project.",
        "The human decision is the reviewers' label only. Their reasons are not recorded yet, so a mismatch does not show who is right.",
    ]
    for i, t in enumerate(notes, 1):
        c = nt.cell(i, 1, t)
        c.font = Font(name=FONT, size=12 if i == 1 else 10, bold=i == 1)
        c.alignment = Alignment(wrap_text=True, vertical="top")

    wb.calculation.fullCalcOnLoad = True
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"wrote {out.resolve()}  ({len(ids)} photos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
