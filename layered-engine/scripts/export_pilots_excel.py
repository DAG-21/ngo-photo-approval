"""Excel analysis of the pilots run with the layered engine (offline; reads runs/, the pilot lists and the private keys; calls nothing).

    python scripts/export_pilots_excel.py --out reports/pilots_analysis.xlsx                       # every pilot
    python scripts/export_pilots_excel.py --pilots pilot3 --out reports/pilot3_50_photos.xlsx      # one pilot

Same layout as the earlier review sheets: one row per photo with the image, the human decision and tags, the engine's decision, a Match column and an
Outcome column (true/false positive/negative, positive = approved), both as Excel formulas. Extra sheets: Summary, Gate rates, Gate evidence, Notes.
The engine counts as APPROVED only when it auto-approves (APPROVE or APPROVED_PENDING_PUBLICATION_CHECKS); REJECT, HUMAN_REVIEW and RESELECT count as
DISAPPROVED (the photo is not let through automatically).
"""

import argparse
import csv
import io
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from PIL import Image

from layered_engine.annotation import read_rows
from layered_engine.rulebook import ROOT, Rulebook

FONT, THIN = "Arial", Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
HEAD = PatternFill("solid", start_color="1F3864")
AUTO = {"APPROVE", "APPROVED_PENDING_PUBLICATION_CHECKS"}

# name -> (label, key file, list file, run folder, legacy run folder or None, short description)
PILOTS = {
    "pilot1": ("Pilot 1", "key_PRIVATE.csv", "list.csv", "pilot_layered_proposed_v1", "pilot_checklist5_legacy", "10 disapproved photos with taggers' reasons (also run under the legacy profile)"),
    "pilot2": ("Pilot 2", "key2_PRIVATE.csv", "list2.csv", "pilot2_layered", None, "10 approved + 10 disapproved"),
    "pilot3": ("Pilot 3", "key3_PRIVATE.csv", "list3.csv", "pilot3_layered", None, "25 approved + 25 disapproved, random, none used before"),
}


def latest(run: str) -> dict:
    out = {}
    for r in read_rows(ROOT / "runs" / run / "results.jsonl"):
        if r.get("ok"):
            out[r["image_id"]] = r
    return out


def thumb(path: str, size=(250, 190)) -> io.BytesIO:
    im = Image.open(path).convert("RGB")
    im.thumbnail(size)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    buf.seek(0)
    return buf


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def reasoning(d: dict) -> str:
    parts = []
    for g in d["gates"]:
        if g["hard"] and g["status"] in ("NO", "CANNOT_TELL"):
            parts.append(f"{g['gate_id']} {'FAILED' if g['status'] == 'NO' else 'UNSURE'}: {g['evidence'] or '(no evidence given)'}")
    parts += [r for r in d["decision_reasons"] if r.startswith(("QUALITY", "PUBLICATION", "INPUT"))]
    if not parts:
        parts = ["no mandatory gate failed" + ("; publication checks outstanding" if d["recommendation"] == "APPROVED_PENDING_PUBLICATION_CHECKS" else "")]
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilots", nargs="+", default=list(PILOTS), choices=list(PILOTS))
    args = ap.parse_args()
    chosen = sorted(args.pilots, reverse=True)  # newest first

    rb = Rulebook()
    prof = rb.resolve("layered_proposed_v1")
    recs = []
    for name in chosen:
        label, kf, lf, run, legacy_run, _ = PILOTS[name]
        key = {r["opaque_id"]: r for r in read_csv(ROOT / "pilot" / kf)}
        paths = {r["image_id"]: r["path"] for r in read_csv(ROOT / "pilot" / lf)}
        runs, leg_runs = latest(run), latest(legacy_run) if legacy_run else {}
        for oid in sorted(runs, key=lambda i: (key[i]["label"] != "approved", i)):
            r, k = runs[oid], key[oid]
            d = r["decision"]
            leg = leg_runs.get(oid)
            recs.append({
                "pilot": label, "oid": oid, "path": paths[oid], "human": "APPROVED" if k["label"] == "approved" else "DISAPPROVED",
                "tags": k["tags"] or ("(approved: no tags)" if k["label"] == "approved" else "(not tagged)"),
                "engine": "APPROVED" if d["recommendation"] in AUTO else "DISAPPROVED", "rec": d["recommendation"], "score": d["quality"]["overall_score"], "why": reasoning(d),
                "failed": ", ".join(d["failed_gates"]), "unsure": ", ".join(d["uncertain_gates"]), "leg_rec": leg["decision"]["recommendation"] if leg else "",
                "leg_failed": ", ".join(leg["decision"]["failed_gates"]) if leg else "", "scene": r["validated"]["layer_1_scene_understanding"]["activity_description"],
                "program": k["program"], "image_id": k["image_id"], "gates": [g for g in d["gates"] if g["status"] in ("NO", "CANNOT_TELL") and g["hard"]]})

    wb = Workbook()
    ws = wb.active
    ws.title = "Photos"
    heads = ["#", "Image", "Pilot", "Human decision", "Human tags (why disapproved)", "Engine decision", "Engine recommendation (detail)", "Quality score (0-100)",
             "Engine reasoning (gates that failed, with Gemini's evidence)", "Match", "Outcome", "Gates failed", "Gates unsure", "Legacy profile decision (pilot 1 only)",
             "Legacy gates failed", "What Gemini says the photo shows", "Program", "Opaque ID sent to Gemini", "Image ID"]
    widths = [5, 36, 9, 15, 26, 15, 24, 11, 80, 11, 16, 24, 14, 20, 22, 44, 16, 18, 18]
    for c, (h, w) in enumerate(zip(heads, widths), 1):
        cell = ws.cell(1, c, h)
        cell.font, cell.fill, cell.border = Font(name=FONT, bold=True, color="FFFFFF"), HEAD, BOX
        cell.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
        ws.column_dimensions[cell.column_letter].width = w
    ws.row_dimensions[1].height = 44
    ws.freeze_panes = "C2"
    for n, d in enumerate(recs, 1):
        row = n + 1
        vals = [n, None, d["pilot"], d["human"], d["tags"], d["engine"], d["rec"], d["score"], d["why"], f'=IF(D{row}=F{row},"Match","No match")',
                f'=IF(AND(D{row}="APPROVED",F{row}="APPROVED"),"True positive",IF(AND(D{row}="DISAPPROVED",F{row}="DISAPPROVED"),"True negative",'
                f'IF(AND(D{row}="DISAPPROVED",F{row}="APPROVED"),"False positive","False negative")))',
                d["failed"], d["unsure"], d["leg_rec"], d["leg_failed"], d["scene"], d["program"], d["oid"], d["image_id"]]
        for c, v in enumerate(vals, 1):
            cell = ws.cell(row, c, v)
            cell.font, cell.border = Font(name=FONT, size=10), BOX
            cell.alignment = Alignment(wrap_text=True, vertical="top" if c in (5, 9, 12, 16) else "center", horizontal="center" if c in (1, 3, 4, 6, 8, 10, 11) else None)
        ws.row_dimensions[row].height = max(145, 15 * (len(d["why"].splitlines()) + 2))
        img = XLImage(thumb(d["path"]))
        img.anchor = f"B{row}"
        ws.add_image(img)
    last = len(recs) + 1
    for text, color in (("No match", "F8CBAD"), ("Match", "C6E0B4")):
        ws.conditional_formatting.add(f"J2:J{last}", CellIsRule(operator="equal", formula=[f'"{text}"'], fill=PatternFill("solid", bgColor=color)))
    for text, color in (("False positive", "F4B183"), ("False negative", "FFE699"), ("True positive", "C6E0B4"), ("True negative", "C6E0B4")):
        ws.conditional_formatting.add(f"K2:K{last}", CellIsRule(operator="equal", formula=[f'"{text}"'], fill=PatternFill("solid", bgColor=color)))
    rng = lambda col: f"Photos!${col}$2:${col}${last}"  # noqa: E731

    # ---------------------------------------------------------------- Summary: formulas over the Photos sheet
    sm = wb.create_sheet("Summary", 0)
    sm.column_dimensions["A"].width = 70
    labels = [PILOTS[p][0] for p in chosen]
    cols = ["B"] + [chr(ord("C") + i) for i in range(len(labels))]
    for col in cols:
        sm.column_dimensions[col].width = 16
    header = ["Measure", "All shown"] + labels
    spec = [("Photos", None, None), ("Human APPROVED", "D", "APPROVED"), ("Human DISAPPROVED", "D", "DISAPPROVED"),
            ("True positive (humans approved, engine approved)", "K", "True positive"),
            ("True negative (humans disapproved, engine did not auto-approve)", "K", "True negative"),
            ("False positive (humans disapproved, engine AUTO-APPROVED): the costly error", "K", "False positive"),
            ("False negative (humans approved, engine did not auto-approve)", "K", "False negative"),
            ("Match (same decision as the humans)", "J", "Match")]
    for c, h in enumerate(header, 1):
        cell = sm.cell(1, c, h)
        cell.font, cell.fill = Font(name=FONT, bold=True, color="FFFFFF"), HEAD
    for r_, (label, col, val) in enumerate(spec, 2):
        sm.cell(r_, 1, label).font = Font(name=FONT, size=10)
        for ci, who in enumerate([None] + labels):
            f = (f'=COUNTIFS({rng("C")},"{who}")' if who else f"=COUNTA({rng('A')})") if col is None else \
                f'=COUNTIFS({rng(col)},"{val}"' + (f',{rng("C")},"{who}"' if who else "") + ")"
            sm.cell(r_, 2 + ci, f).font = Font(name=FONT, size=10)
    r0 = len(spec) + 3
    rate_rows = [("Approved photos the engine did NOT auto-approve (false rejection rate)", "{c}8", "{c}3"),
                 ("Disapproved photos the engine auto-approved (false approval rate)", "{c}7", "{c}4"),
                 ("Share of decisions matching the humans", "{c}9", "{c}2")]
    for i, (label, num, den) in enumerate(rate_rows):
        sm.cell(r0 + i, 1, label).font = Font(name=FONT, size=10, bold=True)
        for ci, col in enumerate(cols):
            cell = sm.cell(r0 + i, 2 + ci, f'=IF({den.format(c=col)}=0,"n/a",{num.format(c=col)}/{den.format(c=col)})')
            cell.font, cell.number_format = Font(name=FONT, size=10), "0%"
    for i, p in enumerate(chosen):
        sm.cell(r0 + 4 + i, 1, f"{PILOTS[p][0]}: {PILOTS[p][5]}").font = Font(name=FONT, size=10, italic=True)
    sm.cell(r0 + 5 + len(chosen), 1, "Read with care: small samples, mostly one programme, one run each. Percentages are imprecise. See the Notes sheet.").font = Font(name=FONT, size=10, italic=True)

    # ---------------------------------------------------------------- Gate rates
    gr = wb.create_sheet("Gate rates", 1)
    for col, w in zip("ABCDEFG", (8, 62, 18, 18, 18, 18, 16)):
        gr.column_dimensions[col].width = w
    n_app = sum(1 for d in recs if d["human"] == "APPROVED")
    n_dis = sum(1 for d in recs if d["human"] == "DISAPPROVED")
    hdr = ["Gate", "What it checks (operational wording)", f"Failed on approved (of {n_app})", f"Failed on disapproved (of {n_dis})", "Share of approved", "Share of disapproved", "Active in"]
    for c, h in enumerate(hdr, 1):
        cell = gr.cell(1, c, h)
        cell.font, cell.fill, cell.border = Font(name=FONT, bold=True, color="FFFFFF"), HEAD, BOX
        cell.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
    gr.row_dimensions[1].height = 36
    for r_, g in enumerate(rb.gates, 2):
        c = rb.criteria[g.criterion_id]
        text = c.description_operational
        if g.components:
            text = f"Parent of: {', '.join(g.components)}. " + text
        gp = f'"*{g.gate_id}*"'
        vals = [g.gate_id, text, f'=COUNTIFS({rng("D")},"APPROVED",{rng("L")},{gp})', f'=COUNTIFS({rng("D")},"DISAPPROVED",{rng("L")},{gp})',
                f'=IF({n_app}=0,"n/a",C{r_}/{n_app})', f'=IF({n_dis}=0,"n/a",D{r_}/{n_dis})', "both profiles" if g.active_in is None else "layered only"]
        for col, v in enumerate(vals, 1):
            cell = gr.cell(r_, col, v)
            cell.font, cell.border = Font(name=FONT, size=10), BOX
            cell.alignment = Alignment(wrap_text=True, vertical="top", horizontal="center" if col != 2 else None)
            if col in (5, 6):
                cell.number_format = "0%"
        gr.row_dimensions[r_].height = 48
    end = len(rb.gates) + 3
    gr.cell(end, 2, "G010 and G011 are soft in the layered profile, so they never fail there. Gates G016-G018 exist only in the layered profile.").font = Font(name=FONT, size=10, italic=True)
    gr.cell(end + 1, 2, "A gate that fires about as often on approved as on disapproved photos does not separate them; one that fires only on disapproved photos might.").font = Font(name=FONT, size=10, italic=True)

    # ---------------------------------------------------------------- Gate evidence (long format)
    ev = wb.create_sheet("Gate evidence")
    for col, w in zip("ABCDEFG", (5, 10, 14, 8, 26, 14, 100)):
        ev.column_dimensions[col].width = w
    for c, h in enumerate(["#", "Pilot", "Human decision", "Gate", "Criterion", "Status", "Gemini's evidence"], 1):
        cell = ev.cell(1, c, h)
        cell.font, cell.fill, cell.border = Font(name=FONT, bold=True, color="FFFFFF"), HEAD, BOX
    rr = 2
    for n, d in enumerate(recs, 1):
        for g in d["gates"]:
            for c, v in enumerate([n, d["pilot"], d["human"], g["gate_id"], g["criterion_id"], g["status"], g["evidence"] or "(no evidence given)"], 1):
                cell = ev.cell(rr, c, v)
                cell.font, cell.border = Font(name=FONT, size=10), BOX
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            rr += 1
    ev.freeze_panes = "A2"

    nt = wb.create_sheet("Notes")
    nt.column_dimensions["A"].width = 130
    notes = [
        "Photos shown: " + "; ".join(f"{PILOTS[p][0]} ({PILOTS[p][5]})" for p in chosen) + ". Layered engine, Gemini 3.8 Flash, one run per photo. The # matches the Gate evidence sheet.",
        "Engine decision: APPROVED only when the engine auto-approves (APPROVE or APPROVED_PENDING_PUBLICATION_CHECKS). REJECT, HUMAN_REVIEW and RESELECT count as DISAPPROVED, because the photo is not let through automatically.",
        "Positive = approved. True positive: both approved. True negative: both not approved. False positive: humans disapproved but the engine auto-approved (the costly error). False negative: humans approved but the engine did not auto-approve. Match and Outcome are Excel formulas.",
        "Human tags are the taggers' reading of why a photo was disapproved; they are not ground truth, only some disapproved photos have tags, and several tagged reasons (posed, clutter, dignity, focus) have no gate by design.",
        "The photos are the resolution-normalised copies sent to Gemini (long side 1280 px), shown as thumbnails. Only the opaque ID, the photo and a fixed instruction were sent: no label, tag or file name.",
        "Small samples, mostly one programme (Arham Aahar), one run per photo: percentages are imprecise and run-to-run variation has not been measured. Which gates separate the classes should only be trusted if it holds on photos that were not used to find it.",
        "The quality score is uncalibrated. Costs in the reports use placeholder prices. This file contains photos of real people: share only with authorised people.",
    ]
    for i, t in enumerate(notes, 1):
        c = nt.cell(i, 1, t)
        c.font = Font(name=FONT, size=10)
        c.alignment = Alignment(wrap_text=True, vertical="top")

    wb.calculation.fullCalcOnLoad = True
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"wrote {out.resolve()} ({len(recs)} photos, {rr - 2} gate-evidence rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
