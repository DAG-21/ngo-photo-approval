"""A BLIND tagging sheet: approved and disapproved photos mixed, labels hidden, same reason list as the first round.

Why: the first tagging round showed taggers only disapproved photos, and every one of them got a rule reason. If taggers
find the same 'problems' on approved photos, those reasons say little about why the NGO disapproved anything.

Nothing in the workbook may reveal a photo's label: no file names, no programme, no counts of copies. The key (row ->
image id, label, programme) is written to a separate file that is NOT shared with taggers. The main project is only read.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image as XlImage
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

FONT = "Arial"
TAG, CODES, INFO = "Tagging", "Reason codes", "Instructions"
H = ["#", "Photo", "Would you approve this photo? (required)", "Main problem, if any (required)", "Second problem (optional)",
     "Third problem (optional)", "Comment", "Confidence", "Tagger (your name)", "ID (do not edit)"]
APPROVE = ("Yes", "No", "Not sure")
CONFIDENCE = ("Sure", "Unsure")
LOOKS_FINE = "X4"
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
YELLOW = PatternFill("solid", start_color="FFF2CC")
HEAD = PatternFill("solid", start_color="1F3864")
THUMB_LONG_SIDE = 420
BOX_W, BOX_H = 285, 255


def reason_list(options: list[dict[str, str]], exclude_codes: set[str]) -> list[dict[str, str]]:
    """The first round's reason list, with the 'looks fine' wording made neutral (the old one said it was disapproved)."""
    out = []
    for o in options:
        if o["code"] in exclude_codes:
            continue
        o = dict(o)
        if o["code"] == LOOKS_FINE:
            o["label"] = "Looks fine, I see no problem"
            o["meaning"] = "You checked the photo against all the options and see nothing wrong with it."
            o["text"] = f"{o['code']} - {o['label']}"
        out.append(o)
    return out


def select_rows(paired_ids: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Shuffle (seeded) so that any prefix of the sheet is a fair mix of both classes."""
    order = np.random.default_rng(seed).permutation(len(paired_ids))
    out = paired_ids.iloc[order].reset_index(drop=True)
    out.insert(0, "row", np.arange(1, len(out) + 1))
    return out


def thumb_bytes(path: Path) -> bytes:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((THUMB_LONG_SIDE, THUMB_LONG_SIDE))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
    return buf.getvalue()


def build_workbook(rows: pd.DataFrame, options: list[dict[str, str]], out: Path, photo_path: Callable[[str], Path]) -> Path:
    wb = Workbook()
    info = wb.active
    info.title = INFO
    info.column_dimensions["A"].width = 26
    info.column_dimensions["B"].width = 110
    text = [
        ("Tagging sheet (blind)", None),
        ("What this is", "Photos from past submissions to the NGO. Some were approved and some were not; you are NOT told which. We want to see what problems a careful person finds, without knowing the outcome."),
        ("What to do", "Work from the top of the 'Tagging' sheet downwards. For each photo: (1) say whether you would approve it for the NGO's social media; (2) pick the main problem you see, or 'X4 - Looks fine, I see no problem' if there is none; (3) add a second or third problem only if clearly present."),
        ("Please be honest", "Many photos have no problem. Choosing X4 is a normal, expected answer. Do not look for a problem that is not there, and do not guess what the outcome was."),
        ("How to judge", "Use the reason list (sheet 'Reason codes' gives the client's wording). Judge only what you can see. If you cannot tell, choose 'S - Cannot judge'."),
        ("Cells to fill (yellow)", "Columns C to I. Everything else is information. Do not sort or edit column A, B or the hidden ID column."),
        ("Order", "The photos are in random order on purpose. Tag as many as you can from the top; any number from the top is a fair sample. Please do not skip around."),
        ("Privacy", "This file contains photos of real people, including beneficiaries. Share it only with people authorised by the NGO, do not forward it, and delete your copy when told to."),
    ]
    for r, (a, b) in enumerate(text, 1):
        info.cell(r, 1, a).font = Font(name=FONT, bold=True, size=14 if r == 1 else 10)
        if b:
            c = info.cell(r, 2, b)
            c.font = Font(name=FONT, size=10)
            c.alignment = Alignment(wrap_text=True, vertical="top")

    codes = wb.create_sheet(CODES)
    for c, (h, w) in enumerate(zip(["Reason (what you pick)", "What it means (the client's wording where it is a rule)", "Extra team guidance (not in the document)"], [58, 90, 70]), 1):
        cell = codes.cell(1, c, h)
        cell.font, cell.fill = Font(name=FONT, bold=True, color="FFFFFF"), HEAD
        codes.column_dimensions[cell.column_letter].width = w
    for r, o in enumerate(options, 2):
        for c, v in enumerate([o["text"], o["meaning"], o.get("guidance", "")], 1):
            cell = codes.cell(r, c, v)
            cell.font = Font(name=FONT, size=10)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    n_opts = len(options) + 1

    ws = wb.create_sheet(TAG, 1)
    widths = [6, 42, 20, 46, 40, 40, 36, 12, 18, 18]
    for c, (h, w) in enumerate(zip(H, widths), 1):
        cell = ws.cell(1, c, h)
        cell.font, cell.fill, cell.border = Font(name=FONT, bold=True, color="FFFFFF"), HEAD, BOX
        cell.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
        ws.column_dimensions[cell.column_letter].width = w
    ws.row_dimensions[1].height = 42
    ws.freeze_panes = "C2"
    last = len(rows) + 1
    dv_reason = DataValidation(type="list", formula1=f"='{CODES}'!$A$2:$A${n_opts}", allow_blank=True)
    dv_app = DataValidation(type="list", formula1='"' + ",".join(APPROVE) + '"', allow_blank=True)
    dv_conf = DataValidation(type="list", formula1='"' + ",".join(CONFIDENCE) + '"', allow_blank=True)
    for dv in (dv_reason, dv_app, dv_conf):
        ws.add_data_validation(dv)
    for r in rows.itertuples():
        row = r.row + 1
        vals = [r.row, None, None, None, None, None, None, None, None, r.image_id]
        for c, v in enumerate(vals, 1):
            cell = ws.cell(row, c, v)
            cell.font, cell.border = Font(name=FONT, size=10), BOX
            cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center" if c in (1, 3, 8) else None)
            if 3 <= c <= 9:
                cell.fill = YELLOW
        ws.row_dimensions[row].height = 195
        img = XlImage(io.BytesIO(thumb_bytes(photo_path(r.image_id))))
        scale = min(BOX_W / img.width, BOX_H / img.height, 1.0)
        img.width, img.height = int(img.width * scale), int(img.height * scale)
        img.anchor = f"B{row}"
        ws.add_image(img)
    dv_app.add(f"C2:C{last}")
    for col in "DEF":
        dv_reason.add(f"{col}2:{col}{last}")
    dv_conf.add(f"H2:H{last}")
    ws.conditional_formatting.add(f"A2:J{last}", FormulaRule(formula=['AND($C2<>"",$D2<>"")'], fill=PatternFill("solid", bgColor="E2EFDA")))
    ws.column_dimensions["J"].hidden = True
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out


def workbook_text(path: Path) -> list[str]:
    """Every text value in the workbook (used by tests to prove nothing reveals a label)."""
    wb = load_workbook(path)
    return [str(c.value) for ws in wb for row in ws.iter_rows() for c in row if c.value is not None]


def read_returned(path: Path) -> pd.DataFrame:
    wb = load_workbook(path, data_only=True)
    ws = wb[TAG]
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] is None:
            continue
        n, _photo, approve, p1, p2, p3, comment, conf, tagger, image_id = r[:10]
        code = lambda v: (str(v).split(" - ", 1)[0].strip() if v else None)  # noqa: E731
        if approve is None and p1 is None:
            continue
        rows.append({"row": int(n), "image_id": image_id, "approve": approve, "primary": code(p1), "second": code(p2),
                     "third": code(p3), "comment": comment, "confidence": conf, "tagger": tagger})
    return pd.DataFrame(rows, columns=["row", "image_id", "approve", "primary", "second", "third", "comment", "confidence", "tagger"])


def analyse(blind: pd.DataFrame, key: pd.DataFrame, earlier: pd.DataFrame | None = None) -> dict[str, Any]:
    """Compare what taggers found, by true label. Positive = approved."""
    d = blind.merge(key[["image_id", "label", "program"]], on="image_id", how="left")
    d["is_app"] = d["label"] == "approved"
    out: dict[str, Any] = {"n": len(d), "n_approved": int(d.is_app.sum()), "n_disapproved": int((~d.is_app).sum())}
    # 1. would a careful person reproduce the NGO's decision?
    dd = d[d["approve"].isin(["Yes", "No"])]
    pred = dd["approve"] == "Yes"
    out["human_vs_label"] = {
        "answered": len(dd), "not_sure": int((d["approve"] == "Not sure").sum()),
        "agree": float((pred == dd.is_app).mean()) if len(dd) else np.nan,
        "false_approvals": int((pred & ~dd.is_app).sum()), "false_disapprovals": int((~pred & dd.is_app).sum()),
        "disapproved_that_taggers_would_approve": float((pred & ~dd.is_app).sum() / max((~dd.is_app).sum(), 1)),
        "approved_that_taggers_would_not": float((~pred & dd.is_app).sum() / max(dd.is_app.sum(), 1)),
    }
    # 2. how often is a problem found at all, by class
    d["has_problem"] = ~d["primary"].isin([LOOKS_FINE, "S", None])
    out["problem_found"] = {"approved": float(d[d.is_app].has_problem.mean()) if d.is_app.any() else np.nan,
                            "disapproved": float(d[~d.is_app].has_problem.mean()) if (~d.is_app).any() else np.nan}
    # 3. each reason, by class (any of the three slots)
    codes = sorted({c for col in ("primary", "second", "third") for c in d[col].dropna()} - {LOOKS_FINE, "S"})
    rows = []
    for c in codes:
        has = d[["primary", "second", "third"]].eq(c).any(axis=1)
        rows.append({"reason": c, "approved": float(has[d.is_app].mean()) if d.is_app.any() else np.nan,
                     "disapproved": float(has[~d.is_app].mean()) if (~d.is_app).any() else np.nan,
                     "n_approved": int(has[d.is_app].sum()), "n_disapproved": int(has[~d.is_app].sum())})
    out["by_reason"] = pd.DataFrame(rows)
    # 4. did knowing the label change what was found? (same disapproved photos, first round vs blind)
    if earlier is not None and len(earlier):
        e = d[~d.is_app].merge(earlier[["primary"]].rename(columns={"primary": "earlier_primary"}), left_on="image_id", right_index=True)
        out["blind_vs_earlier"] = {
            "n": len(e), "same_primary": float((e["primary"] == e["earlier_primary"]).mean()) if len(e) else np.nan,
            "now_looks_fine": float((e["primary"] == LOOKS_FINE).mean()) if len(e) else np.nan,
            "earlier_looks_fine": 0.0}
    return out


def render(res: dict[str, Any]) -> str:
    f = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.0%}"  # noqa: E731
    h = res["human_vs_label"]
    lines = [f"# Blind tagging: {res['n']} photos ({res['n_approved']} approved, {res['n_disapproved']} disapproved)", "",
             "## Could a careful person reproduce the NGO's decision?", "",
             f"Answered Yes/No on {h['answered']} photos ({h['not_sure']} 'Not sure'). Agreement with the NGO: {f(h['agree'])}.",
             f"Disapproved photos the taggers would approve: {f(h['disapproved_that_taggers_would_approve'])}. "
             f"Approved photos the taggers would not approve: {f(h['approved_that_taggers_would_not'])}.", "",
             "## Is a problem found more often on disapproved photos?", "",
             f"Photos where a problem was named: approved {f(res['problem_found']['approved'])}, disapproved {f(res['problem_found']['disapproved'])}.", "",
             "| reason | on approved | on disapproved | n appr | n disap |", "|---|---:|---:|---:|---:|"]
    for r in res["by_reason"].sort_values("disapproved", ascending=False).itertuples():
        lines.append(f"| {r.reason} | {f(r.approved)} | {f(r.disapproved)} | {r.n_approved} | {r.n_disapproved} |")
    if "blind_vs_earlier" in res:
        b = res["blind_vs_earlier"]
        lines += ["", "## Did knowing the photo was disapproved change what was found?", "",
                  f"On the {b['n']} disapproved photos tagged in both rounds: same main reason {f(b['same_primary'])}; "
                  f"'looks fine' now {f(b['now_looks_fine'])} (0% when taggers knew they were disapproved)."]
    return "\n".join(lines)
