"""Excel sheet for humans to tag WHY photos were disapproved, and the reader for the returned sheet.

The reason options come from the rules (so they follow the rules document), plus a few non-rule reasons.
Photos are embedded as thumbnails so the tagger needs no access to the dataset folders. The order is a seeded random
shuffle, so any prefix of the sheet is a fair random sample (the tagger tags "as many as possible", top to bottom).
Model predictions and rule results are deliberately NOT shown, so the tagger is not anchored by them.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image as XlImage
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.datavalidation import DataValidation

from ..exceptions import DataError
from ..schemas import RuleSet

FONT = "Arial"
TAG_SHEET, CODES_SHEET, INFO_SHEET = "Tagging", "Reason codes", "Instructions"

H_ORDER, H_PHOTO, H_FILE, H_PROGRAM, H_COPIES = "#", "Photo", "File", "Program", "Identical copies"
H_PRIMARY, H_SECOND, H_THIRD = "Primary reason (required)", "Second reason (optional)", "Third reason (optional)"
H_COMMENT, H_CONF, H_TAGGER, H_ID = "Comment", "Confidence", "Tagger", "ID (do not edit)"
HEADERS = [H_ORDER, H_PHOTO, H_FILE, H_PROGRAM, H_COPIES, H_PRIMARY, H_SECOND, H_THIRD, H_COMMENT, H_CONF, H_TAGGER, H_ID]
INPUT_HEADERS = [H_PRIMARY, H_SECOND, H_THIRD, H_COMMENT, H_CONF, H_TAGGER]
CONFIDENCE = ("Sure", "Unsure")

# Short plain-language labels for the rule ids; unknown/new rules fall back to the rule's own name.
FRIENDLY = {
    "R001": "Banner issue (folded, crooked, hidden or unclear)",
    "R002": "Volunteer dress code not followed",
    "R003": "Beneficiary face hidden, cropped or back to the camera",
    "R004": "Looking into the camera / unnatural expressions",
    "R005": "Posed or staged, not candid",
    "R006": "Seva activity not clearly shown",
    "R007": "Bad camera angle (tilted, awkward, shot from behind)",
    "R008": "Volunteer's back prominently visible",
    "R009": "Poor lighting (dark, overexposed, backlit, shadows)",
    "R010": "Blurry, out of focus or grainy",
    "R011": "Cluttered or distracting background",
    "R012": "Dignity of the people not maintained",
    "R013": "Does not reflect humanity and compassion",
}
# The SOP's "Final QC Test" Q1-Q3 restate other rules; offering them separately would only confuse a tagger.
COVERED_ELSEWHERE = {"R014": "R006", "R015": "R002/R003/R008", "R016": "R004/R007/R009/R010"}

SPECIAL_FIRST = [
    ("X1", "Not an activity photo (bill, document, screenshot)",
     "The image is a bill, invoice, spreadsheet, chat/screen screenshot or similar, not a photograph of an activity."),
    ("X2", "Collage of several photos", "Several photos are combined into one image."),
]
# Plausible rejection reasons that are NOT in the client's SOP. They are offered so the tagger can record them (and the
# 'Reason codes' sheet says so); they never become rules unless the client adds them to the rules document.
EXTRA_NON_SOP = [
    ("X5", "Unrelated to the NGO's activity (selfie, scenery, personal photo)",
     "The photo shows something other than an NGO seva activity."),
    ("X6", "No volunteer or beneficiary visible (empty scene or objects only)",
     "No person involved in the activity can be seen."),
    ("X7", "Too many people or a crowd, no clear focus",
     "The frame is crowded and it is unclear who or what the photo is about."),
    ("X8", "Subject too small or too far away", "The activity or the people are tiny in the frame."),
    ("X9", "Image too small or low resolution", "The photo looks pixelated or very small."),
    ("X10", "Text, watermark, date stamp or logo laid over the image", "Something has been added on top of the photo."),
    ("X11", "Sensitive or inappropriate content (privacy, minors, safety)",
     "Content that should not be published for privacy, child-protection or safety reasons. Describe it in Comment."),
    ("X12", "Duplicate or near-duplicate of another photo (give its # in Comment)",
     "Another photo in this sheet shows practically the same moment."),
    ("X13", "Unusual shape or crop (very tall/wide panorama, sideways, awkward crop)",
     "The image shape or crop is unsuitable for posting, regardless of the content."),
]
SPECIAL_LAST = [
    ("X3", "Other reason not covered by the rules (explain in Comment)",
     "You can see a clear reason that none of the options above describe. Write it in the Comment column."),
    ("X4", "Looks fine, cannot tell why it was disapproved",
     "You checked the photo against all the options and see no problem."),
    ("S", "Cannot judge (photo unclear or unreadable)", "The photo is too small, dark or broken to judge."),
]


@dataclass(frozen=True)
class TagRow:
    order: int
    image_id: str
    file: str
    program: str
    copies: int


def reason_options(rule_set: RuleSet) -> list[dict[str, str]]:
    """Dropdown options in display order, each with a code, label, meaning and any extra team guidance."""
    opts = [{"code": c, "label": lab, "meaning": m, "guidance": ""} for c, lab, m in SPECIAL_FIRST]
    for r in rule_set.active():
        if r.rule_id in COVERED_ELSEWHERE:
            continue
        label = FRIENDLY.get(r.rule_id, r.name.replace("_", " ").capitalize())
        opts.append({"code": r.rule_id, "label": label, "meaning": r.original_text or r.description,
                     "guidance": r.additional_guidance})
    opts += [{"code": c, "label": lab, "meaning": m, "guidance": ""} for c, lab, m in EXTRA_NON_SOP]
    opts += [{"code": c, "label": lab, "meaning": m, "guidance": ""} for c, lab, m in SPECIAL_LAST]
    rule_codes = {r.rule_id for r in rule_set.active()}
    for o in opts:
        o["text"] = f"{o['code']} - {o['label']}"
        o["in_sop"] = "Yes" if o["code"] in rule_codes else "No"
    return opts


def code_of(cell_text: str | None) -> str:
    return (cell_text or "").split(" - ", 1)[0].strip()


# --------------------------------------------------------------------------- building


def _style(cell, *, bold=False, color="000000", fill=None, wrap=False, size=10, align=None, valign="center"):
    cell.font = Font(name=FONT, size=size, bold=bold, color=color)
    if fill:
        cell.fill = PatternFill("solid", start_color=fill, end_color=fill)
    cell.alignment = Alignment(wrap_text=wrap, vertical=valign, horizontal=align)


def build_workbook(
    rows: list[TagRow],
    thumb: Callable[[TagRow], tuple[bytes, int, int]],
    rule_set: RuleSet,
    out_path: Path,
) -> Path:
    """Write the workbook. ``thumb(row)`` returns (jpeg bytes, display width px, display height px)."""
    opts = reason_options(rule_set)
    n = len(rows)
    last = n + 1  # last data row on the Tagging sheet
    wb = Workbook()

    # ------------------------------------------------------------------ Tagging
    ws = wb.active
    ws.title = TAG_SHEET
    thin = Side(style="thin", color="BFBFBF")
    widths = {"A": 6, "B": 42, "C": 22, "D": 20, "E": 11, "F": 46, "G": 40, "H": 40, "I": 45, "J": 12, "K": 16, "L": 18}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for j, h in enumerate(HEADERS, start=1):
        c = ws.cell(row=1, column=j, value=h)
        is_input = h in INPUT_HEADERS
        _style(c, bold=True, color="000000" if is_input else "FFFFFF", fill="FFE699" if is_input else "1F3864",
               wrap=True, align="center")
        c.border = Border(top=thin, bottom=thin, left=thin, right=thin)
    ws.row_dimensions[1].height = 34
    for i, r in enumerate(rows, start=2):
        ws.row_dimensions[i].height = 195
        values = [r.order, None, r.file, r.program, r.copies, None, None, None, None, None, None, r.image_id]
        for j, v in enumerate(values, start=1):
            c = ws.cell(row=i, column=j, value=v)
            is_input = HEADERS[j - 1] in INPUT_HEADERS
            _style(c, fill="FFF9E5" if is_input else None, wrap=True, align="center" if j in (1, 5, 10) else None)
            c.border = Border(top=thin, bottom=thin, left=thin, right=thin)
        data, w, h = thumb(r)
        img = XlImage(io.BytesIO(data))
        img.width, img.height = w, h
        ws.add_image(img, f"B{i}")
    ws.column_dimensions["L"].hidden = True
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:K{last}"
    ws.conditional_formatting.add(
        f"A2:K{last}", FormulaRule(formula=['$F2<>""'], fill=PatternFill("solid", start_color="E2EFDA", end_color="E2EFDA")))

    # ------------------------------------------------------------------ Reason codes
    rc = wb.create_sheet(CODES_SHEET)
    for col, w in zip("ABCDE", (62, 80, 60, 10, 10), strict=True):
        rc.column_dimensions[col].width = w
    for j, h in enumerate(["Reason (what you pick)", "What it means (rule wording from the client's document)",
                           "Extra team guidance (not in the document)", "In the SOP?", "Times used"], start=1):
        _style(rc.cell(row=1, column=j, value=h), bold=True, color="FFFFFF", fill="1F3864", wrap=True, align="center")
    for i, o in enumerate(opts, start=2):
        for j, v in enumerate((o["text"], o["meaning"], o["guidance"]), start=1):
            _style(rc.cell(row=i, column=j, value=v), wrap=True, valign="top")
        _style(rc.cell(row=i, column=4, value=o["in_sop"]), align="center", valign="top")
        # count any of the three reason columns, not just the primary
        _style(rc.cell(row=i, column=5, value=f"=COUNTIF({TAG_SHEET}!$F$2:$H${last},A{i})"), align="center", valign="top")
    rc.freeze_panes = "A2"
    wb.defined_names["ReasonList"] = DefinedName("ReasonList", attr_text=f"'{CODES_SHEET}'!$A$2:$A${len(opts) + 1}")
    first_codes_row = len(opts) + 3
    _style(rc.cell(row=first_codes_row, column=1,
                   value="The SOP's 'Final QC Test' questions 1-3 are covered by: " + "; ".join(
                       f"{k} -> {v}" for k, v in COVERED_ELSEWHERE.items())), wrap=True)

    dv_reason = DataValidation(type="list", formula1="=ReasonList", allow_blank=True, showErrorMessage=True,
                               errorTitle="Pick from the list", error="Please choose one of the listed reasons.")
    dv_conf = DataValidation(type="list", formula1=f'"{",".join(CONFIDENCE)}"', allow_blank=True, showErrorMessage=True,
                             errorTitle="Pick from the list", error="Choose Sure or Unsure.")
    ws.add_data_validation(dv_reason)
    ws.add_data_validation(dv_conf)
    dv_reason.add(f"F2:H{last}")
    dv_conf.add(f"J2:J{last}")

    # ------------------------------------------------------------------ Instructions (first sheet)
    ins = wb.create_sheet(INFO_SHEET, 0)
    ins.column_dimensions["A"].width = 34
    ins.column_dimensions["B"].width = 100
    r = 1
    _style(ins.cell(row=r, column=1, value="Why were these photos disapproved? - tagging sheet"), bold=True, size=14)
    lines = [
        ("What this is", "Every photo here was marked DISAPPROVED by the NGO. We need to know the REASON, because the rules alone "
                         "do not explain it. Your tags will be used to improve the automatic checker."),
        ("What to do", "Work from the top of the 'Tagging' sheet downwards (column A '#'). For each photo pick the main reason in "
                       "'Primary reason'. Add a second/third reason only if clearly present. Use 'Comment' for anything unusual."),
        ("Why the order matters", "The photos are in random order on purpose. Tag as many as you can from the top: any number of "
                                  "photos from the top is a fair sample. Please do not skip around or only pick easy ones."),
        ("How to pick", "Use the reason list (sheet 'Reason codes' explains each one with the client's wording; reasons marked 'No' "
                        "under 'In the SOP?' are extra options that are not in the client's document). Judge only what you can see. "
                        "If none fits, use X3 and describe it in Comment. If you cannot tell, choose X4 (looks fine) or S (cannot judge) "
                        "rather than guessing."),
        ("Cells to fill (yellow)", "Columns F-K on the 'Tagging' sheet: Primary reason, Second reason, Third reason, Comment, "
                                   "Confidence, Tagger (your name). Everything else is information. Do not edit or sort the "
                                   "columns A-E or the hidden ID column (filtering is fine)."),
        ("Rows turn green", "A row turns green when its Primary reason is filled in, so you can see where you are."),
        ("Privacy", "This file contains photos of real people, including beneficiaries. Share it only with people authorised by the "
                    "NGO, do not forward it, and delete your copy when told to."),
    ]
    for label, text in lines:
        r += 2 if label == "What this is" else 1
        _style(ins.cell(row=r, column=1, value=label), bold=True, valign="top")
        _style(ins.cell(row=r, column=2, value=text), wrap=True, valign="top")
        ins.row_dimensions[r].height = 15 * (1 + len(text) // 105)

    r += 2
    _style(ins.cell(row=r, column=1, value="Progress (updates automatically)"), bold=True, size=11)
    prog = [
        ("Photos in this sheet", f"=COUNTA({TAG_SHEET}!$C$2:$C${last})"),
        ("Photos tagged", f"=COUNTA({TAG_SHEET}!$F$2:$F${last})"),
        ("Photos remaining", f"=B{r + 1}-B{r + 2}"),
        ("Share done", f"=IF(B{r + 1}=0,0,B{r + 2}/B{r + 1})"),
    ]
    for k, (label, formula) in enumerate(prog, start=1):
        _style(ins.cell(row=r + k, column=1, value=label), bold=True)
        c = ins.cell(row=r + k, column=2, value=formula)
        _style(c, align="left")
        if label == "Share done":
            c.number_format = "0.0%"
    r += len(prog) + 2
    _style(ins.cell(row=r, column=1, value="EXAMPLE of a filled row (not real data)"), bold=True, size=11)
    ex_head = ["Primary reason", "Second reason", "Comment", "Confidence", "Tagger"]
    ex_vals = [opts[2]["text"] if len(opts) > 2 else "", opts[6]["text"] if len(opts) > 6 else "",
               "Volunteer is facing away; banner is folded at the left edge", "Sure", "Asha"]
    for j, (h, v) in enumerate(zip(ex_head, ex_vals, strict=True)):
        _style(ins.cell(row=r + 1 + j, column=1, value=h), bold=True)
        _style(ins.cell(row=r + 1 + j, column=2, value=v), wrap=True, fill="F2F2F2")
    ins.sheet_view.showGridLines = False

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    return out_path


# --------------------------------------------------------------------------- reading the returned sheet


def read_tags(path: Path, rule_set: RuleSet) -> tuple[pd.DataFrame, list[str]]:
    """Read tagged rows. Returns (rows with a primary reason, problems found). Untagged rows are ignored."""
    wb = load_workbook(path, data_only=True)
    if TAG_SHEET not in wb.sheetnames:
        raise DataError(f"sheet '{TAG_SHEET}' not found")
    ws = wb[TAG_SHEET]
    head = {str(c.value): i for i, c in enumerate(ws[1]) if c.value is not None}
    missing = [h for h in HEADERS if h not in head]
    if missing:
        raise DataError(f"missing columns: {missing}")
    valid = {o["code"] for o in reason_options(rule_set)}
    rows, problems = [], []
    for row in ws.iter_rows(min_row=2, values_only=True):
        get = lambda h, row=row: row[head[h]]  # noqa: E731
        primary, second, third = (code_of(get(h)) for h in (H_PRIMARY, H_SECOND, H_THIRD))
        extras = [x for x in (second, third) if x]
        who = f"row #{get(H_ORDER)} ({get(H_FILE)})"
        if not primary:
            if extras or (get(H_COMMENT) or "").strip():
                problems.append(f"{who}: has other tags/comment but no primary reason")
            continue
        for code in (primary, *extras):
            if code not in valid:
                problems.append(f"{who}: unknown reason '{code}'")
        if len({primary, *extras}) != 1 + len(extras):
            problems.append(f"{who}: the same reason is selected more than once")
        conf = (get(H_CONF) or "").strip()
        if conf and conf not in CONFIDENCE:
            problems.append(f"{who}: confidence must be Sure or Unsure, got '{conf}'")
        rows.append({"image_id": get(H_ID), "file": get(H_FILE), "tag_order": get(H_ORDER), "program": get(H_PROGRAM),
                     "primary": primary, "second": second or "", "third": third or "",
                     "comment": (get(H_COMMENT) or "").strip(), "confidence": conf, "tagger": (get(H_TAGGER) or "").strip()})
    return pd.DataFrame(rows), problems


def tagged_share(df: pd.DataFrame, total: int) -> float:
    return len(df) / total if total else 0.0


def iter_codes(df: pd.DataFrame) -> Iterable[str]:
    for col in ("primary", "second", "third"):
        yield from (c for c in df[col] if c)
