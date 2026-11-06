from __future__ import annotations

import io

import pytest
from openpyxl import load_workbook
from PIL import Image

from ngo_photo_approval.data.tagging import (
    CODES_SHEET,
    H_COMMENT,
    H_CONF,
    H_ID,
    H_PRIMARY,
    H_SECOND,
    H_TAGGER,
    H_THIRD,
    HEADERS,
    INFO_SHEET,
    TAG_SHEET,
    TagRow,
    build_workbook,
    code_of,
    read_tags,
    reason_options,
)
from ngo_photo_approval.exceptions import DataError
from ngo_photo_approval.rules.loader import load_rule_set


def thumb(row: TagRow):
    buf = io.BytesIO()
    Image.new("RGB", (120, 90), (200, 100, 50)).save(buf, "JPEG")
    return buf.getvalue(), 120, 90


ROWS = [TagRow(i, f"img_{i:04d}", f"disapproved_{i}.jpg", "arham_aahar" if i % 2 else "", 1 + i % 3) for i in range(1, 6)]


@pytest.fixture
def rules():
    return load_rule_set()


@pytest.fixture
def sheet(tmp_path, rules):
    return build_workbook(ROWS, thumb, rules, tmp_path / "sheet.xlsx")


def test_options_follow_the_rules_and_add_non_rule_reasons(rules):
    opts = reason_options(rules)
    codes = [o["code"] for o in opts]
    assert codes[:2] == ["X1", "X2"] and codes[-3:] == ["X3", "X4", "S"]
    assert [c for c in codes if c.startswith("R")] == [f"R{n:03d}" for n in range(1, 14)]  # R014-R016 restate others
    assert len(set(codes)) == len(codes)
    extra = [o for o in opts if o["code"].startswith("X") and o["code"] not in ("X1", "X2", "X3", "X4")]
    assert [o["code"] for o in extra] == [f"X{n}" for n in range(5, 14)] and all(o["in_sop"] == "No" for o in extra)
    assert all(o["in_sop"] == "Yes" for o in opts if o["code"].startswith("R"))
    r3 = next(o for o in opts if o["code"] == "R003")
    assert r3["meaning"].startswith("Ensure the beneficiary")  # the client's wording, verbatim
    assert "from the front" in r3["guidance"]  # team guidance shown separately
    assert all(o["text"] == f"{o['code']} - {o['label']}" for o in opts)
    assert code_of("R007 - Bad camera angle") == "R007" and code_of(None) == "" and code_of("S - Cannot judge") == "S"


def test_workbook_structure_photos_dropdowns_and_formulas(sheet):
    wb = load_workbook(sheet)
    assert wb.sheetnames[0] == INFO_SHEET and {TAG_SHEET, CODES_SHEET} <= set(wb.sheetnames)
    ws = wb[TAG_SHEET]
    assert [c.value for c in ws[1]] == HEADERS
    assert len(ws._images) == len(ROWS)  # every photo is embedded
    assert [ws.cell(row=i, column=1).value for i in range(2, 7)] == [1, 2, 3, 4, 5]
    assert ws.cell(row=2, column=3).value == "disapproved_1.jpg" and ws.column_dimensions["L"].hidden
    assert ws.freeze_panes == "C2"
    formulas = {str(dv.sqref): dv.formula1 for dv in ws.data_validations.dataValidation}
    assert formulas["F2:H6"] == "=ReasonList" and formulas["J2:J6"] == '"Sure,Unsure"'
    assert "ReasonList" in wb.defined_names
    # progress and usage counts are formulas, not typed numbers
    info = wb[INFO_SHEET]
    texts = [str(c.value) for row in info.iter_rows() for c in row if c.value]
    assert any(t.startswith("=COUNTA(Tagging!$F$2:$F$6)") for t in texts)
    assert wb[CODES_SHEET]["E2"].value.startswith("=COUNTIF(Tagging!$F$2:$H$6,A2)")  # counts all three reason columns
    assert wb[CODES_SHEET]["D2"].value == "No" and wb[CODES_SHEET]["D4"].value == "Yes"  # X1 is not an SOP rule, R001 is
    assert any("EXAMPLE" in t for t in texts)  # the example lives on the Instructions sheet, never in the data
    assert all(ws.cell(row=i, column=6).value is None for i in range(2, 7))  # no pre-filled tags


def fill(sheet, rules, tags):
    """tags: {row_number: dict(header -> value)} written like a human tagger would."""
    wb = load_workbook(sheet)
    ws = wb[TAG_SHEET]
    head = {c.value: c.column for c in ws[1]}
    for row, vals in tags.items():
        for h, v in vals.items():
            ws.cell(row=row, column=head[h], value=v)
    wb.save(sheet)


def test_round_trip_reads_only_tagged_rows(sheet, rules):
    o = {x["code"]: x["text"] for x in reason_options(rules)}
    fill(sheet, rules, {
        2: {H_PRIMARY: o["X1"], H_CONF: "Sure", H_TAGGER: "Asha"},
        4: {H_PRIMARY: o["R007"], H_SECOND: o["R003"], H_COMMENT: " shot from behind ", H_CONF: "Unsure", H_TAGGER: "Asha"},
    })
    df, problems = read_tags(sheet, rules)
    assert problems == [] and len(df) == 2
    assert list(df.image_id) == ["img_0001", "img_0003"] and list(df.primary) == ["X1", "R007"]
    assert df.iloc[1].second == "R003" and df.iloc[1].comment == "shot from behind" and df.iloc[1].confidence == "Unsure"
    assert df.iloc[0].tagger == "Asha" and df.iloc[0].file == "disapproved_1.jpg"


def test_problems_are_reported_not_silently_accepted(sheet, rules):
    o = {x["code"]: x["text"] for x in reason_options(rules)}
    fill(sheet, rules, {
        2: {H_SECOND: o["R001"], H_COMMENT: "forgot the primary"},  # second without primary
        3: {H_PRIMARY: "Z9 - made up"},  # unknown code
        4: {H_PRIMARY: o["R002"], H_SECOND: o["R002"]},  # duplicated reason
        5: {H_PRIMARY: o["R005"], H_CONF: "Maybe"},  # bad confidence
        6: {H_PRIMARY: o["R006"], H_THIRD: o["R010"]},  # fine
    })
    df, problems = read_tags(sheet, rules)
    text = " | ".join(problems)
    assert "no primary reason" in text and "unknown reason 'Z9'" in text and "more than once" in text and "Maybe" in text
    assert len(problems) == 4 and len(df) == 4 and df.iloc[-1].third == "R010"


def test_reader_rejects_a_foreign_workbook(tmp_path, rules):
    from openpyxl import Workbook

    wb = Workbook()
    wb.active.title = "Other"
    wb.save(tmp_path / "x.xlsx")
    with pytest.raises(DataError, match="Tagging"):
        read_tags(tmp_path / "x.xlsx", rules)
    wb = Workbook()
    wb.active.title = TAG_SHEET
    wb.active.append(["#", "File"])
    wb.save(tmp_path / "y.xlsx")
    with pytest.raises(DataError, match="missing columns"):
        read_tags(tmp_path / "y.xlsx", rules)
    assert H_ID in HEADERS and {H_PRIMARY, H_SECOND, H_THIRD, H_COMMENT, H_CONF, H_TAGGER} <= set(HEADERS)
