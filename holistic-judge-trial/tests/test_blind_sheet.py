from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from judge_trial.blind_sheet import (TAG, analyse, build_workbook, read_returned, reason_list, select_rows, workbook_text)

from conftest import make_image

OPTIONS = [
    {"code": "R006", "label": "Seva activity not clearly shown", "meaning": "Clearly showcase the activity.", "guidance": "", "text": "R006 - Seva activity not clearly shown"},
    {"code": "X4", "label": "Looks fine, cannot tell why it was disapproved", "meaning": "You see no problem.", "guidance": "", "text": "X4 - Looks fine, cannot tell why it was disapproved"},
    {"code": "X5", "label": "Unrelated", "meaning": "extra", "guidance": "", "text": "X5 - Unrelated"},
]


@pytest.fixture
def paired(cfg):
    rows = [{"image_id": f"img_{i:02d}", "label": "approved" if i % 2 else "not_approved", "program": "arham_aahar" if i < 6 else "other",
             "group_id": f"g{i}", "tagged_primary": "R006" if i % 2 == 0 else None} for i in range(10)]
    for r in rows:
        make_image(cfg, r["image_id"])
    return pd.DataFrame(rows)


def test_reason_list_rewords_the_looks_fine_option_and_drops_extras():
    out = reason_list(OPTIONS, {"X5"})
    assert [o["code"] for o in out] == ["R006", "X4"]
    x4 = out[1]
    assert "disapproved" not in x4["text"] and "disapproved" not in x4["meaning"] and x4["text"].startswith("X4 - ")


def test_selection_is_seeded_complete_and_mixes_both_classes(paired):
    a, b = select_rows(paired, 3), select_rows(paired, 3)
    assert a.image_id.tolist() == b.image_id.tolist() and sorted(a.image_id) == sorted(paired.image_id)
    assert a.row.tolist() == list(range(1, 11))
    assert select_rows(paired, 4).image_id.tolist() != a.image_id.tolist()


def test_the_workbook_never_reveals_a_label(cfg, paired, tmp_path):
    rows = select_rows(paired, 3)
    path = build_workbook(rows, reason_list(OPTIONS, {"X5"}), tmp_path / "s.xlsx", lambda i: cfg.normalized_dir / f"{i}.jpg")
    text = " ".join(workbook_text(path)).lower()
    for leak in ("disapproved", "not_approved", "not approved", "arham", "program", "copies", "label"):
        assert leak not in text, leak
    ws = load_workbook(path)[TAG]
    assert ws.max_row == 11 and ws.column_dimensions["J"].hidden and len(ws._images) == 10


def test_a_returned_sheet_round_trips(cfg, paired, tmp_path):
    rows = select_rows(paired, 3)
    path = build_workbook(rows, reason_list(OPTIONS, {"X5"}), tmp_path / "s.xlsx", lambda i: cfg.normalized_dir / f"{i}.jpg")
    wb = load_workbook(path)
    ws = wb[TAG]
    ws["C2"], ws["D2"], ws["E2"], ws["H2"], ws["I2"] = "No", "R006 - Seva activity not clearly shown", "X4 - Looks fine, I see no problem", "Sure", "Asha"
    ws["C3"], ws["D3"] = "Yes", "X4 - Looks fine, I see no problem"
    wb.save(tmp_path / "back.xlsx")
    got = read_returned(tmp_path / "back.xlsx")
    assert len(got) == 2 and got.iloc[0].primary == "R006" and got.iloc[0].second == "X4" and got.iloc[1].approve == "Yes"
    assert got.iloc[0].image_id == rows.image_id.iloc[0] and got.iloc[0].tagger == "Asha"


def test_analysis_separates_a_real_signal_from_tagging_everything(paired):
    key = paired.assign(row=range(1, 11))
    blind = pd.DataFrame({"row": range(1, 11), "image_id": paired.image_id,
                          "approve": ["Yes" if i % 2 else "No" for i in range(10)],          # taggers agree with the labels exactly
                          "primary": ["X4" if i % 2 else "R006" for i in range(10)],
                          "second": [None] * 10, "third": [None] * 10})
    earlier = pd.DataFrame({"primary": ["R006"] * 5}, index=[f"img_{i:02d}" for i in range(0, 10, 2)])
    res = analyse(blind, key, earlier)
    assert res["human_vs_label"]["agree"] == 1.0 and res["problem_found"] == {"approved": 0.0, "disapproved": 1.0}
    r = res["by_reason"].set_index("reason").loc["R006"]
    assert r.disapproved == 1.0 and r.approved == 0.0
    assert res["blind_vs_earlier"]["same_primary"] == 1.0 and res["blind_vs_earlier"]["now_looks_fine"] == 0.0
    # taggers who find a problem on everything look the same on both classes
    everything = blind.assign(primary="R006", approve="No")
    res2 = analyse(everything, key, earlier)
    assert res2["problem_found"]["approved"] == res2["problem_found"]["disapproved"] == 1.0
    assert res2["human_vs_label"]["disapproved_that_taggers_would_approve"] == 0.0 and res2["human_vs_label"]["approved_that_taggers_would_not"] == 1.0
