"""Excel of MY OWN audit (Claude looking at the photos, not Gemini): which of the client's rules do the 41 approved photos break?

    python scripts/claude_audit_excel.py

The judgements below were made by looking at each photo (numbered 1-41 in sorted image-id order, the same 41 approved photos as the
Gemini preview) against the client's own rule text only: R001-R013 (R014-R016 restate other rules). A rule is listed only when the
violation is visible; borderline cases are left out or marked (low confidence). Gemini's flags from the checklist1 run are shown beside
mine so the two readings can be compared.
"""

import io
import sys
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from PIL import Image

from judge_trial.checklist import load_checks
from judge_trial.config import TRIAL_ROOT, ensure_main_importable, load_trial_config
from judge_trial.data import image_path
from judge_trial.runner import read_results, run_dir_for
from audit_more_data import MORE
from audit_data_p3 import P3
from audit_data_p4 import P4
from audit_data_p5 import P5
from audit_data_p6 import P6
from audit_data_p7 import P7
from audit_data_p8 import P8
from audit_data_p9 import P9

# photo number (1-41) -> [(rule, what I can see)]
DATA: dict[int, list[tuple[str, str]]] = {
    1: [("R006", "Only a person eating alone; no serving or giving is shown"), ("R003", "Face partly hidden by the hand and plate while eating")],
    2: [("R006", "Men standing and eating; no volunteer or serving visible"), ("R011", "Motorbike and a pole in the middle of the frame")],
    3: [],
    4: [("R003", "Beneficiaries' faces turned away or covered by the sari"), ("R011", "Pole in the foreground and wet, cluttered ground")],
    5: [("R004", "Volunteer looks straight into the camera"), ("R011", "Dirty barn floor with scattered waste")],
    6: [("R002", "Main volunteer in a pale striped shirt and dark trousers, not white-on-white or white with jeans (low confidence)")],
    7: [("R002", "Volunteer in a grey T-shirt and navy trousers"), ("R003", "Child's face hidden by a hood and turned down")],
    8: [("R001", "Banner text partly hidden behind the volunteers"), ("R004", "Everyone looks into the camera"), ("R005", "Lined-up posed group"),
        ("R006", "No beneficiary and no serving in progress")],
    9: [("R003", "Children's faces turned away; one shows the back of the head"), ("R007", "Shot from behind the children")],
    10: [("R006", "Only pigeons; no volunteer or feeding action visible")],
    11: [("R004", "Looks straight into the camera"), ("R005", "Posed portrait holding the dish"), ("R006", "No seva activity shown")],
    12: [("R002", "Woman volunteer in a coloured sari, not white"), ("R003", "Beneficiaries' faces turned away, backs to the camera")],
    13: [("R002", "Volunteers in a navy polo and a floral top, not white"), ("R004", "Girl looks straight into the camera"), ("R005", "Posed with the tray")],
    14: [("R004", "Looks at the camera"), ("R005", "Posed portrait"), ("R006", "No seva activity shown"), ("R011", "Rubble and waste behind him")],
    15: [("R002", "Woman volunteer in a pink top, not white")],
    16: [("R003", "Beneficiary's face turned away and partly covered by cloth"), ("R011", "Construction site and tarp in the background")],
    17: [("R002", "Children not in white"), ("R004", "All look into the camera"), ("R005", "Posed group"), ("R006", "No activity shown")],
    18: [("R004", "Girl looks straight into the camera"), ("R006", "No volunteer or giving shown"), ("R011", "Rusty vehicle clutter behind her")],
    19: [("R004", "Child looks into the camera"), ("R006", "Only a child eating; no giving shown")],
    20: [],
    21: [("R001", "Banner on the table front is wrinkled"), ("R002", "Volunteer in a navy polo, not white"), ("R003", "Beneficiary leaning in profile, face partly hidden")],
    22: [("R002", "Several volunteers in yellow or other non-white tops"), ("R004", "Everyone looks into the camera"), ("R005", "Posed group photo"), ("R006", "No activity shown")],
    23: [("R004", "Boy looks into the camera"), ("R006", "No serving or giving visible"), ("R008", "Volunteer seen from behind")],
    24: [("R001", "Standee banner cut off at the frame edge and curved"), ("R003", "Beneficiaries' faces turned away")],
    25: [("R003", "Face partly covered by cap and hand"), ("R006", "No volunteer or giving shown"), ("R011", "Parked vehicles clutter the frame")],
    26: [("R004", "Everyone looks into the camera"), ("R005", "Lined-up posed group"), ("R006", "Line-up; no serving to beneficiaries")],
    27: [("R004", "Beneficiary looks straight into the camera"), ("R011", "Scooter in the foreground and construction clutter")],
    28: [("R004", "Looking at the camera"), ("R005", "Posed with the bags"), ("R006", "No activity; people small and far away"), ("R011", "Building and cars behind")],
    29: [("R003", "Children's faces covered by the books they hold up"), ("R005", "Staged hold-up for the camera"), ("R007", "Wide-angle, tilted perspective")],
    30: [("R003", "Beneficiaries' faces turned away from the camera")],
    31: [("R003", "Beneficiaries' faces turned down or away"), ("R008", "Volunteer bent over with her back to the camera")],
    32: [("R003", "Child's back is to the camera")],
    33: [("R003", "Children's faces hidden or turned away"), ("R006", "No volunteer or giving shown"), ("R011", "Cluttered slum background")],
    34: [("R001", "Banner partly hidden behind the people"), ("R004", "Volunteer looks into the camera"), ("R005", "Posed handover"), ("R011", "Play-area signboards behind")],
    35: [("R001", "Lower part of the banner hidden by heads"), ("R002", "Volunteer in an orange top, not white"), ("R008", "Volunteer's back to the camera")],
    36: [("R008", "Volunteer in a white sari with her back to the camera")],
    37: [("R003", "Woman's face turned away and covered by the scarf"), ("R006", "Unclear what is happening; people small and far"), ("R011", "Pole in the middle of the frame")],
    38: [("R007", "Scooter handlebars block the foreground"), ("R011", "Tarps and clutter behind")],
    39: [("R004", "Child looks into the camera"), ("R006", "Only a child eating; no giving shown")],
    40: [("R006", "Only food seen from above; no people or serving")],
    41: [("R001", "Banner partly hidden by the volunteers"), ("R002", "Two volunteers not in white"), ("R004", "All look into the camera"), ("R005", "Posed line-up"),
         ("R006", "No activity shown"), ("R007", "Scooters block the foreground"), ("R011", "Blue sheets and scooters behind")],
}

FONT, THIN = "Arial", Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
HEAD = PatternFill("solid", start_color="1F3864")


def thumb(path: Path, size=(250, 190)) -> io.BytesIO:
    im = Image.open(path).convert("RGB")
    im.thumbnail(size)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    buf.seek(0)
    return buf


def main() -> int:
    cfg = load_trial_config()
    ensure_main_importable(cfg)
    from ngo_photo_approval.rules.loader import load_rule_set

    names = {r.rule_id: r.source_ref for r in load_rule_set().active()}
    key = pd.read_csv(TRIAL_ROOT / "ids" / "paired_tagged_dev.csv")
    ids1 = sorted(key[key.label == "approved"].image_id)
    more = pd.read_csv(TRIAL_ROOT / "ids" / "audit_approved_80.csv")
    ids2 = more.image_id.tolist()  # already in the order the photos were numbered 42-121
    more3 = pd.read_csv(TRIAL_ROOT / "ids" / "audit_approved_179.csv")
    ids3 = more3.image_id.tolist()  # photos 122-300, in file order
    ids = ids1 + ids2 + ids3
    for part in (MORE, P3, P4, P5, P6, P7, P8, P9):
        DATA.update(part)
    assert len(ids) == len(DATA) == 300 and sorted(DATA) == list(range(1, 301)), (len(ids), len(DATA))
    prog = {**key.set_index("image_id").program.to_dict(), **more.set_index("image_id").program.to_dict(), **more3.set_index("image_id").program.to_dict()}

    spec = load_checks("checklist1")
    client = {}
    for c in spec["checks"]:
        if c["source"] == "client" and c["rule"] != "FACT":
            client.setdefault(c["rule"], []).append(c["id"])
    gem = {r["image_id"]: r for r in read_results(run_dir_for(cfg, "dev", "K", "paired-tagged")) if r.get("ok")}

    def gemini_rules(i):
        return sorted(rule for rule, cks in client.items() if any(gem[i]["checks"][c]["answer"] == "NO" for c in cks))

    mine_cnt, gem_cnt, both_cnt = Counter(), Counter(), Counter()
    rows = []
    for n, i in enumerate(ids, 1):
        mine = DATA[n]
        mr = sorted({r for r, _ in mine})
        gr = gemini_rules(i) if i in gem else None
        for r in mr:
            mine_cnt[r] += 1
        if gr is not None:
            for r in gr:
                gem_cnt[r] += 1
            for r in set(mr) & set(gr):
                both_cnt[r] += 1
        text = "\n".join(f"{r} - {names[r]}: {e}" for r, e in mine) or "(no client rule clearly violated)"
        rows.append({"n": n, "id": i, "text": text, "count": len(mr), "gem": (", ".join(gr) or "(none)") if gr is not None else "(not run)", "program": prog[i]})

    wb = Workbook()
    ws = wb.active
    ws.title = "Approved photos"
    heads = ["#", "Image (approved by the NGO)", "Client rules this photo violates (my reading)", "Rules violated (count)", "Rules Gemini flagged (checklist1)", "Program", "Image ID"]
    widths = [5, 36, 82, 12, 22, 18, 18]
    for c, (h, w) in enumerate(zip(heads, widths), 1):
        cell = ws.cell(1, c, h)
        cell.font, cell.fill, cell.border = Font(name=FONT, bold=True, color="FFFFFF"), HEAD, BOX
        cell.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
        ws.column_dimensions[cell.column_letter].width = w
    ws.row_dimensions[1].height = 34
    ws.freeze_panes = "C2"
    for r_, d in enumerate(rows, 2):
        for c, v in enumerate([d["n"], None, d["text"], d["count"], d["gem"], d["program"], d["id"]], 1):
            cell = ws.cell(r_, c, v)
            cell.font, cell.border = Font(name=FONT, size=10), BOX
            cell.alignment = Alignment(wrap_text=True, vertical="top" if c in (3, 5) else "center", horizontal="center" if c in (1, 4) else None)
        ws.row_dimensions[r_].height = max(145, 15 * (len(d["text"].splitlines()) + 1))
        if d["count"]:
            ws.cell(r_, 4).fill = PatternFill("solid", start_color="F8CBAD" if d["count"] >= 2 else "FFE699")
        img = XLImage(thumb(image_path(cfg, d["id"])))
        img.anchor = f"B{r_}"
        ws.add_image(img)

    sm = wb.create_sheet("Summary", 0)
    sm.column_dimensions["A"].width = 56
    for col in "BCDE":
        sm.column_dimensions[col].width = 22
    total = len(rows)
    any_v = sum(d["count"] > 0 for d in rows)
    two = sum(d["count"] >= 2 for d in rows)
    top = [("Approved photos audited (by me, looking at each photo)", total), ("   of which also read by Gemini (first 41 only)", sum(d["gem"] != "(not run)" for d in rows)), ("Photos with at least one client rule clearly violated", f"{any_v} ({any_v / total:.0%})"),
           ("Photos with two or more rules violated", f"{two} ({two / total:.0%})"), ("Photos with no violation", f"{total - any_v} ({(total - any_v) / total:.0%})")]
    for r_, (a, b) in enumerate(top, 1):
        sm.cell(r_, 1, a).font = Font(name=FONT, size=10, bold=True)
        sm.cell(r_, 2, b).font = Font(name=FONT, size=10)
    hdr = ["Rule", f"Photos I flagged (of {total})", "Photos Gemini flagged (of the first 41)", "Flagged by both (of the first 41)"]
    for c, h in enumerate(hdr, 1):
        cell = sm.cell(7, c, h)
        cell.font, cell.fill = Font(name=FONT, bold=True, color="FFFFFF"), HEAD
    for r_, rule in enumerate(sorted(names)[:13], 8):
        sm.cell(r_, 1, f"{rule} - {names[rule]}").font = Font(name=FONT, size=10)
        for c, v in enumerate([mine_cnt[rule], gem_cnt[rule], both_cnt[rule]], 2):
            sm.cell(r_, c, v).font = Font(name=FONT, size=10)
    nt = wb.create_sheet("Notes")
    nt.column_dimensions["A"].width = 120
    for i, t in enumerate([
        "This is my own reading (Claude looking at each photo), not Gemini's. Gemini's flags are shown beside mine for comparison.",
        "Only the client's own rule text is used (R001-R013; R014-R016 restate other rules). The team's extra rules (white-top giver, front angle, etc.) are NOT used here.",
        "I list a rule only when the problem is visible. Several of these rules are subjective (candid, looking into the camera, clean background, activity clearly shown), so another reviewer could mark a few differently; photo 6 (dress code) is low confidence.",
        "I looked at contact sheets at moderate resolution, so I could not judge fine detail such as slight blur or small creases, and I did not mark them.",
        "300 approved photos were audited: the 41 approved photos of the paired test set, plus 80 random ones (seed 2028) and 179 more random ones (seed 2029), all from the unique approved train+validation photos. Not all 2,651.",
        "Interpretation needs a baseline: the same reading of the disapproved photos has not been done. Gemini read only the first 41 photos, so the Gemini columns are blank (not run) for photos 42-300.",
        "This file contains photos of real people: share only with authorised people.",
    ], 1):
        c = nt.cell(i, 1, t)
        c.font = Font(name=FONT, size=10)
        c.alignment = Alignment(wrap_text=True)
    out = TRIAL_ROOT / "reports" / "approved_rule_audit_by_claude.xlsx"
    wb.save(out)
    print(f"wrote {out}\nphotos {total}; >=1 violation {any_v}; >=2 {two}; none {total - any_v}")
    for rule in sorted(names)[:13]:
        print(f"  {rule}: me {mine_cnt[rule]:>2} | Gemini {gem_cnt[rule]:>2} | both {both_cnt[rule]:>2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
