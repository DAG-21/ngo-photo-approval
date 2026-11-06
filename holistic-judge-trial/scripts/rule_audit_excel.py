"""Audit: do the photos humans APPROVED break the client's own rules? (offline; reads finished checklist runs, calls nothing)

    python scripts/rule_audit_excel.py --version checklist1 --pool dev --tag paired-tagged --ids ids/paired_tagged_dev.csv --out reports/approved_rule_audit_preview.xlsx

A rule counts as violated on a photo when any of that rule's questions was answered NO (the same definition as the checklist reports).
Only questions that are a direct breakdown of the client's rule text (source: client) are used; the team's and the taggers' extra
questions are ignored, so this measures the client's 16 rules only. Rules that did not apply to the photo (not applicable) are not counted.
"""

import argparse
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True)
    ap.add_argument("--pool", default="dev", choices=["dev", "confirm", "validation"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", default="K")
    ap.add_argument("--ids", default=None, help="CSV with image_id and label columns; only approved photos in it are listed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_trial_config()
    ensure_main_importable(cfg)
    from ngo_photo_approval.rules.loader import load_rule_set

    names = {r.rule_id: r.source_ref for r in load_rule_set().active()}
    spec = load_checks(args.version)
    client = {}
    for c in spec["checks"]:
        if c["source"] == "client" and c["rule"] != "FACT":
            client.setdefault(c["rule"], []).append(c["id"])
    latest = {r["image_id"]: r for r in read_results(run_dir_for(cfg, args.pool, args.variant, args.tag)) if r.get("ok")}
    if args.ids:
        key = pd.read_csv(TRIAL_ROOT / args.ids).set_index("image_id")
        approved = [i for i in latest if i in key.index and key.loc[i, "label"] == "approved"]
        program = {i: key.loc[i, "program"] for i in approved}
    else:
        approved, program = list(latest), {}

    recs, per_rule, checked = [], Counter(), Counter()
    for i in sorted(approved):
        ans = latest[i]["checks"]
        viol = []
        for rule, ids in client.items():
            graded = [a for a in ids if ans[a]["answer"] in ("YES", "NO")]
            if graded:
                checked[rule] += 1
            no = [a for a in ids if ans[a]["answer"] == "NO"]
            if no:
                per_rule[rule] += 1
                viol.append(f"{rule} - {names[rule]}\n" + "\n".join(f"   - {a}: {ans[a]['evidence'] or '(no evidence given)'}" for a in no))
        recs.append({"id": i, "n": len(viol), "text": "\n".join(viol) if viol else "(no client rule violated)", "scene": latest[i]["scene"],
                     "program": program.get(i, "")})
    recs.sort(key=lambda d: (-d["n"], d["id"]))

    wb = Workbook()
    ws = wb.active
    ws.title = "Approved photos"
    heads = ["#", "Image (approved by the NGO)", "Client rules violated (with Gemini's evidence)", "Rules violated (count)", "What the photo shows (Gemini)", "Program", "Image ID"]
    widths = [5, 36, 80, 12, 44, 18, 18]
    for c, (h, w) in enumerate(zip(heads, widths), 1):
        cell = ws.cell(1, c, h)
        cell.font, cell.fill, cell.border = Font(name=FONT, bold=True, color="FFFFFF"), HEAD, BOX
        cell.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
        ws.column_dimensions[cell.column_letter].width = w
    ws.row_dimensions[1].height = 34
    ws.freeze_panes = "C2"
    for n, d in enumerate(recs, 1):
        row = n + 1
        for c, v in enumerate([n, None, d["text"], d["n"], d["scene"], d["program"], d["id"]], 1):
            cell = ws.cell(row, c, v)
            cell.font, cell.border = Font(name=FONT, size=10), BOX
            cell.alignment = Alignment(wrap_text=True, vertical="top" if c in (3, 5) else "center", horizontal="center" if c in (1, 4) else None)
        ws.row_dimensions[row].height = max(145, 15 * (len(d["text"].splitlines()) + 1))
        if d["n"]:
            ws.cell(row, 4).fill = PatternFill("solid", start_color="F8CBAD" if d["n"] >= 2 else "FFE699")
        img = XLImage(thumb(image_path(cfg, d["id"])))
        img.anchor = f"B{row}"
        ws.add_image(img)

    sm = wb.create_sheet("Summary", 0)
    sm.column_dimensions["A"].width = 52
    for col in "BCD":
        sm.column_dimensions[col].width = 20
    total = len(recs)
    any_v = sum(d["n"] > 0 for d in recs)
    two = sum(d["n"] >= 2 for d in recs)
    rows = [("Approved photos audited", total, ""), ("Photos with at least one client rule violated", any_v, f"{any_v / max(total, 1):.0%}"),
            ("Photos with two or more rules violated", two, f"{two / max(total, 1):.0%}"),
            ("Photos with no violation", total - any_v, f"{(total - any_v) / max(total, 1):.0%}"), ("", "", ""),
            ("Rule (client wording ref)", "Photos it applied to", "Photos violating it"), ]
    for r_, (a, b, c) in enumerate(rows, 1):
        sm.cell(r_, 1, a).font = Font(name=FONT, size=10, bold=r_ in (1, 6))
        sm.cell(r_, 2, b).font = Font(name=FONT, size=10, bold=r_ == 6)
        sm.cell(r_, 3, c).font = Font(name=FONT, size=10, bold=r_ == 6)
    r_ = len(rows) + 1
    for rule in sorted(client):
        sm.cell(r_, 1, f"{rule} - {names[rule]}").font = Font(name=FONT, size=10)
        sm.cell(r_, 2, checked[rule]).font = Font(name=FONT, size=10)
        v = per_rule[rule]
        sm.cell(r_, 3, f"{v} ({v / max(checked[rule], 1):.0%} of those it applied to)").font = Font(name=FONT, size=10)
        r_ += 1
    nt = wb.create_sheet("Notes")
    nt.column_dimensions["A"].width = 120
    for i, t in enumerate([
        "Audit of photos the NGO APPROVED against the client's own rules, using Gemini's yes/no answers to the checklist questions that break each rule down.",
        f"Prompt version {args.version}; run '{args.tag}'. Only questions that follow the client's wording are used (not the team's or taggers' extra questions).",
        "A rule is 'violated' when any of its questions was answered NO. A rule that did not apply to the photo (for example no banner) is not counted.",
        "This is Gemini's reading, not a ground truth: Gemini is noisy (the same photo gets different answers between runs) and may be stricter or looser than a human on subjective rules like 'candid' or 'looking into the camera'.",
        "To interpret the rates you also need the same audit on the disapproved photos: a rule that is 'violated' as often on disapproved photos does not explain the disapprovals.",
        "This file contains photos of real people: share only with authorised people.",
    ], 1):
        c = nt.cell(i, 1, t)
        c.font = Font(name=FONT, size=10)
        c.alignment = Alignment(wrap_text=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"wrote {out.resolve()}  ({total} approved photos; {any_v} with at least one violation; {two} with two or more)")
    for rule in sorted(client):
        print(f"  {rule:<5} {names[rule][:34]:<34} applied to {checked[rule]:>3}, violated on {per_rule[rule]:>3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
