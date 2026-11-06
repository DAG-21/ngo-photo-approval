"""Presentation only: the V1 vs V2 workbook. Called by v1_v2_comparison.py --excel. Outcome and match columns are Excel formulas."""

from __future__ import annotations

import io
from pathlib import Path

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from PIL import Image

from layered_engine.comparison import AUTO
from layered_engine.rulebook import Rulebook

FONT, THIN = "Arial", Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
HEAD = PatternFill("solid", start_color="1F3864")
SUB = PatternFill("solid", start_color="D9E2F3")


def thumb(path: str, size=(250, 190)) -> io.BytesIO:
    im = Image.open(path).convert("RGB")
    im.thumbnail(size)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    buf.seek(0)
    return buf


def head(ws, row, values, widths=None, height=38):
    for c, v in enumerate(values, 1):
        cell = ws.cell(row, c, v)
        cell.font, cell.fill, cell.border = Font(name=FONT, bold=True, color="FFFFFF"), HEAD, BOX
        cell.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
        if widths:
            ws.column_dimensions[cell.column_letter].width = widths[c - 1]
    ws.row_dimensions[row].height = height


def put(ws, row, values, bold=False, fill=None, wrap_cols=()):
    for c, v in enumerate(values, 1):
        cell = ws.cell(row, c, v)
        cell.font, cell.border = Font(name=FONT, size=10, bold=bold), BOX
        cell.alignment = Alignment(wrap_text=True, vertical="top" if c in wrap_cols else "center")
        if fill:
            cell.fill = fill


def reasoning(d) -> str:
    parts = [f"{g.gate_id} {'FAILED' if g.status.value == 'NO' else 'UNSURE'}: {g.evidence or '(no evidence given)'}" for g in d.gates if g.hard and g.status.value in ("NO", "CANNOT_TELL")]
    parts += [r for r in d.decision_reasons if r.startswith(("QUALITY", "PUBLICATION", "INPUT"))]
    return "\n".join(parts) or "no mandatory gate failed" + ("; publication checks outstanding" if d.recommendation.value == "APPROVED_PENDING_PUBLICATION_CHECKS" else "")


def build(out_path: Path, res: dict, key, paths, num, r1, r2, d1, d2, approved, STAGES, NAMES, ALL) -> None:
    wb = Workbook()
    ids = sorted(d1, key=lambda i: num[i])
    n = len(ids)
    last = n + 1

    # ================================================================== Photos
    ws = wb.active
    ws.title = "Photos"
    heads = ["#", "Image", "Human decision", "Human tags (why disapproved)", "V1 decision", "V1 detail", "V1 gates failed", "V2 decision", "V2 detail", "V2 gates failed",
             "V2 gates unsure", "V1 outcome", "V2 outcome", "Change V1 -> V2", "V2 gaze tag", "V2 reasoning (gates that failed, with Gemini's evidence)", "What Gemini says the photo shows (V2 run)",
             "Program", "Opaque ID sent to Gemini", "Image ID"]
    widths = [5, 36, 15, 24, 14, 22, 22, 14, 22, 22, 14, 15, 15, 20, 20, 80, 44, 16, 18, 18]
    head(ws, 1, heads, widths, 44)
    ws.freeze_panes = "C2"
    for k, i in enumerate(ids, 1):
        row = k + 1
        h = "APPROVED" if approved[i] else "DISAPPROVED"
        e1 = "APPROVED" if d1[i].recommendation.value in AUTO else "DISAPPROVED"
        e2 = "APPROVED" if d2[i].recommendation.value in AUTO else "DISAPPROVED"
        outc = lambda col: (f'=IF(AND(C{row}="APPROVED",{col}{row}="APPROVED"),"True positive",IF(AND(C{row}="DISAPPROVED",{col}{row}="DISAPPROVED"),"True negative",'  # noqa: E731
                            f'IF(AND(C{row}="DISAPPROVED",{col}{row}="APPROVED"),"False positive","False negative")))')
        change = (f'=IF(L{row}=M{row},"same",IF(AND(L{row}="False negative",M{row}="True positive"),"rescued (approved photo now passes)",'
                  f'IF(AND(L{row}="True negative",M{row}="False positive"),"NEW FALSE APPROVAL",IF(AND(L{row}="True positive",M{row}="False negative"),"approved photo now blocked",'
                  f'IF(AND(L{row}="False positive",M{row}="True negative"),"false approval removed","changed")))))')
        vals = [num[i], None, h, key[i]["tags"] or ("(approved: no tags)" if approved[i] else "(not tagged)"), e1, d1[i].recommendation.value, ", ".join(d1[i].failed_gates), e2,
                d2[i].recommendation.value, ", ".join(d2[i].failed_gates), ", ".join(d2[i].uncertain_gates), outc("E"), outc("H"), change,
                r2[i]["validated"]["layer_1_scene_understanding"]["scene_facts"].get("f_gaze_tag") or "", reasoning(d2[i]),
                r2[i]["validated"]["layer_1_scene_understanding"]["activity_description"], key[i]["program"], i, key[i]["image_id"]]
        for c, v in enumerate(vals, 1):
            cell = ws.cell(row, c, v)
            cell.font, cell.border = Font(name=FONT, size=10), BOX
            cell.alignment = Alignment(wrap_text=True, vertical="top" if c in (4, 7, 10, 16, 17) else "center", horizontal="center" if c in (1, 3, 5, 8, 12, 13, 14, 15) else None)
        ws.row_dimensions[row].height = max(145, 15 * (len(vals[15].splitlines()) + 2))
        img = XLImage(thumb(paths[i]))
        img.anchor = f"B{row}"
        ws.add_image(img)
    for col in ("L", "M"):
        for text, color in (("False positive", "F4B183"), ("False negative", "FFE699"), ("True positive", "C6E0B4"), ("True negative", "C6E0B4")):
            ws.conditional_formatting.add(f"{col}2:{col}{last}", CellIsRule(operator="equal", formula=[f'"{text}"'], fill=PatternFill("solid", bgColor=color)))
    ws.conditional_formatting.add(f"N2:N{last}", CellIsRule(operator="equal", formula=['"NEW FALSE APPROVAL"'], fill=PatternFill("solid", bgColor="F4B183")))
    ws.conditional_formatting.add(f"N2:N{last}", CellIsRule(operator="equal", formula=['"rescued (approved photo now passes)"'], fill=PatternFill("solid", bgColor="C6E0B4")))
    P = lambda col: f"Photos!${col}$2:${col}${last}"  # noqa: E731

    # ================================================================== Summary (formulas over Photos)
    sm = wb.create_sheet("Summary", 0)
    head(sm, 1, ["Measure (same 50 photos: 25 approved + 25 disapproved)", "V1 (as run)", "V2 (as run)", "Reading"], [64, 18, 18, 70], 30)
    spec = [("True positive: humans approved, engine auto-approved", "TP", "co", "True positive"), ("False negative: humans approved, engine did NOT auto-approve", "FN", "co", "False negative"),
            ("True negative: humans disapproved, engine did not auto-approve", "TN", "co", "True negative"), ("False positive: humans disapproved, engine AUTO-APPROVED (the costly error)", "FP", "co", "False positive")]
    r = 2
    cell_of = {}
    for label, k, _, text in spec:
        put(sm, r, [label, f'=COUNTIFS({P("L")},"{text}")', f'=COUNTIFS({P("M")},"{text}")', ""])
        cell_of[k] = r
        r += 1
    put(sm, r, ["Approved photos explicitly REJECTED", f'=COUNTIFS({P("C")},"APPROVED",{P("F")},"REJECT")', f'=COUNTIFS({P("C")},"APPROVED",{P("I")},"REJECT")', "the rest of the false negatives went to human review or reselect"])
    cell_of["RA"] = r
    r += 1
    put(sm, r, ["Disapproved photos explicitly REJECTED", f'=COUNTIFS({P("C")},"DISAPPROVED",{P("F")},"REJECT")', f'=COUNTIFS({P("C")},"DISAPPROVED",{P("I")},"REJECT")', ""])
    cell_of["RD"] = r
    r += 1
    rates = [("False-rejection rate (approved not auto-approved)", "=B{FN}/(B{TP}+B{FN})", "=C{FN}/(C{TP}+C{FN})", "lower is better; with 25 photos one photo is 4 points"),
             ("False-approval rate (disapproved auto-approved)", "=B{FP}/(B{TN}+B{FP})", "=C{FP}/(C{TN}+C{FP})", "lower is better; the costly error"),
             ("Approval precision TP / (TP + FP)", "=IF(B{TP}+B{FP}=0,\"n/a\",B{TP}/(B{TP}+B{FP}))", "=IF(C{TP}+C{FP}=0,\"n/a\",C{TP}/(C{TP}+C{FP}))", "of the photos the engine let through, the share humans had approved"),
             ("Rejection precision RD / (RA + RD)", "=IF(B{RA}+B{RD}=0,\"n/a\",B{RD}/(B{RA}+B{RD}))", "=IF(C{RA}+C{RD}=0,\"n/a\",C{RD}/(C{RA}+C{RD}))", "of the explicit rejections, the share humans had disapproved"),
             ("Rejection recall RD / disapproved", "=B{RD}/(B{TN}+B{FP})", "=C{RD}/(C{TN}+C{FP})", "of the disapproved photos, the share explicitly rejected"),
             ("Agreement with the historical decision", "=(B{TP}+B{TN})/(B{TP}+B{FN}+B{TN}+B{FP})", "=(C{TP}+C{TN})/(C{TP}+C{FN}+C{TN}+C{FP})", "not ordinary accuracy: a human-review counts as 'not approved'")]
    for label, f1, f2, note in rates:
        put(sm, r, [label, f1.format(**cell_of), f2.format(**cell_of), note])
        for c in (2, 3):
            sm.cell(r, c).number_format = "0%"
        r += 1
    h1, h2 = res["v1"], res["v2"]
    put(sm, r, ["Human reviews", f'=COUNTIFS({P("F")},"HUMAN_REVIEW")', f'=COUNTIFS({P("I")},"HUMAN_REVIEW")', ""])
    r += 1
    put(sm, r, ["Mean failed hard gates: approved photos", round(h1["mean_failed_gates"]["approved"], 2), round(h2["mean_failed_gates"]["approved"], 2), "computed by the script"])
    r += 1
    put(sm, r, ["Mean failed hard gates: disapproved photos", round(h1["mean_failed_gates"]["disapproved"], 2), round(h2["mean_failed_gates"]["disapproved"], 2), "computed by the script"])
    r += 1
    a1, a2 = h1["auc_failed_gates"], h2["auc_failed_gates"]
    put(sm, r, ["Separation (AUC) of 'fewer failed gates' with 95% range", f"{a1['auc']:.2f} ({a1['ci_low']:.2f}-{a1['ci_high']:.2f})", f"{a2['auc']:.2f} ({a2['ci_low']:.2f}-{a2['ci_high']:.2f})",
                "0.5 = no separation; both ranges include 0.5 means neither separates the classes reliably"])
    r += 1
    put(sm, r, ["Separation (AUC) of the quality score", round(h1["auc_quality_score"], 2), round(h2["auc_quality_score"], 2), "uncalibrated score"])
    r += 1
    put(sm, r, ["Mean quality score: approved / disapproved", f"{h1['mean_quality']['approved']:.1f} / {h1['mean_quality']['disapproved']:.1f}", f"{h2['mean_quality']['approved']:.1f} / {h2['mean_quality']['disapproved']:.1f}", ""])
    r += 2
    sm.cell(r, 1, "Counts above are Excel formulas over the Photos sheet. Rates and the other tables are not statistically meaningful at n=25 per class: a one-photo change moves a rate by 4 points.").font = Font(name=FONT, italic=True, size=10)
    sm.cell(r + 1, 1, "DEVELOPMENT COMPARISON: the V2 changes were chosen after looking at these same 50 photos' gate statistics, so V2 is not independently validated on them.").font = Font(name=FONT, bold=True, size=10, color="C00000")

    # ================================================================== Stages
    st = wb.create_sheet("Ablation stages")
    head(st, 1, ["Stage (cumulative, specification order)", "Answers replayed", "Auto-approved of 25 approved (TP)", "Not auto-approved (FN)", "Disapproved stopped (TN)", "Disapproved auto-approved (FP)",
                 "Explicit REJECT", "Human review", "Approved photos rescued vs previous stage (#)", "NEW false approvals vs previous stage (#)", "Approved photos newly blocked (#)"],
         [58, 18, 14, 14, 14, 14, 11, 11, 30, 30, 24], 58)
    row = 2
    for tag, key_ in (("V1's stored answers", "stages_v1_answers"), ("V2's stored answers", "stages_v2_answers")):
        for s in res[key_]:
            c = s["headline"]["confusion"]
            put(st, row, [s["stage"], tag, c["TP"], c["FN"], c["TN"], c["FP"], s["headline"]["ai_rejections"], s["headline"]["ai_human_review"],
                          ", ".join(map(str, s["rescued_approved"])) or "-", ", ".join(map(str, s["new_false_approvals"])) or "-", ", ".join(map(str, s["newly_rejected_approved"])) or "-"], wrap_cols=(1,))
            row += 1
        row += 1
    st.cell(row, 1, "A replay re-decides stored answers in Python. It is exact for changes of hard/soft, applicability and aggregation, and it cannot show a change of wording (E3 and most of E5), because the answers were given to the old wording.").font = Font(name=FONT, italic=True, size=10)
    row += 2
    head(st, row, ["LEAVE ONE OUT on V2's stored answers: the full V2 with this one change switched off, then on", "", "Approved photos it rescues (#)", "NEW false approvals it causes (#)", "Approved photos it newly blocks (#)", "Disapproved photos it newly stops (#)"], None, 44)
    row += 1
    for r_ in res["leave_one_out_v2_answers"]:
        put(st, row, [f"{r_['change']}  {r_['name']}", "", ", ".join(map(str, r_["rescued_approved"])) or "-", ", ".join(map(str, r_["new_false_approvals"])) or "-",
                      ", ".join(map(str, r_["newly_rejected_approved"])) or "-", ", ".join(map(str, r_["newly_stopped_disapproved"])) or "-"], wrap_cols=(1,))
        row += 1
    st.cell(row + 1, 1, "Leave-one-out uses ONE set of answers, so it contains no run-to-run noise. Changes that act through wording (V2-02, V2-06) have no effect here by construction: their effect is in the V2 answers themselves.").font = Font(name=FONT, italic=True, size=10)

    # ================================================================== Gate level
    gs = wb.create_sheet("Gate level")
    head(gs, 1, ["Gate", "What it checks (V2 wording)", "Hard? V1 -> V2", "V1 applicable (approved / disapproved)", "V1 failed (approved / disapproved)", "V1 fail rate approved", "V1 fail rate disapproved",
                 "V1 false rejections with this gate / correct rejections", "V1 rejected for this gate ALONE (approved / disapproved)", "V2 applicable (approved / disapproved)", "V2 failed (approved / disapproved)",
                 "V2 fail rate approved", "V2 fail rate disapproved", "V2 false rejections with this gate / correct rejections", "V2 rejected for this gate ALONE (approved / disapproved)"],
         [7, 60, 11, 15, 15, 11, 11, 18, 18, 15, 15, 11, 11, 18, 18], 70)
    prof2 = Rulebook().resolve("layered_proposed_v2")
    text = {g.gate_id: prof2.criteria[g.criterion_id].description_operational for g in prof2.gates}
    g1 = {g["gate"]: g for g in res["gate_table_v1"]}
    g2 = {g["gate"]: g for g in res["gate_table_v2"]}
    row = 2
    for gid in sorted(set(g1) | set(g2)):
        def part(g):
            if g is None:
                return ["-"] * 6
            a, d = g["approved"], g["disapproved"]
            f = lambda x: x["fail_rate_of_applicable"] if x["fail_rate_of_applicable"] is not None else "n/a"  # noqa: E731
            return [f"{a['applicable']} / {d['applicable']}", f"{a['no']} / {d['no']}", f(a), f(d), f"{g['false_rejections_with_gate']} / {g['correct_rejections_with_gate']}",
                    f"{g['sole_cause_false_rejections']} / {g['sole_cause_correct_rejections']}"]
        p1, p2 = part(g1.get(gid)), part(g2.get(gid))
        hh = lambda g: ("hard" if g and g["hard"] else "soft")  # noqa: E731
        put(gs, row, [gid, text.get(gid, ""), f"{hh(g1.get(gid))} -> {hh(g2.get(gid))}"] + p1[:2] + p1[2:4] + p1[4:] + p2[:2] + p2[2:4] + p2[4:], wrap_cols=(2,))
        for c in (6, 7, 12, 13):
            gs.cell(row, c).number_format = "0%"
        gs.row_dimensions[row].height = 48
        row += 1
    gs.cell(row + 1, 1, "Denominators are the photos on which the gate APPLIES (not NOT_APPLICABLE). A soft gate still shows what the model answered (failed = NO) but cannot reject. 'Alone' = the photo was rejected and this was its only failed gate: those are the photos that stop being rejected if only this gate is relaxed.").font = Font(name=FONT, italic=True, size=10)

    # ================================================================== Upper bound
    ub = wb.create_sheet("Upper bound (gate removed)")
    head(ub, 1, ["Gates deleted outright (NOT a proposal)", "Answers", "Approved auto-approved (of 25)", "Disapproved auto-approved = false approvals (of 25)"], [44, 14, 22, 28], 44)
    for k, u in enumerate(res["upper_bounds"], 2):
        put(ub, k, [u["removed"], u["answers"], u["TP"], u["FP"]])
    ub.cell(len(res["upper_bounds"]) + 3, 1, "Shows how many wrongly rejected photos a gate (or family) is responsible for, and what it costs in false approvals. Even with every gate removed, the quality, publication and human-review steps still run.").font = Font(name=FONT, italic=True, size=10)

    # ================================================================== Changes
    ch = wb.create_sheet("V2 changes")
    head(ch, 1, ["Change", "Applies to", "Original rule", "Proposed rule", "Previous decision effect", "New decision effect", "Why", "Pilot-3 evidence", "Approval status"], [8, 26, 52, 52, 28, 28, 52, 52, 26], 32)
    rb = Rulebook()
    p = rb.profiles["layered_proposed_v2"]
    seen = {}
    for key_ in ("overrides", "gate_overrides"):
        for tgt, ov in p[key_].items():
            pc = ov["policy_change"]
            seen.setdefault(pc["change_id"], {"pc": pc, "t": []})["t"].append(tgt)
    for k, cid in enumerate(sorted(seen), 2):
        pc = seen[cid]["pc"]
        put(ch, k, [cid, ", ".join(seen[cid]["t"]), pc["original_rule"], pc["proposed_rule"], pc["previous_effect"], pc["new_effect"], pc["reason"], pc["evidence"], pc["status"]], wrap_cols=tuple(range(2, 10)))
        ch.row_dimensions[k].height = 120

    # ================================================================== Notes
    nt = wb.create_sheet("Notes")
    nt.column_dimensions["A"].width = 150
    notes = [
        "V1 = profile layered_proposed_v1 (rulebook 1.1.0, prompt hash 11427c79d96c569a) as run in pilot 3. V2 = profile layered_proposed_v2 (rulebook 2.0.0-proposed, prompt photo_evaluation_prompt_v2). Same model (gemini-3.8-flash), same 50 photographs, one run each.",
        "Engine decision: APPROVED only when the engine auto-approves (APPROVE or APPROVED_PENDING_PUBLICATION_CHECKS). REJECT, HUMAN_REVIEW and RESELECT count as DISAPPROVED, because the photo is not let through automatically.",
        "Positive = approved. True positive: both approved. True negative: both not approved. False positive: humans disapproved but the engine auto-approved (the costly error). False negative: humans approved but the engine did not auto-approve.",
        "V1 vs V2 'as run' differs in three ways at once: the changed policy, the changed wording the model was asked, and ordinary run-to-run variation of the model. The 'Ablation stages' sheet separates the policy-only part exactly; it cannot separate wording from noise.",
        "The V2 changes were chosen after looking at gate statistics of these same 50 photos. This is a development comparison. It is not evidence that V2 will do as well on photographs it has not seen.",
        "Every V2 change touches a mandatory requirement and is PROPOSED_PENDING_NGO_APPROVAL. Five of the seven (V2-01 gaze, V2-02 face scope, V2-03 banner folds, V2-04 banner visibility, V2-05 back view) change or re-propose something the NGO decided or withdrew on 2026-11-02 (see RULEBOOK_CHANGELOG.md 1.1.0). V2-06 and V2-07 are wording and applicability changes the NGO has not ruled on. None is NGO policy.",
        "Human tags are the taggers' reading of why a photo was disapproved; they are not ground truth, only some disapproved photos have tags, and the historical decisions themselves may be inconsistent.",
        "The photos are the resolution-normalised copies sent to Gemini (long side 1280 px), shown as thumbnails. Only the opaque ID, the photo and a fixed instruction were sent: no label, tag or file name.",
        "Costs in the reports use placeholder prices. This file contains photos of real people: share only with authorised people.",
    ]
    for k, t in enumerate(notes, 1):
        c = nt.cell(k, 1, t)
        c.font, c.alignment = Font(name=FONT, size=10), Alignment(wrap_text=True, vertical="top")
        nt.row_dimensions[k].height = 32
    wb.save(out_path)
