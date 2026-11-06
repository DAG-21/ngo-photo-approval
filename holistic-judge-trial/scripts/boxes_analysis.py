"""Brightness around people and how much of the frame people / the activity fill, for photos tagged 'poor lighting' (R009) vs approved.

    python scripts/boxes_analysis.py

Offline: reads analysis/people_boxes.jsonl (Gemini's boxes, one per photo) and the photos. Writes analysis/boxes_report.md,
boxes_metrics.csv, boxes_figure.png and boxes_check.jpg (boxes drawn on 8 photos so the boxes can be checked by eye).
"""

import sys

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from brightness_analysis import auc, boot_ci, luminance
from judge_trial.config import TRIAL_ROOT, load_trial_config
from judge_trial.data import image_path
from judge_trial.people_boxes import box_metrics, norm_box, read_rows

OUT = TRIAL_ROOT / "analysis"
SIDE = 320


def main() -> int:
    cfg = load_trial_config()
    sample = pd.read_csv(TRIAL_ROOT / "ids" / "boxes_sample.csv")
    latest = {r["image_id"]: r for r in read_rows(OUT / "people_boxes.jsonl") if r.get("ok")}
    rows = []
    for r in sample.itertuples():
        if r.image_id in latest:
            rows.append({"image_id": r.image_id, "group": r.group, "program": r.program,
                         **box_metrics(luminance(image_path(cfg, r.image_id)), latest[r.image_id])})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "boxes_metrics.csv", index=False)
    g = {k: df[df.group == k] for k in ("R009", "appr")}
    arham = {k: v[v.program == "arham_aahar"] for k, v in g.items()}
    cols = ["n_people", "people_cover", "tallest_person", "median_person_h", "activity_cover", "y_global", "y_people", "y_people_p10",
            "y_head", "y_head_p10", "y_ring", "y_rest", "people_minus_global", "head_minus_global", "ring_minus_global",
            "people_minus_rest", "ring_minus_rest"]
    lines = [f"# People-area brightness and coverage: R009-tagged ({len(g['R009'])}) vs approved ({len(g['appr'])}) photos",
             f"Photos with at least one person box: R009 {int((g['R009'].n_people > 0).sum())}, approved {int((g['appr'].n_people > 0).sum())}. "
             f"Arham Aahar subset: {len(arham['R009'])} / {len(arham['appr'])}.", "",
             "Brightness is 0-255. 'people' = inside the person boxes, 'ring' = the band 30% of a box's size around each box, 'rest' = everything outside that, "
             "'head' = the top quarter of each person box. AUC = chance that an R009 photo has a HIGHER value than an approved one (0.5 = same).", "",
             "| measure | R009 median | approved median | AUC (95% range) | AUC within Arham Aahar |", "|---|---:|---:|---:|---:|"]
    res = []
    for c in cols:
        a, b = g["R009"][c].dropna().values, g["appr"][c].dropna().values
        if len(a) < 8 or len(b) < 8:
            continue
        v = auc(a, b)
        lo, hi = boot_ci(a, b)
        aa, bb = arham["R009"][c].dropna().values, arham["appr"][c].dropna().values
        ar = auc(aa, bb) if len(aa) >= 8 and len(bb) >= 8 else np.nan
        res.append((c, v, lo, hi))
        lines.append(f"| {c} | {np.median(a):.2f} | {np.median(b):.2f} | {v:.2f} ({lo:.2f}-{hi:.2f}) | {ar:.2f} |")
    clear = [r for r in res if r[2] > 0.5 or r[3] < 0.5]
    lines += ["", "Measures whose 95% range excludes 0.5: " + (", ".join(f"{r[0]} ({r[1]:.2f})" for r in clear) if clear else "none.")]
    (OUT / "boxes_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = ["y_global", "y_people", "y_head", "y_ring", "people_minus_rest", "head_minus_global", "tallest_person", "activity_cover"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, c in zip(axes.ravel(), show):
        d = [g["appr"][c].dropna(), g["R009"][c].dropna()]
        ax.boxplot(d, tick_labels=[f"approved\n(n={len(d[0])})", f"R009\n(n={len(d[1])})"], showfliers=False)
        ax.set_title(c)
    fig.suptitle("Brightness around people (0-255) and coverage: approved vs tagged 'poor lighting'")
    fig.tight_layout()
    fig.savefig(OUT / "boxes_figure.png", dpi=110)

    # boxes drawn on photos, to check the detector by eye (4 R009 + 4 approved)
    pick = list(g["R009"].image_id[:4]) + list(g["appr"].image_id[:4])
    T = 330
    sheet = Image.new("RGB", (T * 4, (T + 18) * 2), "white")
    d = ImageDraw.Draw(sheet)
    for n, i in enumerate(pick):
        im = Image.open(image_path(cfg, i)).convert("RGB")
        im.thumbnail((T, T))
        w, h = im.size
        dr = ImageDraw.Draw(im)
        for p in latest[i]["people"]:
            y0, x0, y1, x1 = norm_box(p["box"], h, w)
            dr.rectangle([x0, y0, x1, y1], outline={"volunteer": "lime", "beneficiary": "red", "other": "yellow"}[p["role"]], width=2)
        ab = latest[i]["activity_box"]
        if ab[2] > ab[0]:
            y0, x0, y1, x1 = norm_box(ab, h, w)
            dr.rectangle([x0, y0, x1, y1], outline="cyan", width=2)
        x, y = (n % 4) * T, (n // 4) * (T + 18)
        sheet.paste(im, (x, y + 18))
        d.text((x + 3, y + 3), f"{'R009' if n < 4 else 'approved'} {i[4:12]}  green=volunteer red=beneficiary yellow=other cyan=activity", fill="black")
    sheet.save(OUT / "boxes_check.jpg", quality=85)
    print(f"\nsaved to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
