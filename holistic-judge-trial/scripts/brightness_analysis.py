"""Do photos that humans tagged 'poor lighting' (R009) really differ in brightness from approved photos? (offline; no API call)

    python scripts/brightness_analysis.py

Groups (train + validation splits only; the test split is never read):
  R009   disapproved photos a person tagged with R009 in any of the three slots
  other  disapproved photos that were tagged, but not with R009
  appr   every unique approved photo
For each photo it measures plain pixel brightness: overall, in a 3x3 grid, the dark-tile share, and top-versus-bottom.
It does NOT locate people (no person detector is available offline), so 'the area around the people' is approximated by
the lower two thirds of the frame, where the activity usually is. Writes analysis/brightness_metrics.csv, report.md, figure.png.
"""

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from PIL import Image

from judge_trial import data
from judge_trial.config import TRIAL_ROOT, load_trial_config

OUT = TRIAL_ROOT / "analysis"
SIDE = 320


def luminance(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((SIDE, SIDE))
        a = np.asarray(im, dtype=np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]  # 0-255, display (gamma) scale


def metrics(y: np.ndarray) -> dict[str, float]:
    h, w = y.shape
    cells = np.array([[y[i * h // 3:(i + 1) * h // 3, j * w // 3:(j + 1) * w // 3].mean() for j in range(3)] for i in range(3)])
    tiles = [y[i:i + 32, j:j + 32] for i in range(0, h - 31, 32) for j in range(0, w - 31, 32)]
    tile_mean = np.array([t.mean() for t in tiles])
    tile_std = np.array([t.std() for t in tiles])
    top, rest = y[: h // 3].mean(), y[h // 3:].mean()
    lower = y[h // 3:]
    return {
        "mean": float(y.mean()), "p05": float(np.percentile(y, 5)), "p50": float(np.percentile(y, 50)), "p95": float(np.percentile(y, 95)),
        "std": float(y.std()), "frac_dark": float((y < 60).mean()), "frac_bright": float((y > 235).mean()),
        "cell_min": float(cells.min()), "cell_max": float(cells.max()), "cell_range": float(cells.max() - cells.min()),
        "center_cell": float(cells[1, 1]), "center_minus_mean": float(cells[1, 1] - y.mean()),
        "lower_mean": float(rest), "lower_p10": float(np.percentile(lower, 10)), "lower_frac_dark": float((lower < 60).mean()),
        "top_minus_lower": float(top - rest), "dark_tile_share": float((tile_mean < 70).mean()), "median_tile_std": float(np.median(tile_std)),
    }


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a random R009 photo has a higher value than a random approved one); 0.5 = no difference."""
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_ci(pos, neg, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> int:
    cfg = load_trial_config()
    _, _, unique = data.load_main(cfg)
    tags = pd.read_csv(cfg.main_root / "data" / "annotations" / "disapproval_reasons.csv")
    tags["has_r009"] = tags[["primary", "second", "third"]].eq("R009").any(axis=1)
    u = unique[unique.split.isin(["train", "validation"])].set_index("image_id")
    rows = []
    for t in tags.itertuples():
        if t.image_id in u.index:
            rows.append((t.image_id, "R009" if t.has_r009 else "other", u.loc[t.image_id, "program"]))
    for i, r in u[u.label == "approved"].iterrows():
        rows.append((i, "appr", r["program"]))
    df = pd.DataFrame(rows, columns=["image_id", "group", "program"])
    m = []
    for r in df.itertuples():
        p = data.image_path(cfg, r.image_id)
        m.append(metrics(luminance(p)) if p.exists() else {})
    df = pd.concat([df, pd.DataFrame(m)], axis=1).dropna(subset=["mean"])
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "brightness_metrics.csv", index=False)

    g = {k: df[df.group == k] for k in ("R009", "other", "appr")}
    cols = [c for c in df.columns if c not in ("image_id", "group", "program")]
    arham = {k: v[v.program == "arham_aahar"] for k, v in g.items()}
    lines = [f"# Brightness: photos tagged 'poor lighting' (R009) vs approved photos",
             f"R009-tagged {len(g['R009'])}, other-tagged disapproved {len(g['other'])}, approved {len(g['appr'])} (train+validation only). "
             f"Arham Aahar subset: {len(arham['R009'])} / {len(arham['other'])} / {len(arham['appr'])}.", "",
             "AUC = chance that an R009-tagged photo has a HIGHER value than an approved photo (0.5 = same; below 0.5 = the tagged photos are lower). "
             "95% range from a bootstrap. 'vs other' compares with disapproved photos tagged for something else.", "",
             "| measure | R009 median | approved median | AUC vs approved (range) | AUC within Arham Aahar | AUC vs other-tagged |", "|---|---:|---:|---:|---:|---:|"]
    res = []
    for c in cols:
        a = auc(g["R009"][c].values, g["appr"][c].values)
        lo, hi = boot_ci(g["R009"][c].values, g["appr"][c].values)
        ar = auc(arham["R009"][c].values, arham["appr"][c].values) if len(arham["R009"]) > 5 else np.nan
        ot = auc(g["R009"][c].values, g["other"][c].values) if len(g["other"]) > 5 else np.nan
        res.append((c, a, lo, hi, ar, ot))
        lines.append(f"| {c} | {g['R009'][c].median():.2f} | {g['appr'][c].median():.2f} | {a:.2f} ({lo:.2f}-{hi:.2f}) | {ar:.2f} | {ot:.2f} |")
    flag = [r for r in res if r[2] > 0.5 or r[3] < 0.5]
    lines += ["", "Measures whose 95% range excludes 0.5 (a difference that is unlikely to be chance): "
              + (", ".join(f"{r[0]} ({r[1]:.2f})" for r in flag) if flag else "none.")]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = ["mean", "p05", "lower_mean", "lower_p10", "dark_tile_share", "cell_min", "cell_range", "top_minus_lower"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, c in zip(axes.ravel(), show):
        data_ = [g["appr"][c], g["other"][c], g["R009"][c]]
        ax.boxplot(data_, tick_labels=[f"approved\n(n={len(data_[0])})", f"other tags\n(n={len(data_[1])})", f"R009\n(n={len(data_[2])})"], showfliers=False)
        ax.set_title(c)
    fig.suptitle("Pixel brightness (0-255): approved vs disapproved photos tagged for other reasons vs tagged 'poor lighting'")
    fig.tight_layout()
    fig.savefig(OUT / "figure.png", dpi=110)
    print(f"\nsaved {OUT / 'report.md'}, {OUT / 'brightness_metrics.csv'}, {OUT / 'figure.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
