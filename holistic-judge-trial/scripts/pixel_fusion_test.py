"""Do plain pixel brightness measurements add anything to the embeddings model? (offline; no API call; test split never read)

    python scripts/pixel_fusion_test.py

Same data and protocol as the main project's embedding probe: the 3,388 unique train+validation photos, label = approved, 5-fold
cross-validation with every photo group kept in one fold (StratifiedGroupKFold), CLIP B/32 + DINOv2-base embeddings, standardised logistic
regression. Three repeats with different fold seeds.

Models compared (same folds):
  pixels      the 18 brightness measures alone
  embeddings  CLIP + DINOv2 (the baseline, about 0.77 AUC in the main project)
  stacked     the embeddings model's score (taken out-of-fold inside each training fold) plus the 18 brightness measures, in a second small
              logistic regression - so the brightness measures are not drowned out by 2,048 embedding numbers
Reports ROC-AUC, approved photos accepted at 5% false approvals, and a paired group-bootstrap range for stacked minus embeddings.
"""

import sys

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from brightness_analysis import luminance, metrics
from judge_trial.config import TRIAL_ROOT, ensure_main_importable, load_trial_config
from judge_trial.data import image_path

OUT = TRIAL_ROOT / "analysis"
PIX = ["mean", "p05", "p50", "p95", "std", "frac_dark", "frac_bright", "cell_min", "cell_max", "cell_range", "center_cell",
       "center_minus_mean", "lower_mean", "lower_p10", "lower_frac_dark", "top_minus_lower", "dark_tile_share", "median_tile_std"]


def pixel_table(cfg, ids) -> pd.DataFrame:
    cache = OUT / "brightness_metrics_all.csv"
    have = pd.read_csv(cache).drop_duplicates("image_id").set_index("image_id") if cache.exists() else pd.DataFrame(columns=PIX)
    rows = {}
    for i in ids:
        if i in have.index and not have.loc[i, PIX].isna().any():
            rows[i] = have.loc[i, PIX].astype(float).to_dict()
        else:
            rows[i] = metrics(luminance(image_path(cfg, i)))
    return pd.DataFrame.from_dict(rows, orient="index")[PIX]


def lr(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))


def recall_at_far(y, p, far=0.05):
    thr = np.sort(p[y == 0])[::-1][int(np.floor(far * (y == 0).sum()))]  # ties never push false approvals over the limit
    return float((p[y == 1] > thr).mean())


def run_once(Xe, Xp, y, g, seed, c_emb=0.002, c_pix=0.3):
    n = len(y)
    oof = {k: np.zeros(n) for k in ("pixels", "embeddings", "stacked")}
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xe, y, g):
        oof["pixels"][te] = lr(c_pix).fit(Xp[tr], y[tr]).decision_function(Xp[te])
        m = lr(c_emb).fit(Xe[tr], y[tr])
        oof["embeddings"][te] = m.decision_function(Xe[te])
        inner = np.zeros(len(tr))
        for itr, ite in StratifiedGroupKFold(4, shuffle=True, random_state=seed + 100).split(Xe[tr], y[tr], g[tr]):
            inner[ite] = lr(c_emb).fit(Xe[tr][itr], y[tr][itr]).decision_function(Xe[tr][ite])
        s2 = lr(c_pix).fit(np.column_stack([inner, Xp[tr]]), y[tr])
        oof["stacked"][te] = s2.decision_function(np.column_stack([oof["embeddings"][te], Xp[te]]))
    return oof


def main() -> int:
    cfg = load_trial_config()
    ensure_main_importable(cfg)
    from ngo_photo_approval.config import load_project_config
    from ngo_photo_approval.scoring.offline import embedding_cache_path, unique_trainval

    project = load_project_config()
    d = unique_trainval(project)
    embs = []
    for path in (embedding_cache_path(project), project.output_dir / "cache" / "embeddings_dinov2_base.npz"):
        z = np.load(path, allow_pickle=False)
        embs.append(dict(zip(z["ids"].tolist(), z["emb"], strict=True)))
    d = d[d.image_id.isin(set.intersection(*[set(e) for e in embs]))].reset_index(drop=True)
    Xe = np.hstack([np.stack([e[i] for i in d.image_id]) for e in embs])
    Xp = pixel_table(cfg, d.image_id).loc[d.image_id].to_numpy()
    y = (d.label == "approved").astype(int).to_numpy()
    g = d.group_id.to_numpy()
    print(f"{len(d)} unique train+validation photos: {int(y.sum())} approved / {int((1 - y).sum())} not approved; embeddings {Xe.shape[1]}, pixel measures {Xp.shape[1]}")

    res: dict[str, list] = {k: [] for k in ("pixels", "embeddings", "stacked")}
    rec: dict[str, list] = {k: [] for k in res}
    first = None
    for seed in (0, 1, 2):
        oof = run_once(Xe, Xp, y, g, seed)
        first = first or oof
        for k, v in oof.items():
            res[k].append(roc_auc_score(y, v))
            rec[k].append(recall_at_far(y, v))
    lines = ["# Do brightness measures add to the embeddings model?", "",
             f"{len(d)} unique train+validation photos ({int(y.sum())} approved / {int((1 - y).sum())} not approved), 5-fold group-aware CV, 3 repeats. "
             "Test split not used.", "", "| model | ROC-AUC (mean, sd over 3 repeats) | approved accepted at 5% false approvals |", "|---|---:|---:|"]
    for k, label in (("pixels", "brightness measures alone"), ("embeddings", "embeddings (CLIP + DINOv2)"), ("stacked", "embeddings + brightness measures")):
        lines.append(f"| {label} | {np.mean(res[k]):.3f} (sd {np.std(res[k]):.3f}) | {np.mean(rec[k]):.1%} |")
    # paired bootstrap over whole groups, first repeat
    rng = np.random.default_rng(0)
    ug = np.unique(g)
    idx = {u: np.flatnonzero(g == u) for u in ug}
    diffs = []
    for _ in range(300):
        pick = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        if len(np.unique(y[pick])) < 2:
            continue
        diffs.append(roc_auc_score(y[pick], first["stacked"][pick]) - roc_auc_score(y[pick], first["embeddings"][pick]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    lines += ["", f"Stacked minus embeddings, AUC difference (repeat 1, bootstrap over whole groups): {np.mean(res['stacked']) - np.mean(res['embeddings']):+.3f} "
              f"(95% range {lo:+.3f} to {hi:+.3f}).", ""]
    # which brightness measures matter in the stacked model, trained on everything (for reading only; not a performance figure)
    inner = np.zeros(len(y))
    for itr, ite in StratifiedGroupKFold(5, shuffle=True, random_state=7).split(Xe, y, g):
        inner[ite] = lr(0.002).fit(Xe[itr], y[itr]).decision_function(Xe[ite])
    m = lr(0.3).fit(np.column_stack([inner, Xp]), y)
    coef = pd.Series(m[-1].coef_[0], index=["embeddings_score", *PIX]).sort_values()
    lines += ["Largest weights in the second-stage model (standardised inputs; positive = more likely approved; reading aid only):", ""]
    lines += [f"- {k}: {v:+.2f}" for k, v in list(coef.head(4).items()) + list(coef.tail(4).items())]
    text = "\n".join(lines)
    (OUT / "fusion_report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
