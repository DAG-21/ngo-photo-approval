"""Signal check: do the (resolution-normalized) photos contain anything that predicts the human decision?

Embeds each unique train+validation photo with a pretrained vision model, then fits a plain logistic regression to the
approve / not-approve label with group-aware cross-validation. Several models can be compared; their numbers can also be
concatenated. The TEST split is never embedded. Local only: nothing is sent anywhere.

    python scripts/embedding_probe.py embed    --model dinov2_base [--bench]   # compute + cache (resumable); --bench = speed test
    python scripts/embedding_probe.py evaluate --models dinov2_base [siglip_base ...]   # several => concatenated features
    python scripts/embedding_probe.py compare                                   # table of every evaluated model

Models: clip_b32 (baseline), dinov2_small, dinov2_base, siglip_base, clip_l14. Input is the already-normalized copy
(data/derived/tuning_images_norm<N>/), squashed to 224x224 with no center crop so framing and cropping stay visible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ngo_photo_approval.config import load_model_config, load_project_config, setup_logging
from ngo_photo_approval.scoring.offline import embedding_cache_path, unique_trainval

CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Spec:
    name: str
    hf_id: str
    kind: str  # clip | siglip | dinov2
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


SPECS = {s.name: s for s in (
    Spec("clip_b32", "openai/clip-vit-base-patch32", "clip", CLIP_MEAN, CLIP_STD),
    Spec("dinov2_small", "facebook/dinov2-small", "dinov2", IMNET_MEAN, IMNET_STD),
    Spec("dinov2_base", "facebook/dinov2-base", "dinov2", IMNET_MEAN, IMNET_STD),
    Spec("siglip_base", "google/siglip-base-patch16-224", "siglip", (0.5,) * 3, (0.5,) * 3),
    Spec("clip_l14", "openai/clip-vit-large-patch14", "clip", CLIP_MEAN, CLIP_STD),
)}


def cache_path(project, name: str) -> Path:
    return embedding_cache_path(project) if name == "clip_b32" else project.output_dir / "cache" / f"embeddings_{name}.npz"


def tensor_for(spec: Spec, path: Path) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB").resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
    return ((a - np.array(spec.mean, dtype=np.float32)) / np.array(spec.std, dtype=np.float32)).transpose(2, 0, 1)


def load_model(spec: Spec):
    from transformers import AutoModel, CLIPModel

    return (CLIPModel if spec.kind == "clip" else AutoModel).from_pretrained(spec.hf_id).eval()


def embed_batch(model, spec: Spec, x):
    import torch

    if spec.kind in ("clip", "siglip"):
        f = model.get_image_features(pixel_values=x)
    else:  # dinov2: class token + mean of patch tokens (a common strong choice for linear probes)
        h = model(pixel_values=x).last_hidden_state
        f = torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=-1)
    f = f if isinstance(f, torch.Tensor) else f.pooler_output
    return torch.nn.functional.normalize(f, dim=-1)


def cmd_embed(project, model_cfg, args) -> int:
    import torch

    spec = SPECS[args.model]
    norm = model_cfg.image_input.tuning_normalize_long_side_px
    img_dir = project.output_dir / "derived" / f"tuning_images_norm{norm}"
    cache = cache_path(project, spec.name)
    cache.parent.mkdir(parents=True, exist_ok=True)
    d = unique_trainval(project)
    done: dict[str, np.ndarray] = {}
    if cache.exists() and not args.bench:
        z = np.load(cache, allow_pickle=False)
        done = dict(zip(z["ids"].tolist(), z["emb"], strict=True))
    todo = d[~d.image_id.isin(done)]
    ids = todo.image_id.tolist()[: args.bench_n] if args.bench else todo.image_id.tolist()
    print(f"{spec.name} ({spec.hf_id}): unique train+val photos {len(d)} | cached {len(done)} | to embed {len(ids)}", flush=True)
    if not ids:
        return 0
    torch.set_num_threads(max(1, args.threads))
    t_load = time.time()
    model = load_model(spec)
    print(f"model loaded in {time.time() - t_load:.0f}s", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex, torch.no_grad():
        for i in range(0, len(ids), args.batch):
            chunk = ids[i:i + args.batch]
            arrs = list(ex.map(lambda iid: tensor_for(spec, img_dir / f"{iid}.jpg"), chunk))
            feats = embed_batch(model, spec, torch.from_numpy(np.stack(arrs))).numpy()
            done.update(zip(chunk, feats, strict=True))
            n = min(i + args.batch, len(ids))
            rate = n / (time.time() - t0)
            if not args.bench and ((i // args.batch) % 5 == 0 or n == len(ids)):
                np.savez(cache, ids=np.array(list(done)), emb=np.stack(list(done.values())))
                print(f"embedded {n}/{len(ids)}  ({rate:.2f} img/s, ~{(len(ids) - n) / rate / 60:.0f} min left)", flush=True)
    rate = len(ids) / (time.time() - t0)
    if args.bench:
        total = len(d) / rate / 60
        print(f"BENCH {spec.name}: {rate:.2f} img/s on this CPU -> about {total:.0f} min for all {len(d)} photos "
              f"(feature size {feats.shape[1]})")
        return 0
    np.savez(cache, ids=np.array(list(done)), emb=np.stack(list(done.values())))
    print("done ->", cache)
    return 0


def cv_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, c: float, seed: int = 0) -> np.ndarray:
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000)).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def far_recall(y: np.ndarray, p: np.ndarray, max_far: float = 0.05) -> tuple[float, float]:
    """Best approval recall at false-approval rate <= max_far (threshold from these OOF scores: optimistic)."""
    thr = np.quantile(p[y == 0], 1 - max_far) if (y == 0).any() else 1.0
    return float((p[y == 1] > thr).mean()), float(thr)


def load_features(project, names: list[str], d: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    embs = []
    for n in names:
        p = cache_path(project, n)
        if not p.exists():
            raise SystemExit(f"no embeddings for {n}: run `embedding_probe.py embed --model {n}` first")
        z = np.load(p, allow_pickle=False)
        embs.append(dict(zip(z["ids"].tolist(), z["emb"], strict=True)))
    d = d[d.image_id.isin(set.intersection(*[set(e) for e in embs]))].reset_index(drop=True)
    return d, np.hstack([np.stack([e[i] for i in d.image_id]) for e in embs])


def cmd_evaluate(project, model_cfg, args) -> int:
    names = args.models
    label = "+".join(names)
    d, X = load_features(project, names, unique_trainval(project))
    y = (d.label == "approved").astype(int).to_numpy()
    g = d.group_id.to_numpy()
    print(f"[{label}] photos: {len(d)} unique (train+validation only): {int(y.sum())} approved / {int((1 - y).sum())} not approved; "
          f"feature size {X.shape[1]}")

    res: dict = {"models": names, "n": int(len(d)), "feature_size": int(X.shape[1]), "by_C": {}}
    best = None
    for c in (0.0005, 0.002, 0.01, 0.05):
        oof = cv_scores(X, y, g, c)
        auc = float(roc_auc_score(y, oof))
        res["by_C"][str(c)] = {"roc_auc": auc, "pr_auc": float(average_precision_score(y, oof))}
        if best is None or auc > best[0]:
            best = (auc, c, oof)
    auc, c, oof = best
    rec, _ = far_recall(y, oof)
    res.update(best_C=c, roc_auc=auc, pr_auc=res["by_C"][str(c)]["pr_auc"], recall_at_far_5pct=rec,
               approved_base_rate=float(y.mean()))

    rng = np.random.default_rng(0)  # chance level: shuffle labels at the GROUP level
    first = d.groupby("group_id").label.first()
    ug = np.unique(g)
    null = []
    for s in range(3):
        perm = dict(zip(ug, rng.permutation(ug), strict=True))
        ys = np.array([int(first[perm[x]] == "approved") for x in g])
        null.append(roc_auc_score(ys, cv_scores(X, ys, g, c, seed=s + 1)))
    res["shuffled_label_auc_mean"], res["shuffled_label_auc_sd"] = float(np.mean(null)), float(np.std(null))

    mp = d.megapixels.astype(float).to_numpy()
    res["resolution_only_auc"] = float(roc_auc_score(y, mp))
    res["auc_within_resolution_band"] = {}
    for name, m in {"<2 MP": mp < 2, "2-8 MP": (mp >= 2) & (mp < 8), ">=8 MP": mp >= 8}.items():
        if m.sum() >= 30 and len(np.unique(y[m])) == 2 and (y[m] == 0).sum() >= 10:
            res["auc_within_resolution_band"][name] = {
                "n": int(m.sum()), "not_approved": int((y[m] == 0).sum()), "roc_auc": float(roc_auc_score(y[m], oof[m]))}
        else:
            res["auc_within_resolution_band"][name] = {"n": int(m.sum()), "note": "too few not-approved photos"}

    out = project.output_dir / "results" / "embedding_probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{label}_cv.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    pd.DataFrame({"image_id": d.image_id, "label": d.label, "oof_p_approved": oof}).to_csv(out / f"{label}_oof.csv", index=False)

    print(f"  cross-validated ROC-AUC (grouped 5-fold), best C={c}: {auc:.3f}   PR-AUC {res['pr_auc']:.3f} (base rate {y.mean():.2f})")
    print("  all C tried: " + ", ".join(f"C={k}: {v['roc_auc']:.3f}" for k, v in res["by_C"].items()))
    print(f"  chance level (labels shuffled by group): {res['shuffled_label_auc_mean']:.3f} +/- {res['shuffled_label_auc_sd']:.3f}")
    print(f"  approved recall at 5% false approvals: {rec:.1%}   | resolution alone: {res['resolution_only_auc']:.3f}")
    for k, v in res["auc_within_resolution_band"].items():
        print(f"    inside {k:<8} n={v['n']:<5} " + (f"AUC={v['roc_auc']:.3f}" if "roc_auc" in v else v["note"]))
    return 0


def cmd_compare(project, model_cfg, args) -> int:
    rows = []
    for p in sorted((project.output_dir / "results" / "embedding_probe").glob("*_cv.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        bands = r.get("auc_within_resolution_band", {})
        rows.append({"features": "+".join(r.get("models", [p.stem[:-3]])), "dims": r.get("feature_size", 512),
                     "photos": r["n"], "ROC-AUC": r["roc_auc"], "PR-AUC": r["pr_auc"],
                     "recall@5%FAR": r["recall_at_far_5pct"],
                     "min band AUC": min((v["roc_auc"] for v in bands.values() if "roc_auc" in v), default=float("nan"))})
    df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--model", choices=sorted(SPECS), default="clip_b32")
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--threads", type=int, default=4)
    e.add_argument("--bench", action="store_true", help="time a small sample, save nothing")
    e.add_argument("--bench-n", type=int, default=32)
    v = sub.add_parser("evaluate")
    v.add_argument("--models", nargs="+", choices=sorted(SPECS), default=["clip_b32"])
    sub.add_parser("compare")
    args = ap.parse_args()
    project = load_project_config()
    setup_logging(project.logging["level"])
    model_cfg = load_model_config()
    return {"embed": cmd_embed, "evaluate": cmd_evaluate, "compare": cmd_compare}[args.command](project, model_cfg, args)


if __name__ == "__main__":
    sys.exit(main())
