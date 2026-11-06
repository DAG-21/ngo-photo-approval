"""Choosing the worked examples that go into the prompt.

Examples come from the TRAIN split only. For each class the photos are clustered on their CLIP embeddings (already
computed by the main project) and the photo closest to each cluster centre is taken, so the examples cover the
different kinds of photo instead of repeating one. Approved and disapproved examples are interleaved in a seeded
random order so position does not signal the label. Reasons are only attached if a reasons file supplies them. Disapproved examples may instead be fixed by hand
(``examples.fixed_not_approved``), e.g. photos whose reason a person has tagged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TrialConfig, ensure_main_importable
from .data import image_path, load_main


def select_examples(cfg: TrialConfig) -> list[dict]:
    from sklearn.cluster import KMeans

    ensure_main_importable(cfg)
    from ngo_photo_approval.scoring.offline import load_embeddings

    project, _, unique = load_main(cfg)
    emb = load_embeddings(project)
    pool = unique[(unique["split"] == "train") & unique["label"].isin(["approved", "not_approved"])]
    pool = pool[pool["image_id"].isin(emb)].sort_values("image_id").reset_index(drop=True)
    picks: list[dict] = []
    for label, k in (("approved", cfg.examples.k_approved), ("not_approved", cfg.examples.k_not_approved)):
        g = pool[pool["label"] == label].reset_index(drop=True)
        if label == "not_approved" and cfg.examples.fixed_not_approved:
            fixed = list(cfg.examples.fixed_not_approved)
            bad = [i for i in fixed if i not in set(g["image_id"])]
            if bad:
                raise ValueError(f"fixed_not_approved photos are not disapproved train photos with embeddings: {bad}")
            for r in g.set_index("image_id").loc[fixed].reset_index().itertuples():
                picks.append({"image_id": r.image_id, "label": label, "program": r.program, "group_id": r.group_id,
                              "sha256": r.sha256, "method": "fixed (chosen by hand from the tagged photos)", "reason": None})
            continue
        if len(g) < k:
            raise ValueError(f"only {len(g)} {label} candidates for {k} examples")
        x = np.stack([emb[i] for i in g["image_id"]]).astype("float64")
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        km = KMeans(n_clusters=k, n_init=10, random_state=cfg.examples.seed).fit(x)
        for c in range(k):
            members = np.flatnonzero(km.labels_ == c)
            best = members[np.argmin(np.linalg.norm(x[members] - km.cluster_centers_[c], axis=1))]
            r = g.iloc[best]
            picks.append({"image_id": r.image_id, "label": label, "program": r.program, "group_id": r.group_id,
                          "sha256": r.sha256, "method": f"kmeans-medoid(k={k}, seed={cfg.examples.seed})",
                          "reason": None})
    order = np.random.RandomState(cfg.examples.seed).permutation(len(picks))
    return [picks[i] for i in order]


def save_examples(cfg: TrialConfig, picks: list[dict]) -> Path:
    cfg.examples_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.examples_path.write_text(json.dumps(picks, indent=2), encoding="utf-8")
    return cfg.examples_path


def load_examples(cfg: TrialConfig) -> list[dict]:
    """Saved examples with reasons filled in from ``examples.reasons_file`` when one is configured."""
    if not cfg.examples_path.exists():
        return []
    picks = json.loads(cfg.examples_path.read_text(encoding="utf-8"))
    if cfg.examples.reasons_file:
        rf = Path(cfg.examples.reasons_file)
        rf = rf if rf.is_absolute() else cfg.examples_path.parent / rf
        reasons = pd.read_csv(rf).set_index("image_id")["reason"].dropna().to_dict()
        for p in picks:
            if p["image_id"] in reasons:
                p["reason"] = str(reasons[p["image_id"]]).strip()
    return picks


def contact_sheet(cfg: TrialConfig, picks: list[dict], out: Path, thumb: int = 300) -> Path:
    """One image with every example (label + number) so the choice can be reviewed by eye."""
    from PIL import Image, ImageDraw

    cols = 4
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 22)), "white")
    d = ImageDraw.Draw(sheet)
    for n, p in enumerate(picks):
        with Image.open(image_path(cfg, p["image_id"])) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb, thumb))
            x, y = (n % cols) * thumb, (n // cols) * (thumb + 22)
            sheet.paste(im, (x, y + 22))
        d.text((x + 4, y + 4), f"#{n + 1} {p['label']} ({p['program']})", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out
