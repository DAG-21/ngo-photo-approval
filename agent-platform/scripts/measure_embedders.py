"""Which local embedder finds the most useful similar photographs? A free, local measurement on the 50 pilot photographs (no model call, no network).

    python scripts/measure_embedders.py

For each embedder and each photograph, the other 49 photographs are ranked by similarity (leave-one-out: a photograph is never compared with itself).
We then ask: does the nearest neighbour (or the majority of the 3 nearest) have the same staff decision (approved / disapproved) as the photograph?
By chance that is 50% (25 approved, 25 disapproved). With 50 photographs the intervals are wide, so this can only rule embedders out, not prove one best.
Only aggregate numbers are printed; the labels come from the pilot's private key file and stay on this machine.

Caveat: photographs from the same event can look alike and share a decision, which flatters every embedder equally.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE / "src"), str(HERE.parent / "layered-engine" / "src")]

from ngo_agent.embeddings import make_embedder  # noqa: E402

ENGINE = HERE.parent / "layered-engine"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def main() -> int:
    key = {r["opaque_id"]: r["label"] for r in csv.DictReader((ENGINE / "pilot" / "key3_PRIVATE.csv").open(encoding="utf-8"))}
    paths = {r["image_id"]: Path(r["path"]) for r in csv.DictReader((ENGINE / "pilot" / "list3.csv").open(encoding="utf-8"))}
    ids = sorted(i for i in paths if i in key and paths[i].exists())
    y = np.array([key[i] == "approved" for i in ids])
    print(f"{len(ids)} photographs, {int(y.sum())} approved and {int((~y).sum())} disapproved by staff\n")
    for kind in ("thumbnail", "clip", "dinov2"):
        emb = make_embedder(kind)
        t0 = time.time()
        try:
            X = np.array([emb.embed(paths[i].read_bytes()) for i in ids], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            print(f"{emb.name}: unavailable ({type(exc).__name__}: {str(exc)[:120]})\n")
            continue
        sims = X @ X.T
        np.fill_diagonal(sims, -2.0)
        order = np.argsort(-sims, axis=1)
        top1 = order[:, 0]
        top3 = order[:, :3]
        hit1 = int((y[top1] == y).sum())
        hit3 = int(((y[top3].sum(axis=1) >= 2) == y).sum())
        nn_sim = sims[np.arange(len(ids)), top1]
        lo1, hi1 = wilson(hit1, len(ids))
        lo3, hi3 = wilson(hit3, len(ids))
        hi_sim = nn_sim >= np.median(nn_sim)
        print(f"{emb.name}  ({time.time() - t0:.0f}s to embed {len(ids)} photographs, {X.shape[1]} dimensions)")
        print(f"  nearest neighbour has the same decision: {hit1}/{len(ids)} = {hit1 / len(ids):.0%}  (95% interval {lo1:.0%}-{hi1:.0%}); chance is 50%")
        print(f"  majority of the 3 nearest has the same decision: {hit3}/{len(ids)} = {hit3 / len(ids):.0%}  ({lo3:.0%}-{hi3:.0%})")
        print(f"  nearest-neighbour similarity: min {nn_sim.min():.2f}, 25th pct {np.percentile(nn_sim, 25):.2f}, median {np.median(nn_sim):.2f}, 75th pct {np.percentile(nn_sim, 75):.2f}, max {nn_sim.max():.2f}")
        print(f"  agreement when the neighbour is closer than the median: {int((y[top1] == y)[hi_sim].sum())}/{int(hi_sim.sum())}; when farther: {int((y[top1] == y)[~hi_sim].sum())}/{int((~hi_sim).sum())}")
        print(f"  mean similarity to a random other photograph: {float(np.mean(np.where(sims > -2, sims, np.nan)[~np.eye(len(ids), dtype=bool)])):.2f}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
