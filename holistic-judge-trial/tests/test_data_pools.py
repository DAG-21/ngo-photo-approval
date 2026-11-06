from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from judge_trial import data
from judge_trial.data import build_pools, load_pool, related_mask, save_pools, stratified_sample


def synthetic_full() -> pd.DataFrame:
    """60 groups of 3 photos; every 5th group is not_approved; some photos are exact copies of each other."""
    rows = []
    for g in range(60):
        for k in range(3):
            n = g * 3 + k
            rows.append({"image_id": f"img_{n:04d}", "label": "not_approved" if g % 5 == 0 else "approved", "program": f"p{g % 4}",
                         "group_id": f"grp_{g:03d}", "sha256": f"sha_{n:04d}" if k < 2 else f"sha_{n - 1:04d}",  # k==2 copies k==1
                         "duplicate_cluster_id": f"dup_{g:03d}" if k == 2 else None,
                         "split": "validation" if g >= 50 else "train"})
    return pd.DataFrame(rows)


@pytest.fixture
def fake_main(monkeypatch):
    full = synthetic_full()
    unique = full.sort_values("image_id").drop_duplicates("sha256").reset_index(drop=True)
    monkeypatch.setattr(data, "load_main", lambda cfg: (None, full, unique))
    return full, unique


def test_related_mask_covers_group_cluster_and_copies(fake_main):
    full, unique = fake_main
    mask = related_mask(unique, full, {"img_0000"})
    assert set(unique[mask]["group_id"]) == {"grp_000"}  # the whole submission group goes, nothing else
    assert not related_mask(unique, full, set()).any()


def test_exact_copy_in_another_group_is_pulled_in():
    full = pd.DataFrame([
        {"image_id": "a", "group_id": "g1", "sha256": "S", "duplicate_cluster_id": None},
        {"image_id": "b", "group_id": "g2", "sha256": "S", "duplicate_cluster_id": None},   # same content, other group
        {"image_id": "c", "group_id": "g2", "sha256": "T", "duplicate_cluster_id": None},   # shares b's group
        {"image_id": "d", "group_id": "g3", "sha256": "U", "duplicate_cluster_id": None}])
    mask = related_mask(full, full, {"a"})
    assert set(full[mask]["image_id"]) == {"a", "b", "c"}  # b via content, c via b's group; d untouched


def test_a_giant_component_is_not_treated_as_one_occasion():
    rows = [{"image_id": f"big_{i}", "group_id": "giant", "sha256": f"s{i}", "duplicate_cluster_id": None} for i in range(40)]
    rows += [{"image_id": "small_a", "group_id": "tiny", "sha256": "t1", "duplicate_cluster_id": None},
             {"image_id": "small_b", "group_id": "tiny", "sha256": "t2", "duplicate_cluster_id": None}]
    full = pd.DataFrame(rows)
    mask = related_mask(full, full, {"big_0", "small_a"})
    assert set(full[mask]["image_id"]) == {"big_0", "small_a", "small_b"}  # the 39 other giant-group photos stay usable


def test_similar_mask_flags_only_near_identical_scenes():
    from judge_trial.data import similar_mask

    emb = {"anchor": np.array([1.0, 0.0]), "twin": np.array([0.99, 0.05]), "other": np.array([0.2, 1.0])}
    frame = pd.DataFrame({"image_id": ["twin", "other"]})
    assert similar_mask(frame, emb, {"anchor"}, 0.95).tolist() == [True, False]
    assert not similar_mask(frame, emb, set(), 0.95).any()


def test_pools_drop_photos_that_look_like_an_example(cfg, fake_main, monkeypatch):
    full, unique = fake_main
    cfg.pools.dev_size_per_class = cfg.pools.confirm_size_per_class = 50
    ids = unique["image_id"].tolist()
    emb = {i: np.array([0.0, 1.0]) for i in ids}  # everything far from the example ...
    emb["img_0000"] = np.array([1.0, 0.0])         # ... which is img_0000
    emb["img_0100"] = np.array([0.99, 0.02])       # ... except one near-identical scene
    monkeypatch.setattr(data, "load_pool_embeddings", lambda c: emb)
    pools = build_pools(cfg, {"img_0000"})
    assert all("img_0100" not in set(df.image_id) for df in pools.values())
    cfg.pools.example_similarity_max = None
    assert any("img_0100" in set(df.image_id) for df in build_pools(cfg, {"img_0000"}).values())


def test_sampling_is_seeded_stratified_and_capped(fake_main):
    _, unique = fake_main
    train = unique[unique.split == "train"]
    a, b = stratified_sample(train, 5, 1), stratified_sample(train, 5, 1)
    assert a.equals(b) and a.label.value_counts().to_dict() == {"approved": 5, "not_approved": 5}
    assert len(stratified_sample(train, 10_000, 1)) == len(train)  # never more than exist


def test_pools_exclude_examples_and_relatives_and_are_disjoint(cfg, fake_main):
    full, _ = fake_main
    cfg.pools.dev_size_per_class = 8
    cfg.pools.confirm_size_per_class = 8
    cfg.pools.example_similarity_max = None  # similarity filter is tested separately
    example_ids = {"img_0000", "img_0003"}  # groups 000 and 001
    pools = build_pools(cfg, example_ids)
    banned_groups = {"grp_000", "grp_001"}
    for name, df in pools.items():
        assert not set(df.group_id) & banned_groups, name
        assert not set(df.image_id) & example_ids, name
        assert df.sha256.is_unique, f"{name} counts a photo twice"
    assert not set(pools["dev"].group_id) & set(pools["confirm"].group_id)
    assert set(pools["dev"].split) == {"train"} and set(pools["confirm"].split) == {"train"}
    assert set(pools["validation"].split) == {"validation"}
    assert len(pools["dev"]) == 16
    # the synthetic data runs short of disapproved photos for a second full pool: it takes what exists, never more
    assert 0 < len(pools["confirm"]) <= 16 and set(pools["confirm"].label) == {"approved", "not_approved"}


@pytest.fixture
def tmp_pool_dir(cfg, tmp_path, monkeypatch):
    """Keep pool CSVs in a temp folder so tests never read or overwrite the real pools/."""
    monkeypatch.setattr(type(cfg), "pool_path", lambda self, name: tmp_path / "pools" / f"{name}.csv")


def test_load_pool_refuses_missing_and_unknown_pools(cfg, tmp_pool_dir):
    with pytest.raises(FileNotFoundError, match="build_pools"):
        load_pool(cfg, "dev")
    with pytest.raises(ValueError, match="unknown pool"):
        load_pool(cfg, "test")  # the test split has no pool at all


def test_load_pool_detects_an_example_inside_a_pool(cfg, fake_main, tmp_pool_dir):
    cfg.pools.example_similarity_max = None
    pools = build_pools(cfg, set())
    save_pools(cfg, pools)
    leaked = pools["dev"].image_id.iloc[0]
    with pytest.raises(ValueError, match="contains example photos"):
        load_pool(cfg, "dev", {leaked})
    assert len(load_pool(cfg, "dev", set())) == len(pools["dev"])
    assert np.isin(["image_id", "label", "group_id"], load_pool(cfg, "validation", set()).columns).all()
