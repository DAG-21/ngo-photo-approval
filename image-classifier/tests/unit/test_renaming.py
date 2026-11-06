from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from ngo_photo_approval.data.audit import build_audit
from ngo_photo_approval.data.duplicates import build_duplicate_report
from ngo_photo_approval.data.inventory import build_inventory, make_image_id
from ngo_photo_approval.data.renaming import (
    apply_renames,
    legacy_lookup,
    load_map,
    natural_key,
    patch_derived_tables,
    pending,
    plan_new_entries,
    revert_renames,
    verify,
    write_map,
)
from ngo_photo_approval.exceptions import DataError

from ..conftest import make_image

CLASSES = ["approved", "disapproved", "pending-approval"]


def snapshot(root: Path) -> dict[str, str]:
    """relative path -> sha256 of every file."""
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_natural_order():
    names = ["x-10 (1).jpg", "x-2 (1).jpg", "x-2 (10).jpg", "X-1.jpg"]
    assert sorted(names, key=natural_key) == ["X-1.jpg", "x-2 (1).jpg", "x-2 (10).jpg", "x-10 (1).jpg"]


def test_plan_names_and_counts_per_class(dataset_dir):
    plan = plan_new_entries(dataset_dir, CLASSES, [])
    by_class: dict[str, list] = {}
    for e in plan:
        by_class.setdefault(e.class_folder, []).append(e)
    assert {c: len(v) for c, v in by_class.items()} == {"approved": 13, "disapproved": 9, "pending-approval": 4}
    for c, entries in by_class.items():
        assert [e.index for e in entries] == list(range(1, len(entries) + 1))  # 1..count, no gaps
        assert all(e.new_rel.rsplit("/", 1)[1].startswith(f"{c}_") for e in entries)
    assert next(e for e in plan if e.old_rel.endswith(".mp4")).new_rel.endswith(".mp4")  # extension preserved
    assert all(Path(e.new_rel).parent == Path(e.old_rel).parent for e in plan)  # renamed in place, never moved


def test_apply_preserves_every_byte_and_is_revertible(dataset_dir):
    before = snapshot(dataset_dir)
    plan = plan_new_entries(dataset_dir, CLASSES, [])
    assert apply_renames(dataset_dir, plan) == len(plan)
    after = snapshot(dataset_dir)
    assert sorted(after.values()) == sorted(before.values())  # same contents, nothing lost or duplicated
    assert not set(before) & set(after)  # every file got a new name
    assert verify(dataset_dir, plan) == {"renamed_ok": len(plan), "still_old_name": 0, "missing": 0, "size_mismatch": 0}

    assert revert_renames(dataset_dir, plan) == len(plan)
    assert snapshot(dataset_dir) == before  # exact original state restored


def test_never_overwrites_an_unrelated_file(dataset_dir):
    (dataset_dir / "approved" / "approved_3.jpg").write_bytes(b"x")  # a class file that is itself renamed: no loss
    before = snapshot(dataset_dir)
    plan = plan_new_entries(dataset_dir, CLASSES, [])
    apply_renames(dataset_dir, plan)
    assert sorted(snapshot(dataset_dir).values()) == sorted(before.values())

    fresh = make_image(dataset_dir / "approved" / "fresh-9.jpg", seed=991)
    more = plan_new_entries(dataset_dir, CLASSES, plan)
    (dataset_dir / more[0].new_rel).write_bytes(b"appeared after planning")  # a stranger takes the target name
    with pytest.raises(DataError, match="refusing to overwrite"):
        apply_renames(dataset_dir, more)
    assert fresh.exists() and (dataset_dir / more[0].new_rel).read_bytes() == b"appeared after planning"


def test_name_chains_are_resolved_without_clobbering(tmp_path):
    """Files already named like targets (e.g. from a half-finished earlier rename) must not overwrite each other."""
    root = tmp_path / "raw"
    for name, seed in (("aaa-1.jpg", 1), ("aaa-2.jpg", 2), ("approved_2.jpg", 3), ("approved_1.jpg", 4)):
        make_image(root / "approved" / name, seed=seed)
    before = snapshot(root)
    plan = plan_new_entries(root, ["approved"], [])
    assert sorted(e.new_rel for e in plan) == [f"approved/approved_{i}.jpg" for i in range(1, 5)]
    assert apply_renames(root, plan) == 4  # includes targets that were occupied by not-yet-renamed files
    assert sorted(snapshot(root).values()) == sorted(before.values())  # nothing lost
    assert sorted(snapshot(root)) == [f"approved/approved_{i}.jpg" for i in range(1, 5)]
    assert not any(p.name.endswith(".__renaming__") for p in root.rglob("*"))
    assert revert_renames(root, plan) == 4
    assert snapshot(root) == before  # chains also revert exactly


def test_crash_between_phases_is_resumed(tmp_path):
    root = tmp_path / "raw"
    for name, seed in (("aaa-1.jpg", 1), ("aaa-2.jpg", 2), ("approved_2.jpg", 3)):
        make_image(root / "approved" / name, seed=seed)
    before = snapshot(root)
    plan = plan_new_entries(root, ["approved"], [])
    # simulate a crash after phase 1: every chained file parked at its temp name
    for e in plan:
        if (root / e.old_rel).exists():
            (root / e.old_rel).rename(root / (e.new_rel + ".__renaming__"))
    assert not (root / "approved" / "approved_1.jpg").exists()
    todo = pending(root, plan)
    assert len(todo) == 3  # parked files still count as pending
    apply_renames(root, todo)
    assert sorted(snapshot(root).values()) == sorted(before.values())
    assert pending(root, plan) == []


def test_detects_file_modified_after_planning(dataset_dir):
    plan = plan_new_entries(dataset_dir, CLASSES, [])
    (dataset_dir / plan[0].old_rel).write_bytes(b"changed")
    with pytest.raises(DataError, match="changed since the plan"):
        apply_renames(dataset_dir, plan[:1])


def test_resume_after_partial_run_and_cumulative_numbering(dataset_dir, tmp_path):
    map_path = tmp_path / "rename_map.csv"
    plan = plan_new_entries(dataset_dir, CLASSES, [])
    write_map(map_path, plan)  # journal first
    apply_renames(dataset_dir, plan[:5])  # simulated crash after 5 renames
    entries = load_map(map_path)
    todo = pending(dataset_dir, entries)
    assert len(todo) == len(plan) - 5
    apply_renames(dataset_dir, todo)
    assert pending(dataset_dir, entries) == []

    # new photos arrive later: only they are named, numbering continues per class
    make_image(dataset_dir / "approved" / "fresh-1.jpg", seed=777)
    make_image(dataset_dir / "disapproved" / "fresh-2.jpg", seed=778)
    more = plan_new_entries(dataset_dir, CLASSES, entries)
    assert sorted(e.new_rel for e in more) == ["approved/approved_14.jpg", "disapproved/disapproved_10.jpg"]


def test_map_file_roundtrip_and_legacy_lookup(dataset_dir, tmp_path):
    plan = plan_new_entries(dataset_dir, CLASSES, [])
    path = tmp_path / "m.csv"
    write_map(path, plan)
    assert load_map(path) == plan
    row = next(csv.DictReader(open(path, encoding="utf-8")))
    assert row["legacy_image_id"] == make_image_id(row["old_rel_path"])
    assert legacy_lookup(plan)[plan[0].new_rel] == plan[0].old_rel


def test_inventory_after_rename_keeps_ids_and_program_metadata(project, dataset_dir):
    before = build_inventory(dataset_dir, project.dataset, workers=2)
    plan = plan_new_entries(dataset_dir, CLASSES, [])
    apply_renames(dataset_dir, plan)

    after_naive = build_inventory(dataset_dir, project.dataset, workers=2)
    assert set(after_naive.image_id).isdisjoint(before.image_id)  # without the map, ids would silently change

    after = build_inventory(dataset_dir, project.dataset, cache=before, workers=2, legacy=legacy_lookup(plan))
    assert sorted(after.image_id) == sorted(before.image_id)  # ids stable
    cols = ["image_id", "program", "submission_idx", "filename_group", "label", "sha256", "readable"]
    a = after.set_index("image_id")[cols[1:]].sort_index()
    b = before.set_index("image_id")[cols[1:]].sort_index()
    pd.testing.assert_frame_equal(a, b)  # program/group/hash/readability identical
    assert after.rel_path.str.match(r"^(approved|disapproved|pending-approval)/.*_\d+\.(jpg|mp4)$").all()
    # splitting inputs (groups) are therefore unchanged
    dup_b, cl_b = build_duplicate_report(before, 4)
    dup_a, cl_a = build_duplicate_report(after, 4)
    assert set(build_audit(before, dup_b, cl_b, project.dataset).eligible_for_training.index) == set(
        build_audit(after, dup_a, cl_a, project.dataset).eligible_for_training.index)


def test_patch_derived_tables_updates_paths_only_and_backs_up(dataset_dir, tmp_path):
    out = tmp_path / "out"
    (out / "manifests").mkdir(parents=True)
    (out / "splits").mkdir()
    (out / "tuning").mkdir()
    plan = plan_new_entries(dataset_dir, CLASSES, [])
    e = plan[0]
    pd.DataFrame({"image_id": ["i1", "i2"], "rel_path": [e.old_rel, "other/x.jpg"], "width": ["12.0", "7"]}).to_csv(
        out / "manifests" / "image_manifest.csv", index=False)
    pd.DataFrame({"image_id": ["i1"], "rel_path": [e.old_rel], "label": ["approved"]}).to_csv(out / "splits" / "train.csv", index=False)
    pd.DataFrame({"image_id": ["i1"], "local_path": [str(dataset_dir / e.old_rel)], "derived_copy": ["False"]}).to_csv(
        out / "tuning" / "upload_plan.csv", index=False)

    stats = patch_derived_tables(out, dataset_dir, plan)
    assert stats["image_manifest.csv"] == 1 and stats["train.csv"] == 1 and stats["upload_plan.csv"] == 1
    m = pd.read_csv(out / "manifests" / "image_manifest.csv", dtype=str)
    assert list(m.rel_path) == [e.new_rel, "other/x.jpg"] and list(m.width) == ["12.0", "7"]  # untouched cells preserved
    assert pd.read_csv(out / "tuning" / "upload_plan.csv", dtype=str).local_path[0] == str(dataset_dir / e.new_rel)
    assert any((out / "manifests").glob("backup_before_rename_*/image_manifest.csv"))

    patch_derived_tables(out, dataset_dir, plan, reverse=True)
    assert pd.read_csv(out / "splits" / "train.csv", dtype=str).rel_path[0] == e.old_rel
