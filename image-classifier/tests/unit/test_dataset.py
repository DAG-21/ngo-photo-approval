from __future__ import annotations

import io

import pandas as pd
import pytest
from PIL import Image

from ngo_photo_approval.config import SplitConfig
from ngo_photo_approval.data.audit import EXCL_CONFLICT, EXCL_PENDING, EXCL_UNREADABLE, EXCL_UNSUPPORTED, build_audit
from ngo_photo_approval.data.duplicates import build_duplicate_report
from ngo_photo_approval.data.inventory import build_inventory, make_image_id, parse_filename
from ngo_photo_approval.data.preprocessing import open_image, prepare_for_model
from ngo_photo_approval.data.splitting import make_splits, verify_no_leakage, verify_no_pending
from ngo_photo_approval.exceptions import DataError, InvalidImageError, LeakageError

from ..conftest import make_image


@pytest.fixture
def inventory(project):
    return build_inventory(project.data_dir, project.dataset, workers=2)


def audit_of(inventory, project):
    dup, cluster = build_duplicate_report(inventory, project.dataset.near_duplicate_hamming_threshold)
    return build_audit(inventory, dup, cluster, project.dataset), dup


# ------------------------------------------------------------------ labels / inventory


def test_label_mapping(inventory):
    by_folder = inventory.groupby("folder")["label"].first().to_dict()
    assert by_folder == {"approved": "approved", "disapproved": "not_approved", "pending-approval": "pending"}
    assert (inventory.loc[inventory.folder == "pending-approval", "label"] == "pending").all()
    assert "not_approved" not in set(inventory.loc[inventory.folder == "pending-approval", "label"])


def test_ids_are_unique_even_for_same_filename_in_two_folders(project, tmp_path):
    make_image(project.data_dir / "approved" / "same-1.jpg", seed=901)
    make_image(project.data_dir / "disapproved" / "same-1.jpg", seed=902)
    inv = build_inventory(project.data_dir, project.dataset, workers=2)
    ids = inv[inv.rel_path.str.endswith("same-1.jpg")]["image_id"]
    assert len(ids) == 2 and ids.nunique() == 2
    assert make_image_id("approved/x.jpg") != make_image_id("disapproved/x.jpg")


def test_filename_parsing():
    assert parse_filename("arham_aahar-10 (56)") == {"program": "arham_aahar", "submission_idx": 10, "copy_idx": 56}
    assert parse_filename("jeevdaya-1 - 2026-10-29T162333.988")["program"] == "jeevdaya"
    assert parse_filename("Photo from Someone(3)")["program"] is None


def test_corrupt_and_non_image_detected(inventory):
    corrupt = inventory[inventory.rel_path.str.endswith("prog_b-9 (1).jpg")].iloc[0]
    assert not corrupt.readable and "decode error" in corrupt.error
    assert "/" not in corrupt.error and "\\" not in corrupt.error  # no local paths leak
    video = inventory[inventory.rel_path.str.endswith(".mp4")].iloc[0]
    assert video.media_kind == "unsupported_media" and not video.readable


def test_inventory_missing_directory_raises(project, tmp_path):
    with pytest.raises(DataError):
        build_inventory(tmp_path / "nope", project.dataset)


def test_inventory_does_not_modify_dataset(project, inventory):
    before = {p: p.stat().st_mtime_ns for p in project.data_dir.rglob("*") if p.is_file()}
    build_inventory(project.data_dir, project.dataset, workers=2)
    after = {p: p.stat().st_mtime_ns for p in project.data_dir.rglob("*") if p.is_file()}
    assert before == after


def test_pending_excluded_from_training(inventory, project):
    audit, _ = audit_of(inventory, project)
    pending = audit[audit.label == "pending"]
    assert len(pending) == 4 and not pending.eligible_for_training.any()
    assert set(pending.exclusion_reason) == {EXCL_PENDING}
    reasons = dict(zip(audit.rel_path, audit.exclusion_reason, strict=True))
    assert reasons["disapproved/prog_b-9 (1).jpg"] == EXCL_UNREADABLE
    assert reasons["pending-approval/clip-1.mp4"] in {EXCL_PENDING, EXCL_UNSUPPORTED}


# ------------------------------------------------------------------ duplicates


def test_exact_duplicates_detected(inventory, project):
    _, dup = audit_of(inventory, project)
    row = dup[dup.rel_path == "approved/prog_a-9 (1).jpg"].iloc[0]
    cluster = dup[dup.cluster_id == row.cluster_id]
    assert set(cluster.rel_path) == {"approved/prog_a-1 (1).jpg", "approved/prog_a-9 (1).jpg"}
    assert cluster.is_exact_duplicate.all() and not cluster.label_conflict.any()


def test_near_duplicate_detected_but_not_deleted(project):
    src = project.data_dir / "approved" / "prog_a-1 (1).jpg"
    im = Image.open(src)
    buf = io.BytesIO()
    im.resize((300, 225)).save(buf, "JPEG", quality=60)  # recompressed + resized copy
    (project.data_dir / "approved" / "prog_a-8 (1).jpg").write_bytes(buf.getvalue())
    inv = build_inventory(project.data_dir, project.dataset, workers=2)
    dup, _ = build_duplicate_report(inv, 6)
    near = dup[dup.rel_path == "approved/prog_a-8 (1).jpg"]
    assert len(near) == 1 and near.iloc[0].is_near_duplicate
    assert (inv.rel_path == "approved/prog_a-8 (1).jpg").sum() == 1  # still in the manifest


def test_distinct_images_not_grouped(inventory, project):
    _, dup = audit_of(inventory, project)
    assert len(dup) == 2  # only the one exact pair; random-noise images are all distinct


def test_exact_label_conflict_excluded(project):
    src = project.data_dir / "approved" / "prog_a-2 (4).jpg"
    (project.data_dir / "disapproved" / "prog_b-7 (1).jpg").write_bytes(src.read_bytes())
    inv = build_inventory(project.data_dir, project.dataset, workers=2)
    audit, dup = audit_of(inv, project)
    conflicted = audit[audit.exclusion_reason == EXCL_CONFLICT]
    assert set(conflicted.rel_path) == {"approved/prog_a-2 (4).jpg", "disapproved/prog_b-7 (1).jpg"}
    assert dup.exact_label_conflict.any()


# ------------------------------------------------------------------ splitting


def _split(inventory, project, seed=42):
    audit, _ = audit_of(inventory, project)
    return make_splits(inventory, audit, SplitConfig(seed=seed, train=0.5, validation=0.25, test=0.25))


def test_splits_reproducible(inventory, project):
    a, ra = _split(inventory, project)
    b, rb = _split(inventory, project)
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])
    assert ra == rb and ra["seed"] == 42


def test_splits_have_no_pending_and_no_overlap(inventory, project):
    splits, report = _split(inventory, project)
    verify_no_pending(splits)
    verify_no_leakage(splits)
    all_ids = pd.concat(splits.values())["image_id"]
    assert all_ids.is_unique
    assert set(pd.concat(splits.values())["label"]) <= {"approved", "not_approved"}
    assert report["n_eligible"] == len(all_ids)


def test_groups_never_cross_splits(inventory, project):
    splits, _ = _split(inventory, project)
    seen = {}
    for name, df in splits.items():
        for g in df.group_id:
            assert seen.setdefault(g, name) == name
    # duplicate pair shares a group
    dup_rows = pd.concat(splits.values())
    g = dup_rows[dup_rows.rel_path.isin(["approved/prog_a-1 (1).jpg", "approved/prog_a-9 (1).jpg"])].group_id
    assert g.nunique() == 1


def test_filename_group_members_stay_together(inventory, project):
    splits, _ = _split(inventory, project)
    allrows = pd.concat(splits.values())
    for _, g in allrows.groupby(allrows.rel_path.str.extract(r"(prog_\w-\d+) ")[0]):
        assert g.split.nunique() == 1


def test_leakage_detectors_raise():
    a = pd.DataFrame({"image_id": ["1", "2"], "group_id": ["g1", "g2"], "sha256": ["x", "y"]})
    b = pd.DataFrame({"image_id": ["3"], "group_id": ["g2"], "sha256": ["z"]})
    with pytest.raises(LeakageError):
        verify_no_leakage({"train": a, "test": b})
    c = pd.DataFrame({"image_id": ["4"], "group_id": ["g9"], "sha256": ["x"]})  # same bytes
    with pytest.raises(LeakageError):
        verify_no_leakage({"train": a, "test": c})
    with pytest.raises(LeakageError):
        verify_no_pending({"train": pd.DataFrame({"label": ["approved", "pending"]})})


# ------------------------------------------------------------------ image loading


def test_missing_file_handled_safely(tmp_path, model_cfg):
    with pytest.raises(InvalidImageError):
        open_image(tmp_path / "ghost.jpg")
    with pytest.raises(InvalidImageError):
        prepare_for_model(tmp_path / "ghost.jpg", model_cfg.image_input)


def test_corrupt_file_raises(tmp_path, model_cfg):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image at all")
    with pytest.raises(InvalidImageError):
        open_image(bad)
    with pytest.raises(InvalidImageError):
        prepare_for_model(bad, model_cfg.image_input)


def test_prepare_passes_through_small_jpeg_untouched(tmp_path, model_cfg):
    p = make_image(tmp_path / "ok.jpg", size=(400, 300))
    prep = prepare_for_model(p, model_cfg.image_input)
    assert prep.data == p.read_bytes() and not prep.resized and prep.mime_type == "image/jpeg"


def test_prepare_downscales_only_oversized(tmp_path, model_cfg):
    model_cfg.image_input.max_side_px = 200
    p = make_image(tmp_path / "big.jpg", size=(800, 400))
    prep = prepare_for_model(p, model_cfg.image_input)
    assert prep.resized and max(prep.width, prep.height) == 200
    assert Image.open(p).size == (800, 400)  # original untouched


def test_heic_with_jpg_extension_is_converted(tmp_path, model_cfg):
    pytest.importorskip("pillow_heif")
    p = tmp_path / "phone.jpg"
    try:
        Image.new("RGB", (64, 48), (200, 30, 30)).save(p, format="HEIF")
    except Exception:  # pragma: no cover - encoder unavailable on this platform
        pytest.skip("HEIF encoder unavailable")
    prep = prepare_for_model(p, model_cfg.image_input)
    assert prep.mime_type == "image/jpeg" and prep.resized and prep.data[:2] == b"\xff\xd8"
    assert open_image(p).size == (64, 48)


def test_mpo_jpeg_variant_is_accepted_and_normalized(tmp_path, model_cfg):
    """Some cameras write MPO (a JPEG container variant); it must not be dropped as 'unsupported'."""
    p = tmp_path / "cam.jpg"
    first, second = Image.new("RGB", (64, 48), (5, 50, 150)), Image.new("RGB", (64, 48), (50, 5, 150))
    first.save(p, format="MPO", save_all=True, append_images=[second])
    assert Image.open(p).format == "MPO"
    prep = prepare_for_model(p, model_cfg.image_input, ["image/jpeg", "image/png", "image/webp"])
    assert prep.mime_type == "image/jpeg" and prep.data[:2] == b"\xff\xd8"
    from ngo_photo_approval.data.inventory import _EXT_FORMATS

    assert "MPO" in _EXT_FORMATS[".jpg"]


def test_normalization_gives_every_image_the_same_long_side_and_encoding(tmp_path, model_cfg):
    from ngo_photo_approval.data.preprocessing import normalize_for_tuning, prepare_for_tuned_model

    cfg = model_cfg.image_input
    cfg.tuning_normalize_long_side_px = 640
    sizes = {"huge_landscape": (4000, 3000), "small_portrait": (300, 450), "tall_panorama": (900, 2000), "tiny": (80, 60)}
    outs = {}
    for name, size in sizes.items():
        p = make_image(tmp_path / f"{name}.jpg", seed=hash(name) % 1000, size=size)
        before = p.read_bytes()
        prep = normalize_for_tuning(p, cfg)
        outs[name] = prep
        assert p.read_bytes() == before  # original untouched
        assert max(prep.width, prep.height) == 640  # scaled DOWN for big images and UP for small ones
        im = Image.open(io.BytesIO(prep.data))
        assert im.format == "JPEG" and im.size == (prep.width, prep.height) and not im.getexif()
        assert abs(prep.width / prep.height - size[0] / size[1]) < 0.02  # aspect ratio preserved
        assert prepare_for_tuned_model(p, cfg).data == prep.data  # the shared entry point applies it
    assert outs["huge_landscape"].width == 640 and outs["small_portrait"].height == 640
    cfg.tuning_normalize_long_side_px = None  # opt-out falls back to the normal pass-through
    p = tmp_path / "huge_landscape.jpg"
    assert prepare_for_tuned_model(p, cfg).data == p.read_bytes() or prepare_for_tuned_model(p, cfg).resized
