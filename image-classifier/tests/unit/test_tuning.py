from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from ngo_photo_approval.config import SplitConfig
from ngo_photo_approval.data.audit import build_audit
from ngo_photo_approval.data.duplicates import build_duplicate_report
from ngo_photo_approval.data.inventory import build_inventory
from ngo_photo_approval.data.splitting import make_splits
from ngo_photo_approval.exceptions import AuthorizationError, ConfigError, DataError, TuningError
from ngo_photo_approval.tuning.dataset_builder import (
    TRACK_A_LABEL_TO_TARGET,
    build_tuning_files,
    downsample_majority,
    make_record,
    validate_jsonl_file,
    validate_record,
)
from ngo_photo_approval.tuning.job_manager import (
    cancel_job,
    idempotency_key,
    poll_job,
    run_preflight,
    submit_job,
    upload_to_gcs,
)
from ngo_photo_approval.tuning.model_registry import ModelRegistry

from ..conftest import make_image


def rec(uri="gs://b/p/images/x.jpg", mime="image/jpeg", target='{"decision": "approved"}'):
    return make_record(uri, mime, "q?", target, "sys", "MEDIA_RESOLUTION_HIGH")


@pytest.fixture
def prepared(project):
    """Inventory -> audit -> splits written to project.output_dir/splits."""
    inv = build_inventory(project.data_dir, project.dataset, workers=2)
    dup, cluster = build_duplicate_report(inv, 4)
    audit = build_audit(inv, dup, cluster, project.dataset)
    splits, _ = make_splits(inv, audit, SplitConfig(seed=7, train=0.5, validation=0.25, test=0.25))
    out = project.output_dir / "splits"
    out.mkdir(parents=True)
    for k, df in splits.items():
        df.to_csv(out / f"{k}.csv", index=False)
    return splits


# ------------------------------------------------------------------ record schema


def test_record_matches_documented_schema(model_cfg):
    r = rec()
    assert set(r) == {"systemInstruction", "contents", "generationConfig"}
    user, model = r["contents"]
    assert user["role"] == "user" and model["role"] == "model"
    assert user["parts"][0] == {"fileData": {"mimeType": "image/jpeg", "fileUri": "gs://b/p/images/x.jpg"}}
    assert set(user["parts"][1]) == {"text"}
    assert validate_record(r, model_cfg, "A") == []


@pytest.mark.parametrize(("mutate", "needle"), [
    (lambda r: r["contents"][0]["parts"][0]["fileData"].update(fileUri="https://x/y.jpg"), "gs://"),
    (lambda r: r["contents"][0]["parts"][0]["fileData"].update(mimeType="image/heic"), "not allowed"),
    (lambda r: r["contents"][0]["parts"].pop(0), "no image"),
    (lambda r: r["contents"].pop(), "contents"),
    (lambda r: r["contents"][-1]["parts"][0].update(text="not json"), "valid JSON"),
    (lambda r: r["contents"][-1]["parts"][0].update(text='{"decision": "waiting"}'), "decision label"),
    (lambda r: r["contents"][0]["parts"][1].update(text="   "), "empty text"),
    (lambda r: r["contents"][0].update(role="model"), "roles"),
])
def test_invalid_records_are_caught(model_cfg, mutate, needle):
    r = rec()
    mutate(r)
    assert any(needle in p for p in validate_record(r, model_cfg, "A")), validate_record(r, model_cfg, "A")


def test_image_count_limit_enforced(model_cfg):
    r = rec()
    fd = r["contents"][0]["parts"][0]
    r["contents"][0]["parts"] = [dict(fd) for _ in range(31)] + [r["contents"][0]["parts"][-1]]
    assert any("exceed" in p for p in validate_record(r, model_cfg, "A"))


def test_track_b_target_must_cover_all_rules(model_cfg):
    t = json.dumps({"checks": [{"rule_id": "R001", "status": "PASS", "evidence": ""}]})
    assert validate_record(rec(target=t), model_cfg, "B", rule_ids=["R001"]) == []
    assert any("every active rule" in p for p in validate_record(rec(target=t), model_cfg, "B", rule_ids=["R001", "R002"]))


def test_label_to_target_mapping_never_includes_pending():
    assert set(TRACK_A_LABEL_TO_TARGET) == {"approved", "not_approved"}
    assert json.loads(TRACK_A_LABEL_TO_TARGET["not_approved"]) == {"decision": "not_approved"}


# ------------------------------------------------------------------ builder


def test_builder_dry_run_exports_valid_files_without_test_images(project, model_cfg, rule_set, prepared):
    rep = build_tuning_files(project, model_cfg, rule_set, track="A", workers=2)
    out = project.output_dir / "tuning"
    assert rep.dry_run and rep.uris_are_placeholders and any("placeholder" in w for w in rep.warnings)
    test_ids = set(prepared["test"].image_id)
    used = set(pd.read_csv(out / "upload_plan.csv").image_id)
    assert used and not (used & test_ids)
    for name in ("train", "validation"):
        lines = (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(prepared[name]) - sum(1 for s in rep.skipped if s["split"] == name)
        assert validate_jsonl_file(out / f"{name}.jsonl", model_cfg, "A")["invalid"] == 0
    label_of = pd.concat(prepared.values()).set_index("image_id")["label"].to_dict()
    for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines():
        uri = json.loads(line)["contents"][0]["parts"][0]["fileData"]["fileUri"]
        iid = uri.rsplit("/", 1)[1].split(".")[0]
        assert json.loads(json.loads(line)["contents"][1]["parts"][0]["text"])["decision"] == label_of[iid]
        assert "prog_" not in uri  # original file names never leak into URIs
    assert (out / "export_report.json").exists()


def test_builder_uses_configured_bucket_when_set(project, model_cfg, rule_set, prepared):
    project.cloud.gcs_bucket = "ngo-bucket"
    rep = build_tuning_files(project, model_cfg, rule_set, workers=2)
    assert not rep.uris_are_placeholders
    plan = pd.read_csv(project.output_dir / "tuning" / "upload_plan.csv")
    assert plan.gcs_uri.str.startswith("gs://ngo-bucket/ngo-photo-approval/images/").all()


def test_builder_reports_missing_and_unreadable_files(project, model_cfg, rule_set, prepared):
    victim = prepared["train"].iloc[0]
    (project.data_dir / victim.rel_path).unlink()
    rep = build_tuning_files(project, model_cfg, rule_set, workers=2)
    skipped = {s["image_id"]: s["reason"] for s in rep.skipped}
    assert victim.image_id in skipped and "not found" in skipped[victim.image_id]
    assert (project.output_dir / "tuning" / "skipped_examples.csv").exists()


def test_builder_converts_heic_to_separate_derived_copy(project, model_cfg, rule_set, prepared):
    pytest.importorskip("pillow_heif")
    victim = prepared["train"].iloc[0]
    path = project.data_dir / victim.rel_path
    try:
        Image.new("RGB", (80, 60), (10, 200, 10)).save(path, format="HEIF")
    except Exception:  # pragma: no cover
        pytest.skip("HEIF encoder unavailable")
    original = path.read_bytes()
    rep = build_tuning_files(project, model_cfg, rule_set, workers=2)
    assert rep.derived_images >= 1
    assert path.read_bytes() == original  # raw dataset untouched
    plan = pd.read_csv(project.output_dir / "tuning" / "upload_plan.csv").set_index("image_id")
    row = plan.loc[victim.image_id]
    assert row.derived_copy and row.local_path.endswith(".jpg") and str(project.data_dir) not in row.local_path


def test_track_b_requires_verified_annotations(project, model_cfg, rule_set, prepared, tmp_path):
    with pytest.raises(DataError, match="verified"):
        build_tuning_files(project, model_cfg, rule_set, track="B", workers=2)
    ids = list(prepared["train"].image_id[:2])
    rows = []
    for i, iid in enumerate(ids):
        for rid in ("R001", "R002", "R003", "R004"):
            rows.append(dict(image_id=iid, rule_id=rid, status="PASS", verified=(i == 0), evidence="note"))
    ann = tmp_path / "ann.csv"
    pd.DataFrame(rows).to_csv(ann, index=False)
    rep = build_tuning_files(project, model_cfg, rule_set, track="B", annotations_path=ann, workers=2)
    assert rep.files["train"]["examples"] == 1  # only the verified, fully annotated image
    assert any("missing verified annotation" in s["reason"] for s in rep.skipped)
    # unverified suggestions (e.g. VLM pre-labels) never become training targets


def test_track_b_rejects_invalid_annotation_status(project, model_cfg, rule_set, prepared, tmp_path):
    ann = tmp_path / "ann.csv"
    pd.DataFrame([dict(image_id="x", rule_id="R001", status="MAYBE", verified=True)]).to_csv(ann, index=False)
    with pytest.raises(DataError):
        build_tuning_files(project, model_cfg, rule_set, track="B", annotations_path=ann)


def test_builder_requires_splits(project, model_cfg, rule_set):
    with pytest.raises(DataError, match="split manifests"):
        build_tuning_files(project, model_cfg, rule_set)


def test_downsample_is_seeded_and_train_only():
    df = pd.DataFrame({"image_id": [f"i{i}" for i in range(30)], "label": ["approved"] * 20 + ["not_approved"] * 10})
    a, b = downsample_majority(df, 1), downsample_majority(df, 1)
    pd.testing.assert_frame_equal(a, b)
    assert a.label.value_counts().to_dict() == {"approved": 10, "not_approved": 10}


# ------------------------------------------------------------------ job manager (mocked provider)


class FakeTunings:
    def __init__(self, fail_submit=False, states=None, error=None):
        self.fail_submit, self.error = fail_submit, error
        self.states = list(states or ["JOB_STATE_SUCCEEDED"])
        self.tune_calls, self.cancelled = [], []

    def tune(self, **kw):
        self.tune_calls.append(kw)
        if self.fail_submit:
            raise RuntimeError("quota exceeded")
        return SimpleNamespace(name="projects/p/locations/us/tuningJobs/1", state=SimpleNamespace(name="JOB_STATE_PENDING"))

    def get(self, name):
        st = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return SimpleNamespace(
            name=name, state=SimpleNamespace(name=st), error=self.error,
            tuned_model=SimpleNamespace(model="projects/p/models/m1@1", endpoint="projects/p/locations/us/endpoints/e1"))

    def cancel(self, name):
        self.cancelled.append(name)


class FakeClient:
    def __init__(self, **kw):
        self.tunings = FakeTunings(**kw)
        self.models = SimpleNamespace(get=lambda model: SimpleNamespace(name=model))


@pytest.fixture
def ready(project, model_cfg, rule_set, prepared):
    project.cloud.gcs_bucket = "ngo-bucket"
    project.cloud.project_id = "proj"
    build_tuning_files(project, model_cfg, rule_set, workers=2)
    return project.output_dir / "tuning"


URIS = {"train": "gs://ngo-bucket/x/train.jsonl", "validation": "gs://ngo-bucket/x/validation.jsonl"}


def authorize(project):
    project.cloud.allow_cloud_upload = True
    project.cloud.allow_paid_tuning = True


def test_preflight_blocks_by_default_and_names_every_gap(project, model_cfg, ready):
    project.cloud.project_id = None
    project.cloud.gcs_bucket = None
    rep = run_preflight(project, model_cfg, ready, "A", check_credentials=False)
    failed = {c.name for c in rep.checks if not c.ok and c.blocking}
    assert not rep.can_submit
    assert {"project_id", "bucket", "authorize_upload", "authorize_paid_tuning"} <= failed
    assert "NOT READY" in rep.render()


def test_preflight_rejects_placeholder_uris(project, model_cfg, rule_set, prepared):
    authorize(project)
    project.cloud.project_id = "proj"
    build_tuning_files(project, model_cfg, rule_set, workers=2)  # no bucket -> placeholder URIs
    project.cloud.gcs_bucket = "ngo-bucket"
    rep = run_preflight(project, model_cfg, project.output_dir / "tuning", "A", check_credentials=False)
    assert any(c.name == "validate_train.jsonl" and not c.ok for c in rep.checks)


def test_preflight_ok_when_everything_present(project, model_cfg, ready):
    authorize(project)
    rep = run_preflight(project, model_cfg, ready, "A", check_credentials=False, client=FakeClient())
    assert rep.can_submit, rep.render()


def test_submit_requires_explicit_flag_and_authorization(project, model_cfg, ready, tmp_path):
    reg = ModelRegistry(tmp_path / "reg.json")
    client = FakeClient()
    with pytest.raises(AuthorizationError, match="--submit-cloud-job"):
        submit_job(project, model_cfg, ready, "A", URIS, reg, submit=False, client=client)
    with pytest.raises(AuthorizationError, match="NGO_ALLOW_PAID_TUNING"):
        submit_job(project, model_cfg, ready, "A", URIS, reg, submit=True, client=client)
    assert client.tunings.tune_calls == [] and reg.jobs() == []


def test_submit_blocked_when_preflight_fails(project, model_cfg, ready, tmp_path):
    project.cloud.allow_paid_tuning = True  # but upload not authorized / etc.
    reg = ModelRegistry(tmp_path / "reg.json")
    client = FakeClient()
    with pytest.raises(TuningError, match="preflight failed"):
        submit_job(project, model_cfg, ready, "A", URIS, reg, submit=True, client=client)
    assert client.tunings.tune_calls == []


def test_submit_records_job_and_is_idempotent(project, model_cfg, ready, tmp_path):
    authorize(project)
    reg = ModelRegistry(tmp_path / "reg.json")
    client = FakeClient()
    a = submit_job(project, model_cfg, ready, "A", URIS, reg, submit=True, client=client)
    b = submit_job(project, model_cfg, ready, "A", URIS, reg, submit=True, client=client)
    assert a["job_name"] == b["job_name"] and len(client.tunings.tune_calls) == 1 and len(reg.jobs()) == 1
    call = client.tunings.tune_calls[0]
    assert call["base_model"] == model_cfg.models["primary"].id  # whatever is configured, never a hidden default
    assert call["training_dataset"].gcs_uri == URIS["train"]
    assert call["config"].validation_dataset.gcs_uri == URIS["validation"]
    c = submit_job(project, model_cfg, ready, "A", URIS, reg, submit=True, client=client, force_new=True)
    assert len(client.tunings.tune_calls) == 2 and c["job_name"]  # explicit override only


def test_changed_hyperparameters_change_the_key(model_cfg, ready):
    k1 = idempotency_key(ready / "train.jsonl", ready / "validation.jsonl", "m", {"epoch_count": 1}, "A")
    k2 = idempotency_key(ready / "train.jsonl", ready / "validation.jsonl", "m", {"epoch_count": 2}, "A")
    assert k1 != k2


def test_failed_submission_is_reported_and_not_recorded(project, model_cfg, ready, tmp_path):
    authorize(project)
    reg = ModelRegistry(tmp_path / "reg.json")
    with pytest.raises(TuningError, match="submission failed"):
        submit_job(project, model_cfg, ready, "A", URIS, reg, submit=True, client=FakeClient(fail_submit=True))
    assert reg.jobs() == []


def test_poll_success_records_model_and_activation(project, model_cfg, ready, tmp_path):
    authorize(project)
    reg = ModelRegistry(tmp_path / "reg.json")
    client = FakeClient(states=["JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_SUCCEEDED"])
    job = submit_job(project, model_cfg, ready, "A", URIS, reg, submit=True, client=client)
    with pytest.raises(TuningError):
        reg.activate(job["job_name"])  # not succeeded yet
    done = poll_job(reg, job["job_name"], client, interval_s=0, sleep=lambda s: None)
    assert done["state"] == "JOB_STATE_SUCCEEDED" and done["tuned_endpoint"].endswith("/e1")
    assert reg.activate(job["job_name"])["job_name"] == job["job_name"]
    assert reg.active()["tuned_endpoint"].endswith("/e1")


def test_poll_failed_job_raises_and_is_recorded(project, model_cfg, ready, tmp_path):
    authorize(project)
    reg = ModelRegistry(tmp_path / "reg.json")
    client = FakeClient(states=["JOB_STATE_RUNNING", "JOB_STATE_FAILED"], error=SimpleNamespace(message="bad data"))
    job = submit_job(project, model_cfg, ready, "A", URIS, reg, submit=True, client=client)
    with pytest.raises(TuningError, match="bad data"):
        poll_job(reg, job["job_name"], client, sleep=lambda s: None)
    assert reg.get(job["job_name"])["state"] == "JOB_STATE_FAILED"
    # a failed job does not block resubmission of identical data
    again = submit_job(project, model_cfg, ready, "A", URIS, reg, submit=True, client=client)
    assert len(client.tunings.tune_calls) == 2 and again["job_name"]


def test_poll_timeout_and_cancel(project, model_cfg, ready, tmp_path):
    authorize(project)
    reg = ModelRegistry(tmp_path / "reg.json")
    client = FakeClient(states=["JOB_STATE_RUNNING"])
    job = submit_job(project, model_cfg, ready, "A", URIS, reg, submit=True, client=client)
    t = iter(range(0, 1000, 100))
    with pytest.raises(TuningError, match="timed out"):
        poll_job(reg, job["job_name"], client, timeout_s=150, sleep=lambda s: None, clock=lambda: next(t))
    assert cancel_job(reg, job["job_name"], client)["state"] == "JOB_STATE_CANCELLING"
    assert client.tunings.cancelled == [job["job_name"]]


def test_poll_succeeded_without_model_is_an_error(project, model_cfg, ready, tmp_path):
    authorize(project)
    reg = ModelRegistry(tmp_path / "reg.json")
    client = FakeClient()
    job = submit_job(project, model_cfg, ready, "A", URIS, reg, submit=True, client=client)
    client.tunings.get = lambda name: SimpleNamespace(state=SimpleNamespace(name="JOB_STATE_SUCCEEDED"), tuned_model=None, error=None)
    with pytest.raises(TuningError, match="no tuned model"):
        poll_job(reg, job["job_name"], client, sleep=lambda s: None)


class FakeBlob:
    def __init__(self, store, key):
        self.store, self.key = store, key

    def exists(self):
        return self.key in self.store

    def reload(self):
        self.size = self.store[self.key]

    def upload_from_filename(self, path, content_type=None):
        import os

        self.store[self.key] = os.path.getsize(path)


class FakeStorage:
    def __init__(self):
        self.store = {}

    def bucket(self, name):
        return SimpleNamespace(blob=lambda key: FakeBlob(self.store, key))

    def list_blobs(self, bucket, prefix=""):
        return [SimpleNamespace(name=k, size=v) for k, v in list(self.store.items()) if k.startswith(prefix)]


def test_upload_requires_authorization_and_is_idempotent(project, model_cfg, ready):
    with pytest.raises(AuthorizationError):
        upload_to_gcs(project, ready, FakeStorage())
    project.cloud.allow_cloud_upload = True
    st = FakeStorage()
    uris = upload_to_gcs(project, ready, st)
    n = len(st.store)
    assert n > 2 and uris["train"].startswith("gs://ngo-bucket/") and uris["train"].endswith("train.jsonl")
    upload_to_gcs(project, ready, st)  # second run: nothing new, nothing breaks
    assert len(st.store) == n


def test_upload_needs_bucket(project, ready):
    project.cloud.allow_cloud_upload = True
    project.cloud.gcs_bucket = None
    with pytest.raises(ConfigError):
        upload_to_gcs(project, ready, FakeStorage())


def test_refresh_job_records_state_without_raising(project, model_cfg, ready, tmp_path):
    from ngo_photo_approval.tuning.job_manager import refresh_job

    authorize(project)
    reg = ModelRegistry(tmp_path / "reg.json")
    client = FakeClient(states=["JOB_STATE_RUNNING"])
    job = submit_job(project, model_cfg, ready, "A", URIS, reg, submit=True, client=client)
    assert refresh_job(reg, job["job_name"], client)["state"] == "JOB_STATE_RUNNING"
    client.tunings.states = ["JOB_STATE_FAILED"]
    client.tunings.error = SimpleNamespace(message="bad data")
    failed = refresh_job(reg, job["job_name"], client)
    assert failed["state"] == "JOB_STATE_FAILED" and failed["error"] == "bad data"
    client.tunings.states = ["JOB_STATE_SUCCEEDED"]
    assert refresh_job(reg, job["job_name"], client)["tuned_endpoint"].endswith("/e1")


def test_tuning_blocked_in_api_key_mode(project, model_cfg, ready):
    authorize(project)
    project.cloud.auth_mode = "api_key"
    rep = run_preflight(project, model_cfg, ready, "A", check_credentials=False, client=FakeClient())
    assert not rep.can_submit
    assert any(c.name == "auth_mode_supports_tuning" and not c.ok for c in rep.checks)


def test_tuning_export_uses_normalized_images_and_says_so(project, model_cfg, rule_set, prepared):
    from PIL import Image

    model_cfg.image_input.tuning_normalize_long_side_px = 512
    rep = build_tuning_files(project, model_cfg, rule_set, workers=2)
    assert rep.image_normalization == "long side 512px, JPEG q90"
    plan = pd.read_csv(project.output_dir / "tuning" / "upload_plan.csv")
    assert plan.derived_copy.all()  # every uploaded image is a normalized derived copy, never an original
    assert plan.local_path.str.contains("tuning_images_norm512").all()
    for path in plan.local_path.head(8):
        with Image.open(path) as im:
            assert max(im.size) == 512 and im.format == "JPEG"
    assert not any(str(project.data_dir) in p for p in plan.local_path)  # raw dataset untouched


def test_tuned_model_classifier_applies_the_same_normalization(model_cfg, tmp_path):
    import io

    from PIL import Image

    model_cfg.image_input.tuning_normalize_long_side_px = 320
    backend = FakeBackendForTuned()
    from ngo_photo_approval.models.tuned_model import TunedDecisionClassifier

    img = make_image(tmp_path / "big.jpg", seed=5, size=(1600, 900))
    TunedDecisionClassifier(backend, model_cfg, "tuned").classify(img, "i1")
    sent = Image.open(io.BytesIO(backend.requests[0].image_bytes))
    assert max(sent.size) == 320  # same preprocessing the training examples had


class FakeBackendForTuned:
    def __init__(self):
        self.requests = []

    def generate(self, request):
        from ngo_photo_approval.models.cloud_client import BackendResponse
        from ngo_photo_approval.schemas import UsageInfo

        self.requests.append(request)
        return BackendResponse(text='{"decision": "approved"}', usage=UsageInfo(total_tokens=1))


def test_upload_is_parallel_resumable_and_skips_present_blobs(project, model_cfg, ready):
    import threading

    project.cloud.allow_cloud_upload = True
    st = FakeStorage()
    threads = set()
    orig = FakeBlob.upload_from_filename

    def spy(self, path, content_type=None):
        threads.add(threading.current_thread().name)
        return orig(self, path, content_type)

    FakeBlob.upload_from_filename = spy
    try:
        upload_to_gcs(project, ready, st, workers=4)
        first = len(st.store)
        # simulate an interrupted run: drop a few images, keep the rest
        for key in [k for k in st.store if "/images/" in k][:5]:
            del st.store[key]
        threads.clear()
        upload_to_gcs(project, ready, st, workers=4)
    finally:
        FakeBlob.upload_from_filename = orig
    assert len(st.store) == first  # the missing ones were re-uploaded, nothing duplicated
