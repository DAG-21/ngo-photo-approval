"""Phase 8: cloud tuning workflow with hard authorization gates.

Sub-commands:
  preflight   offline readiness report (default; spends nothing, sends nothing)
  upload      upload images + JSONL to GCS            (needs NGO_ALLOW_CLOUD_UPLOAD=true)
  submit      submit a PAID tuning job                (needs --submit-cloud-job AND NGO_ALLOW_PAID_TUNING=true)
  status      poll a job and record the tuned model   (read-only API call)
  activate    mark a succeeded job's model as active
  cancel      cancel a running job

Examples:
    python scripts/manage_tuning_job.py preflight
    python scripts/manage_tuning_job.py submit --submit-cloud-job
"""

from __future__ import annotations

import argparse
import sys

from ngo_photo_approval.config import load_model_config, load_project_config, setup_logging
from ngo_photo_approval.exceptions import NgoPhotoError
from ngo_photo_approval.rules.loader import load_rule_set
from ngo_photo_approval.tuning.job_manager import (
    cancel_job,
    make_client,
    poll_job,
    refresh_job,
    run_preflight,
    submit_job,
    upload_to_gcs,
)
from ngo_photo_approval.tuning.model_registry import ModelRegistry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["preflight", "upload", "submit", "status", "activate", "cancel"],
                    nargs="?", default="preflight")
    ap.add_argument("--track", choices=["A", "B"], default=None)
    ap.add_argument("--submit-cloud-job", action="store_true", help="REQUIRED to submit a paid job")
    ap.add_argument("--force-new", action="store_true", help="submit even if an identical job already exists")
    ap.add_argument("--job", help="job resource name (status/activate/cancel)")
    ap.add_argument("--online", action="store_true", help="preflight: also verify the base model via the API")
    ap.add_argument("--base-model", default=None, help="override base model (default: models.primary in model.yaml)")
    ap.add_argument("--wait", action="store_true", help="status: keep polling until finished")
    args = ap.parse_args()

    project = load_project_config()
    setup_logging(project.logging["level"])
    model_cfg = load_model_config()
    track = args.track or model_cfg.tuning.task_track
    tuning_dir = project.output_dir / "tuning"
    registry = ModelRegistry(project.artifacts_dir / "tuning_registry.json")
    rule_ids = [r.rule_id for r in load_rule_set().active()] if track == "B" else None

    try:
        if args.command == "preflight":
            client = make_client(project) if args.online else None
            print(run_preflight(project, model_cfg, tuning_dir, track, args.base_model, registry, True, client, rule_ids).render())
        elif args.command == "upload":
            print("Uploading to", project.cloud.gcs_bucket)
            print(upload_to_gcs(project, tuning_dir))
        elif args.command == "submit":
            uris = upload_to_gcs(project, tuning_dir)
            job = submit_job(project, model_cfg, tuning_dir, track, uris, registry, args.submit_cloud_job,
                             base_model=args.base_model, force_new=args.force_new, rule_ids=rule_ids)
            print(f"job: {job['job_name']} state={job['state']}")
        elif args.command == "status":
            job_name = args.job or (registry.jobs()[-1]["job_name"] if registry.jobs() else None)
            if not job_name:
                print("no job recorded")
                return 1
            client = make_client(project)
            job = poll_job(registry, job_name, client) if args.wait else refresh_job(registry, job_name, client)
            print(f"{job_name}: {job['state']} endpoint={job['tuned_endpoint']} error={job['error']}")
        elif args.command == "activate":
            print("active:", registry.activate(args.job)["job_name"])
        elif args.command == "cancel":
            print(cancel_job(registry, args.job, make_client(project))["state"])
    except NgoPhotoError as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
