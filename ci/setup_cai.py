#!/usr/bin/env python3
"""
One-off CAI setup over the API v2, from a laptop or a CAI session. Idempotent:
each step finds by name first and only creates what is missing.

  1. Project CAI_PROJECT_NAME from this GitHub repo, if absent; waits for the clone.
  2. Project environment variables: HF_HOME, SPEND_IMPALA_USER and
     SPEND_IMPALA_PASSWORD (from the caller's environment, never printed).
  3. The jobs in ci/cai_jobs.py (script, vCPU, memory, runtime, manual schedule, no
     arguments, no dependencies).

Prints the project and job IDs for the Airflow Variables. Standard library only.

  set -a; source .env; set +a
  python ci/setup_cai.py              # SPEND_CAI_HOST, SPEND_CAI_API_KEY, SPEND_IMPALA_*
  python ci/setup_cai.py --dry-run    # what would be created
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.cai_jobs import JOBS  # noqa: E402
from ci.trigger_cai_pipeline import Api  # noqa: E402

CAI_PROJECT_NAME = "rsingh-spend-analytics"
GIT_URL = "https://github.com/partomia/Spend-Analytics"
RUNTIME = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-jupyterlab-python3.11-standard:2026.08.1-b5"
PROJECT_ENV_FROM_CALLER = ("SPEND_IMPALA_USER", "SPEND_IMPALA_PASSWORD")
PROJECT_ENV = {"HF_HOME": "/home/cdsw/.hf_cache"}


class Workbench(Api):
    """Api is rooted at /api/v2/projects/<id>; the workbench-level calls need the bare /api/v2."""

    def __init__(self, url: str, key: str):
        super().__init__(url, key, "")
        self.base = f"{url.rstrip('/')}/api/v2"


def find_project(wb: Workbench, name: str) -> dict | None:
    found = wb("GET", "/projects", params={"search_filter": json.dumps({"name": name}), "page_size": 100})
    return next((p for p in found.get("projects", []) if p["name"] == name), None)


def ensure_project(wb: Workbench, dry_run: bool) -> dict | None:
    project = find_project(wb, CAI_PROJECT_NAME)
    if project:
        print(f"project {CAI_PROJECT_NAME}: exists ({project['id']})")
        return project
    if dry_run:
        print(f"project {CAI_PROJECT_NAME}: would create from {GIT_URL}")
        return None
    project = wb("POST", "/projects", body={
        "name": CAI_PROJECT_NAME, "template": "git", "git_url": GIT_URL, "visibility": "private",
        "default_project_engine_type": "ml_runtime",
        "description": "Spend Analytics: categorise, personas, forecast, signals (github.com/partomia/Spend-Analytics)"})
    print(f"project {CAI_PROJECT_NAME}: created ({project['id']}), cloning", end="", flush=True)
    for _ in range(60):
        status = str(wb("GET", f"/projects/{project['id']}").get("creation_status", "")).lower()
        if status in ("success", "succeeded", ""):
            break
        if "fail" in status or "error" in status:
            raise SystemExit(f"\nproject creation {status}")
        print(".", end="", flush=True)
        time.sleep(5)
    print(" done")
    return project


def ensure_env(wb: Workbench, project: dict, dry_run: bool) -> None:
    missing = [k for k in PROJECT_ENV_FROM_CALLER if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"set {missing} in the environment (source .env) first")
    current = json.loads(wb("GET", f"/projects/{project['id']}").get("environment") or "{}")
    wanted = {**PROJECT_ENV, **{k: os.environ[k] for k in PROJECT_ENV_FROM_CALLER}}
    changed = sorted(k for k, v in wanted.items() if current.get(k) != v)
    if not changed:
        print("project environment: up to date")
        return
    if dry_run:
        print(f"project environment: would set {changed}")
        return
    wb("PATCH", f"/projects/{project['id']}", body={"environment": json.dumps({**current, **wanted})})
    print(f"project environment: set {changed}")


def ensure_jobs(wb: Workbench, project: dict, dry_run: bool) -> dict:
    existing = {j["name"]: j for j in wb("GET", f"/projects/{project['id']}/jobs",
                                         params={"page_size": 200}).get("jobs", [])}
    ids = {}
    for job in JOBS:
        if job["name"] in existing:
            ids[job["name"]] = existing[job["name"]]["id"]
            print(f"job {job['name']}: exists ({ids[job['name']]})")
            continue
        if dry_run:
            print(f"job {job['name']}: would create ({job['script']}, {job['cpu']} vCPU / {job['memory']} GB)")
            continue
        created = wb("POST", f"/projects/{project['id']}/jobs", body={
            "name": job["name"], "script": job["script"], "cpu": job["cpu"], "memory": job["memory"],
            "runtime_identifier": RUNTIME, "arguments": "", "kill_on_timeout": True})
        ids[job["name"]] = created["id"]
        print(f"job {job['name']}: created ({created['id']})")
    return ids


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    args, _ = p.parse_known_args()
    wb = Workbench(os.environ["SPEND_CAI_HOST"], os.environ["SPEND_CAI_API_KEY"])
    project = ensure_project(wb, args.dry_run)
    if project is None:
        return 0
    ensure_env(wb, project, args.dry_run)
    ids = ensure_jobs(wb, project, args.dry_run)
    print(f"\nSPEND_CAI_PROJECT_ID = {project['id']}")
    for name, var in (("spend-categorise", "SPEND_CAI_CATEGORISE_JOB_ID"),
                      ("spend-feature-engineering", "SPEND_CAI_FEATURES_JOB_ID"),
                      ("spend-model-training", "SPEND_CAI_TRAIN_JOB_ID"),
                      ("spend-model-evaluation", "SPEND_CAI_EVALUATE_JOB_ID")):
        print(f"{var} = {ids.get(name, '(not created)')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
