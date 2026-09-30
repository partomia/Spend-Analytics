#!/usr/bin/env python3
"""
Deploy the champion spend-persona version from the Cloudera AI Registry as the
workbench model spend-persona: Registered Models -> Model Deployments, with no
code file (the build packages the registered MLflow model).

  - No workbench model yet: creates it from the registered model, built from the
    champion (highest) version and deployed when built.
  - Otherwise: if the running build is not of the champion version, a new build
    of it, deployed when built (it replaces the running deployment).

It serves one call: the 27 PERSONA_FEATURES of one or more customers, in order,
to persona_id (spend/registry.py). spend-scorer stays the full endpoint.
Standard library only.

  set -a; source .env; set +a
  python ci/deploy_registered_model.py            # SPEND_CAI_HOST, SPEND_CAI_API_KEY, SPEND_CAI_PROJECT_ID
  python ci/deploy_registered_model.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.cai_jobs import RUNTIME  # noqa: E402
from ci.setup_cai import Workbench  # noqa: E402

REGISTERED = "spend-persona"
WORKBENCH_MODEL = {"name": "spend-persona", "cpu": 1, "memory": 2,
                   "description": "Champion spend-persona version from the AI Registry: 27 features -> persona_id"}
POLL_S, DEADLINE_S = 20, 45 * 60


def champion(wb: Workbench) -> tuple[dict, dict]:
    reg = next((m for m in wb("GET", "/registry/models", params={"page_size": 100}).get("models", [])
                if m.get("name") == REGISTERED), None)
    if reg is None:
        raise SystemExit(f"{REGISTERED} is not in the registry yet: a published daily run registers it")
    versions = wb("GET", f"/registry/models/{reg['model_id']}").get("model_versions", [])
    return reg, max(versions, key=lambda v: v.get("number") or 0)


def wait(fetch, ok: set, what: str) -> str:
    t_end, last = time.time() + DEADLINE_S, None
    while time.time() < t_end:
        status = str(fetch().get("status", "")).lower()
        if status != last:
            print(f"  {what}: {status}", flush=True)
            last = status
        if status in ok or "fail" in status or status in {"stopped", "timedout"}:
            return status
        time.sleep(POLL_S)
    return "timedout"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args, _ = p.parse_known_args()
    wb = Workbench(os.environ["SPEND_CAI_HOST"], os.environ["SPEND_CAI_API_KEY"])
    pid = os.environ["SPEND_CAI_PROJECT_ID"]
    reg, version = champion(wb)
    print(f"champion: {REGISTERED} version {version['number']} ({version['model_version_id']})")
    build_cfg = {"registered_model_version_id": version["model_version_id"], "runtime_identifier": RUNTIME,
                 "kernel": "python3", "comment": f"registry {REGISTERED} v{version['number']}"}
    deploy_cfg = {"cpu": WORKBENCH_MODEL["cpu"], "memory": WORKBENCH_MODEL["memory"], "replicas": 1}

    base = f"/projects/{pid}/models"
    model = next((m for m in wb("GET", base, params={"page_size": 100}).get("models", [])
                  if m["name"] == WORKBENCH_MODEL["name"]), None)
    if model is None:
        if args.dry_run:
            print(f"would create model {WORKBENCH_MODEL['name']} from the registry, build v{version['number']}")
            return 0
        model = wb("POST", base, body={
            "project_id": pid, "name": WORKBENCH_MODEL["name"], "description": WORKBENCH_MODEL["description"],
            "registered_model_id": reg["model_id"], "disable_authentication": False,
            "auto_build_model": True, "auto_build_config": build_cfg,
            "auto_deploy_model": True, "auto_deployment_config": deploy_cfg})
        print(f"model {model['name']}: created ({model['id']}) from the registry")
    else:
        builds = wb("GET", f"{base}/{model['id']}/builds", params={"page_size": 100}).get("model_builds", [])
        if any(b.get("registered_model_version_id") == version["model_version_id"] for b in builds):
            print(f"model {model['name']}: already built from version {version['number']}")
            return 0
        if args.dry_run:
            print(f"would build {model['name']} from version {version['number']} and deploy it")
            return 0
        wb("POST", f"{base}/{model['id']}/builds", body={
            "project_id": pid, "model_id": model["id"], **build_cfg,
            "auto_deploy_model": True, "auto_deployment_config": deploy_cfg})
        print(f"model {model['name']}: new build of version {version['number']}")

    def latest_build():
        builds = wb("GET", f"{base}/{model['id']}/builds", params={"page_size": 100}).get("model_builds", [])
        return max(builds, key=lambda b: b.get("created_at") or "") if builds else {"status": "waiting"}

    t0 = time.time()
    if wait(latest_build, {"built", "succeeded"}, "build") not in {"built", "succeeded"}:
        return 1
    build = latest_build()

    def deployment():
        deps = wb("GET", f"{base}/{model['id']}/builds/{build['id']}/deployments").get("model_deployments", [])
        return deps[0] if deps else {"status": "waiting"}

    if wait(deployment, {"deployed"}, "deployment") != "deployed":
        return 1
    print(f"{model['name']} serves {REGISTERED} version {version['number']} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
