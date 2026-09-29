#!/usr/bin/env python3
"""
Sets the seven SPEND_CAI_* Airflow Variables of cde/dags/spend_dag.py on the CDE
virtual cluster, through its Airflow REST API. Only these keys are created or
updated; the other projects' variables in the same Airflow are never touched.
Values are never printed.

  set -a; source .env; set +a      # SPEND_CAI_HOST, SPEND_CAI_API_KEY, SPEND_IMPALA_USER/_PASSWORD
  python cde/scripts/set_airflow_variables.py --dry-run
  python cde/scripts/set_airflow_variables.py

The job IDs are looked up by name in the CAI project (ci/setup_cai.py). The
vcluster endpoint comes from ~/.cde/config.yaml (or CDE_VCLUSTER_ENDPOINT); the
Airflow API is called with a Knox token for the workload user. Standard library only.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci.setup_cai import CAI_PROJECT_NAME, Workbench, find_project  # noqa: E402

JOB_VARIABLES = {"SPEND_CAI_CATEGORISE_JOB_ID": "spend-categorise",
                 "SPEND_CAI_FEATURES_JOB_ID": "spend-feature-engineering",
                 "SPEND_CAI_TRAIN_JOB_ID": "spend-model-training",
                 "SPEND_CAI_EVALUATE_JOB_ID": "spend-model-evaluation"}


def wanted_variables(wb, host: str, key: str) -> dict:
    project = find_project(wb, CAI_PROJECT_NAME)
    if not project:
        raise SystemExit(f"CAI project {CAI_PROJECT_NAME} not found: run ci/setup_cai.py first")
    jobs = {j["name"]: j["id"] for j in wb("GET", f"/projects/{project['id']}/jobs",
                                           params={"page_size": 200}).get("jobs", [])}
    missing = [n for n in JOB_VARIABLES.values() if n not in jobs]
    if missing:
        raise SystemExit(f"CAI jobs not found: {missing} (ci/setup_cai.py)")
    host = host.rstrip("/")
    return {"SPEND_CAI_HOST": host if host.startswith("https://") else f"https://{host}",
            "SPEND_CAI_PROJECT_ID": project["id"], "SPEND_CAI_API_KEY": key,
            **{var: jobs[name] for var, name in JOB_VARIABLES.items()}}


def vcluster_endpoint() -> str:
    if os.environ.get("CDE_VCLUSTER_ENDPOINT"):
        return os.environ["CDE_VCLUSTER_ENDPOINT"]
    found = re.search(r"vcluster-endpoint:\s*(\S+)", (Path.home() / ".cde" / "config.yaml").read_text())
    if not found:
        raise SystemExit("no vcluster-endpoint in ~/.cde/config.yaml; set CDE_VCLUSTER_ENDPOINT")
    return found.group(1)


class Airflow:
    def __init__(self, vcluster: str, user: str, password: str):
        root = vcluster.split("/dex")[0]
        service = re.sub(r"^https://[^.]+\.", "https://service.", root)
        basic = base64.b64encode(f"{user}:{password}".encode()).decode()
        req = urllib.request.Request(f"{service}/gateway/authtkn/knoxtoken/api/v1/token",
                                     headers={"Authorization": f"Basic {basic}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            token = json.load(r)["access_token"]
        self.base = f"{root}/airflow/api/v1"
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def __call__(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, headers=self.headers, method=method)
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode()
        return json.loads(text) if text else {}


def apply(airflow, wanted: dict, dry_run: bool) -> None:
    for key, value in wanted.items():
        try:
            current = airflow("GET", f"/variables/{key}").get("value")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            current = None
        if current == value:
            print(f"{key}: up to date")
        elif dry_run:
            print(f"{key}: would {'create' if current is None else 'update'}")
        elif current is None:
            airflow("POST", "/variables", body={"key": key, "value": value})
            print(f"{key}: created")
        else:
            airflow("PATCH", f"/variables/{key}", body={"key": key, "value": value})
            print(f"{key}: updated")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    args, _ = p.parse_known_args()
    host, key = os.environ["SPEND_CAI_HOST"], os.environ["SPEND_CAI_API_KEY"]
    wanted = wanted_variables(Workbench(host if host.startswith("http") else f"https://{host}", key), host, key)
    airflow = Airflow(vcluster_endpoint(), os.environ["SPEND_IMPALA_USER"], os.environ["SPEND_IMPALA_PASSWORD"])
    apply(airflow, wanted, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
