#!/usr/bin/env python3
"""
GitHub -> Train & Validate -> KPI Gate, on Cloudera AI. Runs on the GitHub
Actions runner and talks only to the CAI API v2; data and models stay in the
workbench.

  1. Finds the GITHUB_CHAIN jobs (ci/cai_jobs.py) by name in the project.
  2. Starts them one at a time, each with its environment (github_env): sync-code
     to the pushed commit, then feature engineering and training into
     artifacts/github/, then evaluation as a dry run (the KPI gate; writes
     nothing).
  3. Follows each run to the end; stops at the first job that does not succeed
     and fails the check, naming the KPI gate when that is what stopped it. The
     published run stays live either way; the nightly DAG is what publishes.
  4. MODEL_CODE_CHANGED=true (the workflow sets it when the push touches
     ci/cai_jobs.MODEL_CODE) and the gate passed: a new spend-scorer build of this
     commit, deployed when built (it replaces the running deployment); red if the
     build or the deployment fails.

Env: CAI_URL, CAI_API_KEY, CAI_PROJECT_ID, GITHUB_SHA (set by Actions), optional
MODEL_CODE_CHANGED, CAI_CA_BUNDLE. Without CAI_URL it prints a notice and succeeds, so the workflow
stays green until the secrets are added. Standard library only.
"""
from __future__ import annotations

import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.cai_jobs import DEADLINE_MIN, GATE_JOB, GITHUB_CHAIN, MODEL, github_env, model_build_body  # noqa: E402

OK = {"succeeded"}
BAD = {"failed", "stopped", "timedout", "killed"}
POLL_S = 20


def status_of(run: dict) -> str:
    """Run status as the API returns it, lower-cased without "engine_" (ENGINE_SUCCEEDED -> succeeded)."""
    return str(run.get("status", "")).lower().replace("engine_", "")


class Api:
    def __init__(self, url: str, key: str, project: str, ca_bundle: str | None = None):
        self.base = f"{url.rstrip('/')}/api/v2/projects/{project}"
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.ctx = ssl.create_default_context(cafile=ca_bundle) if ca_bundle else None

    def __call__(self, method: str, path: str, body: dict | None = None, params: dict | None = None) -> dict:
        url = self.base + path + (f"?{urllib.parse.urlencode(params)}" if params else "")
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers=self.headers, method=method)
        with urllib.request.urlopen(req, timeout=60, context=self.ctx) as r:
            text = r.read().decode()
        return json.loads(text) if text else {}


def job_ids(api) -> dict:
    jobs = api("GET", "/jobs", params={"page_size": 200}).get("jobs", [])
    ids = {j["name"]: j["id"] for j in jobs}
    missing = [n for n in GITHUB_CHAIN if n not in ids]
    if missing:
        raise SystemExit(f"::error::CAI jobs not found in the project: {missing} (docs/DEMO_RUNBOOK.md, CAI jobs)")
    return {n: ids[n] for n in GITHUB_CHAIN}


def run_job(api, name: str, job_id: str, env: dict, poll_s: float = POLL_S,
            deadline_s: float = DEADLINE_MIN * 60) -> str:
    run = api("POST", f"/jobs/{job_id}/runs", body={"environment": env})
    print(f"{name}: started run {run['id']} with environment {env}", flush=True)
    t_end, last = time.time() + deadline_s, None
    while time.time() < t_end:
        time.sleep(poll_s)
        status = status_of(api("GET", f"/jobs/{job_id}/runs/{run['id']}"))
        if status != last:
            print(f"{name}: {status}", flush=True)
            last = status
        if status in OK | BAD:
            return status
    print(f"{name}: no result after {deadline_s / 60:.0f} min", flush=True)
    return "timedout"


def summary(rows: list[tuple[str, str]]) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write("| CAI job | Result |\n|---|---|\n" + "".join(f"| `{n}` | {s} |\n" for n, s in rows))


def wait_for(fetch, ok: set, what: str, poll_s: float, deadline_s: float) -> str:
    t_end, last = time.time() + deadline_s, None
    while time.time() < t_end:
        status = str(fetch().get("status", "")).lower()
        if status != last:
            print(f"{what}: {status}", flush=True)
            last = status
        if status in ok or "fail" in status or status in BAD:
            return status
        time.sleep(poll_s)
    return "timedout"


def rebuild_model(api, sha: str, poll_s: float = POLL_S, deadline_s: float = DEADLINE_MIN * 60) -> int:
    """A new spend-scorer build of this commit, deployed when built."""
    model = next((m for m in api("GET", "/models", params={"page_size": 100}).get("models", [])
                  if m["name"] == MODEL["name"]), None)
    if model is None:
        print(f"::notice::model {MODEL['name']} not created yet (ci/setup_cai.py): nothing to rebuild")
        return 0
    mid, pid = model["id"], api.base.rsplit("/", 1)[-1]
    build = api("POST", f"/models/{mid}/builds", body=model_build_body(pid, mid, f"github {sha[:7]}"))
    print(f"{MODEL['name']}: build {build['id']} of commit {sha[:7]}", flush=True)
    status = wait_for(lambda: api("GET", f"/models/{mid}/builds/{build['id']}"), {"built", "succeeded"},
                      "build", poll_s, deadline_s)
    if status not in {"built", "succeeded"}:
        print(f"::error::{MODEL['name']} build {build['id']} {status}; the running deployment is unchanged")
        return 1

    def deployment():
        deps = api("GET", f"/models/{mid}/builds/{build['id']}/deployments").get("model_deployments", [])
        return deps[0] if deps else {"status": "waiting"}

    status = wait_for(deployment, {"deployed"}, "deployment", poll_s, deadline_s)
    if status != "deployed":
        print(f"::error::{MODEL['name']} build {build['id']} deployment {status}")
        return 1
    print(f"{MODEL['name']} redeployed with commit {sha[:7]}")
    return 0


def run_chain(api, sha: str, poll_s: float = POLL_S, rebuild: bool = False) -> int:
    ids = job_ids(api)
    rows = []
    for i, name in enumerate(GITHUB_CHAIN):
        status = run_job(api, name, ids[name], github_env(name, sha), poll_s)
        rows.append((name, status))
        if status not in OK:
            summary(rows + [(n, "not run") for n in GITHUB_CHAIN[i + 1:]])
            what = "KPI GATE REJECTED the candidate" if name == GATE_JOB else f"{name} {status}"
            print(f"::error::{what} for commit {sha[:7]}. The published run stays live; "
                  f"see the {name} run log in Cloudera AI.")
            return 1
    print(f"KPI gate passed for commit {sha[:7]}; the nightly DAG publishes with this code")
    if rebuild:
        rc = rebuild_model(api, sha, poll_s)
        rows.append((f"{MODEL['name']} rebuild", "deployed" if rc == 0 else "failed"))
        summary(rows)
        return rc
    summary(rows)
    return 0


def main() -> int:
    url = os.environ.get("CAI_URL", "")
    if not url:
        print("::notice::CAI_URL secret not set: skipping the CAI chain (tests and train-validate still ran)")
        return 0
    api = Api(url, os.environ["CAI_API_KEY"], os.environ["CAI_PROJECT_ID"], os.environ.get("CAI_CA_BUNDLE") or None)
    rebuild = os.environ.get("MODEL_CODE_CHANGED", "").lower() == "true"
    return run_chain(api, os.environ.get("GITHUB_SHA", ""), rebuild=rebuild)


if __name__ == "__main__":
    sys.exit(main())
