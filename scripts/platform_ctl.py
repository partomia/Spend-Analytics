#!/usr/bin/env python3
"""
Stop, start or show this project's live Cloudera resources, in dependency order (config/platform.json).

Two scopes:
  project   (default) this project's workloads only: the CAI application, the spend-scorer and
            spend-persona-registry deployments, active spend-* job runs; the rsingh-spend-anl-*
            Airflow schedule and active CDE runs. Iceberg tables, the CDE jobs and repository,
            the CAI jobs, the Airflow Variables and the Data Visualization dashboard stay as they are.
  platform  the project scope plus the environment's shared services: the CDW Impala virtual
            warehouse, the CDE virtual cluster and the CAI workspace. Sibling projects use the same
            services, so this needs --yes-shared.

Stop runs in stop_order (CDE first, so the DAG starts no new CAI job; then CAI; then CDW); start in
start_order (CDW, then CAI, then the CDE schedule, whose unpause starts a run for the latest
closed interval at once). With platform scope each layer's service starts before its workloads
and stops after them. --only limits the layers. Data Visualization has no suspend in the CDP CLI
and is left running.

Credentials are read from .env (SPEND_CAI_HOST, SPEND_CAI_API_KEY); the cde and cdp CLIs use their
own profiles (~/.cde, ~/.cdp). Nothing here prints an environment value.

  python scripts/platform_ctl.py status [--scope platform]
  python scripts/platform_ctl.py stop  [--only cai,cde] [--dry-run]
  python scripts/platform_ctl.py start [--dry-run]
  python scripts/platform_ctl.py stop --scope platform --yes-shared
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]

CFG = json.loads((ROOT / "config" / "platform.json").read_text())
PRJ, PLT = CFG["project"], CFG["platform"]
LAYERS = ("cdw", "cde", "cai")
CAI_ACTIVE = ("ENGINE_SCHEDULING", "ENGINE_STARTING", "ENGINE_RUNNING")
DRY = False
ACTIONS = 0


def say(msg: str) -> None:
    print(msg, flush=True)


def act(what: str, fn, *a, **kw):
    global ACTIONS
    ACTIONS += 1
    if DRY:
        say(f"  would {what}")
        return None
    say(f"  {what}")
    return fn(*a, **kw)


def wait_for(what: str, fn, ok: set, timeout: int = 2400, poll: int = 20) -> str:
    if DRY:
        return ""
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        st = str(fn())
        if st != last:
            say(f"    {what}: {st} ({time.time() - t0:.0f} s)")
            last = st
        if st.lower() in {s.lower() for s in ok}:
            return st
        time.sleep(poll)
    raise SystemExit(f"{what}: still {last} after {timeout} s")


def dotenv(path: Path = ROOT / ".env") -> dict:
    """KEY=VALUE pairs of the git-ignored .env; the shell environment is not consulted."""
    if not path.is_file():
        raise SystemExit(f"{path.name} not found: copy .env.example and fill it in")
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        out[k.strip()] = v.strip().strip("'\"")
    return out


# ------------------------------------------------------------------ CLIs
def cdp(*args: str) -> dict:
    exe = shutil.which("cdp") or next(iter(glob.glob(os.path.expanduser("~/Library/Python/*/bin/cdp"))), None)
    if not exe:
        raise SystemExit("cdp CLI not found: pip install cdpcli")
    out = subprocess.run([exe, *args], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(f"cdp {' '.join(args[:2])}: {out.stderr.strip()[:300]}")
    return json.loads(out.stdout) if out.stdout.strip() else {}


def cde(*args: str):
    out = subprocess.run(["cde", *args], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(f"cde {' '.join(args[:3])}: {out.stderr.strip()[:300]}")
    return json.loads(out.stdout) if out.stdout.strip().startswith(("{", "[")) else out.stdout


class Cai:
    def __init__(self):
        env = dotenv()
        missing = [k for k in ("SPEND_CAI_HOST", "SPEND_CAI_API_KEY") if not env.get(k)]
        if missing:
            raise SystemExit(f"missing from .env: {', '.join(missing)}")
        self.base = env["SPEND_CAI_HOST"].rstrip("/") + "/api/v2"
        self.h = {"Authorization": f"Bearer {env['SPEND_CAI_API_KEY']}"}
        found = self("GET", "/projects", params={"search_filter": json.dumps({"name": PRJ["cai_project"]}),
                                                 "page_size": 100})
        self.pid = next(p["id"] for p in found["projects"] if p["name"] == PRJ["cai_project"])

    def __call__(self, method: str, path: str, **kw) -> dict:
        r = requests.request(method, self.base + path, headers=self.h, timeout=60, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path}: HTTP {r.status_code}")
        return r.json() if r.text.strip() else {}

    def p(self, path: str) -> str:
        return f"/projects/{self.pid}{path}"

    def apps(self) -> list[dict]:
        return [a for a in self("GET", self.p("/applications"), params={"page_size": 100})["applications"]
                if a["name"] in PRJ["cai_apps"]]

    def jobs(self) -> list[dict]:
        # job objects carry their environment in clear: never print one
        return [j for j in self("GET", self.p("/jobs"), params={"page_size": 200})["jobs"]
                if j["name"].startswith(PRJ["cai_job_prefix"])]

    def active_runs(self, job: dict) -> list[dict]:
        # the API lists the oldest runs first unless sorted
        runs = self("GET", self.p(f"/jobs/{job['id']}/runs"),
                    params={"page_size": 20, "sort": "-created_at"}).get("job_runs", [])
        return [r for r in runs if r["status"] in CAI_ACTIVE]

    def deployments(self) -> list[tuple[dict, dict, dict]]:
        """(model, build, deployment) for every deployment of the project's models, newest first."""
        names = set(PRJ["cai_models"]) | set(PRJ.get("cai_retired_models", []))
        out = []
        for m in self("GET", self.p("/models"), params={"page_size": 100})["models"]:
            if m["name"] not in names:
                continue
            for b in self("GET", self.p(f"/models/{m['id']}/builds"), params={"page_size": 100})["model_builds"]:
                for d in self("GET", self.p(f"/models/{m['id']}/builds/{b['id']}/deployments"))["model_deployments"]:
                    out.append((m, b, d))
        return sorted(out, key=lambda t: t[2]["created_at"], reverse=True)


# ------------------------------------------------------------------ project workloads
def cai_project(action: str, c: Cai | None = None) -> None:
    c = c or Cai()
    deps = c.deployments()
    live = {"deployed", "deploying", "pending"}
    if action == "status":
        for a in c.apps():
            say(f"  app {a['name']}: {a['status']} ({a['cpu']} vCPU / {a['memory']} GB)")
        for name in PRJ["cai_models"]:
            mine = [(b, d) for m, b, d in deps if m["name"] == name]
            up = [(b, d) for b, d in mine if d["status"] in live]
            say(f"  model {name}: " + (f"{up[0][1]['status']} (build {up[0][0]['id'][:8]})" if up else
                                       f"not deployed ({len(mine)} stopped deployments)"))
        for m, b, d in deps:
            if m["name"] in PRJ.get("cai_retired_models", []) and d["status"] in live:
                say(f"  retired model {m['name']}: {d['status']} (deployment {d['id'][:8]})")
        for j in c.jobs():
            runs = c.active_runs(j)
            sched = f"schedule {j['schedule']}{' PAUSED' if j.get('paused') else ''}" if j.get("schedule") else "manual"
            say(f"  job {j['name']}: {sched}" + (f", {len(runs)} active run(s)" if runs else ""))
        return
    if action == "stop":
        for a in c.apps():
            if a["status"] != "APPLICATION_STOPPED":
                act(f"stop application {a['name']}", c, "POST", c.p(f"/applications/{a['id']}:stop"))
        for m, b, d in deps:
            if d["status"] in live:
                act(f"stop model {m['name']} deployment {d['id'][:8]}", c, "POST",
                    c.p(f"/models/{m['id']}/builds/{b['id']}/deployments/{d['id']}:stop"))
        for j in c.jobs():
            if j["name"] in PRJ["cai_scheduled_jobs"] and not j.get("paused"):
                act(f"pause schedule of {j['name']}", c, "PATCH", c.p(f"/jobs/{j['id']}"), json={"paused": True})
            for r in c.active_runs(j):
                act(f"stop job run {j['name']} {r['id']}", c, "POST", c.p(f"/jobs/{j['id']}/runs/{r['id']}:stop"))
        return
    # start: schedules, the newest deployment of each model, then the applications
    for j in c.jobs():
        if j["name"] in PRJ["cai_scheduled_jobs"] and j.get("paused"):
            act(f"resume schedule of {j['name']}", c, "PATCH", c.p(f"/jobs/{j['id']}"), json={"paused": False})
    for name in PRJ["cai_models"]:
        mine = [(m, b, d) for m, b, d in deps if m["name"] == name]
        if not mine:
            say(f"  model {name}: no deployment to restart")
        elif mine[0][2]["status"] not in live:
            m, b, d = mine[0]
            act(f"restart model {name} deployment {d['id'][:8]} (build {b['id'][:8]})", c, "POST",
                c.p(f"/models/{m['id']}/builds/{b['id']}/deployments/{d['id']}:restart"))
    for a in c.apps():
        if a["status"] not in ("APPLICATION_RUNNING", "APPLICATION_STARTING"):
            act(f"start application {a['name']}", c, "POST", c.p(f"/applications/{a['id']}:restart"))


def _mine(name: str) -> bool:
    return name.startswith(PRJ["cde_job_prefix"])


def cde_project(action: str) -> None:
    # the server-side rlike narrows the list; the startswith check is the guarantee
    runs = [r for r in cde("run", "list", "--filter", "status[in]running,starting", "--filter",
                           f"job[rlike]^{PRJ['cde_job_prefix']}") or [] if _mine(r["job"])]
    scheduled = [j for j in cde("job", "list", "--filter", f"name[rlike]^{PRJ['cde_job_prefix']}") or []
                 if _mine(j["name"]) and (j.get("schedule") or {}).get("enabled")]
    if action == "status":
        say("  active runs: " + (", ".join(f"{r['job']} #{r['id']} {r['status']}" for r in runs) or "none"))
        for j in scheduled:
            s = j["schedule"]
            say(f"  schedule {j['name']} ({s.get('cronExpression')} UTC): {'paused' if s.get('paused') else 'active'}")
        for name in PRJ["cde_dag_jobs"]:
            hist = cde("run", "list", "--filter", f"job[eq]{name}") or []
            if hist:
                r = max(hist, key=lambda r: r["id"])
                say(f"  last run of {name}: #{r['id']} {r['status']} ({r.get('started', '')} - {r.get('ended', '')})")
        return
    if action == "stop":
        for j in scheduled:
            if not j["schedule"].get("paused"):
                act(f"pause CDE schedule {j['name']}", cde, "job", "schedule", "pause", "--name", j["name"])
        # Spark children first, so an orchestration run does not retry them
        for r in sorted(runs, key=lambda r: r["type"] == "airflow"):
            act(f"kill CDE run {r['job']} #{r['id']} ({r['status']})", cde, "run", "kill", "--id", str(r["id"]))
        return
    for j in scheduled:
        if j["schedule"].get("paused"):
            act(f"unpause CDE schedule {j['name']} (Airflow runs the latest closed interval at once)",
                cde, "job", "schedule", "unpause", "--name", j["name"])


PROJECT = {"cai": cai_project, "cde": cde_project}


# ------------------------------------------------------------------ shared services
def cdw(action: str) -> None:
    env_crn = cdp("environments", "describe-environment", "--environment-name", CFG["environment"])["environment"]["crn"]
    cid = next(c["id"] for c in cdp("dw", "list-clusters")["clusters"] if c["environmentCrn"] == env_crn)
    vws = {v["name"]: v for v in cdp("dw", "list-vws", "--cluster-id", cid)["vws"]}
    for name in PLT["cdw_vws"]:
        vw = vws[name]
        status = lambda v=vw: cdp("dw", "describe-vw", "--cluster-id", cid, "--vw-id", v["id"])["vw"]["status"]  # noqa: E731
        if action == "status":
            say(f"  CDW virtual warehouse {name}: {vw['status']}")
        elif action == "stop" and vw["status"].lower() not in ("stopped", "suspended"):
            act(f"suspend CDW virtual warehouse {name}", cdp, "dw", "suspend-vw", "--cluster-id", cid, "--vw-id", vw["id"])
            wait_for(name, status, {"Stopped", "Suspended"})
        elif action == "start" and vw["status"].lower() != "running":
            act(f"start CDW virtual warehouse {name}", cdp, "dw", "start-vw", "--cluster-id", cid, "--vw-id", vw["id"])
            wait_for(name, status, {"Running"})
    if action == "status":
        vizs = {v["name"]: v for v in cdp("dw", "list-data-visualizations", "--cluster-id", cid)["dataVisualizations"]}
        for name in PLT.get("cdw_data_visualizations", []):
            st = vizs[name]["status"] if name in vizs else "not found"
            say(f"  Data Visualization {name}: {st} (no suspend in the CDP CLI, left as is)")


def cde_vc(action: str) -> None:
    svc = next(s for s in cdp("de", "list-services")["services"]
               if s["name"] == PLT["cde_service"] and s["status"] == "ClusterCreationCompleted")
    vc = next(v for v in cdp("de", "list-vcs", "--cluster-id", svc["clusterId"])["vcs"] if v["vcName"] == PLT["cde_vc"])
    ids = ("--cluster-id", svc["clusterId"], "--vc-id", vc["vcId"])
    status = lambda: cdp("de", "describe-vc", *ids)["vc"]["status"]  # noqa: E731
    st = vc["status"]
    if action == "status":
        say(f"  CDE virtual cluster {PLT['cde_vc']}: {st}")
    elif action == "stop" and "suspend" not in st.lower():
        act(f"suspend CDE virtual cluster {PLT['cde_vc']}", cdp, "de", "suspend-vc", *ids)
        wait_for(PLT["cde_vc"], status, {"AppSuspended", "Suspended"})
    elif action == "start" and "suspend" in st.lower():
        act(f"resume CDE virtual cluster {PLT['cde_vc']}", cdp, "de", "resume-vc", *ids)
        wait_for(PLT["cde_vc"], status, {"AppInstalled", "Running"})


def cai_workspace(action: str) -> None:
    def ws():
        return next(w for w in cdp("ml", "list-workspaces")["workspaces"]
                    if w["instanceName"] == PLT["cai_workspace"] and w["environmentName"] == CFG["environment"])
    w = ws()
    st = w["instanceStatus"]
    if action == "status":
        say(f"  CAI workspace {PLT['cai_workspace']}: {st}")
    elif action == "stop" and "suspend" not in st:
        act(f"suspend CAI workspace {PLT['cai_workspace']}", cdp, "ml", "suspend-workspace", "--workspace-crn", w["crn"])
        wait_for(PLT["cai_workspace"], lambda: ws()["instanceStatus"], {"suspend:finished"}, timeout=3600)
    elif action == "start" and "suspend" in st:
        act(f"resume CAI workspace {PLT['cai_workspace']}", cdp, "ml", "resume-workspace", "--workspace-crn", w["crn"])
        wait_for(PLT["cai_workspace"], lambda: ws()["instanceStatus"], {"resume:finished", "installation:finished"},
                 timeout=3600)


SHARED = {"cdw": cdw, "cde": cde_vc, "cai": cai_workspace}


def layer(name: str, action: str, scope: str) -> None:
    steps = []
    if action == "start" and scope == "platform":
        steps.append(SHARED[name])
    if name in PROJECT:
        steps.append(PROJECT[name])
    if action != "start" and scope == "platform":
        steps.append(SHARED[name])
    if not steps:
        return
    say(f"[{name}]")
    before = ACTIONS
    for step in steps:
        try:
            step(action)
        except Exception as e:  # noqa: BLE001  one layer's failure is reported; the others still run
            say(f"  ERROR {type(e).__name__}: {e}")
            if action != "status":
                raise
    if action != "status" and ACTIONS == before:
        say(f"  nothing to {action}")


def main(argv: list[str] | None = None) -> int:
    global DRY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["status", "stop", "start"])
    ap.add_argument("--scope", choices=["project", "platform"], default="project")
    ap.add_argument("--only", default="", help=f"comma-separated layers from {', '.join(LAYERS)}")
    ap.add_argument("--yes-shared", action="store_true", help="required to stop or start the shared services")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    DRY = a.dry_run
    if a.scope == "platform" and a.action != "status" and not (a.yes_shared or a.dry_run):
        raise SystemExit("--scope platform stops/starts services other projects share: add --yes-shared")
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    if only - set(LAYERS):
        raise SystemExit(f"unknown layers {sorted(only - set(LAYERS))}")
    say(f"{a.action} --scope {a.scope}{' (dry run)' if DRY else ''}")
    order = CFG["start_order"] if a.action == "start" else CFG["stop_order"]
    for name in order:
        if not only or name in only:
            layer(name, a.action, a.scope)
    return 0


if __name__ == "__main__":
    sys.exit(main())
