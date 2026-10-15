"""scripts/platform_ctl.py against fake cde/cdp CLIs and a fake CAI API: what stop and start touch,
in which order, and that nothing outside this project's prefixes is ever acted on."""

import importlib.util
import re

import pytest

from spend.config import ROOT


def _load():
    spec = importlib.util.spec_from_file_location("spend_platform", ROOT / "scripts" / "platform_ctl.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plat(monkeypatch):
    p = _load()
    monkeypatch.setattr(p, "DRY", False)
    return p


def test_no_script_shadows_a_stdlib_module():
    # scripts/ is first on sys.path for every script run from it (CI broke on scripts/platform.py)
    import sys

    assert not {p.stem for p in (ROOT / "scripts").glob("*.py")} & sys.stdlib_module_names


def test_config_names_match_the_deploy_scripts():
    from ci.cai_jobs import JOBS, MODEL
    from ci.deploy_registered_model import WORKBENCH_MODEL

    p = _load()
    prj = p.PRJ
    assert prj["cai_project"] == re.search(r'^CAI_PROJECT_NAME = "([\w-]+)"',
                                           (ROOT / "ci" / "setup_cai.py").read_text(), re.M).group(1)
    assert all(j["name"].startswith(prj["cai_job_prefix"]) for j in JOBS)
    assert {MODEL["name"], WORKBENCH_MODEL["name"]} == set(prj["cai_models"])
    deploy = (ROOT / "cde" / "scripts" / "deploy_jobs.sh").read_text()
    assert prj["cde_job_prefix"] == re.search(r'JOB_PREFIX:-([\w-]+)\}', deploy).group(1) + "-"
    assert prj["cde_dag_jobs"] == [re.search(r'DAG_JOB_NAME:-([\w-]+)\}',
                                             (ROOT / "cde" / "scripts" / "deploy_dag.sh").read_text()).group(1)]
    assert set(p.CFG["stop_order"]) == set(p.LAYERS) and p.CFG["start_order"] == p.CFG["stop_order"][::-1]
    assert p.CFG["stop_order"][0] == "cde"          # the DAG must not start CAI jobs while CAI is stopped


class FakeCde:
    def __init__(self, runs, jobs):
        self.runs, self.jobs, self.calls = runs, jobs, []

    def __call__(self, *args):
        self.calls.append(args)
        if args[:2] == ("run", "list"):
            return self.runs
        if args[:2] == ("job", "list"):
            return self.jobs
        return ""

    def actions(self):
        return [c for c in self.calls if c[1] not in ("list",)]


RUNS = [
    {"id": 7, "job": "rsingh-spend-anl-orchestration", "type": "airflow", "status": "running"},
    {"id": 8, "job": "rsingh-spend-anl-generate-bronze", "type": "spark", "status": "starting"},
    {"id": 9, "job": "rsingh-spend-other-job", "type": "spark", "status": "running"},
    {"id": 10, "job": "rsingh-ogx-stream-bronze", "type": "spark", "status": "running"},
]
JOBS = [
    {"name": "rsingh-spend-anl-orchestration", "schedule": {"enabled": True, "paused": False}},
    {"name": "rsingh-spend-anl-dq-check", "schedule": {"enabled": False}},
    {"name": "rsingh-spend-other-orchestration", "schedule": {"enabled": True, "paused": False}},
]


def test_cde_stop_pauses_and_kills_only_this_project_children_first(plat, monkeypatch):
    fake = FakeCde(RUNS, JOBS)
    monkeypatch.setattr(plat, "cde", fake)
    plat.cde_project("stop")
    assert fake.actions() == [
        ("job", "schedule", "pause", "--name", "rsingh-spend-anl-orchestration"),
        ("run", "kill", "--id", "8"),
        ("run", "kill", "--id", "7"),
    ]
    run_list = next(c for c in fake.calls if c[:2] == ("run", "list"))
    assert "status[in]running,starting" in run_list and "job[rlike]^rsingh-spend-anl-" in run_list


def test_cde_start_unpauses_and_dry_run_acts_on_nothing(plat, monkeypatch):
    jobs = [{**j, "schedule": {**j["schedule"], "paused": True}} for j in JOBS]
    fake = FakeCde([], jobs)
    monkeypatch.setattr(plat, "cde", fake)
    plat.cde_project("start")
    assert fake.actions() == [("job", "schedule", "unpause", "--name", "rsingh-spend-anl-orchestration")]

    fake = FakeCde(RUNS, JOBS)
    monkeypatch.setattr(plat, "cde", fake)
    monkeypatch.setattr(plat, "DRY", True)
    plat.cde_project("stop")
    assert fake.actions() == []


class FakeCai:
    SECRET = "s3cr3t-token-value"

    def __init__(self, app_status, dep_status):
        self.calls = []
        self.app = {"id": "a1", "name": "Spend Analytics Application - Live", "status": app_status,
                    "cpu": 4, "memory": 8}
        self.deps = [
            ({"id": "m1", "name": "spend-scorer"}, {"id": "b2"}, {"id": "d-new", "status": dep_status,
                                                                  "created_at": "2026-10-05"}),
            ({"id": "m1", "name": "spend-scorer"}, {"id": "b1"}, {"id": "d-old", "status": "stopped",
                                                                  "created_at": "2026-10-01"}),
            ({"id": "m3", "name": "spend-persona"}, {"id": "b3"}, {"id": "d-ret", "status": dep_status,
                                                                   "created_at": "2026-09-30"}),
        ]
        self.job = {"id": "j1", "name": "spend-model-training", "schedule": "", "paused": False,
                    "environment": {"TOKEN": self.SECRET}}

    def __call__(self, method, path, **kw):
        self.calls.append((method, path))
        return {}

    def p(self, path):
        return path

    def apps(self):
        return [self.app]

    def jobs(self):
        return [self.job]

    def active_runs(self, job):
        return [{"id": "r1", "status": "ENGINE_RUNNING"}] if self.app["status"] != "APPLICATION_STOPPED" else []

    def deployments(self):
        return self.deps


def test_cai_stop_stops_app_every_deployment_and_active_runs(plat):
    c = FakeCai("APPLICATION_RUNNING", "deployed")
    plat.cai_project("stop", c)
    assert c.calls == [
        ("POST", "/applications/a1:stop"),
        ("POST", "/models/m1/builds/b2/deployments/d-new:stop"),
        ("POST", "/models/m3/builds/b3/deployments/d-ret:stop"),
        ("POST", "/jobs/j1/runs/r1:stop"),
    ]


def test_cai_start_restarts_newest_deployment_never_the_retired_model(plat):
    c = FakeCai("APPLICATION_STOPPED", "stopped")
    plat.cai_project("start", c)
    assert c.calls == [
        ("POST", "/models/m1/builds/b2/deployments/d-new:restart"),
        ("POST", "/applications/a1:restart"),
    ]


def test_status_never_prints_a_job_environment(plat, capsys):
    plat.cai_project("status", FakeCai("APPLICATION_RUNNING", "deployed"))
    out = capsys.readouterr().out
    assert "spend-model-training" in out and FakeCai.SECRET not in out


def test_layers_run_in_order_shared_services_around_the_workloads(plat, monkeypatch):
    seen = []
    monkeypatch.setattr(plat, "PROJECT", {k: (lambda a, k=k: seen.append(("project", k, a))) for k in ("cai", "cde")})
    monkeypatch.setattr(plat, "SHARED", {k: (lambda a, k=k: seen.append(("shared", k, a))) for k in plat.LAYERS})
    plat.main(["stop"])
    assert seen == [("project", "cde", "stop"), ("project", "cai", "stop")]
    seen.clear()
    plat.main(["stop", "--scope", "platform", "--yes-shared"])
    assert seen == [("project", "cde", "stop"), ("shared", "cde", "stop"), ("project", "cai", "stop"),
                    ("shared", "cai", "stop"), ("shared", "cdw", "stop")]
    seen.clear()
    plat.main(["start", "--scope", "platform", "--yes-shared", "--only", "cai"])
    assert seen == [("shared", "cai", "start"), ("project", "cai", "start")]


def test_platform_scope_needs_yes_shared_and_layers_are_checked(plat):
    with pytest.raises(SystemExit, match="--yes-shared"):
        plat.main(["stop", "--scope", "platform"])
    with pytest.raises(SystemExit, match="--yes-shared"):
        plat.main(["start", "--scope", "platform"])
    with pytest.raises(SystemExit, match="unknown layers"):
        plat.main(["stop", "--only", "nifi"])


def test_credentials_come_from_dotenv_only(plat, tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nSPEND_CAI_HOST=https://cai.example\nexport SPEND_CAI_API_KEY='k'\n\nBAD LINE\n")
    monkeypatch.setenv("SPEND_CAI_HOST", "https://from-the-shell")
    assert plat.dotenv(env) == {"SPEND_CAI_HOST": "https://cai.example", "SPEND_CAI_API_KEY": "k"}
    with pytest.raises(SystemExit, match="not found"):
        plat.dotenv(tmp_path / "missing")
