"""GitHub -> CAI: the job definitions, the trigger against a fake CAI API v2, the
sync-code job on a throwaway git repo, and the cai-pipeline workflow job."""

import subprocess

import pytest

from ci import cai_jobs, trigger_cai_pipeline as trig
from tests.conftest import ROOT, _load

SHA = "0123456789abcdef0123"


class FakeApi:
    """Job list plus runs that go running -> the final status given per job name."""

    def __init__(self, final: dict | None = None, names=cai_jobs.GITHUB_CHAIN):
        self.jobs = {n: f"id-{i}" for i, n in enumerate(names)}
        self.final = final or {}
        self.started, self.polls = [], {}

    def __call__(self, method, path, body=None, params=None):
        if method == "GET" and path == "/jobs":
            return {"jobs": [{"name": n, "id": i} for n, i in self.jobs.items()]}
        name = next(n for n, i in self.jobs.items() if f"/jobs/{i}/" in path)
        if method == "POST":
            self.started.append((name, body["environment"]))
            return {"id": f"run-{name}"}
        self.polls[name] = self.polls.get(name, 0) + 1
        done = self.polls[name] > 1
        return {"status": self.final.get(name, "ENGINE_SUCCEEDED") if done else "ENGINE_RUNNING"}


def test_chain_runs_in_order_with_each_jobs_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    api = FakeApi()
    assert trig.run_chain(api, SHA, poll_s=0) == 0
    assert [n for n, _ in api.started] == cai_jobs.GITHUB_CHAIN
    env = dict(api.started)
    assert env["spend-sync-code"] == {"EXPECTED_GIT_SHA": SHA[:12]}
    for name in cai_jobs.GITHUB_CHAIN[1:]:
        assert env[name]["SPEND_TRIGGERED_BY"] == "github"
        assert env[name]["SPEND_STORAGE_ARTIFACTS_DIR"] == cai_jobs.GITHUB_ARTIFACTS_DIR
    assert env["spend-model-evaluation"]["SPEND_DRY_RUN"] == "1"
    assert all("SPEND_DRY_RUN" not in env[n] for n in cai_jobs.GITHUB_CHAIN[:-1])
    assert "KPI gate passed" in capsys.readouterr().out
    assert (tmp_path / "summary.md").read_text().count("succeeded") == 4


def test_a_rejecting_gate_fails_the_check_by_name(capsys):
    api = FakeApi({"spend-model-evaluation": "ENGINE_FAILED"})
    assert trig.run_chain(api, SHA, poll_s=0) == 1
    assert "::error::KPI GATE REJECTED the candidate" in capsys.readouterr().out


def test_a_failed_job_stops_the_chain(capsys):
    api = FakeApi({"spend-sync-code": "ENGINE_FAILED"})
    assert trig.run_chain(api, SHA, poll_s=0) == 1
    assert [n for n, _ in api.started] == ["spend-sync-code"]
    out = capsys.readouterr().out
    assert "::error::spend-sync-code failed" in out and "KPI GATE" not in out


def test_a_run_that_never_ends_times_out():
    api = FakeApi({"spend-sync-code": "ENGINE_RUNNING"})
    assert trig.run_job(api, "spend-sync-code", "id-0", {}, poll_s=0, deadline_s=0.05) == "timedout"


def test_missing_jobs_are_named():
    with pytest.raises(SystemExit, match="spend-model-evaluation"):
        trig.job_ids(FakeApi(names=cai_jobs.GITHUB_CHAIN[:-1]))


def test_no_cai_url_is_a_notice_not_a_failure(monkeypatch, capsys):
    monkeypatch.delenv("CAI_URL", raising=False)
    assert trig.main() == 0 and "::notice::" in capsys.readouterr().out


def test_every_job_is_in_the_runbook_and_its_script_exists():
    runbook = (ROOT / "docs" / "DEMO_RUNBOOK.md").read_text()
    for job in cai_jobs.JOBS:
        assert (ROOT / job["script"]).exists(), job["script"]
        row = f"| `{job['name']}` | `{job['script']}` | {job['cpu']} vCPU / {job['memory']} GB |"
        assert row in runbook, row
    assert "spend-categorise" not in cai_jobs.GITHUB_CHAIN          # writes merchant_category_map


def _git(cwd, *args):
    return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                                   text=True).strip()


def test_sync_code_resets_to_the_pushed_commit(tmp_path, monkeypatch, capsys):
    origin, project = tmp_path / "origin", tmp_path / "project"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    (origin / "a.txt").write_text("v1")
    _git(origin, "add", "a.txt")
    _git(origin, "commit", "-qm", "v1")
    _git(tmp_path, "clone", "-q", str(origin), str(project))
    (origin / "a.txt").write_text("v2")
    _git(origin, "commit", "-qam", "v2")
    pushed = _git(origin, "rev-parse", "HEAD")
    (project / "a.txt").write_text("edited in the project")
    (project / "artifacts").mkdir()
    (project / "artifacts" / "keep.txt").write_text("untracked")

    job = _load(ROOT / "cai" / "jobs" / "sync_code.py")
    installs = []
    monkeypatch.setattr(job, "_repo_root", lambda: project)
    monkeypatch.setattr(job, "install_requirements", lambda root: installs.append(root))
    monkeypatch.setenv("EXPECTED_GIT_SHA", pushed[:12])
    assert job.main() == 0 and installs == [project]
    assert (project / "a.txt").read_text() == "v2" and (project / "artifacts" / "keep.txt").exists()

    monkeypatch.setenv("EXPECTED_GIT_SHA", "deadbeef")                  # a newer push moved main on
    assert job.main() == 1 and "stopping the chain" in capsys.readouterr().out
    assert installs == [project]                                        # a stopped chain installs nothing


def test_sync_code_installs_requirements_only_when_they_change(tmp_path):
    job = _load(ROOT / "cai" / "jobs" / "sync_code.py")
    (tmp_path / "requirements.txt").write_text("pyyaml\n")
    calls = []
    assert job.install_requirements(tmp_path, pip=lambda: calls.append(1)) and calls == [1]
    assert not job.install_requirements(tmp_path, pip=lambda: calls.append(1)) and calls == [1]
    (tmp_path / "requirements.txt").write_text("pyyaml\nimpyla\n")
    assert job.install_requirements(tmp_path, pip=lambda: calls.append(1)) and calls == [1, 1]


def test_workflow_runs_the_cai_chain_after_train_validate():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    job = ci.split("\n  cai-pipeline:")[1]
    assert "needs: train-validate" in job and "github.event_name != 'pull_request'" in job
    assert "group: spend-analytics-cai" in job and "cancel-in-progress: false" in job
    assert "python ci/trigger_cai_pipeline.py" in job
    for secret in ("CAI_URL", "CAI_API_KEY", "CAI_PROJECT_ID"):
        assert f"{secret}: ${{{{ secrets.{secret} }}}}" in job


class FakeWorkbench:
    def __init__(self, env="{}"):
        self.calls, self.env, self.jobs, self.project = [], env, [], None
        self.models, self.apps, self.base = [], [], "https://ml-x.example/api/v2"

    def __call__(self, method, path, body=None, params=None):
        self.calls.append((method, path, body))
        for kind, items in (("models", self.models), ("applications", self.apps)):
            if path == f"/projects/p1/{kind}" and method == "GET":
                return {kind: items}
            if path == f"/projects/p1/{kind}" and method == "POST":
                items.append({"id": f"{kind[0]}{len(items)}", "access_key": "mk-123", **body})
                return items[-1]
            if path.startswith(f"/projects/p1/{kind}/") and method == "GET":
                return next(i for i in items if i["id"] == path.rsplit("/", 1)[1])
        if path.startswith("/projects/p1/models/") and path.endswith("/builds") and method == "POST":
            return {"id": "b0", **body}
        if path == "/projects" and method == "GET":
            return {"projects": [self.project] if self.project else []}
        if path == "/projects" and method == "POST":
            self.project = {"id": "p1", "name": body["name"]}
            return self.project
        if path == "/projects/p1" and method == "GET":
            return {"creation_status": "success", "environment": self.env}
        if path == "/projects/p1" and method == "PATCH":
            self.env = body["environment"]
            return {}
        if path == "/projects/p1/jobs" and method == "GET":
            return {"jobs": self.jobs}
        if path == "/projects/p1/jobs" and method == "POST":
            self.jobs.append({"name": body["name"], "id": f"j{len(self.jobs)}", **body})
            return self.jobs[-1]
        if path.startswith("/projects/p1/jobs/") and method == "PATCH":
            next(j for j in self.jobs if j["id"] == path.rsplit("/", 1)[1]).update(body)
            return {}
        raise AssertionError((method, path))


def test_setup_cai_creates_once_and_never_prints_secrets(monkeypatch, capsys):
    import json
    from ci import setup_cai
    monkeypatch.setenv("SPEND_IMPALA_USER", "u")
    monkeypatch.setenv("SPEND_IMPALA_PASSWORD", "s3cret-value")
    wb = FakeWorkbench()
    project = setup_cai.ensure_project(wb, dry_run=False)
    setup_cai.ensure_env(wb, project, dry_run=False)
    ids = setup_cai.ensure_jobs(wb, project, dry_run=False)
    assert project["name"] == setup_cai.CAI_PROJECT_NAME and set(ids) == {j["name"] for j in cai_jobs.JOBS}
    assert json.loads(wb.env) == {"HF_HOME": "/home/cdsw/.hf_cache", "SPEND_IMPALA_USER": "u",
                                  "SPEND_IMPALA_PASSWORD": "s3cret-value"}
    assert all(j["runtime_identifier"] == cai_jobs.RUNTIME and j["arguments"] == "" for j in wb.jobs)
    n = len(wb.calls)
    setup_cai.ensure_project(wb, dry_run=False)
    setup_cai.ensure_env(wb, project, dry_run=False)
    setup_cai.ensure_jobs(wb, project, dry_run=False)
    assert not [c for c in wb.calls[n:] if c[0] in ("POST", "PATCH")]       # idempotent
    wb.jobs[0]["memory"] = 2                                                  # sized before a change
    setup_cai.ensure_jobs(wb, project, dry_run=False)
    assert wb.calls[-1] == ("PATCH", "/projects/p1/jobs/j0", {"cpu": cai_jobs.JOBS[0]["cpu"],
                                                               "memory": cai_jobs.JOBS[0]["memory"]})
    assert "s3cret-value" not in capsys.readouterr().out


def test_setup_cai_deploys_the_model_then_the_app_once_the_model_key_is_set(monkeypatch, capsys):
    import json
    from ci import setup_cai
    monkeypatch.setenv("SPEND_IMPALA_USER", "u")
    monkeypatch.setenv("SPEND_IMPALA_PASSWORD", "p")
    monkeypatch.delenv("SPEND_ENDPOINT_API_KEY", raising=False)
    wb = FakeWorkbench()
    project = setup_cai.ensure_project(wb, dry_run=False)
    model = setup_cai.ensure_model(wb, project, dry_run=False)
    build = next(b for m, p, b in wb.calls if p.endswith("/builds"))
    assert build["file_path"] == "cai/model/predict.py" and build["function_name"] == "predict"
    assert build["auto_deploy_model"] and build["auto_deployment_config"]["memory"] == 16
    assert wb.models[0]["disable_authentication"] is False
    endpoint = setup_cai.endpoint_env(wb, model)
    assert endpoint == {"SPEND_ENDPOINT_URL": "https://modelservice.ml-x.example/model",
                        "SPEND_ENDPOINT_ACCESS_KEY": "mk-123"}
    setup_cai.ensure_app(wb, project, "SPEND_ENDPOINT_API_KEY" in endpoint, dry_run=False)
    assert wb.apps == []                                                     # no Model API key yet
    monkeypatch.setenv("SPEND_ENDPOINT_API_KEY", "model-key-value")
    model = setup_cai.ensure_model(wb, project, dry_run=False)
    assert len(wb.models) == 1
    endpoint = setup_cai.endpoint_env(wb, model)
    setup_cai.ensure_env(wb, project, dry_run=False, extra=endpoint)
    assert json.loads(wb.env)["SPEND_ENDPOINT_API_KEY"] == "model-key-value"
    setup_cai.ensure_app(wb, project, "SPEND_ENDPOINT_API_KEY" in endpoint, dry_run=False)
    setup_cai.ensure_app(wb, project, True, dry_run=False)
    assert [a["script"] for a in wb.apps] == ["app/run.py"]
    out = capsys.readouterr().out
    assert "model-key-value" not in out and "mk-123" not in out


def test_airflow_variables_upsert_only_the_spend_keys_and_never_print_values(capsys):
    import urllib.error
    av = _load(ROOT / "cde" / "scripts" / "set_airflow_variables.py")
    store = {"SPEND_CAI_HOST": "https://old", "SPEND_CAI_PROJECT_ID": "p1", "MULE_CAI_HOST": "https://mule"}
    calls = []

    def airflow(method, path, body=None):
        calls.append((method, path))
        key = path.rsplit("/", 1)[1]
        if method == "GET":
            if key not in store:
                raise urllib.error.HTTPError(path, 404, "not found", {}, None)
            return {"key": key, "value": store[key]}
        store[body["key"]] = body["value"]
        return {}

    wb = FakeWorkbench()
    wb.project = {"id": "p1", "name": "rsingh-spend-analytics"}
    wb.jobs = [{"name": j["name"], "id": f"id-{j['name']}"} for j in cai_jobs.JOBS]
    wanted = av.wanted_variables(wb, "ml-x.example", "api-key-value")
    assert set(wanted) == {"SPEND_CAI_HOST", "SPEND_CAI_PROJECT_ID", "SPEND_CAI_API_KEY",
                           *av.JOB_VARIABLES} and len(wanted) == 7
    assert wanted["SPEND_CAI_HOST"] == "https://ml-x.example"
    assert wanted["SPEND_CAI_TRAIN_JOB_ID"] == "id-spend-model-training"
    av.apply(airflow, wanted, dry_run=True)
    assert all(m == "GET" for m, _ in calls) and store["SPEND_CAI_HOST"] == "https://old"
    av.apply(airflow, wanted, dry_run=False)
    assert {k: store[k] for k in wanted} == wanted and store["MULE_CAI_HOST"] == "https://mule"
    assert ("PATCH", "/variables/SPEND_CAI_HOST") in calls
    assert not [p for m, p in calls if m != "GET" and "SPEND_CAI_PROJECT_ID" in p]   # unchanged
    assert "api-key-value" not in capsys.readouterr().out


class FakeModelApi(FakeApi):
    """FakeApi plus the spend-scorer model: a build that goes built, then its deployment deployed."""

    def __init__(self, build_final="built", deploy_final="deployed"):
        super().__init__()
        self.base = "https://cai/api/v2/projects/proj-1"
        self.build_final, self.deploy_final, self.builds, self.n = build_final, deploy_final, [], 0

    def __call__(self, method, path, body=None, params=None):
        if not path.startswith("/models"):
            return super().__call__(method, path, body, params)
        if path == "/models":
            return {"models": [{"name": cai_jobs.MODEL["name"], "id": "mod-1"}]}
        if method == "POST":
            self.builds.append(body)
            return {"id": "b-1"}
        self.n += 1
        if path.endswith("/deployments"):
            return {"model_deployments": [{"status": self.deploy_final if self.n > 3 else "pending"}]}
        return {"status": self.build_final if self.n > 1 else "building"}


def test_a_model_code_push_rebuilds_the_scorer_after_the_gate(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    api = FakeModelApi()
    assert trig.run_chain(api, SHA, poll_s=0, rebuild=True) == 0
    b = api.builds[0]
    assert b["project_id"] == "proj-1" and b["model_id"] == "mod-1" and b["auto_deploy_model"]
    assert b["file_path"] == cai_jobs.MODEL["file"] and b["comment"] == f"github {SHA[:7]}"
    assert "redeployed with commit" in capsys.readouterr().out
    assert "rebuild` | deployed" in (tmp_path / "summary.md").read_text()


def test_no_rebuild_without_model_code_or_after_a_rejected_gate():
    api = FakeModelApi()
    assert trig.run_chain(api, SHA, poll_s=0) == 0 and api.builds == []
    api = FakeModelApi()
    api.final = {"spend-model-evaluation": "ENGINE_FAILED"}
    assert trig.run_chain(api, SHA, poll_s=0, rebuild=True) == 1 and api.builds == []


def test_a_failed_build_fails_the_check(capsys):
    assert trig.run_chain(FakeModelApi(build_final="build failed"), SHA, poll_s=0, rebuild=True) == 1
    assert "the running deployment is unchanged" in capsys.readouterr().out


def test_workflow_passes_the_model_code_check_to_the_chain():
    job = (ROOT / ".github" / "workflows" / "ci.yml").read_text().split("\n  cai-pipeline:")[1]
    assert "fetch-depth: 0" in job and "touches_model" in job
    assert "MODEL_CODE_CHANGED: ${{ steps.changes.outputs.model_code_changed }}" in job
