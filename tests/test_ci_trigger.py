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
    monkeypatch.setattr(job, "_repo_root", lambda: project)
    monkeypatch.setenv("EXPECTED_GIT_SHA", pushed[:12])
    assert job.main() == 0
    assert (project / "a.txt").read_text() == "v2" and (project / "artifacts" / "keep.txt").exists()

    monkeypatch.setenv("EXPECTED_GIT_SHA", "deadbeef")                  # a newer push moved main on
    assert job.main() == 1 and "stopping the chain" in capsys.readouterr().out


def test_workflow_runs_the_cai_chain_after_train_validate():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    job = ci.split("\n  cai-pipeline:")[1]
    assert "needs: train-validate" in job and "github.event_name != 'pull_request'" in job
    assert "group: spend-analytics-cai" in job and "cancel-in-progress: false" in job
    assert "python ci/trigger_cai_pipeline.py" in job
    for secret in ("CAI_URL", "CAI_API_KEY", "CAI_PROJECT_ID"):
        assert f"{secret}: ${{{{ secrets.{secret} }}}}" in job
