"""
The CAI jobs of this project, read by ci/trigger_cai_pipeline.py and checked
against docs/DEMO_RUNBOOK.md by the tests. Pure Python, so the GitHub runner
can import it without cmlapi.

The jobs have no CAI dependencies: on the nightly path Airflow starts them one
by one (cde/dags/spend_dag.py), and on a push GitHub Actions starts GITHUB_CHAIN
one by one through the CAI API v2. A job run ignores arguments, so both pass
settings through the run's environment.

The GitHub chain is a candidate check, not a publish: feature engineering and
training write to their own artifacts folder, and evaluation runs the KPI gate
with SPEND_DRY_RUN, so it writes no table and no MLflow run. The nightly DAG
publishes.
"""
from __future__ import annotations

GITHUB_ARTIFACTS_DIR = "artifacts/github"
RUNTIME = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-jupyterlab-python3.11-standard:2026.08.1-b5"

MODEL = {"name": "spend-scorer", "file": "cai/model/predict.py", "cpu": 4, "memory": 16,
         "description": "Spend profile, 12-week forecast, what-if and live categorise for one customer"}
# A push touching these rebuilds and redeploys spend-scorer once its KPI gate passes: the
# build snapshots the project files and installs requirements.txt (cdsw-build.sh).
MODEL_CODE = ("spend/", "cai/model/", "config/", "requirements.txt", "cdsw-build.sh")

JOBS = [
    # name, script, vCPU, memory GB (the runbook's job table)
    {"name": "spend-sync-code", "script": "cai/jobs/sync_code.py", "cpu": 2, "memory": 8},
    # 4 / 16 never left scheduling on 5-8 Oct while the per-user quota was full; a night's keys take seconds
    {"name": "spend-categorise", "script": "cai/jobs/categorise.py", "cpu": 2, "memory": 8},
    {"name": "spend-feature-engineering", "script": "cai/jobs/feature_engineering.py", "cpu": 2, "memory": 8},
    {"name": "spend-model-training", "script": "cai/jobs/train_models.py", "cpu": 4, "memory": 16},
    {"name": "spend-model-evaluation", "script": "cai/jobs/evaluate_models.py", "cpu": 2, "memory": 8},
]
BY_NAME = {j["name"]: j for j in JOBS}
DEADLINE_MIN = 90                     # per job run, as the DAG (cde/dags/spend_dag.py CAI_DEADLINE_MIN)

# categorise is left out: it writes merchant_category_map, which is a publish
GITHUB_CHAIN = ["spend-sync-code", "spend-feature-engineering", "spend-model-training", "spend-model-evaluation"]
GATE_JOB = "spend-model-evaluation"


def model_build_body(project_id: str, model_id: str, comment: str = "") -> dict:
    """A spend-scorer build that deploys itself when built."""
    return {"project_id": project_id, "model_id": model_id, "file_path": MODEL["file"], "function_name": "predict",
            "kernel": "python3", "runtime_identifier": RUNTIME, "comment": comment, "auto_deploy_model": True,
            "auto_deployment_config": {"cpu": MODEL["cpu"], "memory": MODEL["memory"], "replicas": 1}}


def touches_model(paths: list[str]) -> bool:
    return any(p == m or p.startswith(m) for p in paths for m in MODEL_CODE)


def github_env(name: str, sha: str) -> dict:
    """The environment of one job run in the GitHub chain."""
    if name == "spend-sync-code":
        return {"EXPECTED_GIT_SHA": sha[:12]}
    env = {"SPEND_TRIGGERED_BY": "github", "SPEND_STORAGE_ARTIFACTS_DIR": GITHUB_ARTIFACTS_DIR}
    if name == GATE_JOB:
        env["SPEND_DRY_RUN"] = "1"
    return env
