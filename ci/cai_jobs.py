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

JOBS = [
    # name, script, vCPU, memory GB (the runbook's job table)
    {"name": "spend-sync-code", "script": "cai/jobs/sync_code.py", "cpu": 1, "memory": 2},
    {"name": "spend-categorise", "script": "cai/jobs/categorise.py", "cpu": 4, "memory": 16},
    {"name": "spend-feature-engineering", "script": "cai/jobs/feature_engineering.py", "cpu": 2, "memory": 8},
    {"name": "spend-model-training", "script": "cai/jobs/train_models.py", "cpu": 8, "memory": 32},
    {"name": "spend-model-evaluation", "script": "cai/jobs/evaluate_models.py", "cpu": 2, "memory": 8},
]
BY_NAME = {j["name"]: j for j in JOBS}
DEADLINE_MIN = 90                     # per job run, as the DAG (cde/dags/spend_dag.py CAI_DEADLINE_MIN)

# categorise is left out: it writes merchant_category_map, which is a publish
GITHUB_CHAIN = ["spend-sync-code", "spend-feature-engineering", "spend-model-training", "spend-model-evaluation"]
GATE_JOB = "spend-model-evaluation"


def github_env(name: str, sha: str) -> dict:
    """The environment of one job run in the GitHub chain."""
    if name == "spend-sync-code":
        return {"EXPECTED_GIT_SHA": sha[:12]}
    env = {"SPEND_TRIGGERED_BY": "github", "SPEND_STORAGE_ARTIFACTS_DIR": GITHUB_ARTIFACTS_DIR}
    if name == GATE_JOB:
        env["SPEND_DRY_RUN"] = "1"
    return env
