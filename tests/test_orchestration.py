"""The DAG and the drill script name CDE jobs, CAI jobs and Airflow Variables by
string; a mismatch with deploy_jobs.sh or the job files only shows up on the
cluster (404 "job not found", or a CAI job reading the wrong env var). Airflow
is not installed locally or in CI, so these checks read the files as text."""

import re

from spend.config import ROOT, settings

DAG = (ROOT / "cde" / "dags" / "spend_dag.py").read_text()
DEPLOY = (ROOT / "cde" / "scripts" / "deploy_jobs.sh").read_text()
DRILL = (ROOT / "cde" / "scripts" / "backfill_drill.sh").read_text()
DEPLOY_DAG = (ROOT / "cde" / "scripts" / "deploy_dag.sh").read_text()
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

CAI_VARS = ["SPEND_CAI_HOST", "SPEND_CAI_PROJECT_ID", "SPEND_CAI_API_KEY", "SPEND_CAI_CATEGORISE_JOB_ID",
            "SPEND_CAI_FEATURES_JOB_ID", "SPEND_CAI_TRAIN_JOB_ID", "SPEND_CAI_EVALUATE_JOB_ID"]


def _const(text: str, name: str) -> str:
    return re.search(rf'^{name}="?\$?{{?(?:{name}:-)?([\w-]+)', text, re.M).group(1)


def _deployed_suffixes() -> set[str]:
    return set(re.findall(r'^create_job "\$\{JOB_PREFIX\}-([\w-]+)"', DEPLOY, re.M))


def test_prefixes_agree():
    prefix = settings()["databases"]["prefix"]
    assert re.search(r'^DB_PREFIX = "(\w+)"', DAG, re.M).group(1) == prefix
    assert _const(DEPLOY, "DB_PREFIX") == _const(DRILL, "DB_PREFIX") == prefix
    job_prefix = re.search(r'^JOB_PREFIX = "([\w-]+)"', DAG, re.M).group(1)
    assert _const(DEPLOY, "JOB_PREFIX") == _const(DRILL, "JOB_PREFIX") == job_prefix == "rsingh-spend-anl"


def test_deployed_jobs_point_at_existing_files():
    files = re.findall(r'^create_job "\$\{JOB_PREFIX\}-[\w-]+"\s+"([\w/.]+)"', DEPLOY, re.M)
    assert len(files) == 4
    for f in files:
        assert (ROOT / f).is_file(), f


def test_dag_and_drill_only_run_deployed_jobs():
    deployed = _deployed_suffixes()
    assert deployed == {"generate-bronze", "dq-check", "build-silver", "build-gold-features"}
    assert set(re.findall(r'job_name=f"\{JOB_PREFIX\}-([\w-]+)"', DAG)) == deployed
    assert set(re.findall(r'"\$\{JOB_PREFIX\}-([\w-]+)"', DRILL)) == deployed


def test_dag_has_twelve_tasks_in_pipeline_order():
    chain = re.search(r"^\s+\((generate >> [^)]+)\)", DAG, re.M | re.S).group(1)
    assert [t.strip() for t in chain.split(">>")] == [
        "generate", "dq_bronze", "silver", "dq_silver", "categorise", "gold", "dq_gold",
        "features", "train", "evaluate", "refresh"]
    assert 'task_id="refresh_endpoint", python_callable=refresh_endpoint' in DAG
    assert re.findall(r'dq_task\("(\w+)", "(\w+)"\)', DAG) == [
        ("dq_bronze", "bronze"), ("dq_silver", "silver"), ("dq_gold", "gold")]
    assert re.findall(r'cai_task\("(\w+)", "(\w+)"\)', DAG) == [
        ("cai_categorise", "SPEND_CAI_CATEGORISE_JOB_ID"),
        ("cai_feature_engineering", "SPEND_CAI_FEATURES_JOB_ID"),
        ("cai_model_training", "SPEND_CAI_TRAIN_JOB_ID"),
        ("cai_model_evaluation", "SPEND_CAI_EVALUATE_JOB_ID")]
    task_ids = re.findall(r'task_id="(\w+)"', DAG) + [t for t, _ in re.findall(r'_task\("(\w+)", "(\w+)"\)', DAG)]
    assert 'task_id="deploy_registry_model", python_callable=deploy_registry_model' in DAG
    assert re.search(r"^\s+evaluate >> registry_deploy$", DAG, re.M)
    assert len(set(task_ids)) == 12


def test_every_overridden_run_repeats_the_db_prefix():
    # run-time args replace the job's own args
    for args in re.findall(r'"args": \[(.+?)\]', DAG, re.S):
        assert args.lstrip().startswith('"--db-prefix", DB_PREFIX')
    assert 'run() { cde job run --name "$1" --arg=--db-prefix --arg="${DB_PREFIX}"' in DRILL


def test_cai_variables_and_environment():
    assert set(re.findall(r'''["'](SPEND_CAI_\w+)["']''', DAG.split('"""', 2)[2])) == set(CAI_VARS)
    for var in CAI_VARS:
        assert var in DAG.split('"""')[1], f"{var} missing from the DAG docstring"
    assert 'env = {"SPEND_RUN_DATE": as_of, "SPEND_TRIGGERED_BY": "airflow"}' in DAG
    # every CAI job the DAG triggers reads both variables
    for job in ("categorise", "feature_engineering", "train_models", "evaluate_models"):
        text = (ROOT / "cai" / "jobs" / f"{job}.py").read_text()
        assert 'os.environ.get("SPEND_RUN_DATE")' in text, job
    # triggered_by enters at feature engineering and travels in the manifest to the model run
    for job in ("categorise", "feature_engineering"):
        text = (ROOT / "cai" / "jobs" / f"{job}.py").read_text()
        assert 'os.environ.get("SPEND_TRIGGERED_BY")' in text, job
    assert all(v in DRILL for v in CAI_VARS if v not in {"SPEND_CAI_FEATURES_JOB_ID", "SPEND_CAI_TRAIN_JOB_ID",
                                                          "SPEND_CAI_EVALUATE_JOB_ID"})


def test_schedule_and_dag_registration():
    assert 'DAILY = "0 21 * * *"' in DAG
    assert 'dag_id="spend_analytics_pipeline"' in DAG
    assert 'DAG_PATH="cde/dags/spend_dag.py"' in DEPLOY_DAG
    assert "rsingh-spend-anl-orchestration" in DEPLOY_DAG and "rsingh-spend-anl-pipeline" in DEPLOY_DAG
    assert _const(DEPLOY, "REPO_NAME") == "rsingh-spend-anl-pipeline"
    assert _const(DEPLOY, "PYTHON_ENV") == "rsingh-spend-anl-python-env"


def test_ci_runs_the_kpi_gate_after_the_tests():
    assert "train-validate:" in CI and "needs: test" in CI
    train = CI.split("train-validate:")[1]
    assert "requirements-train-ci.txt" in train
    assert "--ignore-gate" not in train
    assert "--stub" not in train
    assert "--inject-bad-data" in CI.split("train-validate:")[0]


def test_cde_python_env_has_what_the_jobs_import():
    reqs = (ROOT / "cde" / "resources" / "requirements.txt").read_text()
    jobs = "".join(p.read_text() for p in (ROOT / "cde" / "jobs").glob("*.py"))
    if re.search(r"mapInArrow|mapInPandas|applyInPandas|pandas_udf|toPandas", jobs):
        assert re.search(r"^pyarrow==", reqs, re.M)       # PySpark checks for it on the driver
    assert re.search(r"^great_expectations==1\.23\.2", reqs, re.M)


def test_endpoint_refresh_names_the_model_and_the_registry_model():
    from ci.cai_jobs import MODEL

    assert re.search(r'^ENDPOINT_MODEL = "([\w-]+)"', DAG, re.M).group(1) == MODEL["name"]
    assert re.search(r'^REGISTRY_MODEL = "([\w-]+)"', DAG, re.M).group(1) == settings()["registry"]["model_name"]
    assert ":restart" in DAG and '"action": "categorise"' in DAG


def test_registry_deploy_matches_the_script_and_the_features():
    from ci.deploy_registered_model import WORKBENCH_MODEL
    from spend.features import PERSONA_FEATURES

    assert re.search(r'^REGISTRY_ENDPOINT = "([\w-]+)"', DAG, re.M).group(1) == WORKBENCH_MODEL["name"]
    assert int(re.search(r"^PERSONA_FEATURE_COUNT = (\d+)", DAG, re.M).group(1)) == len(PERSONA_FEATURES)
    assert (f'"cpu": {WORKBENCH_MODEL["cpu"]}, "memory": {WORKBENCH_MODEL["memory"]}, "replicas": 1'
            in DAG.split("def deploy_registry_model")[1])
