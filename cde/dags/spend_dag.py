"""
Airflow DAG (CDE): daily spend analytics pipeline, 11 tasks.

  generate_spend_bronze -> dq_bronze -> build_silver -> dq_silver            (CDE Spark)
    -> cai_categorise                                                        (CAI Job: new merchant keys -> map)
    -> build_gold_features -> dq_gold                                        (CDE Spark)
    -> cai_feature_engineering -> cai_model_training -> cai_model_evaluation (CAI Jobs: the daily ML, PLAN decision 1)
    -> refresh_endpoint                                                      (CAI API: restart spend-scorer, verify)

dq_bronze/dq_silver/dq_gold are the same CDE job (cde/jobs/dq_check.py, Great
Expectations) run three times with --layer overridden per task. Critical
expectation failures fail the task (and stop the DAG before the next layer);
warnings are recorded but let the run continue. Every result, pass or fail, is
appended to <db_prefix>_ref.dq_results (Iceberg) tagged with this DAG run's
run_id, so a run's three layers can be grouped in the app's Data quality tab.

cai_categorise runs between silver and gold because gold's category shares join
silver.spend_event to ref.merchant_category_map. The three ML jobs hand off
through artifacts/<run_date>/ on the CAI project filesystem; a failed KPI gate
fails cai_model_evaluation (exit 1) after recording the run, and yesterday's
personas, forecast and signals stay live.

refresh_endpoint runs only after a passed gate: it restarts the spend-scorer
deployment (it loads the latest published run at start-up), then calls it until
it answers with the run of the champion (highest) spend-persona version in the
Cloudera AI Registry, or, with no registry version, a run dated on or after as_of.
Red if that does not happen within ENDPOINT_DEADLINE_MIN.

Airflow Variables (CDE Airflow UI > Admin > Variables), set before the DAG is
deployed:
  SPEND_CAI_HOST               https://ml-xxxx.<env>.cloudera.site  (CAI workbench URL)
  SPEND_CAI_PROJECT_ID         project id (from the project URL or API)
  SPEND_CAI_API_KEY            CAI API v2 key (User settings > API keys; not the Model API key)
  SPEND_CAI_CATEGORISE_JOB_ID  id of the spend-categorise job
  SPEND_CAI_FEATURES_JOB_ID    id of the spend-feature-engineering job
  SPEND_CAI_TRAIN_JOB_ID       id of the spend-model-training job
  SPEND_CAI_EVALUATE_JOB_ID    id of the spend-model-evaluation job
If SPEND_CAI_HOST is not set the CAI steps are skipped, so the Spark part can be
tested on its own (gold then joins the map as it stands).

Scheduled daily at 21:00 UTC (02:30 IST), between the other daily DAGs on the
cluster: Mule (20:30), Churn (22:00) and Collections (00:30). Manual trigger (Trigger DAG w/ config):
{"as_of": "2026-09-25"}; empty = yesterday.

Job names must match cde/scripts/deploy_jobs.sh exactly (CDEJobRunOperator
fails with 404 "job not found" otherwise).
"""

import time
from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from cloudera.cdp.airflow.operators.cde_operator import CDEJobRunOperator

JOB_PREFIX = "rsingh-spend-anl"
DB_PREFIX = "rsingh_spend_anl"
# Scheduled runs: the day before the interval end; manual runs: the as_of param (empty = yesterday).
AS_OF = ("{{ params.as_of or ((data_interval_end - macros.timedelta(days=1)).strftime('%Y-%m-%d') "
         "if dag_run.run_type == 'scheduled' else (macros.datetime.utcnow() - macros.timedelta(days=1))"
         ".strftime('%Y-%m-%d')) }}")
TERMINAL_OK = {"succeeded"}
TERMINAL_BAD = {"failed", "stopped", "timedout"}
DAILY = "0 21 * * *"
CAI_DEADLINE_MIN = 90
ENDPOINT_MODEL = "spend-scorer"
REGISTRY_MODEL = "spend-persona"
ENDPOINT_DEADLINE_MIN = 30


def trigger_cai_job(as_of: str, job_var: str, **_):
    host = Variable.get("SPEND_CAI_HOST", default_var="").rstrip("/")
    if not host:
        raise AirflowSkipException("SPEND_CAI_HOST not set: skipping the CAI step")
    project = Variable.get("SPEND_CAI_PROJECT_ID")
    job = Variable.get(job_var)
    headers = {"Authorization": f"Bearer {Variable.get('SPEND_CAI_API_KEY')}", "Content-Type": "application/json"}
    # A job run ignores "arguments" (the job's own are used); the environment map is applied.
    # categorise records as_of as the map's first_run_date; the ML jobs hand off under
    # artifacts/<as_of>/.
    env = {"SPEND_RUN_DATE": as_of, "SPEND_TRIGGERED_BY": "airflow"}
    url = f"{host}/api/v2/projects/{project}/jobs/{job}/runs"
    resp = requests.post(url, json={"environment": env}, headers=headers, timeout=60)
    resp.raise_for_status()
    run_id = resp.json()["id"]
    print(f"Started CAI job run {run_id} ({job_var}) with environment: {env}")

    deadline = time.time() + CAI_DEADLINE_MIN * 60
    while time.time() < deadline:
        time.sleep(30)
        r = requests.get(f"{url}/{run_id}", headers=headers, timeout=60)
        r.raise_for_status()
        status = str(r.json().get("status", "")).lower().replace("engine_", "")
        print(f"CAI run {run_id}: {status}")
        if status in TERMINAL_OK:
            return run_id
        if status in TERMINAL_BAD:
            raise AirflowException(f"CAI job run {run_id} ended with status {status}")
    raise AirflowException(f"CAI job run {run_id} did not finish within {CAI_DEADLINE_MIN} minutes")


def _champion_run_id(api) -> str | None:
    """The daily run of the highest spend-persona version (its MLflow run's run_id parameter), if any.
    Only a passed, latest-dated run is registered (spend/registry.py), so that is the run to serve."""
    try:
        models = api("GET", "/registry/models", params={"page_size": 100}).get("models", [])
    except requests.RequestException as e:
        print(f"registry not reachable ({e}): checking the run date instead")
        return None
    model = next((m for m in models if m.get("name") == REGISTRY_MODEL), None)
    if model is None:
        print(f"registry has no {REGISTRY_MODEL} ({len(models)} models listed): checking the run date instead")
        return None
    versions = api("GET", f"/registry/models/{model['model_id']}").get("model_versions", [])
    if not versions:
        print(f"registry {REGISTRY_MODEL} has no versions yet: checking the run date instead")
        return None
    best = max(versions, key=lambda v: v.get("number") or 0)
    params = ((best.get("model_version_metadata") or {}).get("mlflow_metadata") or {}).get("params") or []
    run_id = next((p["value"] for p in params if p.get("key") == "run_id"), None)
    print(f"registry: {REGISTRY_MODEL} version {best.get('number')} is champion (run {run_id})")
    return run_id


def refresh_endpoint(as_of: str, **_):
    """Restart the spend-scorer deployment so it loads the run just published, then check that it serves it."""
    host = Variable.get("SPEND_CAI_HOST", default_var="").rstrip("/")
    if not host:
        raise AirflowSkipException("SPEND_CAI_HOST not set: skipping the endpoint refresh")
    project = Variable.get("SPEND_CAI_PROJECT_ID")
    key = Variable.get("SPEND_CAI_API_KEY")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def api(method, path, params=None):
        r = requests.request(method, f"{host}/api/v2{path}", params=params, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json() if r.text else {}

    pid = f"/projects/{project}"
    model = next((m for m in api("GET", f"{pid}/models", params={"page_size": 100}).get("models", [])
                  if m["name"] == ENDPOINT_MODEL), None)
    if model is None:
        raise AirflowSkipException(f"model {ENDPOINT_MODEL} not deployed: nothing to refresh")
    mid = model["id"]
    deployed = []
    for b in api("GET", f"{pid}/models/{mid}/builds", params={"page_size": 100}).get("model_builds", []):
        for d in api("GET", f"{pid}/models/{mid}/builds/{b['id']}/deployments").get("model_deployments", []):
            if str(d.get("status", "")).lower() == "deployed":
                deployed.append((b["id"], d["id"]))
    if not deployed:
        raise AirflowException(f"{ENDPOINT_MODEL} has no running deployment to restart")
    build, dep = deployed[0]
    expected = _champion_run_id(api)
    api("POST", f"{pid}/models/{mid}/builds/{build}/deployments/{dep}:restart")
    print(f"restarted {ENDPOINT_MODEL} deployment {dep} (build {build}); expecting run "
          f"{expected or 'dated on or after ' + as_of}")

    domain = host.split("://", 1)[1]
    url = f"https://modelservice.{domain}/model"
    probe = {"accessKey": model["access_key"],
             "request": {"action": "categorise", "merchant_texts": ["UBER TRIP 8841"]}}
    t0 = time.time()
    deadline = t0 + ENDPOINT_DEADLINE_MIN * 60
    time.sleep(60)
    while time.time() < deadline:
        try:
            r = requests.post(url, json=probe, headers=headers, timeout=120)
            served = (r.json().get("response") or {}).get("model") or {} if r.ok else {}
        except (requests.RequestException, ValueError):
            served = {}
        if served:
            ok = served.get("run_id") == expected if expected else served.get("run_date", "") >= as_of
            print(f"endpoint serves run {served.get('run_id')} ({served.get('run_date')}, feature hash "
                  f"{served.get('feature_hash')}) after {time.time() - t0:.0f} s")
            if ok:
                return served.get("run_id")
        time.sleep(30)
    raise AirflowException(f"{ENDPOINT_MODEL} did not serve the expected run within {ENDPOINT_DEADLINE_MIN} minutes")


default_args = {
    "owner": "spend-analytics",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="spend_analytics_pipeline",
    description="Six source extracts -> bronze/silver/gold with GX gates (CDE) -> categorise, personas, "
                "forecast, signals with a KPI gate (CAI)",
    default_args=default_args,
    schedule_interval=DAILY,
    # In the past, or manual triggers run no tasks. With catchup=False, an unpaused DAG
    # runs the latest closed interval as soon as it is registered, hence paused on
    # creation: unpausing (cde job schedule unpause) is a deliberate step.
    start_date=datetime(2026, 9, 27, 21, 0),
    catchup=False,
    is_paused_upon_creation=True,
    # Each run regenerates bronze and replaces its run date's outputs: two at once would collide.
    max_active_runs=1,
    params={"as_of": ""},
    tags=["spend", "iceberg", "great-expectations", "minilm", "chronos-2", "kmeans"],
) as dag:

    def dq_task(task_id: str, layer: str) -> CDEJobRunOperator:
        # run-time args replace the job's own args, so repeat --db-prefix; --pipeline-run
        # groups this run's three layers in dq_results under the DAG run_id
        return CDEJobRunOperator(
            task_id=task_id,
            job_name=f"{JOB_PREFIX}-dq-check",
            overrides={"spark": {"args": ["--db-prefix", DB_PREFIX, "--layer", layer, "--as-of", AS_OF,
                                          "--pipeline-run", "{{ run_id }}"]}},
            wait=True,
        )

    def cai_task(task_id: str, job_var: str) -> PythonOperator:
        # retries=0: the CAI jobs are not idempotent mid-run and a gate failure is not transient
        return PythonOperator(task_id=task_id, python_callable=trigger_cai_job,
                              op_kwargs={"as_of": AS_OF, "job_var": job_var}, retries=0)

    generate = CDEJobRunOperator(
        task_id="generate_spend_bronze",
        job_name=f"{JOB_PREFIX}-generate-bronze",
        # run-time args replace the job's own args, so repeat --db-prefix
        overrides={"spark": {"args": ["--db-prefix", DB_PREFIX, "--as-of", AS_OF]}},
        wait=True,
    )
    dq_bronze = dq_task("dq_bronze", "bronze")
    silver = CDEJobRunOperator(task_id="build_silver", job_name=f"{JOB_PREFIX}-build-silver", wait=True)
    dq_silver = dq_task("dq_silver", "silver")
    categorise = cai_task("cai_categorise", "SPEND_CAI_CATEGORISE_JOB_ID")
    gold = CDEJobRunOperator(
        task_id="build_gold_features",
        job_name=f"{JOB_PREFIX}-build-gold-features",
        overrides={"spark": {"args": ["--db-prefix", DB_PREFIX, "--as-of", AS_OF]}},
        wait=True,
    )
    dq_gold = dq_task("dq_gold", "gold")
    features = cai_task("cai_feature_engineering", "SPEND_CAI_FEATURES_JOB_ID")
    train = cai_task("cai_model_training", "SPEND_CAI_TRAIN_JOB_ID")
    evaluate = cai_task("cai_model_evaluation", "SPEND_CAI_EVALUATE_JOB_ID")
    # retries=0: a restart that never serves the new run needs a look, not a second restart
    refresh = PythonOperator(task_id="refresh_endpoint", python_callable=refresh_endpoint,
                             op_kwargs={"as_of": AS_OF}, retries=0)

    (generate >> dq_bronze >> silver >> dq_silver >> categorise >> gold >> dq_gold >> features >> train >> evaluate
     >> refresh)
