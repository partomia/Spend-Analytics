"""
Airflow DAG (CDE): daily spend analytics pipeline, 12 tasks.

  generate_spend_bronze -> dq_bronze -> build_silver -> dq_silver            (CDE Spark)
    -> cai_categorise                                                        (CAI Job: new merchant keys -> map)
    -> build_gold_features -> dq_gold                                        (CDE Spark)
    -> cai_feature_engineering -> cai_model_training -> cai_model_evaluation (CAI Jobs: the daily ML, PLAN decision 1)
    -> refresh_endpoint                                                      (CAI API: restart spend-scorer, verify)
     + deploy_registry_model (beside it)          (CAI API: build the new registry version into spend-persona-registry)

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

deploy_registry_model also runs only after a passed gate: it builds the champion
spend-persona version into the workbench model spend-persona-registry (deployed
straight from the registry, no code file), waits for the deployment and calls it
once. Skipped if that model was never created (ci/deploy_registered_model.py) or
the registry lists no version for as_of within REGISTRY_WAIT_MIN.

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
REGISTRY_ENDPOINT = "spend-persona-registry"
PERSONA_FEATURE_COUNT = 27
ENDPOINT_DEADLINE_MIN = 30
REGISTRY_WAIT_MIN = 5
# A run stuck "running" holds the shared vcluster quota and blocks every project's stages
# (runs 617/618 sat for 2.3 days): each task and the whole run have a hard limit.
SPARK_TIMEOUT = timedelta(minutes=90)          # bronze took 11-21 min, plus up to ~25 min queueing
CAI_TIMEOUT = timedelta(minutes=CAI_DEADLINE_MIN + 10)
ENDPOINT_TIMEOUT = timedelta(minutes=REGISTRY_WAIT_MIN + ENDPOINT_DEADLINE_MIN + 10)
DAGRUN_TIMEOUT = timedelta(hours=6)            # a full run takes about 50 min; well inside the daily interval


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
    started = False
    while time.time() < deadline:
        time.sleep(30)
        r = requests.get(f"{url}/{run_id}", headers=headers, timeout=60)
        r.raise_for_status()
        status = str(r.json().get("status", "")).lower().replace("engine_", "")
        print(f"CAI run {run_id}: {status}")
        started = started or status != "scheduling"
        if status in TERMINAL_OK:
            return run_id
        if status in TERMINAL_BAD and not started:
            # CAI fails a run that cannot be placed after 60 min, with no log
            raise AirflowException(
                f"CAI job run {run_id} never left scheduling and ended {status}: no room for its vCPU / memory, "
                "usually the per-user quota taken by running applications and models (scripts/platform_ctl.py stop)")
        if status in TERMINAL_BAD:
            raise AirflowException(f"CAI job run {run_id} ended with status {status}")
    raise AirflowException(f"CAI job run {run_id} did not finish within {CAI_DEADLINE_MIN} minutes")


def _cai_api(what: str):
    """(api, host, project path, headers) for the CAI API v2; skips the task when SPEND_CAI_HOST is not set."""
    host = Variable.get("SPEND_CAI_HOST", default_var="").rstrip("/")
    if not host:
        raise AirflowSkipException(f"SPEND_CAI_HOST not set: skipping {what}")
    headers = {"Authorization": f"Bearer {Variable.get('SPEND_CAI_API_KEY')}", "Content-Type": "application/json"}

    def api(method, path, params=None, body=None):
        r = requests.request(method, f"{host}/api/v2{path}", params=params, json=body, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json() if r.text else {}

    return api, host, f"/projects/{Variable.get('SPEND_CAI_PROJECT_ID')}", headers


def _param(version: dict, key: str) -> str | None:
    params = ((version.get("model_version_metadata") or {}).get("mlflow_metadata") or {}).get("params") or []
    return next((p["value"] for p in params if p.get("key") == key), None)


def _champion(api, as_of: str) -> dict | None:
    """The highest spend-persona version, once one for as_of (or later) is listed, else None.
    Only a passed, latest-dated run is registered (spend/registry.py), so that is the run to serve.
    A version just registered can take a while to be listed: wait up to REGISTRY_WAIT_MIN."""
    deadline, why = time.time() + REGISTRY_WAIT_MIN * 60, ""
    while True:
        try:
            models = api("GET", "/registry/models", params={"page_size": 100}).get("models", [])
            model = next((m for m in models if m.get("name") == REGISTRY_MODEL), None)
            versions = api("GET", f"/registry/models/{model['model_id']}").get("model_versions", []) if model else []
        except requests.RequestException as e:
            print(f"registry not reachable ({e})")
            return None
        best = max(versions, key=lambda v: v.get("number") or 0) if versions else None
        if best and (_param(best, "run_date") or "") >= as_of:
            print(f"registry: {REGISTRY_MODEL} version {best.get('number')} is champion "
                  f"(run {_param(best, 'run_id')}, {_param(best, 'run_date')})")
            return best
        why = (f"no {REGISTRY_MODEL} ({len(models)} models listed)" if model is None else
               f"{REGISTRY_MODEL} has no versions" if best is None else
               f"{REGISTRY_MODEL} version {best.get('number')} is for {_param(best, 'run_date')}")
        if time.time() > deadline:
            print(f"registry: {why}, nothing for {as_of} after {REGISTRY_WAIT_MIN} minutes")
            return None
        print(f"registry: {why}; waiting for the {as_of} version")
        time.sleep(30)


def refresh_endpoint(as_of: str, **_):
    """Restart the spend-scorer deployment so it loads the run just published, then check that it serves it."""
    api, host, pid, headers = _cai_api("the endpoint refresh")
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
    champion = _champion(api, as_of)
    expected = _param(champion, "run_id") if champion else None
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


def deploy_registry_model(as_of: str, **_):
    """Build the champion spend-persona version into REGISTRY_ENDPOINT and check it answers.
    The model itself is created once by ci/deploy_registered_model.py; a new build replaces its deployment."""
    api, host, pid, headers = _cai_api("the registry deployment")
    model = next((m for m in api("GET", f"{pid}/models", params={"page_size": 100}).get("models", [])
                  if m["name"] == REGISTRY_ENDPOINT), None)
    if model is None:
        raise AirflowSkipException(f"{REGISTRY_ENDPOINT} not created (ci/deploy_registered_model.py): skipping")
    champion = _champion(api, as_of)
    if champion is None:
        raise AirflowSkipException(f"no {REGISTRY_MODEL} version for {as_of}: {REGISTRY_ENDPOINT} unchanged")
    base = f"{pid}/models/{model['id']}/builds"
    builds = api("GET", base, params={"page_size": 100}).get("model_builds", [])
    build = next((b for b in builds if b.get("registered_model_version_id") == champion["model_version_id"]), None)
    if build is None:
        last = max(builds, key=lambda b: b.get("created_at") or "")
        build = api("POST", base, body={
            "project_id": pid.rsplit("/", 1)[1], "model_id": model["id"],
            "registered_model_version_id": champion["model_version_id"],
            # a build reports its kernel as "Python 3.11", but a new build accepts only python3 / python2 / r
            "runtime_identifier": last["runtime_identifier"], "kernel": "python3",
            "comment": f"registry {REGISTRY_MODEL} v{champion.get('number')} (DAG {as_of})",
            "auto_deploy_model": True, "auto_deployment_config": {"cpu": 1, "memory": 2, "replicas": 1}})
        print(f"{REGISTRY_ENDPOINT}: new build {build['id']} of version {champion.get('number')}")
    else:
        print(f"{REGISTRY_ENDPOINT}: build {build['id']} of version {champion.get('number')} exists")

    # A registry build can report "pushing" after its deployment is up: the deployment is what counts.
    t0, last_status = time.time(), None
    while time.time() < t0 + ENDPOINT_DEADLINE_MIN * 60:
        status = str(api("GET", f"{base}/{build['id']}").get("status", "")).lower()
        if "fail" in status:
            raise AirflowException(f"{REGISTRY_ENDPOINT} build {build['id']}: {status}")
        deps = api("GET", f"{base}/{build['id']}/deployments").get("model_deployments", [])
        status = str(deps[0].get("status", "")).lower() if deps else f"build {status}"
        if status != last_status:
            print(f"{REGISTRY_ENDPOINT}: {status} after {time.time() - t0:.0f} s")
            last_status = status
        if status == "deployed":
            break
        if status in TERMINAL_BAD:
            raise AirflowException(f"{REGISTRY_ENDPOINT} deployment {status}")
        time.sleep(30)
    else:
        raise AirflowException(f"{REGISTRY_ENDPOINT} not deployed within {ENDPOINT_DEADLINE_MIN} minutes")

    url = f"https://modelservice.{host.split('://', 1)[1]}/model"
    # Not whole numbers: a request of only whole numbers (0.0, 1.0) reaches MLflow as int64 and the
    # float64 signature answers HTTP 400; one fractional value anywhere in the request is enough.
    probe = {"accessKey": model["access_key"], "request": {"inputs": [[0.5] * PERSONA_FEATURE_COUNT]}}
    while time.time() < t0 + ENDPOINT_DEADLINE_MIN * 60:
        try:
            r = requests.post(url, json=probe, headers=headers, timeout=120)
            prediction = (r.json().get("response") or {}).get("prediction") if r.ok else None
        except (requests.RequestException, ValueError):
            prediction = None
        if isinstance(prediction, list) and len(prediction) == 1:
            print(f"{REGISTRY_ENDPOINT} serves {REGISTRY_MODEL} version {champion.get('number')}: "
                  f"a row of 0.5s is persona {prediction[0]} ({time.time() - t0:.0f} s)")
            return champion.get("number")
        time.sleep(30)
    raise AirflowException(f"{REGISTRY_ENDPOINT} deployed but did not answer within {ENDPOINT_DEADLINE_MIN} minutes")


default_args = {
    "owner": "spend-analytics",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": SPARK_TIMEOUT,
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
    dagrun_timeout=DAGRUN_TIMEOUT,
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
                              op_kwargs={"as_of": AS_OF, "job_var": job_var}, retries=0,
                              execution_timeout=CAI_TIMEOUT)

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
                             op_kwargs={"as_of": AS_OF}, retries=0, execution_timeout=ENDPOINT_TIMEOUT)
    # beside refresh_endpoint, so a slow registry build never delays spend-scorer
    registry_deploy = PythonOperator(task_id="deploy_registry_model", python_callable=deploy_registry_model,
                                     op_kwargs={"as_of": AS_OF}, retries=0, execution_timeout=ENDPOINT_TIMEOUT)

    (generate >> dq_bronze >> silver >> dq_silver >> categorise >> gold >> dq_gold >> features >> train >> evaluate
     >> refresh)
    evaluate >> registry_deploy
