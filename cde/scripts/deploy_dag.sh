#!/usr/bin/env bash
# Register/update the Airflow DAG as a `--type airflow` CDE job sourced from
# the repository. Re-run after every DAG change: `cde repository sync` alone
# does not refresh an already-registered DAG.
#
# Set the seven SPEND_CAI_* Airflow Variables first (cde/dags/spend_dag.py).
# The DAG is created unpaused; its first scheduled run is the interval closing
# after start_date, so registering it does not start a run by itself.

set -euo pipefail

REPO_NAME="${REPO_NAME:-rsingh-spend-anl-pipeline}"
DAG_JOB_NAME="${DAG_JOB_NAME:-rsingh-spend-anl-orchestration}"
DAG_PATH="cde/dags/spend_dag.py"

cde repository sync --name "${REPO_NAME}"

if cde job describe --name "${DAG_JOB_NAME}" &>/dev/null; then
  type="$(cde job describe --name "${DAG_JOB_NAME}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('type'))")"
  if [[ "${type}" != "airflow" ]]; then
    echo "==> ${DAG_JOB_NAME} has type ${type}; recreating as airflow"
    cde job delete --name "${DAG_JOB_NAME}"
  else
    echo "==> Updating ${DAG_JOB_NAME}"
    cde job update --name "${DAG_JOB_NAME}" --dag-file "${DAG_PATH}" --mount-1-resource "${REPO_NAME}"
    echo "Give Airflow ~30s to re-parse before triggering."
    exit 0
  fi
fi

echo "==> Creating ${DAG_JOB_NAME}"
cde job create --name "${DAG_JOB_NAME}" --type airflow --dag-file "${DAG_PATH}" --mount-1-resource "${REPO_NAME}"
echo "DAG spend_analytics_pipeline registered (daily, 21:00 UTC). Run now:"
echo "  cde job run --name ${DAG_JOB_NAME}"
