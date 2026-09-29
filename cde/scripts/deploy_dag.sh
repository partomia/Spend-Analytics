#!/usr/bin/env bash
# Register/update the Airflow DAG as a `--type airflow` CDE job sourced from
# the repository. Re-run after every DAG change: `cde repository sync` alone
# does not refresh an already-registered DAG.
#
# Set the seven SPEND_CAI_* Airflow Variables first (cde/dags/spend_dag.py).
# The DAG is created paused (is_paused_upon_creation=True in the DAG; CDE takes an
# Airflow job's schedule from the DAG only): with catchup=False, unpausing starts
# a run for the latest closed interval at once. An update keeps the paused state.

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
echo "DAG spend_analytics_pipeline registered paused (daily, 21:00 UTC). To arm it:"
echo "  cde job schedule unpause --name ${DAG_JOB_NAME}"
