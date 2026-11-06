#!/usr/bin/env bash
# The Databricks side of the export, simulated on CDE with open-source Delta Lake: two
# on-demand Spark jobs (not in the nightly DAG) that apply the tokenized Parquet batches
# (databricks_export/) to SCD type 2 Delta tables under databricks_sim/, then check them.
#
#   rsingh-spend-anl-delta-sim        databricks/ingest_manifest_driven.py, the same script a
#                                     Databricks job runs, with path-based Delta targets
#   rsingh-spend-anl-delta-sim-check  cde/jobs/delta_sim_check.py: manifests vs versions, open
#                                     rows vs source, tokens decrypt, keys = Iceberg at the
#                                     last batch's snapshot (key from the workload credential)
#
#   ./cde/scripts/deploy_delta_sim.sh
#   cde job run --name rsingh-spend-anl-delta-sim --wait
#   cde job run --name rsingh-spend-anl-delta-sim-check --wait
#
# Delta Lake 3.3.x is the line built for Spark 3.5 / Scala 2.12 (CDE's Spark). The jars and
# the delta-spark Python package go into a files resource. Delta takes over spark_catalog,
# so Cloudera's Iceberg tables are read through a second catalog, ice (type hive). The Delta
# tables are path-based and never registered in the shared metastore.
#
# Remove: cde job delete --name <job> for both jobs, cde resource delete --name
# rsingh-spend-anl-delta-sim, and the databricks_sim/ prefix in S3.

set -euo pipefail

JOB_PREFIX="${JOB_PREFIX:-rsingh-spend-anl}"
RESOURCE="${RESOURCE:-${JOB_PREFIX}-delta-sim}"
TOKEN_CREDENTIAL="${TOKEN_CREDENTIAL:-rsingh-spend-anl-token-key}"
DELTA_VERSION="${DELTA_VERSION:-3.3.2}"
BUCKET_ROOT="${BUCKET_ROOT:-s3a://federal-buk-574bcea0/data/rsingh_spend_anl}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$(mktemp -d)"
trap 'rm -rf "${BUILD}"' EXIT

MAVEN="https://repo1.maven.org/maven2/io/delta"
JARS=("delta-spark_2.12-${DELTA_VERSION}.jar" "delta-storage-${DELTA_VERSION}.jar")
echo "==> Delta Lake ${DELTA_VERSION} jars and Python package"
curl -fsSL -o "${BUILD}/${JARS[0]}" "${MAVEN}/delta-spark_2.12/${DELTA_VERSION}/${JARS[0]}"
curl -fsSL -o "${BUILD}/${JARS[1]}" "${MAVEN}/delta-storage/${DELTA_VERSION}/${JARS[1]}"
python3 -m pip install --quiet --no-deps --target "${BUILD}/py" "delta-spark==${DELTA_VERSION}"
(cd "${BUILD}/py" && zip -qr "${BUILD}/delta_python.zip" delta)

echo "==> Files resource: ${RESOURCE}"
cde resource create --name "${RESOURCE}" --type files 2>/dev/null || true
for f in "${BUILD}/${JARS[0]}" "${BUILD}/${JARS[1]}" "${BUILD}/delta_python.zip" \
         "${REPO}/databricks/ingest_manifest_driven.py" "${REPO}/cde/jobs/delta_sim_check.py"; do
  cde resource upload --name "${RESOURCE}" --local-path "${f}" --hide-progress-bars
done

COMMON=(--type spark --mount-1-resource "${RESOURCE}"
        --jar "${JARS[0]}" --jar "${JARS[1]}" --py-file delta_python.zip
        --driver-cores 2 --driver-memory 4g --executor-cores 2 --executor-memory 4g
        --min-executors 1 --initial-executors 1 --max-executors "${MAX_EXECUTORS:-4}"
        --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension,org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions
        --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog
        --conf spark.sql.catalog.ice=org.apache.iceberg.spark.SparkCatalog
        --conf spark.sql.catalog.ice.type=hive
        --conf spark.sql.shuffle.partitions=64)
ARGS=(--arg --root --arg "${BUCKET_ROOT}/databricks_export" --arg --target --arg "${BUCKET_ROOT}/databricks_sim")

create() {  # name, then job-specific flags
  local name="$1"; shift
  cde job delete --name "${name}" 2>/dev/null || true
  cde job create --name "${name}" "${COMMON[@]}" "$@"
  echo "    ${name}"
}
echo "==> Jobs"
create "${JOB_PREFIX}-delta-sim" --application-file ingest_manifest_driven.py "${ARGS[@]}"
create "${JOB_PREFIX}-delta-sim-check" --application-file delta_sim_check.py "${ARGS[@]}" \
  --arg --credential --arg "${TOKEN_CREDENTIAL}" --workload-credential "${TOKEN_CREDENTIAL}"
