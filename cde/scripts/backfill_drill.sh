#!/usr/bin/env bash
# Load the lakehouse day by day so the gold tables have one Iceberg snapshot per
# daily load, just like real overnight extracts: each load adds the week's
# spend_features snapshot and MERGEs the new weeks into cohort_weekly and
# mix_weekly. Between silver and gold the CAI categorise job maps the load's new
# merchant keys, as the DAG does; gold's category shares join that map.
# Afterwards run the CAI backfill (cai/jobs/backfill_history.py) to create the
# matching model runs, forecasts and signals.
#
#   ./cde/scripts/backfill_drill.sh                         # the last 3 Fridays, then yesterday
#   ./cde/scripts/backfill_drill.sh 2026-09-11 2026-09-18   # explicit dates
#
# The categorise step calls the CAI API v2 with the same four values the DAG
# reads from Airflow Variables, taken here from the environment (never commit them):
#   export SPEND_CAI_HOST=https://ml-xxxx... SPEND_CAI_PROJECT_ID=... \
#          SPEND_CAI_API_KEY=... SPEND_CAI_CATEGORISE_JOB_ID=...
# SKIP_CATEGORISE=1 skips it (only when the map already covers every load).
#
# Run-time --arg values replace the job's own args, so --db-prefix is passed
# again here. Run the chain back to back: the shared vcluster scales down
# after ~15 min idle and a cold scale-up can take 20+ minutes.

set -euo pipefail

JOB_PREFIX="${JOB_PREFIX:-rsingh-spend-anl}"
DB_PREFIX="${DB_PREFIX:-rsingh_spend_anl}"

if [[ $# -gt 0 ]]; then
  dates=("$@")
else
  read -r -a dates <<< "$(python3 - <<'EOF'
from datetime import date, timedelta
y = date.today() - timedelta(days=1)
friday = y - timedelta(days=(y.weekday() - 4) % 7)   # latest Friday on or before yesterday
out = sorted({friday - timedelta(weeks=k) for k in range(3)} | {y})
print(" ".join(d.isoformat() for d in out))
EOF
)"
fi

if [[ "${SKIP_CATEGORISE:-}" != "1" ]]; then
  for v in SPEND_CAI_HOST SPEND_CAI_PROJECT_ID SPEND_CAI_API_KEY SPEND_CAI_CATEGORISE_JOB_ID; do
    [[ -n "${!v:-}" ]] || { echo "${v} is not set (or SKIP_CATEGORISE=1)"; exit 1; }
  done
fi

run() { cde job run --name "$1" --arg=--db-prefix --arg="${DB_PREFIX}" "${@:2}" --wait; }

dq() { run "${JOB_PREFIX}-dq-check" --arg=--layer --arg="$1" --arg=--as-of --arg="$2" --arg=--pipeline-run --arg="drill-$2"; }

# Start the CAI categorise job for one load date and wait for it (stdlib only).
categorise() {
  [[ "${SKIP_CATEGORISE:-}" == "1" ]] && { echo "    categorise skipped"; return; }
  AS_OF="$1" python3 - <<'EOF'
import json, os, sys, time, urllib.request

host = os.environ["SPEND_CAI_HOST"].rstrip("/")
url = f"{host}/api/v2/projects/{os.environ['SPEND_CAI_PROJECT_ID']}/jobs/{os.environ['SPEND_CAI_CATEGORISE_JOB_ID']}/runs"
headers = {"Authorization": f"Bearer {os.environ['SPEND_CAI_API_KEY']}", "Content-Type": "application/json"}


def call(u, body=None):
    req = urllib.request.Request(u, data=json.dumps(body).encode() if body else None, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


env = {"SPEND_RUN_DATE": os.environ["AS_OF"], "SPEND_TRIGGERED_BY": "backfill-drill"}
run_id = call(url, {"environment": env})["id"]
print(f"    CAI categorise run {run_id} ({env['SPEND_RUN_DATE']})")
for _ in range(180):
    time.sleep(20)
    status = str(call(f"{url}/{run_id}").get("status", "")).lower().replace("engine_", "")
    if status == "succeeded":
        sys.exit(0)
    if status in {"failed", "stopped", "timedout"}:
        sys.exit(f"CAI categorise run {run_id} ended with status {status}")
sys.exit(f"CAI categorise run {run_id} did not finish within 60 minutes")
EOF
}

for as_of in "${dates[@]}"; do
  echo "==================== as of ${as_of} ($(date '+%H:%M:%S'))"
  run "${JOB_PREFIX}-generate-bronze" --arg=--as-of --arg="${as_of}"
  dq bronze "${as_of}"
  run "${JOB_PREFIX}-build-silver"
  dq silver "${as_of}"
  categorise "${as_of}"
  run "${JOB_PREFIX}-build-gold-features" --arg=--as-of --arg="${as_of}"
  dq gold "${as_of}"
done

echo ""
echo "Gold snapshots, one per load. In Hue (Impala):"
echo "  DESCRIBE HISTORY ${DB_PREFIX}_gold.spend_features;"
echo "Now create the matching model runs from a CAI session:"
echo "  python cai/jobs/backfill_history.py --weeks 3"
