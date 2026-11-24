# Cloudera to Databricks: the Parquet push pipeline

How the nightly tokenized Parquet export on Cloudera reaches a **Databricks Free
Edition** workspace, and how it was built and tested, step by step (24 Nov 2026).
The export itself (tokens, change detection, manifests) is in
[`databricks/README.md`](../databricks/README.md) sections 1-7; the design reasoning in
[`DATABRICKS_EXPORT_BLUEPRINT.md`](DATABRICKS_EXPORT_BLUEPRINT.md).

```
CDE Airflow DAG (nightly, 21:00 UTC)
  dq_gold -> export_databricks -> push_databricks
             (export_parquet.py)  (push_databricks.py)
                   |                    |
                   v                    |  Files API (HTTPS, Databricks PAT)
   s3://federal-buk-574bcea0/.../       |
   databricks_export/silver/<table>/ ---+--> /Volumes/workspace/spend_landing/cloudera_export/silver/<table>/
     data/<batch>/part-*.parquet                data/<batch>/part-*.parquet
     _manifests/<batch>.json                    _manifests/<batch>.json      (written last)
                                                        |
                                                        |  pipeline update started by push_databricks
                                                        v
                                   Lakeflow pipeline spend-cloudera-scd2 (02_ingest_scd2.sql, serverless)
                                     Auto Loader (STREAM read_files) -> spend_bronze.<table>_changes
                                     AUTO CDC ... STORED AS SCD TYPE 2 -> spend_silver.<table>
                                                        |
                                     spend_silver_v.<table>: current rows, detok() for spend_pii_readers
```

## 1. Why a push

The original design (`databricks/01_setup.sql`) has Databricks read the export
straight from S3 through a Unity Catalog storage credential and external location.
**Free Edition does not support that**: "custom workspace storage locations" are not
supported, and storage credential validation fails at Assume Role, however the IAM role
is set up. The options weighed:

| Option | Verdict |
|---|---|
| Make the bucket public | Rejected. It is the shared data-lake bucket of `federal-cdp-env` (sibling projects too), the export holds 30 M transactions, and Free Edition still cannot read `s3://` without an external location |
| Copy by hand from a laptop (`aws s3 sync`, `databricks fs cp`) | Fine for a one-off test, not nightly |
| **Cloudera pushes into a Unity Catalog Volume over the Databricks Files API** | **Chosen**: no AWS key in Databricks, the bucket stays private, and it runs in the nightly DAG |
| A paid or trial workspace | Needed only to test `01_setup.sql` as written |

## 2. What was in place before

- Cloudera: the nightly export (`cde/jobs/export_parquet.py`, CDE job
  `rsingh-spend-anl-export-parquet`, DAG task `export_databricks`) had written batches
  1-13 of `silver.customer`, `silver.account` and `silver.spend_event` to S3.
- The token key: CDE workload credential `rsingh-spend-anl-token-key`; laptop backup
  `~/.spend_tokens/rsingh-spend-anl-token-key-v1.json` (mode 600, never in git).
- The Databricks SQL: `databricks/02_ingest_scd2.sql` (pipeline) and
  `databricks/03_detokenize.sql` (`detok()` and views).

## 3. Databricks workspace setup (once)

Workspace: `https://dbc-4fc428bf-54a6.cloud.databricks.com` (Free Edition, AWS). Free
Edition has a single catalog, `workspace`, used everywhere for `<catalog>`.

### 3.1 CLI and login

```bash
brew install databricks/tap/databricks          # Databricks CLI (v1.27 used)
databricks auth login --host https://dbc-4fc428bf-54a6.cloud.databricks.com --profile spend-free
databricks -p spend-free current-user me        # check: an admin of the workspace
```

### 3.2 Schemas and the landing Volume (SQL editor)

```sql
CREATE SCHEMA IF NOT EXISTS workspace.spend_landing;
CREATE VOLUME IF NOT EXISTS workspace.spend_landing.cloudera_export;
CREATE SCHEMA IF NOT EXISTS workspace.spend_bronze;     -- raw change batches
CREATE SCHEMA IF NOT EXISTS workspace.spend_silver;     -- SCD2 tables, tokens
CREATE SCHEMA IF NOT EXISTS workspace.spend_silver_v;   -- current rows, detokenized for members
CREATE SCHEMA IF NOT EXISTS workspace.spend_gov;        -- detok()
```

The storage credential, external location and grants of `01_setup.sql` are skipped.

### 3.3 The decryption key and who may use it

Only `enc_key_b64` goes to Databricks, never `iv_key`. The value is piped from the
backup file, so it is not on a command line or in shell history:

```bash
databricks -p spend-free secrets create-scope spend_tokens
python3 -c "import json,os,sys; sys.stdout.write(json.load(open(os.path.expanduser(
  '~/.spend_tokens/rsingh-spend-anl-token-key-v1.json')))['enc_key_b64'])" \
  | databricks -p spend-free secrets put-secret spend_tokens enc_key_v1
```

The group `spend_pii_readers` (Settings > Identity and access, or the CLI), with yourself
as a member, then READ on the scope:

```bash
databricks -p spend-free groups create --json '{"displayName":"spend_pii_readers","members":[{"value":"<your user id>"}]}'
databricks -p spend-free secrets put-acl spend_tokens spend_pii_readers READ
```

### 3.4 The pipeline

Upload the pipeline SQL as a workspace file:

```bash
databricks -p spend-free workspace mkdirs /Workspace/Users/<you>/spend-analytics
databricks -p spend-free workspace import /Workspace/Users/<you>/spend-analytics/02_ingest_scd2.sql \
  --file databricks/02_ingest_scd2.sql --format RAW --overwrite
```

Create it (`databricks pipelines create --json @spend_pipeline.json`):

```json
{
  "name": "spend-cloudera-scd2",
  "catalog": "workspace",
  "schema": "spend_silver",
  "serverless": true,
  "continuous": false,
  "development": true,
  "channel": "CURRENT",
  "root_path": "/Workspace/Users/<you>/spend-analytics",
  "libraries": [{"file": {"path": "/Workspace/Users/<you>/spend-analytics/02_ingest_scd2.sql"}}],
  "configuration": {
    "spend.root": "/Volumes/workspace/spend_landing/cloudera_export",
    "spend.start_batch": "1"
  }
}
```

The SQL is unchanged from the repo: only `spend.root` points at the Volume in place of
S3. Pipeline id: `e4e063c8-717b-4ef9-937a-bbfcb00e2548`.

### 3.5 Detokenization

Run `databricks/03_detokenize.sql` with `<catalog>` replaced by `workspace`. One Free
Edition difference: groups made in the workspace are **workspace groups** (SCIM
`resourceType: WorkspaceGroup`), and `is_account_group_member('spend_pii_readers')` returns
false for them; Free Edition has no account console for account groups. In this
workspace `detok()` therefore checks `is_member()`:

```sql
CREATE OR REPLACE FUNCTION workspace.spend_gov.detok(tok STRING)
  RETURNS STRING
  RETURN CASE
    WHEN tok IS NULL THEN NULL
    WHEN is_member('spend_pii_readers')
      THEN CAST(aes_decrypt(unbase64(tok), unbase64(secret('spend_tokens', 'enc_key_v1')), 'GCM') AS STRING)
    ELSE tok
  END;
```

The repo file keeps `is_account_group_member()`, the right check on a full account.

## 4. First test, before the push existed

To prove the Databricks side on its own, a batch 1 was exported on the laptop from the
local 20,000-customer book with the same token key, and uploaded by hand:

```bash
# export_parquet.py on local Spark + Iceberg (scripts/run_cde_local.py's session), to a local folder
cd scripts && SPEND_EXPORT_KEY_FILE=~/.spend_tokens/rsingh-spend-anl-token-key-v1.json ../.venv/bin/python - <<'EOF'
import os, sys
sys.argv = ["x"]
import run_cde_local as r
spark = r.local_spark(r.ROOT / "data" / "warehouse")
job = r.load_job("export_parquet.py")
job.main(["--db-prefix", "rsingh_spend_anl", "--target", "file://" + os.path.expanduser("~/spend_dbx_export")], spark=spark)
EOF
find ~/spend_dbx_export -name '*.crc' -delete       # local Spark checksums; S3 has none
databricks -p spend-free fs cp -r --overwrite ~/spend_dbx_export/silver \
  dbfs:/Volumes/workspace/spend_landing/cloudera_export/silver
databricks -p spend-free pipelines start-update e4e063c8-717b-4ef9-937a-bbfcb00e2548
```

The pipeline completed in about 2 minutes, and the counts matched the manifests
(19,416 / 22,987 / 2,801,881). One fix came out of it: the manifests are pretty-printed
JSON, so reading them needs `multiLine => true` (fixed in `02_ingest_scd2.sql`).

## 5. The push job

### 5.1 Network check from CDE

Before writing it, a probe (`cde spark submit`, 1 core / 2 GB) called the workspace API
without a token from a CDE pod: **HTTP 401**, so CDE can reach Databricks. The same run
listed the 13 batches per table on S3 (spend_event batch 1: 16 files, 5.5 GB).

### 5.2 `cde/jobs/push_databricks.py`

Per table in `config/export.yaml`:

1. Read the manifests on S3 (Hadoop FS on the driver, the job's own S3 access) and in the
   Volume (Files API: `GET /api/2.0/fs/directories/...`, `GET /api/2.0/fs/files/...`).
2. `plan_push`:
   - a batch in both must have the same `to_snapshot_id` and `exported_at`, otherwise
     the Volume holds another export's batches and the job stops;
   - an empty Volume starts at the latest `full` or `resync` batch;
   - a gap (a batch missing below the Volume's latest) stops the job;
   - otherwise: the missing batches, in order.
3. Per batch: check the number of Parquet files against the manifest, upload them from
   the executors (`binaryFiles`, one task per file, `PUT /api/2.0/fs/files/...`), check
   each file's size in the Volume (`HEAD`), then upload the manifest. The manifest stays
   the commit marker.
4. If anything was pushed: wait for any running pipeline update, start one
   (`POST /api/2.0/pipelines/<id>/updates`), and poll it until COMPLETED (fail on
   FAILED / CANCELED, or after `wait_minutes`). A `resync` pushed on top of earlier
   batches does not start the pipeline: it needs `spend.start_batch` and a full refresh,
   and the job fails to say so.

Flags: `--dry-run` (list only), `--no-trigger` (push without starting the pipeline),
`--tables`. HTTP 429 / 5xx are retried 4 times with backoff.

### 5.3 Settings: `config/export.yaml`

```yaml
databricks_push:
  host: https://dbc-4fc428bf-54a6.cloud.databricks.com
  volume_root: /Volumes/workspace/spend_landing/cloudera_export
  pipeline_id: e4e063c8-717b-4ef9-937a-bbfcb00e2548   # spend-cloudera-scd2
  credential: rsingh-spend-anl-databricks              # CDE workload credential, file "token"
  wait_minutes: 45
```

### 5.4 DAG task (`cde/dags/spend_dag.py`)

```python
push = CDEJobRunOperator(task_id="push_databricks", job_name=f"{JOB_PREFIX}-push-databricks", wait=True)
dq_gold >> export >> push
```

The DAG now has 14 tasks. The push sits beside the ML jobs, like the export: a failure
never holds back the models. It is idempotent, so the DAG's one retry is safe.

### 5.5 Tested on the laptop before deploying

With the token in `SPEND_DATABRICKS_TOKEN` and `target` pointed at the local export:

- a dry run against the Volume: every table "to push none" (the manifests matched);
- a push of `customer` and `account` into a scratch folder `.../_push_test`: uploaded,
  sizes equal, a second run pushed nothing; the scratch folder was then removed.

That found two bugs, fixed before the commit: Hadoop hands paths back as `file:/` for
`file:///` (uploads now match on the file name within a batch), and a duplicate
`headers` argument on the upload.

Unit tests (`tests/test_cde_jobs.py`, `tests/test_orchestration.py`): the push plan
(start at the latest base, no gaps, a rerun pushes nothing, another export stops it),
the resync hold, the config, the job's credential and resources in `deploy_jobs.sh`, and
the DAG (14 tasks, `dq_gold >> export >> push`). pytest: 233 passed.

## 6. Deployment on Cloudera

### 6.1 Commit and push

The CDE repository `rsingh-spend-anl-pipeline` syncs from GitHub `main`: commit
`f34fa74` (job, DAG, config, tests, docs), then `b1009cb` (project log).

### 6.2 The Databricks token as a CDE workload credential

A personal access token, 90 days, created straight into a mode-600 temp file (never
printed), loaded into CDE, and the file deleted:

```bash
umask 077
databricks -p spend-free tokens create --lifetime-seconds 7776000 \
  --comment "CDE rsingh-spend-anl-push-databricks" -o json \
  | python3 -c "import json,sys; json.dump({'token': json.load(sys.stdin)['token_value']}, open('/tmp/dbx_push_cred.json','w'))"
cde credential create --name rsingh-spend-anl-databricks --type workload-credential \
  --workload-cred-json-file /tmp/dbx_push_cred.json
rm -f /tmp/dbx_push_cred.json
```

It is mounted into the job at `/etc/dex/secrets/rsingh-spend-anl-databricks/token`.
Expiry: **2027-02-22**.

### 6.3 The job and the DAG

```bash
cde repository sync --name rsingh-spend-anl-pipeline
cde job create --name rsingh-spend-anl-push-databricks --type spark \
  --mount-1-resource rsingh-spend-anl-pipeline --application-file cde/jobs/push_databricks.py \
  --python-env-resource-name rsingh-spend-anl-python-env \
  --driver-cores 1 --driver-memory 2g --executor-cores 1 --executor-memory 2g \
  --min-executors 1 --initial-executors 1 --max-executors 4 --conf spark.executor.memoryOverhead=2g \
  --arg=--db-prefix --arg=rsingh_spend_anl --workload-credential rsingh-spend-anl-databricks
./cde/scripts/deploy_dag.sh            # refuses while a DAG run is in flight
```

These are the flags `cde/scripts/deploy_jobs.sh` uses for the job; the script was not
re-run as a whole, since it recreates every job and rebuilds the Python environment. The
Python environment needed nothing new (`requests` was already there).

## 7. First live run

| Step | Result |
|---|---|
| Run 1334: `cde job run --name rsingh-spend-anl-push-databricks --arg=--dry-run` | Failed **as intended**: the Volume held the laptop's batch 1 (snapshot 3382918986255363505, S3 has 9123428909333812348). This also proved the S3 read and the token |
| Clear the Volume: `databricks fs rm -r dbfs:/Volumes/workspace/spend_landing/cloudera_export/silver` | The laptop test batches gone |
| Run 1335: `--arg=--no-trigger` | Batches 1-13 of all three tables in 7 min, including spend_event batch 1 (16 files, 5.5 GB) |
| Full refresh: `databricks pipelines start-update <id> --full-refresh` | COMPLETED in under 5 min; the tables rebuilt from the real feed |

The first load used `--no-trigger` and a manual full refresh because the pipeline's
tables still held the laptop test data. From now on the nightly push starts a normal
(incremental) update itself.

## 8. Checks

Run on the SQL warehouse after the full refresh:

```sql
-- current rows vs the manifests' I counts (batches 1-13)
SELECT count(*) FILTER (WHERE __END_AT IS NULL) FROM workspace.spend_silver.customer;
SELECT regexp_extract(_metadata.file_path, 'silver/([a-z_]+)/', 1) t, count(*) batches, sum(counts.I) i
FROM read_files('/Volumes/workspace/spend_landing/cloudera_export/silver/*/_manifests/',
                format => 'json', multiLine => true)
GROUP BY 1;
-- tokens join without decryption
SELECT count(*), count(c.cif) FROM workspace.spend_silver.spend_event e
LEFT JOIN workspace.spend_silver.customer c ON e.cif = c.cif AND c.__END_AT IS NULL
WHERE e.__END_AT IS NULL;
-- detokenized for spend_pii_readers
SELECT count(*), count_if(cif RLIKE '^C[0-9]+$') FROM workspace.spend_silver_v.customer;
```

| Check | Result |
|---|---|
| customer: current rows / manifests | 195,961 / 195,961 |
| account | 231,038 / 231,038 |
| spend_event | 31,046,359 / 31,046,359 |
| Events joined to customers on tokens | 31,046,359 of 31,046,359 |
| `spend_silver_v.customer` as a member | all 195,961 `cif`s decrypted (`C…`) |

All 13 batches so far are inserts only (new customers, accounts and events each night),
so no SCD2 version has been closed yet.

## 9. Operating it

| What | How |
|---|---|
| See what would be pushed | `cde job run --name rsingh-spend-anl-push-databricks --arg=--dry-run` |
| Push without ingesting | `--arg=--no-trigger` |
| Python logs of a run | `cde run logs --id N --download-all`, then `driver_stdout.log` (not in `--type driver/stderr`) |
| Pipeline runs | Databricks > Jobs & Pipelines > `spend-cloudera-scd2`, or `databricks pipelines list-updates <id>` |
| Renew the token before 2027-02-22 | a new PAT as in 6.2, then `cde credential update` (or delete and create) `rsingh-spend-anl-databricks`; an expired token fails the task with HTTP 403 |
| A resync batch (key rotation, rule change) | the push copies it and fails without starting the pipeline: set `spend.start_batch` to the batch and run a full refresh |
| Start the Volume again | `databricks fs rm -r` the Volume's `silver/`, run the push (`--no-trigger`), then a full refresh |
| Another Databricks workspace | `databricks_push` in `config/export.yaml`, a token for it in the credential, and sections 3.2-3.5 there |

## 10. Still to see live

- The push starting a pipeline update on its own: the next nightly batch (14). Run
  1330 of the DAG (23 Nov) failed before any export: `generate_spend_bronze` stayed in
  "starting" and was killed twice at 90 min, for lack of room on the shared CDE
  virtual cluster.
- The first `U` and `D` rows through AUTO CDC, closing SCD2 versions.
