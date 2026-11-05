# Cloudera Iceberg → tokenized Parquet → Databricks Unity Catalog

The silver tables built on Cloudera (CDE Spark, Iceberg v2) are handed to
Databricks as a **tokenized, incremental Parquet change feed** in the project's
S3 bucket. Databricks applies the batches in order to Unity Catalog Delta tables
with **SCD type 2** history, and only members of one account group can turn the
tokens back into real identifiers.

This is the operator's guide. The reasoning behind it (research, options weighed,
validation) and how to apply the pattern to other tables are in
[`docs/DATABRICKS_EXPORT_BLUEPRINT.md`](../docs/DATABRICKS_EXPORT_BLUEPRINT.md).

```
Cloudera (CDE, nightly DAG)                                   S3                                   Databricks
───────────────────────────                                   ──                                   ──────────
rsingh_spend_anl_silver.customer  ┐                                                               spend_bronze.*_changes   (streaming, raw batches)
rsingh_spend_anl_silver.account   ├─ export_databricks ──►  databricks_export/silver/<table>/  ──►       │ AUTO CDC ... STORED AS SCD TYPE 2
rsingh_spend_anl_silver.spend_event┘   (after dq_gold)        data/<batch>/part-*.parquet                spend_silver.<table>     (tokens, full history)
                                                              _manifests/<batch>.json                    │ detok() for spend_pii_readers
                                                                                                         spend_silver_v.<table>   (current rows)
```

| | |
|---|---|
| Job | `cde/jobs/export_parquet.py`, CDE job `rsingh-spend-anl-export-parquet` |
| When | DAG task `export_databricks`, after `dq_gold`, beside the ML jobs (a failed export never holds back the models) |
| What | `config/export.yaml`: tables, primary keys and a rule per column |
| Where | `s3://federal-buk-574bcea0/data/rsingh_spend_anl/databricks_export/` |
| Key | CDE workload credential `rsingh-spend-anl-token-key`; Databricks secret `spend_tokens/enc_key_v1` |

## 1. What leaves Cloudera

`config/export.yaml` lists every column of every exported table, **deny by
default**: a column added to silver later fails the export until someone gives
it a rule.

| Rule | Columns | In the Parquet |
|---|---|---|
| `encrypt` | `cif`, `account_id`, `event_id` | deterministic AES-256-GCM token, base64 |
| `prefix3` | `customer.home_pincode` | first 3 digits, column renamed `home_pincode3` |
| `month` | `customer.onboarded_on`, `account.opened_on` | first day of the month |
| `keep` | everything else, including `merchant_text` | unchanged |

Each row also carries `_op` (I / U / D), `_batch`, `_source_snapshot_id` and
`_exported_at`.

## 2. Tokens

```
iv    = HMAC-SHA256(iv_key, value)[:12]          deterministic: same value, same token
token = base64( iv || AES-256-GCM(enc_key, iv, value) || tag16 )
```

This is the byte layout Spark's and Databricks' `aes_decrypt(..., 'GCM')`
expects, so decryption is one built-in SQL call:

```sql
CAST(aes_decrypt(unbase64(cif), unbase64(secret('spend_tokens', 'enc_key_v1')), 'GCM') AS STRING)
```

- Because the token is deterministic, `cif` in `customer`, `account` and
  `spend_event` is the same token: the tables join on tokens without decrypting.
- There are two keys. `enc_key` (shared with Databricks) decrypts. `iv_key`
  stays on Cloudera; without it nobody can make a token for a guessed value.
  This is the reason the IV comes from a separate HMAC key and not from `enc_key`.
- Deterministic encryption shows which rows share a value (as any join key must).
  It does not reveal the value.
- Tokens don't compress: `spend_event` is about 5 times larger than the plain
  Parquet (about 5 GB for 30 M rows).

### Key custody

| Where | What |
|---|---|
| Cloudera | CDE workload credential `rsingh-spend-anl-token-key`, files `key_version`, `enc_key_b64`, `iv_key_b64`, mounted into the job at `/etc/dex/secrets/rsingh-spend-anl-token-key/` |
| Databricks | secret scope `spend_tokens`, key `enc_key_v1` = `enc_key_b64` only |
| Laptop backup | `~/.spend_tokens/rsingh-spend-anl-token-key-v1.json` (mode 600, never in git) |

The job's AWS identity (the shared EKS node role) cannot read AWS Secrets
Manager (tested: `AccessDenied` on `ListSecrets` and `GetSecretValue`), hence
the CDE credential. No key is ever written to a manifest, a log or a Spark plan;
encryption runs inside a pandas UDF.

**Rotation:** create `v2` (new `enc_key` and `iv_key`), update the CDE
credential, add `enc_key_v2` to the scope. The next run sees a new
`key_version` and writes a **resync** batch (see 4). Every token changes, so the
Databricks tables are rebuilt from that batch.

## 3. Change detection on Cloudera

The silver tables are rebuilt every night (`createOrReplace`), so there is no
row-level change log to read. The job instead compares two Iceberg snapshots:

1. The current snapshot `S_now`, and the snapshot of the last export,
   `S_prev` (`to_snapshot_id` in the latest manifest), read with time travel
   (`VERSION AS OF`).
2. A join on the primary key, with `xxhash64(to_json(struct(other columns)))`
   on both sides:
   - key only in `S_now`: `I`
   - key in both, hash differs: `U` (the full new row)
   - key only in `S_prev`: `D` (key columns only, the rest NULL)
3. The rules and tokens are applied, and the batch is written.

The Iceberg tag `databricks_export` (`RETAIN 30 DAYS`) is moved to `S_now`
after each batch, so `S_prev` cannot expire while it is still needed.

If nothing changed (same snapshot), no batch is written.

## 4. The S3 layout and the manifest

```
databricks_export/silver/customer/
  data/000001/part-*.snappy.parquet      batch 1: full
  data/000002/part-*.snappy.parquet      batch 2: incremental
  _manifests/000001.json
  _manifests/000002.json                 written LAST: the commit marker
```

A manifest (no key material in it):

```json
{
  "batch": 2, "batch_type": "incremental", "reason": "diff against snapshot 5012...",
  "source_table": "rsingh_spend_anl_silver.customer", "primary_key": ["cif"],
  "from_snapshot_id": 5012, "to_snapshot_id": 5377, "source_row_count": 195463,
  "counts": {"I": 120, "U": 3410, "D": 15},
  "key_version": "v1", "rules_hash": "…", "tokenized_columns": ["cif"],
  "token_format": "base64(iv12 || AES-256-GCM ciphertext || tag16)",
  "columns": [{"name": "cif", "type": "string", "nullable": true}, "…"],
  "data_path": "s3a://…/silver/customer/data/000002", "previous_batch": 1,
  "exported_at": "2026-11-05T21:42:10+00:00", "files": 1, "bytes": 18200311
}
```

`data_path` uses the Hadoop scheme `s3a://`. On Databricks, read it as `s3://`.

| `batch_type` | When | Rows |
|---|---|---|
| `full` | first export of a table | every row as `I` |
| `incremental` | normal nightly run | only the I / U / D changes |
| `resync` | key rotated, column rules changed, `S_prev` expired, or `--resync` | every row as `I` |

A failed run leaves a data folder with no manifest. Readers ignore it, and the
next run takes a new batch number.

## 5. The Databricks side

| File | What |
|---|---|
| `01_setup.sql` | storage credential and external location, catalog and schemas, the `spend_tokens` secret scope (CLI), grants |
| `02_ingest_scd2.sql` | Lakeflow pipeline: Auto Loader on the data folders → `AUTO CDC ... STORED AS SCD TYPE 2` |
| `03_detokenize.sql` | `detok()` for the `spend_pii_readers` group, views of current rows, and the column mask alternative |
| `ingest_manifest_driven.py` | the strict alternative: a job that reads only committed batches via their manifests, and handles resync itself |

**Which ingestion to use.** `02_ingest_scd2.sql` is the simple one: the pipeline
reads every Parquet file under `data/`. A batch whose job failed *after* Spark
committed its files (read-back mismatch), but before the manifest, would still
be read. This is rare, but `ingest_manifest_driven.py` excludes it. It also
handles a resync batch by closing keys that are missing from the resync. With
the pipeline, you instead set `spend.start_batch` to the resync batch and run a
full refresh.

**SCD2 columns.** `SEQUENCE BY _batch`. Each key appears at most once per batch,
so the order is total. `__START_AT` and `__END_AT` are batch numbers. Join to
the manifests (`exported_at`) if you need wall-clock time. `D` rows close the
current version (`APPLY AS DELETE WHEN _op = 'D'`).

## 6. Run it

On Cloudera (the nightly DAG does this):

```bash
cde job run --name rsingh-spend-anl-export-parquet                       # all tables in config/export.yaml
cde job run --name rsingh-spend-anl-export-parquet --arg=--dry-run       # counts only, writes nothing
cde job run --name rsingh-spend-anl-export-parquet --arg=--resync --arg=--tables --arg=silver.customer
```

Run-time `--arg`s replace the job's own args. The job default is `--db-prefix
rsingh_spend_anl` anyway.

## 7. Test it with the Databricks side

1. After batch 1, `SELECT count(*) FROM spend_silver.customer WHERE __END_AT IS NULL`
   equals `counts.I` in `_manifests/000001.json`. Do the same for `account` and
   `spend_event`.
2. As a member of `spend_pii_readers`,
   `SELECT cif FROM spend_silver_v.customer LIMIT 1` returns a plain `C000…` id.
   As a non-member, the same query returns the token.
3. The join check: the current rows of `spend_silver.spend_event` joined to
   `spend_silver.customer` `USING (cif)` give the same count as the events alone,
   with no decryption.
4. After the next nightly batch, the SCD2 table has new versions for the batch's
   `U` keys, and closed versions (`__END_AT` set) for its `D` keys.
