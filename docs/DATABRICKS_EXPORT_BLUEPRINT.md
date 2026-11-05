# Blueprint: Cloudera Iceberg tables to a tokenized S3 landing zone for Databricks

This PoC shows how Cloudera lakehouse tables (Iceberg v2 on CDP) can be exported
every night to an S3 landing bucket. Selected columns are tokenized on the way
out, and Databricks builds Unity Catalog Delta tables from the files.
Authorised users on Databricks can reverse the tokens.

The document covers four things:
- what we researched;
- the options we weighed, and why we chose what we did;
- how we validated it;
- the jobs and pipelines that now run.

It is written so that the same pattern can be applied to other tables.
`databricks/README.md` is the operator's guide for this PoC's three tables;
this document explains the reasoning behind it.

| | |
|---|---|
| Source | Iceberg v2 tables in the Hive metastore, CDP Public Cloud on AWS `ap-south-1` |
| Export | CDE Spark 3.5 job `cde/jobs/export_parquet.py`, rules in `config/export.yaml` |
| Schedule | Airflow task `export_databricks` in the nightly DAG, after the gold data-quality gate |
| Landing zone | `s3://federal-buk-574bcea0/data/rsingh_spend_anl/databricks_export/` |
| Consumer | Databricks: Auto Loader + `AUTO CDC ... STORED AS SCD TYPE 2`, or a manifest-driven job |
| Status | Running nightly on the cluster since 2026-11-05; Databricks side tested locally on Delta, not yet on a workspace |

---

## 1. Requirements

1. **Regular, incremental export.** Each run delivers only the rows that were
   inserted, updated or deleted since the last run. It does not deliver a
   full copy every night.
2. **Selective tokenization.** Customer and account identifiers leave Cloudera
   only as tokens, and quasi-identifiers are coarsened. Everything else stays
   usable for analytics.
3. **Reversible on Databricks, for authorised users only.** Databricks must
   be able to turn a token back into the identifier inside a query, without a
   call back to Cloudera.
4. **Tables that are easy to build on Databricks.** The files must be plain
   Parquet with a documented contract, so that standard Databricks ingestion
   can apply them. The tables keep full history (SCD type 2).
5. **Nothing sensitive leaks by default.** A new source column must not reach
   S3 until someone has decided how to treat it.

## 2. Research: facts that shaped the design

We checked these on the cluster or in a local reproduction before designing
around them.

| Question | Finding | How we found out |
|---|---|---|
| Can we copy the Iceberg data files to S3? | No. Gold tables use MERGE with merge-on-read position deletes. On `gold.spend_features` the data files hold 16,551,537 records, of which 3,306,605 are deleted (13,244,932 real rows). A plain Parquet reader ignores delete files and would double-count. | `files` metadata table and a dry run on CDE (run 1048) |
| How do the silver tables change? | They are rebuilt every night with `createOrReplace`. That replaces the table's snapshot lineage: the previous night's snapshot is no longer a "current ancestor", though time travel by snapshot ID still reads it. | Iceberg `history` table, local reproduction |
| Does an Iceberg tag survive `createOrReplace`? | Yes. A tag pinned to the exported snapshot stays readable after the rebuild. The repo never runs `expire_snapshots`. | Local integration test |
| Can a CDE job read AWS Secrets Manager? | No. A job pod's AWS identity is the shared EKS worker-node role, and AWS returned `AccessDenied` for both `ListSecrets` and `GetSecretValue`. | Probe job on CDE (runs 1051 to 1053, since deleted) |
| Can a CDE job receive a secret another way? | Yes. A CDE workload credential is mounted into the job's pods as files under `/etc/dex/secrets/<credential>/`. | Probe job on CDE |
| Is a crypto library available to the job? | `cryptography` 50.0.1, pandas 3.0.6 and pyarrow 25.0.1 are in the CDE Python environment; `boto3` is not. | Probe job on CDE |
| Can Databricks decrypt without custom code? | Yes, if the token uses the layout of Spark's built-in `aes_decrypt(..., 'GCM')`: `iv(12) || ciphertext || tag(16)`. Python-made tokens decrypt with Spark SQL unchanged. | Unit test against local Spark |
| How large is tokenized data? | Tokens are random-looking and don't compress. Locally `spend_event` grew about 5 times (98 MB to 487 MB). On the cluster, 30.15 M rows came to 5.5 GB. | Local run, CDE run 1054 |

Tooling notes that cost time:
- `cde run logs` arrives 30 to 60 seconds late; an incomplete log isn't a failed run.
- `cde spark submit` cannot attach a workload credential, so the export must run as a defined job.

## 3. Options evaluated and decisions

### 3.1 Tokenization scheme

| Option | Joinable? | Reversible on Databricks? | Verdict |
|---|---|---|---|
| Hash (SHA-256 / HMAC) | yes | no | Fails requirement 3 |
| AES with a random IV | no (each row a different token) | yes | Fails joins and merges on keys |
| Format-preserving encryption (FF1/FF3) | yes | only with a third-party library on both sides | More moving parts than needed |
| External vault / tokenization service | yes | yes | Not available in the PoC. It is the production upgrade path if the bank has one (section 9) |
| **Deterministic AES-256-GCM, IV derived from the value** | **yes** | **yes, built-in `aes_decrypt`** | **Chosen** |

The token is

```
iv    = HMAC-SHA256(iv_key, value)[:12]
token = base64( iv || AES-256-GCM(enc_key, iv, value) || tag16 )
```

- **Joins and merges.** The same value always gives the same token, so tokens
  can be join keys, MERGE keys and SCD2 keys.
- **Decryption.** Decrypting needs only `enc_key`, which is shared with
  Databricks.
- **Forging.** Making a token needs `iv_key` as well, and `iv_key` never
  leaves Cloudera. So nobody on the Databricks side can make a token for a
  guessed value, for example to test whether a known customer ID is present.
- **The trade-off.** Deterministic encryption reveals which rows share a
  value. Any join key has to, so this is inherent, not a flaw.

### 3.2 Which columns, and how

Rules live in `config/export.yaml`, one per column, and are **deny by
default**. If a source column isn't listed, the export fails.

| Rule | Applied to | Effect |
|---|---|---|
| `encrypt` | `cif`, `account_id`, `event_id`, in every table where they appear | token |
| `prefix3` | `home_pincode` | first 3 digits; column renamed `home_pincode3` |
| `month` | `onboarded_on`, `opened_on` | first day of the month |
| `keep` | everything else, including `merchant_text` | unchanged |
| `drop` | (available, unused) | not exported |

A unit test checks that the rules for `spend_event` cover exactly the columns
silver builds. A local test showed that an unlisted `pan_number` column stops
the export.

### 3.3 Key custody

| Option | Verdict |
|---|---|
| AWS Secrets Manager read by the job | Not possible: `AccessDenied` (section 2) |
| Key passed as a Spark conf or job argument | Rejected: visible in the Spark UI, job definition and logs |
| **CDE workload credential, mounted as files** | **Chosen**: `rsingh-spend-anl-token-key`, files `key_version`, `enc_key_b64`, `iv_key_b64` |

On Databricks, only `enc_key` goes into the secret scope `spend_tokens`, as
`enc_key_v1`. Encryption runs inside a pandas UDF, so the key is never a
literal in a Spark plan; the cluster's run plan shows only `pythonUDF0`. The
offline backup is `~/.spend_tokens/rsingh-spend-anl-token-key-v1.json` (mode
600, never in git). Every commit was scanned for the key before pushing.

### 3.4 Change detection

| Option | Verdict |
|---|---|
| Full copy every night (this PoC's first version, `1615da0`) | Rejected: 5 GB a night, and Databricks cannot tell what changed |
| Iceberg incremental read (`start-snapshot-id`) | Rejected: it reads appended files only, and silver is rewritten, not appended |
| Iceberg changelog view | Rejected: it walks the snapshot lineage, which `createOrReplace` breaks |
| **Diff two snapshots on the primary key** | **Chosen**: works for any write pattern (append, overwrite, MERGE, rebuild) |

The diff reads the current snapshot `S_now` and the previously exported one
`S_prev` with time travel (`VERSION AS OF`). It joins them on the primary key
and compares `xxhash64(to_json(struct(other columns)))`. `to_json` drops NULL
fields, so `(a=NULL, b=1)` and `(a=1, b=NULL)` hash differently.

| Case | `_op` | Row content |
|---|---|---|
| key only in `S_now` | `I` | the full row |
| key in both, hash differs | `U` | the full new row |
| key only in `S_prev` | `D` | key columns only, the rest NULL |

`S_prev` is the `to_snapshot_id` in the last manifest. After each batch, an
Iceberg tag `databricks_export` (`RETAIN 30 DAYS`) is moved to `S_now`, so the
snapshot the next run diffs against cannot expire.

The job does a full **resync** (every row as `I`) instead of a diff in these
cases:
- the first export;
- the key version changed;
- the column rules changed;
- `S_prev` is no longer readable;
- an operator passes `--resync`.

### 3.5 Delivery format and commit protocol

```
<root>/<layer>/<table>/data/<batch:06d>/part-*.snappy.parquet
<root>/<layer>/<table>/_manifests/<batch:06d>.json      written last: the commit marker
```

- **Format.** Plain Parquet: snappy, `TIMESTAMP_MICROS` in UTC, and no Hive
  partition folders, so readers don't infer a partition column.
- **Metadata columns.** Every row carries `_op`, `_batch`, `_source_snapshot_id`
  and `_exported_at`.
- **Write order.** The job writes with `errorifexists`, reads the data back
  and checks the row count, then writes the manifest.
- **Failed runs.** A batch folder without a manifest is uncommitted and
  readers ignore it. The next run takes a new batch number.
- **Manifest contents.** The manifest records source table, snapshots,
  counts, schema, key version, rules hash, files and bytes. It holds no key
  material.

We kept Parquet rather than writing Delta or Iceberg into the landing zone.
The landing zone is a contract between two platforms, and Parquet plus a JSON
manifest is the lowest common denominator either side can read, audit and
replay. Databricks owns its Delta tables, so Cloudera never writes Databricks
table files.

### 3.6 Databricks ingestion

| Option | When to use |
|---|---|
| **Lakeflow pipeline: Auto Loader on `data/` + `AUTO CDC ... STORED AS SCD TYPE 2`** (`databricks/02_ingest_scd2.sql`) | The default: declarative, and Databricks manages checkpoints and history |
| **Manifest-driven job: read manifests, MERGE into Delta** (`databricks/ingest_manifest_driven.py`) | When only committed batches may ever be read, or resyncs must not need a full refresh |
| `COPY INTO` a staging table + `MERGE` | Possible; the other two cover it |
| Reading Cloudera's Iceberg tables directly (catalog federation) | Out of scope: identifiers would leave in clear |

How the tables are built:
- **SCD type 2.** Both ingest options use `SEQUENCE BY _batch`; each key
  appears at most once per batch, so the order is total.
- **Deletes.** `_op = 'D'` closes the current version.
- **History columns.** `__START_AT` and `__END_AT` are batch numbers. Join to
  the manifests' `exported_at` for wall-clock time.

Detokenization is a Unity Catalog SQL function, `detok(tok)`. It returns
`CAST(aes_decrypt(unbase64(tok), unbase64(secret('spend_tokens','enc_key_v1')), 'GCM') AS STRING)`
for members of the account group `spend_pii_readers`, and the token for
everyone else. There are also views of the current rows, and a column-mask
alternative. All of this is in `databricks/03_detokenize.sql`.

## 4. The pipeline as built

```
Nightly DAG spend_analytics_pipeline (CDE Airflow, 21:00 UTC)
  generate_spend_bronze > dq_bronze > build_silver > dq_silver > cai_categorise
    > build_gold_features > dq_gold > [ML jobs ... ]
                              dq_gold > export_databricks        beside the ML; a failed export never blocks the models

export_databricks = CDE job rsingh-spend-anl-export-parquet
  per table in config/export.yaml:
    1. current snapshot S_now; last manifest -> S_prev; plan the batch (none / full / incremental / resync)
    2. check every column has a rule (deny by default); check the primary key is unique
    3. diff S_prev vs S_now -> I / U / D; tokenize and coarsen; add metadata columns
    4. check no two keys tokenize alike; write Parquet; read back and compare counts
    5. write the manifest (commit); move tag databricks_export to S_now

Databricks (on its own schedule, after ~22:00 UTC)
  read_files / Auto Loader on data/  ->  bronze change tables  ->  AUTO CDC SCD2  ->  spend_silver.<table>
                                                                                    ->  spend_silver_v.<table> (detok for spend_pii_readers)
```

| Component | File |
|---|---|
| Export job | `cde/jobs/export_parquet.py` |
| Column rules, target, credential name | `config/export.yaml` |
| Job definition (with `--workload-credential`) | `cde/scripts/deploy_jobs.sh` |
| DAG task | `cde/dags/spend_dag.py` (`export_databricks`) |
| Demo notebook (throwaway key, scratch folder) | `notebooks/iceberg_to_parquet_export.ipynb` |
| Databricks setup, ingest, detokenization | `databricks/01_setup.sql`, `02_ingest_scd2.sql`, `03_detokenize.sql`, `ingest_manifest_driven.py` |
| Operator guide | `databricks/README.md` |

## 5. Validation

### 5.1 Unit tests (CI, every push)

In `tests/test_cde_jobs.py` and `tests/test_orchestration.py`:
- **Tokens:** they are deterministic, round-trip through Python, and Spark's
  `aes_decrypt` reads the Python-made token.
- **Change set:** I, U and D are detected, including a change from a value to
  NULL.
- **Column rules:** `prefix3`, `month`, `encrypt` and `drop` each do what they
  should, and deny-by-default rejects both unlisted and vanished columns.
- **Batch planning:** full, incremental and resync are each chosen in the
  right case.
- **Configuration:** the rules cover silver's real columns.
- **Deployment:** the DAG task sits after `dq_gold`, and the job mounts the
  credential.

### 5.2 Local integration tests (Spark + Iceberg, real silver data)

| Scenario | Expected | Result |
|---|---|---|
| First export of three tables | `full`, all rows `I` | pass |
| Customer rebuilt with 3 changed, 2 removed, 1 new row | `incremental`, `{"I":1,"U":3,"D":2}`, and the decrypted keys are exactly those rows | pass |
| Account table unchanged | batch type `none`, nothing written | pass |
| Decrypt with Spark SQL `aes_decrypt` | every token decrypts to the source value | pass |
| Join `spend_event` to `customer` on tokens | matches without decrypting | pass |
| Unlisted column (`pan_number`) added | export fails | pass |
| Key rotated to v2 | `resync`, every row `I` | pass |
| Tag `databricks_export` | on the exported snapshot, readable after `createOrReplace` | pass |

### 5.3 Databricks-side logic (local Delta Lake 4.0)

`ingest_manifest_driven.py` was run on synthetic batches:
- a full batch;
- an incremental batch with one U, one D and one I;
- a resync in which one key disappears and one arrives with a NULL column;
- a fourth folder with no manifest.

The SCD2 table came out exactly as expected:
- unchanged keys stayed open through the resync;
- the vanished key was closed;
- the uncommitted folder was ignored;
- a second run changed nothing.

### 5.4 On the cluster

| Run | What | Result |
|---|---|---|
| 1054 (2026-11-04, by hand) | Batch 1, `full` | customer 195,463, account 230,461, spend_event 30,152,902 rows (16 files, 5.5 GB); counts equal the source; tags set; about 8 min |
| 1062 (2026-11-05, DAG task) | Batch 2, `incremental` | customer +40, account +45, spend_event +104,308 (I only, no U or D); batch 1 plus batch 2 equals each source exactly; about 3 min |

## 6. Applying the blueprint to another table

1. **Add the table to `config/export.yaml`** with its primary key and a rule
   for every column. Tokenize identifiers, coarsen quasi-identifiers, and
   keep the rest.
2. **Check the key is unique per snapshot.** The job refuses duplicate keys.
   Use a composite key (a list) if needed.
3. **Run a dry run first:**
   `cde job run --name rsingh-spend-anl-export-parquet --arg=--db-prefix --arg=rsingh_spend_anl --arg=--tables --arg=<layer.table> --arg=--dry-run`.
   It shows the batch type and counts and writes nothing.
4. **Let the nightly task pick it up.** The first run of a new table is a
   `full` batch.
5. **On Databricks, add a bronze streaming table and a flow** to
   `02_ingest_scd2.sql`, or an entry in `TABLES` of the manifest-driven job.
   Route tokenized columns through `detok()` in a view or mask.

Things to decide for each new table:
- **Gold tables.** Large MERGE-loaded gold tables work too: reading through
  Spark applies the delete files. The diff then runs over the full table each
  night.
- **Large append-only tables.** For a very large append-only table, an
  incremental read of appended files would be cheaper than a diff; this is a
  possible optimisation.
- **Schema changes.** Any rule or column change triggers a resync, by design.

## 7. Operations

- **Monitoring.** Each manifest's `counts` should match the rows Databricks
  applied for that batch. Its `source_row_count` should match the current row
  count on Databricks (`__END_AT IS NULL`).
- **Failed export.** Re-run the job. The uncommitted folder is skipped and the
  next batch diffs against the last committed snapshot.
- **Key rotation:**
  1. Create the new key version and update the CDE credential.
  2. Add `enc_key_v2` to the Databricks scope.
  3. The next run writes a resync. Every token changes, so rebuild the
     Databricks tables from that batch: set `spend.start_batch` and run a full
     refresh of the pipeline. The manifest-driven job handles a resync on its
     own.
- **Snapshot retention.** If `expire_snapshots` is ever introduced, keep the
  tag, or keep retention longer than the longest expected pause between
  exports. Otherwise the next run falls back to a resync (safe, but larger).
- **Data retention.** Old `data/<batch>/` folders are not removed today. Add
  an S3 lifecycle rule once Databricks has applied them.

## 8. Limitations of the PoC

- **Not yet run on a workspace.** The Databricks side was validated on local
  Delta Lake. The SQL syntax was checked against the current Databricks
  documentation, but not run on a workspace.
- **`secret()` inside `detok()`.** Whether `secret()` may be called inside a
  SQL function or column mask must be confirmed on the workspace.
  `03_detokenize.sql` includes a fallback.
- **The pipeline option can read uncommitted files.** Auto Loader on `data/`
  can read a batch whose job failed after Spark committed the files but
  before the manifest was written. This is rare; the manifest-driven job
  excludes it.
- **What gets revealed.** Deterministic tokens reveal equality, and
  `merchant_text`, amounts and dates are exported in clear by decision.
- **Synthetic data.** The data is synthetic, and the bucket, keys and groups
  are PoC-scoped.

## 9. Moving to production

| Area | PoC | Production |
|---|---|---|
| Key storage | CDE workload credential + Databricks secret scope | A KMS-backed secret store both sides can reach, or the bank's tokenization vault (same token contract) |
| Key access on Databricks | `spend_pii_readers` group | Groups from the bank's identity provider, column masks or ABAC tags on all tokenized columns |
| Landing bucket | Project folder in the environment bucket | A dedicated bucket with a bucket policy: Cloudera writes, Databricks reads, lifecycle rule on `data/` |
| Ingestion | Pipeline, or manifest-driven job | Manifest-driven for strictness, or the pipeline with a manifest check before each update |
| Monitoring | Manual count checks | An automated check of manifest counts against applied rows, with an alert on mismatch or a missing batch |
