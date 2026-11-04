"""
Databricks job (PySpark, not a pipeline): apply the Cloudera spend batches to SCD type 2 Delta
tables, reading only committed batches (those with a manifest) and handling resync batches.

The stricter alternative to 02_ingest_scd2.sql (databricks/README.md, section 5):
  - a data folder without a manifest (a failed export) is never read;
  - a resync batch closes every current key that is missing from it, so key rotation
    (every token changes) and expired-snapshot resyncs need no full refresh.

Per table, Delta target columns: the exported columns + _batch, __START_AT, __END_AT (batch numbers).
State: the highest applied batch is max(_batch) of the target, so a rerun skips what is applied.
Run it as a job task after the nightly export (~22:00 UTC).
"""

import json

from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = "s3://federal-buk-574bcea0/data/rsingh_spend_anl/databricks_export"
CATALOG = "<catalog>"
TABLES = {
    "customer": {"keys": ["cif"]},
    "account": {"keys": ["account_id"]},
    "spend_event": {"keys": ["event_id"]},
}
META = ["_op", "_source_snapshot_id", "_exported_at"]
SKIP = {"_batch", "__START_AT", "__END_AT"}         # not part of a row's content

spark = SparkSession.builder.getOrCreate()


def manifests(table: str) -> list[dict]:
    path = f"{ROOT}/silver/{table}/_manifests/"
    try:
        files = dbutils.fs.ls(path)  # noqa: F821 - Databricks runtime global
    except Exception:  # noqa: BLE001 - nothing exported yet
        return []
    out = [json.loads(dbutils.fs.head(f.path, 10_000_000)) for f in files if f.name.endswith(".json")]  # noqa: F821
    return sorted(out, key=lambda m: m["batch"])


def applied_batch(target: str) -> int:
    """Highest batch seen in the target: opened (_batch) or closed (__END_AT). Reapplying a
    D-only batch is harmless (its keys are no longer current), so this is enough as state."""
    if not spark.catalog.tableExists(target):
        return 0
    return spark.table(target).agg(F.max(F.greatest("_batch", "__END_AT"))).first()[0] or 0


def row_hash(df, keys: list[str]):
    # NULL-safe comparison of the non-key columns (to_json drops NULL fields)
    return F.xxhash64(F.to_json(F.struct(*[c for c in df.columns if c not in keys and c not in SKIP])))


def open_versions(rows, b: int):
    return rows.withColumn("__START_AT", F.lit(b).cast("bigint")).withColumn("__END_AT", F.lit(None).cast("bigint"))


def close(t: DeltaTable, key_rows, keys: list[str], b: int) -> None:
    on = " AND ".join(f"t.{k} = s.{k}" for k in keys)
    (t.alias("t").merge(key_rows.alias("s"), f"{on} AND t.__END_AT IS NULL")
     .whenMatchedUpdate(set={"__END_AT": F.lit(b).cast("bigint")}).execute())


def apply_batch(target: str, m: dict, keys: list[str]) -> None:
    b = m["batch"]
    df = spark.read.parquet(m["data_path"].replace("s3a://", "s3://"))
    if df.count() != sum(m["counts"].values()):
        raise RuntimeError(f"{m['data_path']}: rows differ from the manifest counts {m['counts']}")
    upserts = df.where("_op IN ('I', 'U')").drop(*META)

    if not spark.catalog.tableExists(target):
        open_versions(upserts, b).write.format("delta").saveAsTable(target)
        return

    t = DeltaTable.forName(spark, target)
    if m["batch_type"] in ("full", "resync"):
        current = spark.table(target).where("__END_AT IS NULL")
        # current keys missing from the batch are gone (all of them after a key rotation: new tokens)
        close(t, current.select(*keys).join(df.select(*keys), keys, "left_anti"), keys, b)
        # a row equal to its current version is not a change: leave that version open
        same = (upserts.select(*keys, row_hash(upserts, keys).alias("_h"))
                .join(current.select(*keys, row_hash(current, keys).alias("_h")), [*keys, "_h"]))
        upserts = upserts.join(same.select(*keys), keys, "left_anti")
        changed = upserts.select(*keys)
    else:
        changed = df.select(*keys)                  # U and D close the current version; I has none
    close(t, changed, keys, b)
    open_versions(upserts, b).write.format("delta").mode("append").saveAsTable(target)


def main() -> None:
    for table, cfg in TABLES.items():
        target = f"{CATALOG}.spend_silver.{table}"
        done = applied_batch(target)
        todo = [m for m in manifests(table) if m["batch"] > done]
        for m in todo:
            print(f"{target}: batch {m['batch']} {m['batch_type']} {m['counts']}")
            apply_batch(target, m, cfg["keys"])
        if not todo:
            print(f"{target}: up to date at batch {done}")


main()
