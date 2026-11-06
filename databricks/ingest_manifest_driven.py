"""
Apply the Cloudera spend batches to SCD type 2 Delta tables, reading only committed batches
(those with a manifest) and handling resync batches. Plain PySpark + Delta Lake: runs as a
Databricks job, and on Cloudera CDE with open-source Delta Lake (cde/scripts/deploy_delta_sim.sh).

The stricter alternative to 02_ingest_scd2.sql (databricks/README.md, section 5):
  - a data folder without a manifest (a failed export) is never read;
  - a resync batch closes every current key that is missing from it, so key rotation
    (every token changes) and expired-snapshot resyncs need no full refresh;
  - each batch is one MERGE (one Delta commit): closing the old versions and opening the
    new ones cannot be split by a failure.

Per table, Delta target columns: the exported columns + _batch, __START_AT, __END_AT (batch numbers).
State: the highest batch the target has seen (opened or closed), so a rerun skips what is applied.

Usage:
  ingest_manifest_driven.py [--root s3://.../databricks_export] [--target <catalog>.spend_silver | s3://.../path]
                            [--tables customer,account,spend_event]
A --target with a scheme (s3://, s3a://, file:/) or starting with / writes path-based Delta tables
<target>/<table>; otherwise Unity Catalog / metastore tables <target>.<table>.
"""

import argparse
import json

from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = "s3://federal-buk-574bcea0/data/rsingh_spend_anl/databricks_export"
TARGET = "<catalog>.spend_silver"
TABLES = {
    "customer": {"layer": "silver", "keys": ["cif"]},
    "account": {"layer": "silver", "keys": ["account_id"]},
    "spend_event": {"layer": "silver", "keys": ["event_id"]},
}
META = ["_op", "_source_snapshot_id", "_exported_at"]
SKIP = {"_batch", "__START_AT", "__END_AT"}         # not part of a row's content

spark = SparkSession.builder.getOrCreate()


class Target:
    """A Delta table by name or by path."""

    def __init__(self, base: str, table: str):
        self.by_path = "://" in base or base.startswith(("/", "file:"))
        self.ref = f"{base.rstrip('/')}/{table}" if self.by_path else f"{base}.{table}"

    def exists(self) -> bool:
        return DeltaTable.isDeltaTable(spark, self.ref) if self.by_path else spark.catalog.tableExists(self.ref)

    def read(self):
        return spark.read.format("delta").load(self.ref) if self.by_path else spark.table(self.ref)

    def delta(self) -> DeltaTable:
        return DeltaTable.forPath(spark, self.ref) if self.by_path else DeltaTable.forName(spark, self.ref)

    def create(self, df) -> None:
        w = df.write.format("delta").mode("errorifexists")
        w.save(self.ref) if self.by_path else w.saveAsTable(self.ref)


def manifests(root: str, layer: str, table: str) -> list[dict]:
    path = f"{root}/{layer}/{table}/_manifests/*.json"
    try:
        texts = [r["value"] for r in spark.read.text(path, wholetext=True).collect()]
    except Exception as e:  # noqa: BLE001 - nothing exported yet
        if "PATH_NOT_FOUND" in str(e) or "does not exist" in str(e):
            return []
        raise
    return sorted((json.loads(t) for t in texts), key=lambda m: m["batch"])


def data_path(root: str, m: dict) -> str:
    """The batch folder under this reader's root (s3:// on Databricks, s3a:// on Cloudera)."""
    rel = f"{m['layer']}/{m['table']}/data/{m['batch']:06d}"
    if not m["data_path"].rstrip("/").endswith(rel):
        raise RuntimeError(f"batch {m['batch']}: data_path {m['data_path']} is not <root>/{rel}")
    return f"{root}/{rel}"


def applied_batch(target: Target) -> int:
    if not target.exists():
        return 0
    return target.read().agg(F.max(F.greatest("_batch", "__END_AT"))).first()[0] or 0


def row_hash(df, keys: list[str]):
    # NULL-safe comparison of the non-key columns (to_json drops NULL fields)
    return F.xxhash64(F.to_json(F.struct(*[c for c in df.columns if c not in keys and c not in SKIP])))


def open_versions(rows, b: int):
    return rows.withColumn("__START_AT", F.lit(b).cast("bigint")).withColumn("__END_AT", F.lit(None).cast("bigint"))


def apply_batch(target: Target, root: str, m: dict, keys: list[str]) -> None:
    b = m["batch"]
    df = spark.read.parquet(data_path(root, m))
    if df.count() != sum(m["counts"].values()):
        raise RuntimeError(f"batch {b}: {df.count()} rows, the manifest counts {m['counts']}")
    upserts = df.where("_op IN ('I', 'U')").drop(*META)

    if not target.exists():
        target.create(open_versions(upserts, b))
        return

    if m["batch_type"] in ("full", "resync"):
        current = target.read().where("__END_AT IS NULL")
        # current keys missing from the batch are gone (all of them after a key rotation: new tokens)
        gone = current.select(*keys).join(df.select(*keys), keys, "left_anti")
        # a row equal to its current version is not a change: leave that version open
        same = (upserts.select(*keys, row_hash(upserts, keys).alias("_h"))
                .join(current.select(*keys, row_hash(current, keys).alias("_h")), [*keys, "_h"]))
        upserts = upserts.join(same.select(*keys), keys, "left_anti")
        closing = gone.unionByName(upserts.select(*keys))
    else:
        closing = df.select(*keys)                  # U and D close the current version; I has none

    # One MERGE: rows with a merge key close the current version, rows without one are inserted.
    new = open_versions(upserts, b)
    cols = new.columns
    staged = (closing.select(*[F.col(k).alias(f"_mk_{k}") for k in keys],
                             *[F.lit(None).cast(new.schema[c].dataType).alias(c) for c in cols],
                             F.lit(False).alias("_ins"))
              .unionByName(new.select(*[F.lit(None).cast(new.schema[k].dataType).alias(f"_mk_{k}") for k in keys],
                                      *cols, F.lit(True).alias("_ins"))))
    on = " AND ".join(f"t.{k} = s._mk_{k}" for k in keys) + " AND t.__END_AT IS NULL"
    (target.delta().alias("t").merge(staged.alias("s"), on)
     .whenMatchedUpdate(condition="NOT s._ins", set={"__END_AT": F.lit(b).cast("bigint")})
     .whenNotMatchedInsert(condition="s._ins", values={c: f"s.`{c}`" for c in cols})
     .execute())


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=ROOT)
    p.add_argument("--target", default=TARGET)
    p.add_argument("--tables", default=",".join(TABLES))
    args, _ = p.parse_known_args(argv)
    root = args.root.rstrip("/")
    for table in [t.strip() for t in args.tables.split(",") if t.strip()]:
        cfg = TABLES[table]
        target = Target(args.target, table)
        done = applied_batch(target)
        todo = [m for m in manifests(root, cfg["layer"], table) if m["batch"] > done]
        for m in todo:
            print(f"{target.ref}: batch {m['batch']} {m['batch_type']} {m['counts']}", flush=True)
            apply_batch(target, root, m, cfg["keys"])
        print(f"{target.ref}: up to date at batch {max([done] + [m['batch'] for m in todo])}", flush=True)


if __name__ == "__main__":
    main()
