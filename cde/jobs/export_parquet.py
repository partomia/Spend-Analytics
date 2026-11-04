"""
Export - Iceberg tables to plain Parquet on S3, a landing zone for Databricks

Reads one pinned Iceberg snapshot per table and writes it as plain Parquet
(snappy, Hive-style partition folders, TIMESTAMP_MICROS) under the project's
export folder, with an _export_manifest.json next to the data. Databricks reads
the folder as a silver layer and builds its Unity Catalog Delta tables from it.

Why through Spark and not a copy of the table's data files: gold's MERGE loads
leave Iceberg v2 position-delete files behind (gold.spend_features has one per
data file), and a plain Parquet reader ignores them, so the raw files would
count every reloaded row twice. Reading the snapshot applies the deletes.

Layout (one folder per table, replaced on every run):
  <target>/<layer>/<table>/<col>_month=YYYY-MM/part-*.snappy.parquet   partitioned tables
  <target>/<layer>/<table>/part-*.snappy.parquet                        small tables
  <target>/<layer>/<table>/_SUCCESS
  <target>/<layer>/<table>/_export_manifest.json   snapshot id, row count, schema, Databricks SQL

Readers skip files whose names start with "_". The folder is replaced in place,
so a consumer should wait for an _export_manifest.json with a newer exported_at
(written last, after the row-count check) rather than poll the data files.

Reads:  <prefix>_<layer>.<table> for each --tables entry (default: the three silver tables)
Writes: <target>/<layer>/<table>/ (default target: the project folder on the environment bucket)

Usage:
  spark-submit export_parquet.py [--db-prefix P] [--tables silver.spend_event,gold.spend_features]
                                 [--target s3a://bucket/path] [--snapshot-id ID] [--dry-run]

Exit: an error (unknown table, row-count mismatch) fails the job.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DB_PREFIX = "rsingh_spend_anl"
DEFAULT_TARGET = "s3a://federal-buk-574bcea0/data/rsingh_spend_anl/databricks_export"
DEFAULT_TABLES = "silver.spend_event,silver.customer,silver.account"
LAYERS = ("bronze", "silver", "gold", "ref")
# table -> date column whose month names the partition folder; other tables are written unpartitioned
PARTITION_BY = {"silver.spend_event": "txn_date", "gold.spend_features": "snapshot_date",
                "gold.spend_cohort_weekly": "week_end", "gold.spend_mix_weekly": "week_end"}
ROWS_PER_FILE = 2_000_000
MANIFEST = "_export_manifest.json"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db-prefix", default=DEFAULT_DB_PREFIX)
    p.add_argument("--tables", default=DEFAULT_TABLES, help="comma-separated layer.table")
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--snapshot-id", type=int, default=None, help="pin one snapshot (single table only)")
    p.add_argument("--dry-run", action="store_true", help="resolve snapshots and count rows; write nothing")
    args, _ = p.parse_known_args(argv)
    return args


def parse_tables(spec: str) -> list[tuple[str, str]]:
    out = []
    for item in (s.strip() for s in spec.split(",") if s.strip()):
        layer, _, table = item.partition(".")
        if layer not in LAYERS or not table:
            raise ValueError(f"{item!r}: expected <layer>.<table> with layer in {LAYERS}")
        out.append((layer, table))
    if not out:
        raise ValueError("no tables to export")
    return out


def table_path(target: str, layer: str, table: str) -> str:
    return f"{target.rstrip('/')}/{layer}/{table}"


def partition_column(layer: str, table: str) -> str | None:
    col = PARTITION_BY.get(f"{layer}.{table}")
    return f"{col}_month" if col else None


def current_snapshot(spark: SparkSession, fq: str) -> tuple[int, str]:
    row = spark.sql(f"SELECT snapshot_id, made_current_at FROM {fq}.history WHERE is_current_ancestor "
                    "ORDER BY made_current_at DESC LIMIT 1").first()
    if row is None:
        raise ValueError(f"{fq} has no snapshot (empty table?)")
    return int(row["snapshot_id"]), str(row["made_current_at"])


def delete_files(spark: SparkSession, fq: str, snapshot_id: int) -> int:
    """Position / equality delete files in the snapshot: what a raw-file copy would get wrong."""
    try:
        rows = spark.sql(f"SELECT content, COUNT(*) AS n FROM {fq}.files VERSION AS OF {snapshot_id} "
                         "GROUP BY content").collect()
    except Exception:  # noqa: BLE001 - informational only
        return -1
    return int(sum(r["n"] for r in rows if r["content"] != 0))


def with_partition(df: DataFrame, layer: str, table: str) -> tuple[DataFrame, str | None]:
    part = partition_column(layer, table)
    if part is None:
        return df, None
    return df.withColumn(part, F.date_format(F.col(PARTITION_BY[f"{layer}.{table}"]), "yyyy-MM")), part


def databricks_sql(layer: str, table: str, path: str, part: str | None) -> str:
    """Unity Catalog statement that turns the export into a managed Delta table (refresh = rerun)."""
    s3 = path.replace("s3a://", "s3://", 1)
    cluster = f"\nCLUSTER BY ({PARTITION_BY[f'{layer}.{table}']})" if part else ""
    drop = f" EXCEPT ({part})" if part else ""
    return (f"CREATE OR REPLACE TABLE <catalog>.spend_{layer}.{table}{cluster}\n"
            f"AS SELECT *{drop} FROM parquet.`{s3}/`;")


def _fs(spark: SparkSession, path: str):
    jvm = spark._jvm
    p = jvm.org.apache.hadoop.fs.Path(path)
    return p, p.getFileSystem(spark._jsc.hadoopConfiguration())


def write_text(spark: SparkSession, path: str, text: str) -> None:
    p, fs = _fs(spark, path)
    out = fs.create(p, True)
    try:
        out.write(bytearray(text.encode("utf-8")))
    finally:
        out.close()


def folder_bytes(spark: SparkSession, path: str) -> int:
    p, fs = _fs(spark, path)
    return int(fs.getContentSummary(p).getLength())


def export_table(spark: SparkSession, db_prefix: str, layer: str, table: str, target: str = DEFAULT_TARGET,
                 snapshot_id: int | None = None, dry_run: bool = False) -> dict:
    fq = f"{db_prefix}_{layer}.{table}"
    sid, committed = (snapshot_id, None) if snapshot_id else current_snapshot(spark, fq)
    path = table_path(target, layer, table)
    src = spark.sql(f"SELECT * FROM {fq} VERSION AS OF {sid}")
    n = src.count()
    deletes = delete_files(spark, fq, sid)
    df, part = with_partition(src, layer, table)
    logger.info("%s @ snapshot %s: %d rows, %s delete files -> %s (partition %s)",
                fq, sid, n, deletes, path, part or "none")
    manifest = {
        "source_table": fq, "layer": layer, "table": table, "iceberg_snapshot_id": sid,
        "snapshot_committed_at": committed, "row_count": n, "source_delete_files": deletes,
        "target_path": path, "format": "parquet", "compression": "snappy", "timestamp_type": "TIMESTAMP_MICROS (UTC)",
        "partition_column": part, "partition_source": PARTITION_BY.get(f"{layer}.{table}"),
        "columns": [{"name": f.name, "type": f.dataType.simpleString(), "nullable": f.nullable}
                    for f in df.schema.fields],
        "databricks_sql": databricks_sql(layer, table, path, part), "spark_version": spark.version,
    }
    if dry_run:
        logger.info("dry run: %s not written", path)
        return {**manifest, "dry_run": True}

    spark.conf.set("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
    w = (df.repartition(part) if part else df.repartition(max(1, math.ceil(n / ROWS_PER_FILE))))
    w = w.write.mode("overwrite").option("compression", "snappy").option("maxRecordsPerFile", ROWS_PER_FILE)
    (w.partitionBy(part) if part else w).parquet(path)

    back = spark.read.parquet(path)
    written = back.count()
    if written != n:
        raise RuntimeError(f"{path}: wrote {written} rows, snapshot {sid} of {fq} has {n}")
    files, size = len(back.inputFiles()), folder_bytes(spark, path)
    manifest.update(files=files, bytes=size,
                    exported_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    write_text(spark, f"{path}/{MANIFEST}", json.dumps(manifest, indent=2))
    logger.info("%s: %d rows in %d files, %.1f MB, row count verified", path, n, files, size / 1e6)
    return manifest


def run(spark: SparkSession, db_prefix: str, tables: list[tuple[str, str]], target: str,
        snapshot_id: int | None = None, dry_run: bool = False) -> list[dict]:
    if snapshot_id and len(tables) > 1:
        raise ValueError("--snapshot-id pins one table's snapshot: pass a single --tables entry")
    out = [export_table(spark, db_prefix, layer, table, target, snapshot_id, dry_run) for layer, table in tables]
    for m in out:
        logger.info("%s %s @ %s: %d rows -> %s", "would export" if dry_run else "exported", m["source_table"],
                    m["iceberg_snapshot_id"], m["row_count"], m["target_path"])
    return out


def main(argv=None, spark: SparkSession | None = None) -> None:
    args = parse_args(argv)
    own = spark is None
    spark = spark or SparkSession.builder.appName("spend-export-parquet").getOrCreate()
    try:
        run(spark, args.db_prefix, parse_tables(args.tables), args.target, args.snapshot_id, args.dry_run)
    finally:
        if own:
            spark.stop()


if __name__ == "__main__":
    main()
