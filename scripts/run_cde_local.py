#!/usr/bin/env python3
"""
Run the CDE Spark jobs, the categorise step and the Great Expectations gates
on a laptop (or in CI) against a local Iceberg (Hadoop) catalog, so the same
job files can be tested without a vcluster. Optionally exports the tables the
CAI jobs and app read to parquet (parquet backend).

  python scripts/run_cde_local.py all --as-of 2026-09-25
  python scripts/run_cde_local.py all --as-of 2026-09-25 --customers 4000 --stub   # CI size
  python scripts/run_cde_local.py generate --as-of 2026-09-25 --inject-bad-data --warehouse /tmp/bad
  python scripts/run_cde_local.py dq                                               # the three dq_* stages
  python scripts/run_cde_local.py gold dq_gold export
  python scripts/run_cde_local.py history --table spend_features

Stages, in DAG order: generate, dq_bronze, silver, dq_silver, categorise, gold,
dq_gold; plus export, all (= the seven stages + export), dq (= the three dq_*
stages), history (list Iceberg snapshots of a gold table).

categorise is a CAI job on the cluster (MiniLM, Impala writes). Here it runs
the same spend.categorise code in-process and writes ref.merchant_category_map
through Spark; --stub swaps MiniLM for the TF-IDF stub.

Never point an experiment at the default data/warehouse / data/parquet: they
hold the local book every recorded number depends on. Use --warehouse and
--parquet-dir with a scratch folder instead.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS = ROOT / "cde" / "jobs"
ICEBERG_PACKAGE = "org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.10.0"
DEFAULT_PREFIX = "rsingh_spend_anl"
# stage -> (job file, extra job args), in DAG order; categorise is not a CDE job
STAGES = {
    "generate": ("generate_spend_bronze.py", []),
    "dq_bronze": ("dq_check.py", ["--layer", "bronze"]),
    "silver": ("build_silver.py", []),
    "dq_silver": ("dq_check.py", ["--layer", "silver"]),
    "categorise": (None, []),
    "gold": ("build_gold_features.py", []),
    "dq_gold": ("dq_check.py", ["--layer", "gold"]),
}
DQ_STAGES = ["dq_bronze", "dq_silver", "dq_gold"]
# (layer, table) exported for the parquet backend: what the CAI jobs, endpoint and app read.
EXPORT_TABLES = [("gold", "spend_features"), ("gold", "spend_cohort_weekly"), ("gold", "spend_mix_weekly"),
                 ("silver", "customer"), ("silver", "spend_event"),
                 ("ref", "merchant_category_map"), ("ref", "merchant_seed"), ("ref", "mcc_category"),
                 ("ref", "category_taxonomy"), ("ref", "calendar_weekly"), ("ref", "pincode_geo"),
                 ("ref", "archetype_truth"), ("ref", "dq_results")]


def local_spark(warehouse: Path, driver_memory: str = "6g"):
    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    # Python workers turn naive datetimes into instants with their local zone; pin it to UTC like CDE,
    # so an event keeps its calendar date on a laptop in IST.
    os.environ["TZ"] = "UTC"
    time.tzset()
    return (SparkSession.builder.appName("spend-local")
            .master("local[*]")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.jars.packages", ICEBERG_PACKAGE)
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
            .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog")
            .config("spark.sql.catalog.local.type", "hadoop")
            .config("spark.sql.catalog.local.warehouse", str(warehouse))
            .config("spark.sql.defaultCatalog", "local")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.driver.memory", driver_memory)
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def load_job(filename: str):
    spec = importlib.util.spec_from_file_location(filename[:-3], JOBS / filename)
    module = importlib.util.module_from_spec(spec)
    # registered only for exec_module: @dataclass needs the module in sys.modules to resolve
    # field types, but leaving it there makes cloudpickle serialize this job's closures/UDFs
    # by reference (unimportable by workers) instead of by value.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


def run_categorise(spark, db_prefix: str, stub: bool) -> None:
    sys.path.insert(0, str(ROOT))
    from spend.categorise import categorise_spark

    categorise_spark(spark, db_prefix, stub=stub)


def export_parquet(spark, db_prefix: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer, name in EXPORT_TABLES:
        full = f"{db_prefix}_{layer}.{name}"
        if not spark.catalog.tableExists(full):
            print(f"skipped {full} (not built yet)")
            continue
        path = out_dir / f"{name}.parquet"
        pdf = spark.table(full).toPandas()
        pdf.to_parquet(path, index=False)
        print(f"exported {len(pdf)} rows {layer}.{name} -> {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stages", nargs="+", choices=[*STAGES, "export", "all", "dq", "history"])
    p.add_argument("--as-of", default=None)
    p.add_argument("--db-prefix", default=DEFAULT_PREFIX)
    p.add_argument("--customers", type=int, default=None)
    p.add_argument("--inject-bad-data", action="store_true")
    p.add_argument("--stub", action="store_true", help="categorise with the TF-IDF stub instead of MiniLM")
    p.add_argument("--warehouse", default=str(ROOT / "data" / "warehouse"))
    p.add_argument("--parquet-dir", default=str(ROOT / "data" / "parquet"))
    p.add_argument("--driver-memory", default="6g")
    p.add_argument("--table", default="spend_features")
    args = p.parse_args()

    stages = [*STAGES, "export"] if "all" in args.stages else [
        s for stage in args.stages for s in (DQ_STAGES if stage == "dq" else [stage])]
    job_args = ["--db-prefix", args.db_prefix]
    if args.as_of:
        job_args += ["--as-of", args.as_of]
    gen_args = list(job_args)
    if args.customers:
        gen_args += ["--customers", str(args.customers)]
    if args.inject_bad_data:
        gen_args += ["--inject-bad-data"]

    # groups the layers of one invocation in dq_results, like the DAG's run_id
    pipeline_run = f"local-{uuid.uuid4().hex[:8]}"

    spark = local_spark(Path(args.warehouse), args.driver_memory)
    spark.sparkContext.setLogLevel("WARN")
    jvm = spark.sparkContext._jvm
    # The Hadoop catalog logs a stack trace for every new table (no version-hint.text yet).
    jvm.org.apache.logging.log4j.core.config.Configurator.setLevel(
        "org.apache.iceberg.hadoop.HadoopTableOperations", jvm.org.apache.logging.log4j.Level.ERROR)
    try:
        for stage in stages:
            print(f"\n=== {stage} ===", flush=True)
            started = time.time()
            if stage == "export":
                export_parquet(spark, args.db_prefix, Path(args.parquet_dir))
            elif stage == "history":
                spark.sql(f"SELECT snapshot_id, committed_at, operation, summary['added-records'] AS added, "
                          f"summary['deleted-records'] AS deleted "
                          f"FROM {args.db_prefix}_gold.{args.table}.snapshots ORDER BY committed_at").show(truncate=False)
            elif stage == "categorise":
                run_categorise(spark, args.db_prefix, args.stub)
            else:
                try:
                    job_file, extra = STAGES[stage]
                    if stage in DQ_STAGES:
                        extra = extra + ["--pipeline-run", pipeline_run]
                    load_job(job_file).main((gen_args if stage == "generate" else job_args) + extra, spark=spark)
                except SystemExit as e:
                    if e.code:
                        sys.exit(f"stage '{stage}' failed (exit {e.code})")
            print(f"--- {stage} took {time.time() - started:.0f}s", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
