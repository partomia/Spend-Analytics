"""
Export - Iceberg tables to a tokenized, incremental Parquet change feed for Databricks

For each table in config/export.yaml, compares the current Iceberg snapshot with
the one exported last (time travel on the primary key plus a hash of the other
columns) and writes only what changed, as one batch:

  _op = I   a new key (every row of the first, full, batch)
  _op = U   a changed row (the full new row)
  _op = D   a key that disappeared (key columns only, the rest NULL)

Databricks applies the batches in order (_batch) to Unity Catalog Delta tables
with SCD type 2 history (databricks/README.md).

Columns follow config/export.yaml, deny by default: a source column not listed
there fails the export. Identifiers are encrypted with deterministic AES-256-GCM:
the IV is HMAC-SHA256(iv_key, value)[:12] and the token is base64(iv ||
ciphertext || tag), the layout Spark's and Databricks' aes_decrypt(..., 'GCM')
read. The same value always gives the same token, so tokens join and merge;
decrypting needs enc_key only. The keys come from the CDE workload credential
named in the config (files enc_key_b64, iv_key_b64, key_version under
/etc/dex/secrets/<name>/), or a JSON file in SPEND_EXPORT_KEY_FILE; encryption
runs in a pandas UDF so the key never appears in a Spark plan or the UI.

Layout per table:
  <target>/<layer>/<table>/data/<batch:06d>/part-*.snappy.parquet
  <target>/<layer>/<table>/_manifests/<batch:06d>.json    written last: the commit marker

The latest manifest's to_snapshot_id is the next run's starting point; the tag
databricks_export on the source table keeps that snapshot from expiring. A
resync batch (the whole table as I rows) replaces the diff when there is no
readable previous snapshot, the key version or the column rules changed, or
--resync is given. A batch without a manifest (a failed run) is ignored by
readers, and the next run takes a new batch number.

Reading through Spark, not copying the table's files, also applies Iceberg v2
position deletes: on gold.spend_features the data files hold 16.6M records of
which 3.3M are deleted.

Usage:
  spark-submit export_parquet.py [--db-prefix P] [--tables silver.customer,...] [--target s3a://...]
                                 [--config config/export.yaml] [--resync] [--dry-run]

Exit: an error (unlisted column, duplicate key, row-count mismatch) fails the job.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DB_PREFIX = "rsingh_spend_anl"
KEY_DIR = Path("/etc/dex/secrets")
RULES = ("encrypt", "prefix3", "month", "keep", "drop")
META = [("_op", "string"), ("_batch", "bigint"), ("_source_snapshot_id", "bigint"), ("_exported_at", "timestamp")]
TAG = "databricks_export"
TAG_RETAIN_DAYS = 30
ROWS_PER_FILE = 2_000_000


def _config_path() -> Path:
    try:
        return Path(__file__).resolve().parents[2] / "config" / "export.yaml"
    except NameError:
        return Path.cwd() / "config" / "export.yaml"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db-prefix", default=DEFAULT_DB_PREFIX)
    p.add_argument("--config", default=str(_config_path()))
    p.add_argument("--tables", default=None, help="comma-separated layer.table (default: every table in the config)")
    p.add_argument("--target", default=None, help="override the config's target root")
    p.add_argument("--resync", action="store_true", help="write the whole table as a resync batch")
    p.add_argument("--dry-run", action="store_true", help="compute the batch and its counts; write nothing")
    args, _ = p.parse_known_args(argv)
    return args


# ---------------------------------------------------------------- config and keys

def load_config(path: str | Path) -> dict:
    text = Path(path).read_text()
    try:
        import yaml
        cfg = yaml.safe_load(text)
    except ImportError:  # the CDE env ships ruamel.yaml (Great Expectations), not always PyYAML
        from ruamel.yaml import YAML
        cfg = YAML(typ="safe").load(text)
    for name, t in cfg["tables"].items():
        layer, _, table = name.partition(".")
        if not table:
            raise ValueError(f"{name}: expected <layer>.<table>")
        bad = {c: r for c, r in t["columns"].items() if r not in RULES}
        if bad:
            raise ValueError(f"{name}: unknown rules {bad} (allowed: {RULES})")
        keys = primary_key(t)
        if any(t["columns"].get(k) in (None, "drop") for k in keys):
            raise ValueError(f"{name}: primary key {keys} must be exported")
    return cfg


def primary_key(table_cfg: dict) -> list[str]:
    pk = table_cfg["primary_key"]
    return [pk] if isinstance(pk, str) else list(pk)


def rules_hash(table_cfg: dict) -> str:
    return hashlib.sha256(json.dumps(table_cfg, sort_keys=True).encode()).hexdigest()[:16]


def load_keys(credential: str, key_dir: Path = KEY_DIR) -> dict:
    """{'version', 'enc', 'iv'}: from SPEND_EXPORT_KEY_FILE (JSON) or the mounted workload credential."""
    src = os.environ.get("SPEND_EXPORT_KEY_FILE")
    if src:
        raw = json.loads(Path(src).read_text())
    else:
        d = key_dir / credential
        if not d.is_dir():
            raise FileNotFoundError(f"{d} not mounted: create the job with --workload-credential {credential}")
        raw = {f: (d / f).read_text().strip() for f in ("key_version", "enc_key_b64", "iv_key_b64")}
    keys = {"version": raw["key_version"], "enc": base64.b64decode(raw["enc_key_b64"]),
            "iv": base64.b64decode(raw["iv_key_b64"])}
    if len(keys["enc"]) != 32 or len(keys["iv"]) != 32:
        raise ValueError("enc_key and iv_key must be 32 bytes each (AES-256)")
    return keys


# ---------------------------------------------------------------- tokens

def tokenize_value(value: str | None, enc: bytes, iv_key: bytes) -> str | None:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if value is None:
        return None
    data = value.encode("utf-8")
    iv = hmac.new(iv_key, data, hashlib.sha256).digest()[:12]
    return base64.b64encode(iv + AESGCM(enc).encrypt(iv, data, None)).decode("ascii")


def detokenize_value(token: str | None, enc: bytes) -> str | None:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if token is None:
        return None
    raw = base64.b64decode(token)
    return AESGCM(enc).decrypt(raw[:12], raw[12:], None).decode("utf-8")


def encrypt_udf(keys: dict):
    enc, iv_key = keys["enc"], keys["iv"]

    @F.pandas_udf("string")
    def encrypt(s):
        import pandas as pd
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        aes = AESGCM(enc)

        def one(v):
            if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA:
                return None
            data = str(v).encode("utf-8")
            iv = hmac.new(iv_key, data, hashlib.sha256).digest()[:12]
            return base64.b64encode(iv + aes.encrypt(iv, data, None)).decode("ascii")

        return pd.Series([one(v) for v in s], index=s.index, dtype="object")

    return encrypt


# ---------------------------------------------------------------- columns

def check_columns(source_cols: list[str], table_cfg: dict, fq: str) -> None:
    rules = table_cfg["columns"]
    unlisted = [c for c in source_cols if c not in rules]
    missing = [c for c in rules if c not in source_cols]
    if unlisted:
        raise ValueError(f"{fq}: columns not in config/export.yaml (deny by default): {unlisted}")
    if missing:
        raise ValueError(f"{fq}: config/export.yaml lists columns the table no longer has: {missing}")


def output_name(col: str, rule: str) -> str:
    return f"{col}3" if rule == "prefix3" else col


def apply_rules(df: DataFrame, table_cfg: dict, encrypt) -> DataFrame:
    out = []
    for c in df.columns:
        rule = table_cfg["columns"].get(c, "keep")     # metadata columns pass through
        if rule == "drop":
            continue
        col = F.col(c)
        if rule == "encrypt":
            col = encrypt(col.cast("string"))
        elif rule == "prefix3":
            col = F.substring(col.cast("string"), 1, 3)
        elif rule == "month":
            col = F.trunc(col, "MM")
        out.append(col.alias(output_name(c, rule)))
    return df.select(*out)


def tokenized_columns(table_cfg: dict) -> list[str]:
    return [c for c, r in table_cfg["columns"].items() if r == "encrypt"]


# ---------------------------------------------------------------- Iceberg and the change set

def current_snapshot(spark: SparkSession, fq: str) -> int:
    row = spark.sql(f"SELECT snapshot_id FROM {fq}.history WHERE is_current_ancestor "
                    "ORDER BY made_current_at DESC LIMIT 1").first()
    if row is None:
        raise ValueError(f"{fq} has no snapshot (empty table?)")
    return int(row["snapshot_id"])


def at_snapshot(spark: SparkSession, fq: str, sid: int) -> DataFrame:
    return spark.sql(f"SELECT * FROM {fq} VERSION AS OF {sid}")


def readable(spark: SparkSession, fq: str, sid: int) -> bool:
    try:
        at_snapshot(spark, fq, sid).schema
        return True
    except Exception as e:  # noqa: BLE001 - expired or unknown snapshot
        logger.warning("%s: snapshot %s not readable (%s)", fq, sid, str(e).splitlines()[0][:200])
        return False


def row_hash(df: DataFrame, keys: list[str]) -> F.Column:
    # to_json drops NULL fields, so (a=NULL, b=1) and (a=1, b=NULL) hash differently
    return F.xxhash64(F.to_json(F.struct(*[F.col(c) for c in df.columns if c not in keys])))


def change_set(cur: DataFrame, prev: DataFrame | None, keys: list[str]) -> DataFrame:
    """Raw source columns + _op: every row as I without a previous snapshot, else I / U / D."""
    if prev is None:
        return cur.withColumn("_op", F.lit("I"))
    others = [c for c in cur.columns if c not in keys]
    cur_h = cur.withColumn("_h", row_hash(cur, keys))
    prev_h = prev.select(*keys, row_hash(prev, keys).alias("_ph"))
    iu = (cur_h.join(prev_h, keys, "left")
          .where(F.col("_ph").isNull() | (F.col("_ph") != F.col("_h")))
          .withColumn("_op", F.when(F.col("_ph").isNull(), "I").otherwise("U"))
          .select(*cur.columns, "_op"))
    types = dict(cur.dtypes)
    d = (prev.select(*keys).join(cur.select(*keys), keys, "left_anti")
         .select(*keys, *[F.lit(None).cast(types[c]).alias(c) for c in others], F.lit("D").alias("_op"))
         .select(*cur.columns, "_op"))
    return iu.unionByName(d)


# ---------------------------------------------------------------- S3 state

def _fs(spark: SparkSession, path: str):
    p = spark._jvm.org.apache.hadoop.fs.Path(path)
    return p, p.getFileSystem(spark._jsc.hadoopConfiguration())


def write_text(spark: SparkSession, path: str, text: str) -> None:
    p, fs = _fs(spark, path)
    out = fs.create(p, True)
    try:
        out.write(bytearray(text.encode("utf-8")))
    finally:
        out.close()


def read_text(spark: SparkSession, path: str) -> str:
    p, fs = _fs(spark, path)
    stream = fs.open(p)
    try:
        reader = spark._jvm.java.io.BufferedReader(spark._jvm.java.io.InputStreamReader(stream, "UTF-8"))
        lines = []
        while (line := reader.readLine()) is not None:
            lines.append(line)
        return "\n".join(lines)
    finally:
        stream.close()


def list_names(spark: SparkSession, path: str) -> list[str]:
    p, fs = _fs(spark, path)
    if not fs.exists(p):
        return []
    return [str(s.getPath().getName()) for s in fs.listStatus(p)]


def folder_bytes(spark: SparkSession, path: str) -> int:
    p, fs = _fs(spark, path)
    return int(fs.getContentSummary(p).getLength())


def table_path(target: str, layer: str, table: str) -> str:
    return f"{target.rstrip('/')}/{layer}/{table}"


def batch_name(batch: int) -> str:
    return f"{batch:06d}"


def latest_manifest(spark: SparkSession, tpath: str) -> dict | None:
    names = sorted(n for n in list_names(spark, f"{tpath}/_manifests") if re.fullmatch(r"\d{6}\.json", n))
    return json.loads(read_text(spark, f"{tpath}/_manifests/{names[-1]}")) if names else None


def next_batch(spark: SparkSession, tpath: str) -> int:
    used = [int(n[:6]) for n in list_names(spark, f"{tpath}/_manifests") if re.fullmatch(r"\d{6}\.json", n)]
    used += [int(n) for n in list_names(spark, f"{tpath}/data") if re.fullmatch(r"\d{6}", n)]
    return max(used, default=0) + 1


def plan_batch(last: dict | None, keys: dict, rhash: str, force_resync: bool, prev_readable) -> tuple[str, str]:
    """(batch_type, reason). prev_readable is called only when an incremental batch is possible."""
    if last is None:
        return "full", "first export"
    if force_resync:
        return "resync", "--resync"
    if last.get("key_version") != keys["version"]:
        return "resync", f"key version {last.get('key_version')} -> {keys['version']}"
    if last.get("rules_hash") != rhash:
        return "resync", "column rules changed"
    if not prev_readable(int(last["to_snapshot_id"])):
        return "resync", f"previous snapshot {last['to_snapshot_id']} not readable"
    return "incremental", f"diff against snapshot {last['to_snapshot_id']}"


# ---------------------------------------------------------------- one table

def export_table(spark: SparkSession, db_prefix: str, name: str, table_cfg: dict, target: str, keys: dict,
                 resync: bool = False, dry_run: bool = False, tag: str | None = TAG) -> dict:
    """tag=None leaves the source table's tags alone (a scratch export must not move the feed's tag)."""
    layer, _, table = name.partition(".")
    fq, tpath, pk = f"{db_prefix}_{layer}.{table}", table_path(target, layer, table), primary_key(table_cfg)
    sid = current_snapshot(spark, fq)
    cur = at_snapshot(spark, fq, sid)
    check_columns(cur.columns, table_cfg, fq)
    rhash = rules_hash(table_cfg)
    last = latest_manifest(spark, tpath)
    btype, reason = plan_batch(last, keys, rhash, resync, lambda s: readable(spark, fq, s))
    from_sid = int(last["to_snapshot_id"]) if btype == "incremental" else None
    if btype == "incremental" and from_sid == sid:
        logger.info("%s: snapshot %s already exported in batch %s; nothing to do", fq, sid, last["batch"])
        return {"source_table": fq, "batch": None, "batch_type": "none", "to_snapshot_id": sid}

    n_source = cur.count()
    n_keys = cur.select(*pk).distinct().count()
    if n_keys != n_source:
        raise ValueError(f"{fq} @ {sid}: {n_source - n_keys} duplicate {pk} values; the diff needs a unique key")

    batch = next_batch(spark, tpath)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    changes = change_set(cur, at_snapshot(spark, fq, from_sid) if from_sid else None, pk)
    out = apply_rules(changes, table_cfg, encrypt_udf(keys)).select(
        "*", F.lit(batch).cast("bigint").alias("_batch"), F.lit(sid).cast("bigint").alias("_source_snapshot_id"),
        F.lit(now.replace(tzinfo=None)).cast("timestamp").alias("_exported_at"))
    out = out.select(*[c for c in out.columns if c != "_op"], "_op").persist()
    try:
        counts = {r["_op"]: int(r["count"]) for r in out.groupBy("_op").count().collect()}
        counts = {op: counts.get(op, 0) for op in ("I", "U", "D")}
        total = sum(counts.values())
        out_keys = [output_name(k, table_cfg["columns"][k]) for k in pk]
        if out.select(*out_keys).distinct().count() != total:
            raise RuntimeError(f"{fq}: batch {batch} has a repeated key after tokenizing (token collision?)")
        manifest = {
            "batch": batch, "batch_type": btype, "reason": reason, "source_table": fq, "layer": layer, "table": table,
            "primary_key": out_keys, "from_snapshot_id": from_sid, "to_snapshot_id": sid,
            "source_row_count": n_source, "counts": counts, "key_version": keys["version"], "rules_hash": rhash,
            "tokenized_columns": tokenized_columns(table_cfg), "token_format": "base64(iv12 || AES-256-GCM ciphertext || tag16)",
            "columns": [{"name": f.name, "type": f.dataType.simpleString(), "nullable": f.nullable}
                        for f in out.schema.fields],
            "data_path": f"{tpath}/data/{batch_name(batch)}", "previous_batch": last and last["batch"],
            "exported_at": now.isoformat(),
        }
        logger.info("%s @ %s: batch %s %s (%s): I %d, U %d, D %d of %d source rows", fq, sid, batch, btype, reason,
                    counts["I"], counts["U"], counts["D"], n_source)
        if dry_run:
            return {**manifest, "dry_run": True}

        spark.conf.set("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        (out.repartition(max(1, math.ceil(total / ROWS_PER_FILE)))
         .write.mode("errorifexists").option("compression", "snappy").option("maxRecordsPerFile", ROWS_PER_FILE)
         .parquet(manifest["data_path"]))
    finally:
        out.unpersist()

    back = spark.read.parquet(manifest["data_path"])
    if back.count() != total:
        raise RuntimeError(f"{manifest['data_path']}: read back {back.count()} rows, wrote {total}")
    manifest.update(files=len(back.inputFiles()), bytes=folder_bytes(spark, manifest["data_path"]))
    write_text(spark, f"{tpath}/_manifests/{batch_name(batch)}.json", json.dumps(manifest, indent=2))
    try:
        if tag:
            spark.sql(f"ALTER TABLE {fq} CREATE OR REPLACE TAG {tag} AS OF VERSION {sid} RETAIN {TAG_RETAIN_DAYS} DAYS")
    except Exception as e:  # noqa: BLE001 - the manifest is the source of truth; the tag only guards expiry
        logger.warning("%s: could not tag snapshot %s (%s)", fq, sid, str(e).splitlines()[0][:200])
    logger.info("%s: committed batch %s, %d rows in %d files, %.1f MB", tpath, batch, total, manifest["files"],
                manifest["bytes"] / 1e6)
    return manifest


def run(spark: SparkSession, db_prefix: str, cfg: dict, tables: list[str] | None = None, target: str | None = None,
        keys: dict | None = None, resync: bool = False, dry_run: bool = False) -> list[dict]:
    keys = keys or load_keys(cfg["key_credential"])
    names = tables or list(cfg["tables"])
    unknown = [n for n in names if n not in cfg["tables"]]
    if unknown:
        raise ValueError(f"not in config/export.yaml: {unknown}")
    out = [export_table(spark, db_prefix, n, cfg["tables"][n], target or cfg["target"], keys, resync, dry_run)
           for n in names]
    for m in out:
        if m.get("batch"):
            logger.info("%s %s batch %s (%s): %s", "would write" if dry_run else "exported", m["source_table"],
                        m["batch"], m["batch_type"], m["counts"])
    return out


def main(argv=None, spark: SparkSession | None = None) -> None:
    args = parse_args(argv)
    own = spark is None
    spark = spark or SparkSession.builder.appName("spend-export-parquet").getOrCreate()
    try:
        tables = [t.strip() for t in args.tables.split(",") if t.strip()] if args.tables else None
        run(spark, args.db_prefix, load_config(args.config), tables, args.target, resync=args.resync,
            dry_run=args.dry_run)
    finally:
        if own:
            spark.stop()


if __name__ == "__main__":
    main()
