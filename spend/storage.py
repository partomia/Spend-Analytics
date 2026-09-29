"""Read gold / silver / ref and write the CAI outputs, on CDW Impala or local parquet.

  impala   CDW Impala virtual warehouse over HTTPS (impyla). A run first
           REFRESHes what it reads (CDE commits Iceberg snapshots Impala has
           not seen yet), then pins reads to one Iceberg snapshot
           (FOR SYSTEM_VERSION AS OF), so a run reads one consistent version
           even if a CDE load commits mid-run. Outputs are Iceberg v2 tables;
           a rerun for a run_date does DELETE + INSERT for that date.
  parquet  one file per table under storage.parquet_dir, for a laptop, Docker,
           CI or an offline demo. Same reads and replace-by-run_date semantics.

Numbers read back through Impala stay numbers: DECIMAL results (SUM, AVG of
ints) come back as Python Decimal and are cast to float here, never to str.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd

from spend.config import ROOT, settings, table
from spend.schema import OUTPUT_TABLES, PARTITION

logger = logging.getLogger(__name__)


def get_storage(backend: str | None = None):
    backend = backend or settings()["storage"]["backend"]
    if backend == "impala":
        return ImpalaStorage(settings()["impala"])
    if backend == "parquet":
        return ParquetStorage(Path(settings()["storage"]["parquet_dir"]))
    raise ValueError(f"unknown storage backend {backend!r} (impala | parquet)")


def _cast(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    for col, typ in columns:
        if col not in df.columns:
            continue
        if typ == "DATE":
            df[col] = pd.to_datetime(df[col]).dt.date
        elif typ == "TIMESTAMP":
            df[col] = pd.to_datetime(df[col])
        elif typ == "DOUBLE":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        elif typ == "INT":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif typ == "BOOLEAN":
            df[col] = df[col].astype("boolean")
    return df


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object and df[c].map(lambda v: isinstance(v, Decimal)).any():
            df[c] = df[c].map(lambda v: float(v) if isinstance(v, Decimal) else v).astype(float)
    return df


def _sql_literal(value, typ: str) -> str:
    if value is None or value is pd.NA or value is pd.NaT or (isinstance(value, float) and math.isnan(value)):
        return "NULL"
    if typ == "DOUBLE":
        return repr(float(value))
    if typ in ("INT", "BIGINT"):
        return str(int(value))
    if typ == "BOOLEAN":
        return "true" if bool(value) else "false"
    if typ == "DATE":
        return f"DATE '{pd.Timestamp(value).date().isoformat()}'"
    if typ == "TIMESTAMP":
        return f"CAST('{pd.Timestamp(value).strftime('%Y-%m-%d %H:%M:%S')}' AS TIMESTAMP)"
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _d(x: date) -> str:
    return f"DATE '{x.isoformat()}'"


class ParquetStorage:
    name = "parquet"

    def __init__(self, directory: Path):
        self.dir = directory if directory.is_absolute() else ROOT / directory
        self._cache: dict[str, pd.DataFrame] = {}

    def _path(self, key: str) -> Path:
        return self.dir / f"{table(key).split('.')[-1]}.parquet"

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def refresh(self, key: str) -> None:
        self._cache.pop(key, None)

    def snapshot_id(self, key: str) -> str | None:
        return None

    def read(self, key: str, where_run_date: date | None = None, columns: list[str] | None = None,
             snapshot_id: str | None = None) -> pd.DataFrame:
        path = self._path(key)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run scripts/run_cde_local.py all, or copy an export")
        if key not in self._cache:
            self._cache[key] = pd.read_parquet(path)
        df = self._cache[key]
        if where_run_date is not None and "run_date" in df.columns:
            df = df[pd.to_datetime(df["run_date"]).dt.date == where_run_date]
        if columns is not None:
            df = df[columns]
        return df.reset_index(drop=True)

    def read_where(self, key: str, column: str, values, columns: list[str] | None = None) -> pd.DataFrame:
        df = self.read(key, columns=None)
        if column in ("snapshot_date", "run_date", "week_end"):
            df = df[pd.to_datetime(df[column]).dt.date.isin(set(values))]
        else:
            df = df[df[column].isin(set(values))]
        return (df[columns] if columns is not None else df).reset_index(drop=True)

    def run_dates(self, key: str) -> list[date]:
        if not self.exists(key):
            return []
        return sorted(pd.to_datetime(self.read(key, columns=["run_date"])["run_date"]).dt.date.unique())

    def replace_run(self, key: str, df: pd.DataFrame, run_date: date) -> None:
        cols = OUTPUT_TABLES[key]
        df = _cast(df[[c for c, _ in cols]], cols)
        path = self._path(key)
        if path.exists():
            old = pd.read_parquet(path)
            old = old[pd.to_datetime(old["run_date"]).dt.date != run_date]
            df = pd.concat([_cast(old, cols), df], ignore_index=True) if len(old) else df
        self.dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        self._cache.pop(key, None)
        logger.info("wrote %d rows for %s to %s", (df["run_date"] == run_date).sum(), run_date, path.name)

    def append(self, key: str, df: pd.DataFrame) -> None:
        cols = OUTPUT_TABLES[key]
        df = _cast(df[[c for c, _ in cols]], cols)
        path = self._path(key)
        if path.exists():
            old = pd.read_parquet(path)
            df = pd.concat([_cast(old, cols), df], ignore_index=True) if len(old) else df
        self.dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        self._cache.pop(key, None)


class ImpalaStorage:
    name = "impala"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.insert_chunk = int(cfg.get("insert_chunk", 500))
        self.bulk_insert_chunk = int(cfg.get("bulk_insert_chunk", 5000))
        self._conn = None
        self._ensured: set[str] = set()

    def _connect(self):
        if self._conn is None:
            from impala.dbapi import connect

            c = self.cfg
            if not c["host"]:
                raise RuntimeError("SPEND_IMPALA_HOST is not set (CDW Impala virtual warehouse host)")
            kwargs = dict(host=c["host"], port=int(c["port"]), use_ssl=c["use_ssl"],
                          use_http_transport=c["use_http_transport"], http_path=c["http_path"],
                          auth_mechanism=c["auth_mechanism"])
            if c["auth_mechanism"].upper() == "GSSAPI":
                kwargs["kerberos_service_name"] = c["kerberos_service_name"]
            if c.get("user"):
                kwargs["user"] = c["user"]
            if c.get("password"):
                kwargs["password"] = c["password"]
            self._conn = connect(**kwargs)
        return self._conn

    def query(self, sql: str) -> pd.DataFrame:
        cur = self._connect().cursor()
        try:
            cur.execute(sql)
            if cur.description is None:
                return pd.DataFrame()
            cols = [d[0].split(".")[-1] for d in cur.description]
            return _numeric(pd.DataFrame(cur.fetchall(), columns=cols))
        finally:
            cur.close()

    def execute(self, sql: str) -> None:
        cur = self._connect().cursor()
        try:
            cur.execute(sql)
        finally:
            cur.close()

    def exists(self, key: str) -> bool:
        db, name = table(key).split(".")
        return not self.query(f"SHOW TABLES IN {db} LIKE '{name}'").empty

    def refresh(self, key: str) -> None:
        """Pick up snapshots committed by CDE since Impala last loaded the table's metadata."""
        self.execute(f"REFRESH {table(key)}")

    def snapshot_id(self, key: str) -> str | None:
        """Current Iceberg snapshot of a table, recorded for lineage and pinned reads."""
        try:
            hist = self.query(f"DESCRIBE HISTORY {table(key)}")
            return str(hist.iloc[-1]["snapshot_id"]) if not hist.empty else None
        except Exception as e:  # lineage is best-effort, never fail the run on it
            logger.warning("could not read snapshot history for %s: %s", key, e)
            return None

    @staticmethod
    def source(key: str, snapshot_id: str | None = None) -> str:
        t = table(key)
        return f"{t} FOR SYSTEM_VERSION AS OF {int(snapshot_id)}" if snapshot_id else t

    def read(self, key: str, where_run_date: date | None = None, columns: list[str] | None = None,
             snapshot_id: str | None = None) -> pd.DataFrame:
        sql = f"SELECT {', '.join(columns) if columns else '*'} FROM {self.source(key, snapshot_id)}"
        if where_run_date is not None:
            sql += f" WHERE run_date = {_d(where_run_date)}"
        return self.query(sql)

    def read_where(self, key: str, column: str, values, columns: list[str] | None = None) -> pd.DataFrame:
        values = sorted(set(values))
        cols = ", ".join(columns) if columns else "*"
        if not values:
            return self.query(f"SELECT {cols} FROM {table(key)} LIMIT 0")
        typ = "DATE" if column in ("snapshot_date", "run_date", "week_end") else "STRING"
        in_list = ", ".join(_sql_literal(v, typ) for v in values)
        return self.query(f"SELECT {cols} FROM {table(key)} WHERE {column} IN ({in_list})")

    def run_dates(self, key: str) -> list[date]:
        if not self.exists(key):
            return []
        df = self.query(f"SELECT DISTINCT run_date FROM {table(key)}")
        return sorted(pd.to_datetime(df["run_date"]).dt.date) if not df.empty else []

    def ensure_table(self, key: str) -> None:
        if key in self._ensured:
            return
        full = table(key)
        cols = ", ".join(f"{c} {t}" for c, t in OUTPUT_TABLES[key])
        part = f" PARTITIONED BY SPEC ({PARTITION[key]})" if PARTITION.get(key) else ""
        self.execute(f"CREATE DATABASE IF NOT EXISTS {full.split('.')[0]}")
        self.execute(f"CREATE TABLE IF NOT EXISTS {full} ({cols}){part} "
                     f"STORED AS ICEBERG TBLPROPERTIES ('format-version'='2')")
        self._ensured.add(key)

    def _insert(self, key: str, df: pd.DataFrame) -> None:
        cols = OUTPUT_TABLES[key]
        full = table(key)
        chunk = self.bulk_insert_chunk if key == "customer_persona" else self.insert_chunk
        records = df[[c for c, _ in cols]].to_dict("records")
        for i in range(0, len(records), chunk):
            values = ",\n".join(
                "(" + ", ".join(_sql_literal(r[c], t) for c, t in cols) + ")"
                for r in records[i:i + chunk])
            self.execute(f"INSERT INTO {full} ({', '.join(c for c, _ in cols)}) VALUES {values}")

    def replace_run(self, key: str, df: pd.DataFrame, run_date: date) -> None:
        self.ensure_table(key)
        self.execute(f"DELETE FROM {table(key)} WHERE run_date = {_d(run_date)}")
        self._insert(key, df)
        logger.info("wrote %d rows for %s to %s", len(df), run_date, table(key))

    def append(self, key: str, df: pd.DataFrame) -> None:
        self.ensure_table(key)
        self._insert(key, df)
