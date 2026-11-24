"""
Push - the tokenized change batches on S3 to a Databricks Unity Catalog Volume, then run the pipeline

For workspaces that cannot read the project bucket (Databricks Free Edition has no
external locations), Cloudera pushes the batches written by export_parquet.py
into a Volume through the Databricks Files API and starts the Lakeflow pipeline
(databricks/02_ingest_scd2.sql) that applies them. Nothing in Databricks holds
an AWS key, and the bucket stays private.

Per table in config/export.yaml:
  1. The batches on S3 and in the Volume, from their manifests (<layer>/<table>/_manifests/<batch>.json).
  2. A batch in both must have the same manifest (to_snapshot_id, exported_at): otherwise the
     Volume holds another export's batches and the job stops.
  3. The batches missing from the Volume, in order. An empty Volume starts at the latest full or
     resync batch: what came before it is superseded. A gap (a missing batch below the Volume's
     latest) stops the job.
  4. Per batch: its Parquet files (uploaded from the executors, one file per task, size checked),
     then its manifest. The manifest stays the commit marker: a reader never sees a batch
     whose files are not all there.

Then, if anything was pushed and a pipeline id is configured, one pipeline update, waited on
until it completes (or wait_minutes). A resync batch pushed on top of earlier ones does not
start the pipeline: it needs spend.start_batch = <batch> and a full refresh
(databricks/README.md, section 5), and the job fails to say so.

The Databricks token is a CDE workload credential (file "token", mounted at
/etc/dex/secrets/<name>/) or SPEND_DATABRICKS_TOKEN. Settings: config/export.yaml, databricks_push.

Usage:
  spark-submit push_databricks.py [--config config/export.yaml] [--tables silver.customer,...]
                                  [--dry-run] [--no-trigger]

Exit: a mismatch, a gap, a failed upload or a failed pipeline update fails the job.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests
from pyspark.sql import SparkSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KEY_DIR = Path("/etc/dex/secrets")
MANIFEST = re.compile(r"\d{6}\.json")
RETRIES = 4
UPDATE_DONE = {"COMPLETED"}
UPDATE_BAD = {"FAILED", "CANCELED"}


def _config_path() -> Path:
    try:
        return Path(__file__).resolve().parents[2] / "config" / "export.yaml"
    except NameError:
        return Path.cwd() / "config" / "export.yaml"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db-prefix", default=None, help="accepted for the shared job args; not used")
    p.add_argument("--config", default=str(_config_path()))
    p.add_argument("--tables", default=None, help="comma-separated layer.table (default: every table in the config)")
    p.add_argument("--dry-run", action="store_true", help="list what would be pushed; upload nothing")
    p.add_argument("--no-trigger", action="store_true", help="push only; do not start the pipeline")
    args, _ = p.parse_known_args(argv)
    return args


def load_config(path: str | Path) -> dict:
    text = Path(path).read_text()
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:  # the CDE env ships ruamel.yaml (Great Expectations), not always PyYAML
        from ruamel.yaml import YAML
        return YAML(typ="safe").load(text)


def load_token(credential: str, key_dir: Path = KEY_DIR) -> str:
    token = os.environ.get("SPEND_DATABRICKS_TOKEN")
    if token:
        return token.strip()
    f = key_dir / credential / "token"
    if not f.is_file():
        raise FileNotFoundError(f"{f} not mounted: create the job with --workload-credential {credential}")
    return f.read_text().strip()


# ---------------------------------------------------------------- planning (no I/O)

def plan_push(s3: dict[int, dict], vol: dict[int, dict], table: str) -> list[int]:
    """The batches to push, in order. s3 / vol: batch -> manifest."""
    for b in sorted(set(s3) & set(vol)):
        mine, theirs = s3[b], vol[b]
        if (mine.get("to_snapshot_id"), mine.get("exported_at")) != (theirs.get("to_snapshot_id"),
                                                                      theirs.get("exported_at")):
            raise ValueError(f"{table}: batch {b} in the Volume is not the one on S3 (snapshot "
                             f"{theirs.get('to_snapshot_id')} vs {mine.get('to_snapshot_id')}): the Volume holds "
                             "another export's batches; clear it and full-refresh the pipeline")
    missing = sorted(set(s3) - set(vol))
    if not vol:
        bases = [b for b in missing if s3[b].get("batch_type") in ("full", "resync")]
        return [b for b in missing if b >= max(bases)] if bases else []
    gaps = [b for b in missing if b < max(vol)]
    if gaps:
        raise ValueError(f"{table}: batches {gaps} are on S3 but not in the Volume, below its latest "
                         f"batch {max(vol)}: push them by hand or start the Volume again")
    return missing


def blocking_resyncs(pushed: dict[str, list[int]], s3: dict[str, dict[int, dict]],
                     had_volume: dict[str, bool]) -> list[str]:
    """Resync batches pushed on top of batches the pipeline already applied."""
    return [f"{t} batch {b}" for t, batches in pushed.items() if had_volume[t]
            for b in batches if s3[t][b].get("batch_type") == "resync"]


# ---------------------------------------------------------------- S3 (Hadoop FS on the driver)

def _fs(spark: SparkSession, path: str):
    p = spark._jvm.org.apache.hadoop.fs.Path(path)
    return p, p.getFileSystem(spark._jsc.hadoopConfiguration())


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


def s3_files(spark: SparkSession, path: str) -> dict[str, int]:
    """name -> size of the Parquet parts in a folder (not _SUCCESS or checksums)."""
    p, fs = _fs(spark, path)
    if not fs.exists(p):
        return {}
    return {str(s.getPath().getName()): int(s.getLen()) for s in fs.listStatus(p)
            if str(s.getPath().getName()).endswith(".parquet") and not str(s.getPath().getName()).startswith((".", "_"))}


def s3_manifests(spark: SparkSession, tpath: str) -> tuple[dict[int, dict], dict[int, str]]:
    p, fs = _fs(spark, f"{tpath}/_manifests")
    if not fs.exists(p):
        return {}, {}
    out, raw = {}, {}
    for s in fs.listStatus(p):
        name = str(s.getPath().getName())
        if MANIFEST.fullmatch(name):
            raw[int(name[:6])] = read_text(spark, f"{tpath}/_manifests/{name}")
            out[int(name[:6])] = json.loads(raw[int(name[:6])])
    return out, raw


# ---------------------------------------------------------------- Databricks REST

class Databricks:
    def __init__(self, host: str, token: str):
        self.host = host.rstrip("/")
        self.h = {"Authorization": f"Bearer {token}"}

    def request(self, method: str, path: str, ok404: bool = False, headers: dict | None = None,
                **kw) -> requests.Response:
        for attempt in range(RETRIES):
            try:
                r = requests.request(method, self.host + path, headers={**self.h, **(headers or {})}, timeout=300,
                                     **kw)
            except requests.RequestException as e:
                if attempt == RETRIES - 1:
                    raise
                logger.warning("%s %s: %s; retrying", method, path, e)
            else:
                if r.status_code == 404 and ok404:
                    return r
                if r.status_code < 400:
                    return r
                if r.status_code not in (429, 500, 502, 503, 504) or attempt == RETRIES - 1:
                    raise RuntimeError(f"{method} {path}: HTTP {r.status_code} {r.text[:300]}")
                logger.warning("%s %s: HTTP %s; retrying", method, path, r.status_code)
            time.sleep(2 ** attempt * 5)
        raise AssertionError("unreachable")

    def list_dir(self, path: str) -> list[str]:
        names, token = [], None
        while True:
            r = self.request("GET", "/api/2.0/fs/directories" + quote(path), ok404=True,
                             params={"page_token": token} if token else None)
            if r.status_code == 404:
                return []
            body = r.json()
            names += [e["name"] for e in body.get("contents", [])]
            token = body.get("next_page_token")
            if not token:
                return names

    def read(self, path: str) -> str:
        return self.request("GET", "/api/2.0/fs/files" + quote(path)).text

    def put(self, path: str, data: bytes) -> None:
        self.request("PUT", "/api/2.0/fs/files" + quote(path), params={"overwrite": "true"}, data=data,
                     headers={"Content-Type": "application/octet-stream"})

    def size(self, path: str) -> int:
        return int(self.request("HEAD", "/api/2.0/fs/files" + quote(path)).headers.get("content-length", -1))

    def manifests(self, tpath: str) -> dict[int, dict]:
        return {int(n[:6]): json.loads(self.read(f"{tpath}/_manifests/{n}"))
                for n in self.list_dir(f"{tpath}/_manifests") if MANIFEST.fullmatch(n)}


def upload_files(spark: SparkSession, host: str, token: str, pairs: list[tuple[str, str]]) -> None:
    """(s3 path, volume path) pairs of one batch folder, one task each: an executor reads the file and PUTs it.
    Matched on the file name: Hadoop hands paths back normalised (file:/ for file:///)."""
    if not pairs:
        return
    dest = {s.rsplit("/", 1)[-1]: d for s, d in pairs}
    if len(dest) != len(pairs):
        raise ValueError("upload_files takes the files of one folder: duplicate names")

    def put(item):
        src, data = item
        path = dest[src.rsplit("/", 1)[-1]]
        Databricks(host, token).put(path, bytes(data))
        return path, len(data)

    rdd = spark.sparkContext.binaryFiles(",".join(s for s, _ in pairs), minPartitions=len(pairs))
    for path, n in rdd.map(put).collect():
        logger.info("uploaded %s (%.1f MB)", path, n / 1e6)


def run_update(dbx: Databricks, pipeline_id: str, wait_minutes: int) -> str:
    base = f"/api/2.0/pipelines/{pipeline_id}"
    deadline = time.time() + wait_minutes * 60
    while True:
        latest = (dbx.request("GET", base).json().get("latest_updates") or [{}])[0]
        if latest.get("state") in (None, *UPDATE_DONE, *UPDATE_BAD):
            break
        if time.time() > deadline:
            raise TimeoutError(f"pipeline {pipeline_id}: update {latest.get('update_id')} still "
                               f"{latest.get('state')} after {wait_minutes} minutes")
        logger.info("pipeline busy (update %s %s); waiting", latest.get("update_id"), latest.get("state"))
        time.sleep(30)
    update = dbx.request("POST", f"{base}/updates", json={"full_refresh": False}).json()["update_id"]
    logger.info("started pipeline %s update %s", pipeline_id, update)
    state, last = "", ""
    while time.time() < deadline:
        state = dbx.request("GET", f"{base}/updates/{update}").json()["update"]["state"]
        if state != last:
            logger.info("pipeline update %s: %s", update, state)
            last = state
        if state in UPDATE_DONE:
            return update
        if state in UPDATE_BAD:
            raise RuntimeError(f"pipeline {pipeline_id} update {update} ended {state}: see the pipeline's event log")
        time.sleep(30)
    raise TimeoutError(f"pipeline {pipeline_id} update {update} still {state} after {wait_minutes} minutes")


# ---------------------------------------------------------------- the job

def run(spark: SparkSession, cfg: dict, tables: list[str] | None = None, dry_run: bool = False,
        trigger: bool = True, token: str | None = None) -> dict[str, list[int]]:
    push = cfg["databricks_push"]
    token = token or load_token(push["credential"])
    dbx = Databricks(push["host"], token)
    names = tables or list(cfg["tables"])
    unknown = [n for n in names if n not in cfg["tables"]]
    if unknown:
        raise ValueError(f"not in config/export.yaml: {unknown}")

    pushed, s3_all, had_volume = {}, {}, {}
    for name in names:
        layer, _, table = name.partition(".")
        src = f"{cfg['target'].rstrip('/')}/{layer}/{table}"
        dst = f"{push['volume_root'].rstrip('/')}/{layer}/{table}"
        s3, raw = s3_manifests(spark, src)
        vol = dbx.manifests(dst)
        todo = plan_push(s3, vol, name)
        s3_all[name], had_volume[name], pushed[name] = s3, bool(vol), todo
        logger.info("%s: S3 batches %s, Volume %s, to push %s", name, _span(s3), _span(vol), todo or "none")
        for b in todo:
            folder = f"{src}/data/{b:06d}"
            files = s3_files(spark, folder)
            if len(files) != s3[b]["files"]:
                raise ValueError(f"{name} batch {b}: {len(files)} Parquet files on S3, the manifest says {s3[b]['files']}")
            if dry_run:
                logger.info("would push %s batch %s (%s, %s files, %.1f MB)", name, b, s3[b]["batch_type"],
                            len(files), sum(files.values()) / 1e6)
                continue
            pairs = [(f"{folder}/{f}", f"{dst}/data/{b:06d}/{f}") for f in sorted(files)]
            upload_files(spark, push["host"], token, pairs)
            for f, n in files.items():
                got = dbx.size(f"{dst}/data/{b:06d}/{f}")
                if got != n:
                    raise ValueError(f"{name} batch {b}: {f} is {got} bytes in the Volume, {n} on S3")
            dbx.put(f"{dst}/_manifests/{b:06d}.json", raw[b].encode("utf-8"))
            logger.info("pushed %s batch %s (%s, counts %s)", name, b, s3[b]["batch_type"], s3[b]["counts"])

    if dry_run or not any(pushed.values()):
        logger.info("nothing pushed%s: pipeline not started", " (dry run)" if dry_run else "")
        return pushed
    blocked = blocking_resyncs(pushed, s3_all, had_volume)
    if blocked:
        raise RuntimeError(f"pushed resync {blocked}: set spend.start_batch to it and run a full refresh of the "
                           "pipeline (databricks/README.md, section 5); not started automatically")
    if trigger and push.get("pipeline_id"):
        run_update(dbx, push["pipeline_id"], int(push.get("wait_minutes", 45)))
    return pushed


def _span(batches: dict) -> str:
    return f"{min(batches)}-{max(batches)}" if batches else "none"


def main(argv=None, spark: SparkSession | None = None) -> None:
    args = parse_args(argv)
    own = spark is None
    spark = spark or SparkSession.builder.appName("spend-push-databricks").getOrCreate()
    try:
        tables = [t.strip() for t in args.tables.split(",") if t.strip()] if args.tables else None
        run(spark, load_config(args.config), tables, dry_run=args.dry_run, trigger=not args.no_trigger)
    finally:
        if own:
            spark.stop()


if __name__ == "__main__":
    main()
