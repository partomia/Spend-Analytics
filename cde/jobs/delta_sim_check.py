"""
Check the Delta simulation: the SCD type 2 tables databricks/ingest_manifest_driven.py writes
on CDE (cde/scripts/deploy_delta_sim.sh), against the export manifests and the Iceberg source.

Per table:
  - the target has applied every batch that has a manifest;
  - per batch, the versions opened and closed match the manifest counts (incremental: I+U
    opened, U+D closed; the first full batch: I opened; a later full/resync batch opens at most I+U);
  - one open version per key, and as many open rows as the last manifest's source_row_count;
  - every token in the open rows decrypts (try_aes_decrypt) with the key from the mounted
    credential, and the decrypted keys equal the Iceberg source keys at the last batch's
    to_snapshot_id, in both directions.
Prints counts only: no key material, no decrypted values. Exits 1 if a check fails.

Usage:
  delta_sim_check.py [--root s3a://.../databricks_export] [--target s3a://.../databricks_sim]
                     [--tables customer,account,spend_event] [--source-catalog ice]
                     [--credential rsingh-spend-anl-token-key | --key-file key.json]
"""

import argparse
import base64
import json
import sys
from pathlib import Path

from pyspark.sql import functions as F

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent.parent / "databricks")]
import ingest_manifest_driven as ing  # noqa: E402

spark = ing.spark
KEY_DIR = Path("/etc/dex/secrets")
ROOT = "s3a://federal-buk-574bcea0/data/rsingh_spend_anl/databricks_export"
TARGET = "s3a://federal-buk-574bcea0/data/rsingh_spend_anl/databricks_sim"
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


def load_key(credential: str, key_file: str | None) -> tuple[str, bytes]:
    if key_file:
        raw = json.loads(Path(key_file).read_text())
    else:
        d = KEY_DIR / credential
        raw = {f: (d / f).read_text().strip() for f in ("key_version", "enc_key_b64")}
    return raw["key_version"], base64.b64decode(raw["enc_key_b64"])


def check_table(args, table: str, key_version: str, key) -> None:
    cfg = ing.TABLES[table]
    target = ing.Target(args.target, table)
    ms = ing.manifests(args.root, cfg["layer"], table)
    if not ms:
        check(f"{table}: manifests", False, "none found")
        return
    last = ms[-1]
    if not target.exists():
        check(f"{table}: target", False, f"{target.ref} is not a Delta table")
        return
    t = target.read().cache()
    applied = ing.applied_batch(target)
    check(f"{table}: applied every batch", applied == last["batch"], f"target at {applied}, last manifest {last['batch']}")

    opened = {r[0]: r[1] for r in t.groupBy("__START_AT").count().collect()}
    closed = {r[0]: r[1] for r in t.where("__END_AT IS NOT NULL").groupBy("__END_AT").count().collect()}
    for m in ms:
        b, c, o, x = m["batch"], m["counts"], opened.get(m["batch"], 0), closed.get(m["batch"], 0)
        if m["batch_type"] == "incremental":
            ok = o == c["I"] + c["U"] and x == c["U"] + c["D"]
            want = f"I+U={c['I'] + c['U']}, U+D={c['U'] + c['D']}"
        elif b == ms[0]["batch"]:
            ok, want = o == c["I"] and x == 0, f"I={c['I']}, 0"
        else:
            ok, want = o <= c["I"] + c["U"], f"at most I+U={c['I'] + c['U']}"
        check(f"{table}: batch {b} {m['batch_type']} versions", ok, f"opened {o}, closed {x} (want {want})")

    keys = cfg["keys"]
    cur = t.where("__END_AT IS NULL")
    n_open = cur.count()
    check(f"{table}: open rows = source rows", n_open == last["source_row_count"],
          f"{n_open} vs {last['source_row_count']} at batch {last['batch']}")
    dup = cur.groupBy(*keys).count().where("count > 1").count()
    check(f"{table}: one open version per key", dup == 0, f"{dup} keys with more than one")

    if last["key_version"] != key_version:
        check(f"{table}: key version", False, f"manifest {last['key_version']}, credential {key_version}")
        return
    tok = [c for c in last["tokenized_columns"] if c in cur.columns]
    # the key joins in as data, so it never appears as a literal in a query plan
    dec = cur.crossJoin(F.broadcast(key))
    for c in tok:
        dec = dec.withColumn(f"_raw_{c}", F.expr(f"CAST(try_aes_decrypt(unbase64(`{c}`), _k, 'GCM') AS STRING)"))
    bad = dec.select(*[F.sum(F.when(F.col(c).isNotNull() & F.col(f"_raw_{c}").isNull(), 1).otherwise(0)).alias(c)
                       for c in tok]).first().asDict() if tok else {}
    check(f"{table}: every token decrypts", not any(bad.values()), json.dumps(bad))

    got = dec.select(*[(F.col(f"_raw_{k}") if k in tok else F.col(k)).alias(k) for k in keys])
    src = (spark.read.option("snapshot-id", last["to_snapshot_id"])
           .table(f"{args.source_catalog}.{last['source_table']}").select(*keys))
    missing, extra = src.exceptAll(got).count(), got.exceptAll(src).count()
    check(f"{table}: open keys = Iceberg keys at snapshot {last['to_snapshot_id']}",
          missing == 0 and extra == 0, f"{missing} source keys missing in Delta, {extra} Delta keys not in source")
    t.unpersist()


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=ROOT)
    p.add_argument("--target", default=TARGET)
    p.add_argument("--tables", default=",".join(ing.TABLES))
    p.add_argument("--source-catalog", default="ice")
    p.add_argument("--credential", default="rsingh-spend-anl-token-key")
    p.add_argument("--key-file")
    args, _ = p.parse_known_args(argv)
    args.root = args.root.rstrip("/")
    key_version, enc = load_key(args.credential, args.key_file)
    key = spark.createDataFrame([(bytearray(enc),)], "_k binary")
    for table in [t.strip() for t in args.tables.split(",") if t.strip()]:
        check_table(args, table, key_version, key)
    print(f"DELTA SIM CHECK {'FAILED: ' + ', '.join(failures) if failures else 'OK'}", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
