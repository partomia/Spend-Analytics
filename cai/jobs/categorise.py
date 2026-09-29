#!/usr/bin/env python3
"""
CAI Job spend-categorise: map new merchant keys to the 14 spend categories.

  1. Reads the merchant keys in silver.spend_event that ref.merchant_category_map
     has not seen (LEFT ANTI JOIN), with each key's MCCs and event counts.
  2. Categorises them MCC -> SEED -> EMBEDDING (all-MiniLM-L6-v2 blended with
     character n-grams, PLAN decision 11) -> FALLBACK (spend/categorise.py).
  3. Appends them to ref.merchant_category_map. The map is append-only: a key
     keeps the category of the run that first saw it (first_run_date, run_id,
     model_id), so gold history never re-categorises behind the reader's back.

It runs between build_silver and build_gold in the DAG, because gold category
shares join spend_event to the map. Accuracy is measured by spend-model-evaluation
(seed holdout, hidden-MCC holdout) and enforced by the KPI gate there.

Jobs > New Job: script cai/jobs/categorise.py, Python 3.11 runtime. Airflow passes
the run date and trigger through the environment (SPEND_RUN_DATE,
SPEND_TRIGGERED_BY), since a job run ignores arguments.

  python cai/jobs/categorise.py --dry-run                   # categorises, writes nothing
  python cai/jobs/categorise.py --backend parquet --stub    # offline, no torch

Exit codes: 0 done (including nothing new); 1 when the map would still leave
keys uncategorised.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path(os.getcwd())


sys.path.insert(0, str(_repo_root()))

from spend.categorise import categorise_new, load_embedder  # noqa: E402
from spend.config import policy  # noqa: E402
from spend.storage import get_storage  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

REF_TABLES = ("merchant_seed", "mcc_category", "category_taxonomy")


def run_categorise(st, run_date: date | None = None, stub: bool = False, write: bool = True,
                   triggered_by: str = "cai-job") -> dict:
    t0 = time.time()
    if st.name == "impala":
        for key in ("spend_event", *REF_TABLES):
            st.refresh(key)
        if st.exists("merchant_category_map"):
            st.refresh("merchant_category_map")
    run_date = run_date or datetime.now(timezone.utc).date()
    run_id = f"categorise-{run_date:%Y%m%d}-{datetime.now(timezone.utc):%H%M%S}"
    candidates = st.new_merchant_keys()
    summary = {"run_id": run_id, "run_date": run_date, "triggered_by": triggered_by,
               "new_keys": int(candidates["merchant_key"].nunique()), "methods": {}, "model_id": None}
    if summary["new_keys"]:
        ref = {k: st.read(k) for k in REF_TABLES}
        embedder = load_embedder(stub)
        out = categorise_new(candidates, ref, embedder, policy()["categorise"], run_date, run_id)
        summary["model_id"] = embedder.model_id
        summary["methods"] = out["method"].value_counts().to_dict()
        summary["other"] = int((out["category"] == "OTHER").sum())
        if write:
            st.append("merchant_category_map", out)
    summary["unmapped_after"] = 0
    if write and summary["new_keys"]:
        st.refresh("merchant_category_map")
        summary["unmapped_after"] = int(st.new_merchant_keys()["merchant_key"].nunique())
    summary["duration_s"] = round(time.time() - t0, 1)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-date", default=os.environ.get("SPEND_RUN_DATE") or None,
                   help="run date recorded as first_run_date, YYYY-MM-DD (default: today, UTC)")
    p.add_argument("--backend", default=None, help="impala | parquet (default: config)")
    p.add_argument("--stub", action="store_true", help="character n-gram stand-in instead of MiniLM (no torch)")
    p.add_argument("--dry-run", action="store_true", help="do not write the map")
    p.add_argument("--triggered-by", default=os.environ.get("SPEND_TRIGGERED_BY") or "cai-job")
    args, _ = p.parse_known_args()  # a Jupyter-kernel job runtime adds -f <kernel.json>

    run_date = datetime.strptime(args.run_date, "%Y-%m-%d").date() if args.run_date else None
    s = run_categorise(get_storage(args.backend), run_date=run_date, stub=args.stub, write=not args.dry_run,
                       triggered_by=args.triggered_by)
    print(f"\nrun {s['run_id']}: {s['new_keys']} new merchant keys, methods {s['methods']}, "
          f"model {s['model_id']}, {s['duration_s']}s" + (" (dry run, nothing written)" if args.dry_run else ""))
    if not args.dry_run and s["unmapped_after"]:
        logger.error("%d merchant keys still unmapped after the append", s["unmapped_after"])
        return 1
    return 0


if __name__ == "__main__":
    # CAI Jobs run a script inside a Jupyter kernel wrapper: any SystemExit,
    # even sys.exit(0), is reported as a failure. Only exit on real failure.
    _rc = main()
    if _rc:
        sys.exit(_rc)
