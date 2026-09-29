#!/usr/bin/env python3
"""
One-off: run the daily run for the last N Friday snapshots, oldest first, so the
app's history, persona stability and flow charts have data from day one. Each
run reads only what was known on its own run date (spend.pipeline), and chains
to the previous published run for stable persona IDs, so the history is what the
daily job would have produced on those days.

  python cai/jobs/backfill_history.py --weeks 8
  python cai/jobs/backfill_history.py --weeks 4 --backend parquet --stub

Every run is recorded (triggered_by = backfill), including gate failures.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path(os.getcwd())


sys.path.insert(0, str(_repo_root()))

from spend.categorise import load_embedder  # noqa: E402
from spend.forecast import load_forecaster  # noqa: E402
from spend.pipeline import run_daily, summary_lines  # noqa: E402
from spend.storage import get_storage  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def backfill_dates(storage, weeks: int) -> list:
    if storage.name == "impala":
        storage.refresh("spend_features")
    snap = storage.snapshot_id("spend_features")
    fridays = [d for d in storage.distinct_dates("spend_features", "snapshot_date", snap) if d.weekday() == 4]
    return fridays[-weeks:]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weeks", type=int, default=8)
    p.add_argument("--backend", default=None)
    p.add_argument("--stub", action="store_true")
    args, _ = p.parse_known_args()  # a Jupyter-kernel job runtime adds -f <kernel.json>

    storage = get_storage(args.backend)
    embedder, forecaster = load_embedder(args.stub), load_forecaster(args.stub)   # once, not per run
    for d in backfill_dates(storage, args.weeks):
        out = run_daily(storage, d, stub=args.stub, triggered_by="backfill", embedder=embedder,
                        forecaster=forecaster)
        print(f"{d}: " + "; ".join(summary_lines(out)[1:]) + f"; gate_passed {out['gate_passed']}")


if __name__ == "__main__":
    main()
