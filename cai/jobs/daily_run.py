#!/usr/bin/env python3
"""
The whole daily run in one process: feature engineering -> model training ->
model evaluation (spend.pipeline.run_daily), the same three stages the DAG runs
as the CAI jobs spend-feature-engineering, spend-model-training and
spend-model-evaluation. For a CAI session, CI and the backfill; also usable as a
single CAI job (GPU or 8 vCPU / 32 GB) if the three-job chain is not wanted.

  python cai/jobs/daily_run.py --dry-run                     # everything but the writes (downloads both models once)
  python cai/jobs/daily_run.py --backend parquet --stub      # offline smoke run, no torch
  python cai/jobs/daily_run.py --backend parquet --stub --ignore-gate   # CI: the stub forecaster *is* naive

Environment: SPEND_RUN_DATE, SPEND_TRIGGERED_BY (a job run ignores arguments).
Exit codes: 0 gate passed (or --ignore-gate); 1 gate failed.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path(os.getcwd())


sys.path.insert(0, str(_repo_root()))

from spend.pipeline import run_daily, summary_lines  # noqa: E402
from spend.storage import get_storage  # noqa: E402
from spend.tracking import mlflow_tracker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-date", default=os.environ.get("SPEND_RUN_DATE") or None,
                   help="snapshot date to run, YYYY-MM-DD (default: latest snapshot in gold)")
    p.add_argument("--backend", default=None, help="impala | parquet (default: config)")
    p.add_argument("--stub", action="store_true", help="n-gram categoriser + seasonal-naive forecaster (no torch)")
    p.add_argument("--dry-run", action="store_true", help="do not write any table or MLflow run")
    p.add_argument("--publish-on-fail", action="store_true", help="publish today's outputs even if the gate fails")
    p.add_argument("--ignore-gate", action="store_true", help="exit 0 even if the KPI gate fails")
    p.add_argument("--triggered-by", default=os.environ.get("SPEND_TRIGGERED_BY") or "cai-job")
    args, _ = p.parse_known_args()  # a Jupyter-kernel job runtime adds -f <kernel.json>

    run_date = datetime.strptime(args.run_date, "%Y-%m-%d").date() if args.run_date else None
    out = run_daily(get_storage(args.backend), run_date, stub=args.stub, triggered_by=args.triggered_by,
                    write=not args.dry_run, publish_on_fail=args.publish_on_fail,
                    tracker=None if args.dry_run else mlflow_tracker)
    print()
    for line in summary_lines(out):
        print(line)
    for line in out["gate_detail"]:
        print(f"gate: {line}")
    if args.dry_run:
        print("dry run: nothing written")
    if not out["gate_passed"] and not args.ignore_gate:
        return 1
    return 0


if __name__ == "__main__":
    # CAI Jobs run a script inside a Jupyter kernel wrapper: any SystemExit,
    # even sys.exit(0), is reported as a failure. Only exit on real failure.
    _rc = main()
    if _rc:
        sys.exit(_rc)
