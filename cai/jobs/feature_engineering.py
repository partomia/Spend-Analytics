#!/usr/bin/env python3
"""
CAI Job spend-feature-engineering: stage 1 of 3 of the daily run (PLAN decision 1).

  1. REFRESHes gold and the ref inputs, then pins the Iceberg snapshots of
     spend_features, spend_cohort_weekly and merchant_category_map.
  2. Reads the run date's snapshot of gold.spend_features and the one 4 weeks
     earlier (RELOCATION), the 224 cohort x category series up to the run date
     with the calendar covariates, the categoriser holdout inputs and the
     previous published run's personas.
  3. Writes them with manifest.json (run_id, snapshot IDs) to
     artifacts/<run_date>/ on the project filesystem, for spend-model-training.

Jobs > New Job: script cai/jobs/feature_engineering.py, Python 3.11 runtime,
2 vCPU / 8 GB. Airflow passes the run date and trigger through the environment
(SPEND_RUN_DATE, SPEND_TRIGGERED_BY), since a job run ignores arguments.

  python cai/jobs/feature_engineering.py                       # latest snapshot in gold
  python cai/jobs/feature_engineering.py --backend parquet --run-date 2026-09-25

Exit codes: 0 prepared; an error (no gold, run date not a snapshot) fails the job.
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

from spend.pipeline import artifacts_dir, feature_engineering  # noqa: E402
from spend.storage import get_storage  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-date", default=os.environ.get("SPEND_RUN_DATE") or None,
                   help="snapshot date to run, YYYY-MM-DD (default: latest snapshot in gold)")
    p.add_argument("--backend", default=None, help="impala | parquet (default: config)")
    p.add_argument("--triggered-by", default=os.environ.get("SPEND_TRIGGERED_BY") or "cai-job")
    args, _ = p.parse_known_args()  # a Jupyter-kernel job runtime adds -f <kernel.json>

    run_date = datetime.strptime(args.run_date, "%Y-%m-%d").date() if args.run_date else None
    m = feature_engineering(get_storage(args.backend), run_date, args.triggered_by)
    print(f"\nrun {m['run_id']} ({m['run_date']}): {m['n_customers']} customers, prior snapshot "
          f"{m['prior_snapshot_date']}, {m['n_series']} series x {m['cohort_weeks']} weeks, {m['map_keys']} map keys, "
          f"previous run {m['prev_run_date']} -> {artifacts_dir(datetime.fromisoformat(m['run_date']).date())}")
    return 0


if __name__ == "__main__":
    # CAI Jobs run a script inside a Jupyter kernel wrapper: any SystemExit,
    # even sys.exit(0), is reported as a failure. Only exit on real failure.
    _rc = main()
    if _rc:
        sys.exit(_rc)
