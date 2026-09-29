#!/usr/bin/env python3
"""
CAI Job spend-model-training: stage 2 of 3 of the daily run (PLAN decision 1).

Reads artifacts/<run_date>/ written by spend-feature-engineering and trains:
  - personas: KMeans on the 27 standardised features (clipped at +-3 sd), k by
    silhouette, IDs matched to the previous run (Hungarian), rule-based names;
  - categoriser holdouts: MiniLM + character n-grams on the 20% seed holdout and
    on card merchants with the MCC hidden;
  - forecast: Chronos-2 rolling backtest against seasonal naive, then the
    12-week P10 / P50 / P90 forecast of the 224 cohort x category series;
  - signals: the policy.signals rules on the run date's features.
Writes the outputs and metrics.json back to the same folder for
spend-model-evaluation. Nothing is written to CDW here.

Jobs > New Job: script cai/jobs/train_models.py, Python 3.11 runtime, GPU or
8 vCPU / 32 GB. Airflow passes the run date through SPEND_RUN_DATE; without it
the job trains the folder feature engineering prepared last.

  python cai/jobs/train_models.py                     # MiniLM + Chronos-2
  python cai/jobs/train_models.py --stub              # n-gram + seasonal-naive stand-ins, no torch

Exit codes: 0 trained; an error (no prepared folder) fails the job. The KPI gate
is applied by spend-model-evaluation, not here.
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

from spend.pipeline import latest_prepared_date, train_models  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-date", default=os.environ.get("SPEND_RUN_DATE") or None,
                   help="run date prepared by feature engineering, YYYY-MM-DD (default: the latest prepared)")
    p.add_argument("--stub", action="store_true", help="n-gram categoriser + seasonal-naive forecaster (no torch)")
    args, _ = p.parse_known_args()  # a Jupyter-kernel job runtime adds -f <kernel.json>

    run_date = datetime.strptime(args.run_date, "%Y-%m-%d").date() if args.run_date else latest_prepared_date()
    m = train_models(run_date, stub=args.stub)
    print(f"\ntrained {run_date}: {m['embedder_id']} + {m['forecaster_id']} on {m['device']}; "
          f"{m['n_customers_profiled']} customers in {m['k']} personas, {m['n_series']} series, "
          f"{m['n_signals']} signals {m['signals_by_rule']}; stages {m['stage_durations']}")
    return 0


if __name__ == "__main__":
    # CAI Jobs run a script inside a Jupyter kernel wrapper: any SystemExit,
    # even sys.exit(0), is reported as a failure. Only exit on real failure.
    _rc = main()
    if _rc:
        sys.exit(_rc)
