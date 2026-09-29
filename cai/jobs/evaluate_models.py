#!/usr/bin/env python3
"""
CAI Job spend-model-evaluation: stage 3 of 3 of the daily run (PLAN decision 1).

  1. Reads metrics.json and the outputs spend-model-training wrote to
     artifacts/<run_date>/.
  2. KPI gate (config/policy.yaml, spend/gate.py): categoriser accuracy on both
     holdouts, WAPE against seasonal naive, P10-P90 coverage, silhouette and
     persona stability.
  3. Logs params, KPIs and the gate detail to MLflow (CAI Experiments) before
     the tables are written, so spend_model_run.mlflow_run_id links the two; a
     missing mlflow does not fail the job.
  4. Writes gold (DELETE + INSERT on run_date). Gate passed: customer_persona,
     persona_profile, persona_flow, spend_forecast, spend_backtest,
     spend_signals, then spend_model_run. Gate failed: only spend_backtest and
     spend_model_run (gate_passed = false) as evidence; yesterday's personas,
     forecast and signals stay live, and the job exits 1 so the DAG shows red.

Jobs > New Job: script cai/jobs/evaluate_models.py, Python 3.11 runtime,
2 vCPU / 8 GB. Airflow passes the run date through SPEND_RUN_DATE; without it
the job evaluates the folder feature engineering prepared last. The GitHub chain
(ci/trigger_cai_pipeline.py) sets SPEND_DRY_RUN=1: the gate decides, nothing is
written.

  python cai/jobs/evaluate_models.py --dry-run                  # gate only, writes nothing
  python cai/jobs/evaluate_models.py --backend parquet --ignore-gate

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

from spend.pipeline import evaluate_models, latest_prepared_date, summary_lines  # noqa: E402
from spend.storage import get_storage  # noqa: E402
from spend.tracking import mlflow_tracker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-date", default=os.environ.get("SPEND_RUN_DATE") or None,
                   help="run date trained by spend-model-training, YYYY-MM-DD (default: the latest prepared)")
    p.add_argument("--backend", default=None, help="impala | parquet (default: config)")
    p.add_argument("--dry-run", action="store_true",
                   default=os.environ.get("SPEND_DRY_RUN", "").lower() in ("1", "true", "yes"),
                   help="do not write any table or MLflow run (env SPEND_DRY_RUN=1, as the GitHub chain sets)")
    p.add_argument("--publish-on-fail", action="store_true", help="publish today's outputs even if the gate fails")
    p.add_argument("--ignore-gate", action="store_true", help="exit 0 even if the KPI gate fails")
    args, _ = p.parse_known_args()  # a Jupyter-kernel job runtime adds -f <kernel.json>

    run_date = datetime.strptime(args.run_date, "%Y-%m-%d").date() if args.run_date else latest_prepared_date()
    out = evaluate_models(get_storage(args.backend), run_date, write=not args.dry_run,
                          publish_on_fail=args.publish_on_fail, tracker=None if args.dry_run else mlflow_tracker)
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
