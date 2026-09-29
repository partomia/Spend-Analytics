"""MLflow (CAI Experiments) logging for the daily run, called by the evaluation
stage before the gold writes so spend_model_run.mlflow_run_id links the two.
A missing mlflow or a tracking error never fails the run."""

from __future__ import annotations

import json
import logging

from spend.config import settings

logger = logging.getLogger(__name__)

MLFLOW_PARAMS = ("run_id", "run_date", "snapshot_date", "triggered_by", "embedder_id", "forecaster_id",
                 "model_versions", "device", "features_snapshot_id", "cohort_snapshot_id", "map_snapshot_id",
                 "k", "n_series", "n_customers_profiled", "map_keys")
MLFLOW_METRICS = ("category_accuracy_seed", "category_accuracy_mcc", "silhouette", "persona_stability",
                  "ari_hidden", "wape_model", "wape_naive", "wape_ratio", "coverage", "n_signals",
                  "uncategorised_share", "duration_s")


def mlflow_tracker(model_run: dict) -> str | None:
    cfg = settings()["tracking"]
    if not cfg["enabled"]:
        return None
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not available - skipping experiment tracking")
        return None
    try:
        mlflow.set_experiment(cfg["experiment"])
        with mlflow.start_run(run_name=model_run["run_id"]) as run:
            mlflow.log_params({k: str(model_run.get(k)) for k in MLFLOW_PARAMS})
            mlflow.log_metrics({k: float(v) for k in MLFLOW_METRICS if (v := model_run.get(k)) is not None})
            mlflow.set_tags({"gate_passed": model_run.get("gate_passed"),
                             "outputs_published": model_run.get("outputs_published")})
            mlflow.log_text((model_run.get("gate_detail") or "").replace(" | ", "\n"), "gate_detail.txt")
            extra = {k: model_run[k] for k in ("silhouette_by_k", "signals_by_rule", "stage_durations_json")
                     if k in model_run}
            mlflow.log_text(json.dumps(extra, indent=2, default=str), "run_detail.json")
        logger.info("logged run %s to MLflow experiment %s", model_run["run_id"], cfg["experiment"])
        return run.info.run_id
    except Exception as e:  # tracking must never fail the run
        logger.warning("MLflow logging failed: %s", e)
        return None
