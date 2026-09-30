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
                 "feature_hash", "k", "n_series", "n_customers_profiled", "map_keys")
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
            if model_run.get("persona_model") is not None:
                _log_persona_model(mlflow, model_run)
        logger.info("logged run %s to MLflow experiment %s", model_run["run_id"], cfg["experiment"])
        return run.info.run_id
    except Exception as e:  # tracking must never fail the run
        logger.warning("MLflow logging failed: %s", e)
        return None


def _log_persona_model(mlflow, model_run: dict) -> None:
    """The run's persona model (spend.registry.persona_pipeline) under artifacts/model, registrable as it is."""
    try:
        import mlflow.sklearn

        from spend.registry import MODEL_PATH, persona_signature

        mlflow.sklearn.log_model(model_run["persona_model"], MODEL_PATH, signature=persona_signature(),
                                 input_example=model_run["persona_example"])
        mlflow.log_dict({str(k): v for k, v in model_run["persona_names"].items()}, "persona_names.json")
        mlflow.set_tag("persona_model", MODEL_PATH)
    except Exception as e:
        logger.warning("persona model not logged: %s", e)
        mlflow.set_tag("persona_model", "not logged")
