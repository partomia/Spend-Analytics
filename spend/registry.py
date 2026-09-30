"""The persona model as a registrable artefact, and its versions in the Cloudera AI Registry.

persona_pipeline() turns a run's scaler and persona centroids into a plain
scikit-learn Pipeline: standardise -> clip at +-PERSONA_CLIP -> nearest centroid.
It assigns exactly what personas.assign() assigns, and it is built only from
stock scikit-learn and numpy parts (a native flavour: registry deployments do
not load custom pyfunc classes). Input: a float64 tensor of the 27
PERSONA_FEATURES in order (log_spend_13w = log1p(spend_13w)); output:
persona_id. float64, because a Model Deployment hands the model JSON numbers,
which MLflow will not narrow to a float32 signature; the AI Inference (Triton)
path would need float32 instead.

register_champion() runs after a published run's outputs are written: the
MLflow run's model becomes a new version of registry.model_name. A failed gate
or a backfilled (older) date registers nothing, so the highest version is the
champion: the run the endpoint serves. Each version carries its MLflow run's
parameters (run_id, snapshot IDs, feature hash, git commit) and KPIs, which is
where champion_run_id() reads the run from. (Updating a version's tags through
the API v2 did not return on the federal workbench, so nothing depends on it.)
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from spend.config import settings
from spend.features import PERSONA_CLIP, PERSONA_FEATURES
from spend.personas import Scaler

logger = logging.getLogger(__name__)

MODEL_PATH = "model"
TAG_KPIS = ("silhouette", "persona_stability", "ari_hidden", "category_accuracy_seed", "category_accuracy_mcc",
            "wape_ratio", "coverage")


def persona_pipeline(scaler: Scaler, centroids_raw: dict[int, np.ndarray]):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    n = len(PERSONA_FEATURES)
    std = StandardScaler().fit(np.zeros((2, n)))
    std.mean_, std.scale_, std.var_ = scaler.mean.copy(), scaler.scale.copy(), scaler.scale ** 2
    clip = FunctionTransformer(np.clip, kw_args={"a_min": -PERSONA_CLIP, "a_max": PERSONA_CLIP}).fit(np.zeros((2, n)))
    ids = sorted(centroids_raw)
    nearest = KNeighborsClassifier(n_neighbors=1).fit(
        scaler.transform(np.stack([centroids_raw[i] for i in ids])), np.array(ids, dtype=np.int64))
    return Pipeline([("standardise", std), ("clip", clip), ("nearest_persona", nearest)])


def persona_signature():
    from mlflow.models.signature import ModelSignature
    from mlflow.types import Schema, TensorSpec

    return ModelSignature(inputs=Schema([TensorSpec(np.dtype(np.float64), (-1, len(PERSONA_FEATURES)))]),
                          outputs=Schema([TensorSpec(np.dtype(np.int64), (-1,))]))


def version_tags(model_run: dict, names: dict[int, str]) -> list[dict]:
    tags = {"run_id": model_run["run_id"], "run_date": str(model_run["run_date"]),
            "feature_hash": model_run["feature_hash"], "k": str(model_run["k"]),
            "triggered_by": model_run["triggered_by"], "gate_passed": "true",
            "features": ",".join(PERSONA_FEATURES), "persona_names": json.dumps({str(k): v for k, v in names.items()}),
            **{k: f"{float(model_run[k]):.4f}" for k in TAG_KPIS if model_run.get(k) is not None}}
    return [{"key": k, "value": v} for k, v in tags.items()]


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def find_model(client, name: str):
    models = _get(client.list_registered_models(page_size=100), "models") or []
    return next((m for m in models if _get(m, "name") == name), None)


def champion_version(client, name: str):
    """The highest registered version, or None."""
    model = find_model(client, name)
    if model is None:
        return None
    versions = _get(client.get_registered_model(_get(model, "model_id")), "model_versions") or []
    return max(versions, key=lambda v: _get(v, "number") or 0) if versions else None


def run_id_of(version) -> str | None:
    """The daily run a version was trained in: the run_id parameter of its MLflow run."""
    mlflow_meta = _get(_get(version, "model_version_metadata") or {}, "mlflow_metadata") or {}
    return next((_get(p, "value") for p in _get(mlflow_meta, "params") or [] if _get(p, "key") == "run_id"), None)


def register_champion(client, model_run: dict, names: dict[int, str], experiment_id: str,
                      project_id: str | None = None, name: str | None = None) -> dict:
    """Register the run's MLflow model as the new version; returns {model_id, version, number, previous}."""
    name = name or settings()["registry"]["model_name"]
    previous = champion_version(client, name)
    created = client.create_registered_model({
        "project_id": project_id or os.environ.get("CDSW_PROJECT_ID", ""), "experiment_id": experiment_id,
        "run_id": model_run["mlflow_run_id"], "model_path": MODEL_PATH, "model_name": name,
        "tags": version_tags(model_run, names),
        "description": "Spend personas: standardise, clip, nearest centroid (published daily runs only)",
        "notes": f"run {model_run['run_id']} ({model_run['run_date']}), gate passed", "visibility": "PRIVATE"})
    model_id = _get(created, "model_id")
    versions = _get(client.get_registered_model(model_id), "model_versions") or []
    new = max(versions, key=lambda v: _get(v, "number") or 0)
    out = {"model_id": model_id, "version": _get(new, "model_version_id"), "number": _get(new, "number"),
           "previous": _get(previous, "number") if previous is not None else None}
    logger.info("registered %s version %s for run %s (previous champion: %s)", name, out["number"],
                model_run["run_id"], out["previous"])
    return out


def registrar(model_run: dict, names: dict[int, str]) -> dict | None:
    """register_champion from inside a CAI job (cmlapi.default_client). Never fails the run."""
    if not settings()["registry"]["enabled"] or not model_run.get("mlflow_run_id"):
        return None
    try:
        import cmlapi
        import mlflow

        experiment_id = mlflow.get_run(model_run["mlflow_run_id"]).info.experiment_id
        return register_champion(cmlapi.default_client(), model_run, names, experiment_id)
    except Exception as e:  # the outputs are published; the registry is a record of them
        logger.warning("model registry: %s", e)
        return None
