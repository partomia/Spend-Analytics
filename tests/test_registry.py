"""The registrable persona model and the Cloudera AI Registry versions (a fake cmlapi client)."""

from datetime import timedelta

import numpy as np
import pytest

from spend import personas as ps
from spend import pipeline, registry
from spend.features import PERSONA_FEATURES
from tests.test_pipeline import D1, D2, store  # noqa: F401  (fixture)


def _fit(seed=0, k=5):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(400, len(PERSONA_FEATURES))) * rng.uniform(0.1, 3, len(PERSONA_FEATURES))
    scaler = ps.Scaler.fit(raw)
    centroids = {int(i): raw[rng.integers(len(raw))] for i in rng.choice(20, size=k, replace=False)}
    return raw, scaler, centroids


def test_persona_pipeline_assigns_what_personas_assign_does():
    raw, scaler, centroids = _fit()
    raw[:5] *= 50                                           # outliers: the clip must matter
    expected, _ = ps.assign(raw, scaler, centroids)
    pipe = registry.persona_pipeline(scaler, centroids)
    assert (pipe.predict(raw) == expected).all()
    assert [s for s, _ in pipe.steps] == ["standardise", "clip", "nearest_persona"]


class FakeRegistry:
    """list / get / create of registered models and versions, as cmlapi returns them (dicts here)."""

    def __init__(self):
        self.models = {}

    def list_registered_models(self, page_size=100):
        return {"models": [{"model_id": mid, "name": m["name"]} for mid, m in self.models.items()]}

    def get_registered_model(self, model_id, page_size=10, page_token=""):
        # as the federal registry does: 10 per page by default, not in version order
        versions = sorted(self.models[model_id]["versions"], key=lambda v: str(v["number"]))
        start = int(page_token or 0)
        rest = start + page_size < len(versions)
        return {"model_id": model_id, "model_versions": versions[start:start + page_size],
                "next_page_token": str(start + page_size) if rest else ""}

    def create_registered_model(self, body):
        mid = next((i for i, m in self.models.items() if m["name"] == body["model_name"]), None)
        if mid is None:
            mid = f"m{len(self.models)}"
            self.models[mid] = {"name": body["model_name"], "versions": []}
        versions = self.models[mid]["versions"]
        run = {"key": "run_id", "value": body["tags"][0]["value"]}        # the MLflow run's params, as CAI copies them
        versions.append({"model_version_id": f"{mid}-v{len(versions) + 1}", "number": len(versions) + 1, "tags": [],
                         "model_version_metadata": {"mlflow_metadata": {"run_id": body["run_id"], "params": [run]}}})
        return {"model_id": mid}


def _model_run(run_id, day):
    return {"run_id": run_id, "run_date": day, "feature_hash": "h", "k": 5, "triggered_by": "airflow",
            "mlflow_run_id": f"ml-{run_id}", "silhouette": 0.16, "wape_ratio": 0.87, "coverage": None}


def test_the_highest_version_is_the_champion_with_its_run():
    fake, names = FakeRegistry(), {0: "Tier-1 traveller"}
    assert registry.champion_version(fake, "spend-persona") is None
    first = registry.register_champion(fake, _model_run("r1", D1), names, "exp", project_id="p", name="spend-persona")
    assert first["number"] == 1 and first["previous"] is None
    second = registry.register_champion(fake, _model_run("r2", D2), names, "exp", project_id="p",
                                        name="spend-persona")
    assert second["number"] == 2 and second["previous"] == 1
    champ = registry.champion_version(fake, "spend-persona")
    assert champ["number"] == 2 and registry.run_id_of(champ) == "r2"
    tags = {t["key"]: t["value"] for t in registry.version_tags(_model_run("r2", D2), names)}
    assert tags["wape_ratio"] == "0.8700" and "coverage" not in tags and tags["run_id"] == "r2"


def test_the_champion_is_found_past_the_first_page():
    # 2026-11: with 18 versions the unpaged GET listed 1-9 and 13, so 14-18 looked missing
    fake, names = FakeRegistry(), {0: "Tier-1 traveller"}
    for i in range(1, 106):
        out = registry.register_champion(fake, _model_run(f"r{i}", D1), names, "exp", project_id="p",
                                         name="spend-persona")
    assert out["number"] == 105 and out["previous"] == 104
    champ = registry.champion_version(fake, "spend-persona")
    assert champ["number"] == 105 and registry.run_id_of(champ) == "r105"


def test_only_a_passed_latest_run_is_registered(store, monkeypatch):  # noqa: F811
    st, art = store
    calls, seen = [], []
    tracker = lambda run: seen.append(run) or "ml-1"        # noqa: E731

    def registrar(run, names):
        calls.append((run["run_date"], names))
        return {"number": len(calls)}

    pipeline.run_daily(st, D2, stub=True, root=art, tracker=tracker, registrar=registrar, publish_on_fail=True)
    assert calls == []                                       # published on fail, but the gate failed
    pipe, ex = seen[-1]["persona_model"], seen[-1]["persona_example"]
    assert ex.dtype == np.float64 and ex.shape[1] == len(PERSONA_FEATURES) and len(pipe.predict(ex)) == len(ex)
    assert set(seen[-1]["persona_names"]) == set(pipe.predict(ex)) | set(seen[-1]["persona_names"])

    monkeypatch.setattr(pipeline.kpi_gate, "evaluate", lambda *a, **k: (True, ["PASS forced"]))
    r = pipeline.run_daily(st, D2, stub=True, root=art, tracker=tracker, registrar=registrar)
    assert r["registered"] == {"number": 1} and calls[0][0] == D2 and calls[0][1]
    r = pipeline.run_daily(st, D1, stub=True, root=art, tracker=tracker, registrar=registrar)
    assert r["gate_passed"] and r["registered"] is None and len(calls) == 1   # a backfill is not the champion


def test_registrar_is_off_without_an_mlflow_run(monkeypatch):
    assert registry.registrar({"mlflow_run_id": None}, {}) is None


@pytest.mark.parametrize("paths, rebuild", [
    (["docs/HOW_IT_WORKS.md", "README.md"], False),
    (["spend/scoring.py"], True), (["requirements.txt"], True), (["cai/model/predict.py"], True),
    (["cai/jobs/evaluate_models.py"], False), (["config/policy.yaml"], True)])
def test_model_code_paths(paths, rebuild):
    from ci.cai_jobs import touches_model

    assert touches_model(paths) is rebuild


def test_d1_is_before_d2():
    assert D1 + timedelta(weeks=1) == D2
