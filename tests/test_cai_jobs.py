"""CAI job wrappers: MLflow tracker, exit codes, env-driven run date, backfill dates, predict.py."""

import sys
import types
from contextlib import contextmanager
from datetime import timedelta

import pytest

from spend import config, tracking
from tests.conftest import ROOT, _load
from tests.test_pipeline import D1, D2, store  # noqa: F401  (fixture)

JOBS = ROOT / "cai" / "jobs"


def _fake_mlflow(fail: bool = False):
    logged = {}
    m = types.ModuleType("mlflow")
    m.set_experiment = lambda name: logged.update(experiment=name)

    @contextmanager
    def start_run(run_name):
        if fail:
            raise RuntimeError("tracking server down")
        logged["run_name"] = run_name
        yield types.SimpleNamespace(info=types.SimpleNamespace(run_id="abc123"))
    m.start_run = start_run
    m.log_params = lambda p: logged.update(params=p)
    m.log_metrics = lambda p: logged.update(metrics=p)
    m.set_tags = lambda t: logged.update(tags=t)
    m.log_text = lambda text, name: logged.setdefault("text", {}).update({name: text})
    return m, logged


RUN = {"run_id": "20260925-x", "k": 8, "silhouette": 0.166, "persona_stability": None, "wape_ratio": 0.775,
       "gate_passed": True, "outputs_published": True, "gate_detail": "PASS a | PASS b",
       "signals_by_rule": {"TRAVEL_SURGE": 3}}


@pytest.fixture
def settings_env(monkeypatch):
    def apply(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        config.settings.cache_clear()
    yield apply
    config.settings.cache_clear()


def test_tracker_logs_and_returns_the_mlflow_id(monkeypatch, settings_env):
    settings_env(SPEND_TRACKING_ENABLED="true")
    m, logged = _fake_mlflow()
    monkeypatch.setitem(sys.modules, "mlflow", m)
    assert tracking.mlflow_tracker(RUN) == "abc123"
    assert logged["experiment"] == "spend-analytics" and logged["run_name"] == "20260925-x"
    assert logged["metrics"] == {"silhouette": 0.166, "wape_ratio": 0.775}       # None metrics dropped
    assert logged["params"]["k"] == "8" and logged["text"]["gate_detail.txt"] == "PASS a\nPASS b"
    assert "TRAVEL_SURGE" in logged["text"]["run_detail.json"]


def test_tracker_failure_or_disabled_never_fails_the_run(monkeypatch, settings_env):
    settings_env(SPEND_TRACKING_ENABLED="true")
    monkeypatch.setitem(sys.modules, "mlflow", _fake_mlflow(fail=True)[0])
    assert tracking.mlflow_tracker(RUN) is None
    settings_env(SPEND_TRACKING_ENABLED="false")
    m, logged = _fake_mlflow()
    monkeypatch.setitem(sys.modules, "mlflow", m)
    assert tracking.mlflow_tracker(RUN) is None and not logged


@pytest.fixture
def jobs_env(store, settings_env, monkeypatch):  # noqa: F811
    st, art = store
    settings_env(SPEND_STORAGE_BACKEND="parquet", SPEND_STORAGE_PARQUET_DIR=str(st.dir),
                 SPEND_STORAGE_ARTIFACTS_DIR=str(art), SPEND_TRACKING_ENABLED="false")
    monkeypatch.delenv("SPEND_RUN_DATE", raising=False)
    monkeypatch.delenv("SPEND_TRIGGERED_BY", raising=False)

    def run(job: str, *argv: str, env: dict | None = None) -> int:
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr(sys, "argv", [job, *argv, "-f", "/tmp/kernel.json"])   # Jupyter kernel extra args
        return _load(JOBS / f"{job}.py").main()
    return st, art, run


def test_three_jobs_chain_through_artifacts(jobs_env):
    st, art, run = jobs_env
    assert run("feature_engineering", env={"SPEND_RUN_DATE": D1.isoformat(), "SPEND_TRIGGERED_BY": "airflow"}) == 0
    assert (art / D1.isoformat() / "manifest.json").exists()
    assert run("train_models", "--stub") == 0
    assert run("evaluate_models", "--dry-run") == 1 and not st.exists("spend_model_run")     # stub fails the gate
    assert run("evaluate_models", "--ignore-gate") == 0
    runs = st.read("spend_model_run")
    assert runs["triggered_by"].tolist() == ["airflow"] and not runs["outputs_published"].any()


def test_train_and_evaluate_default_to_the_last_prepared_date(jobs_env):
    st, art, run = jobs_env
    assert run("feature_engineering", "--run-date", D2.isoformat()) == 0
    assert run("feature_engineering", "--run-date", D1.isoformat()) == 0
    assert run("train_models", "--stub") == 0
    assert (art / D1.isoformat() / "metrics.json").exists() and not (art / D2.isoformat() / "metrics.json").exists()


def test_daily_run_exit_codes(jobs_env):
    st, art, run = jobs_env
    assert run("daily_run", "--stub", "--run-date", D1.isoformat()) == 1
    assert run("daily_run", "--stub", "--publish-on-fail", "--ignore-gate") == 0
    assert st.exists("customer_persona") and st.run_dates("spend_model_run") == [D1, D2]


def test_backfill_takes_the_last_fridays(jobs_env):
    st, art, run = jobs_env
    job = _load(JOBS / "backfill_history.py")
    assert job.backfill_dates(st, 2) == [D1, D2] and job.backfill_dates(st, 1) == [D2]
    assert all(d.weekday() == 4 for d in job.backfill_dates(st, 9)) and len(job.backfill_dates(st, 9)) == 3
    assert (D1 - timedelta(weeks=4)) in job.backfill_dates(st, 9)


def test_predict_serves_the_published_run(jobs_env, monkeypatch):
    st, art, run = jobs_env
    run("daily_run", "--stub", "--publish-on-fail", "--ignore-gate", "--run-date", D2.isoformat())
    import spend.categorise as cat
    import spend.forecast as fc
    monkeypatch.setattr(cat, "load_embedder", lambda stub=False: cat.CharEmbedder())
    monkeypatch.setattr(fc, "load_forecaster", lambda stub=False: fc.StubForecaster())
    monkeypatch.setenv("SPEND_ENDPOINT_CONTEXT", "file")
    predict = _load(ROOT / "cai" / "model" / "predict.py")
    assert predict.CTX.run["run_date"] == D2
    assert predict.predict({"action": "categorise", "merchant_texts": ["hp petrol pump"]})["results"][0]["category"]
    assert "error" in predict.predict({"action": "nope"})
