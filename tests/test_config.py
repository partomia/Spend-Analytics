import importlib

import pytest

from spend import config, features, schema


@pytest.fixture
def fresh(monkeypatch):
    def load(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        config.settings.cache_clear()
        return config.settings()
    yield load
    config.settings.cache_clear()


def test_prefix_only_on_databases(fresh):
    s = fresh()
    assert s["databases"]["gold"] == "rsingh_spend_anl_gold"
    assert s["tables"]["spend_features"] == "rsingh_spend_anl_gold.spend_features"
    assert all(t.split(".")[1] == k or k in ("customer", "account") for k, t in s["tables"].items())


def test_env_overrides(fresh):
    s = fresh(SPEND_DATABASES_PREFIX="demo", SPEND_STORAGE_BACKEND="parquet", SPEND_IMPALA_PORT="21050",
              SPEND_TRACKING_ENABLED="false")
    assert s["tables"]["spend_model_run"] == "demo_gold.spend_model_run"
    assert s["storage"]["backend"] == "parquet"
    assert s["impala"]["port"] == 21050
    assert s["tracking"]["enabled"] is False


def test_every_output_table_is_configured(fresh):
    s = fresh()
    assert set(schema.OUTPUT_TABLES) <= set(s["tables"])
    assert set(schema.RUN_OUTPUTS) <= set(schema.OUTPUT_TABLES)
    assert schema.RUN_OUTPUTS[-1] == "spend_model_run"


def test_no_impala_reserved_column_names():
    reserved = {"rows", "location", "date", "timestamp", "range", "sort", "function", "table", "cast", "comment",
                "default", "partition", "values", "percent", "year", "month", "day", "week", "user", "role"}
    for name, cols in schema.OUTPUT_TABLES.items():
        assert not {c for c, _ in cols} & reserved, name
    assert not set(features.FEATURE_COLUMNS) & reserved


def test_policy_gate_and_signals():
    p = config.policy()
    assert p["gate"]["min_category_accuracy"] == 0.85
    assert p["gate"]["coverage_band"] == [0.60, 0.95]
    assert {s["id"] for s in p["signals"]} == {"TRAVEL_SURGE", "FESTIVE_ELECTRONICS", "RELOCATION"}


def test_feature_lists():
    assert len(features.CATEGORIES) == 14 and len(features.CHANNELS) == 6
    assert len(features.SHARE_COLUMNS) == 24
    assert len(features.PERSONA_FEATURES) == 27
    assert len(set(features.FEATURE_COLUMNS)) == len(features.FEATURE_COLUMNS)
    sid = features.series_id(features.cohort_name("MASS", 2), "GROCERY")
    assert features.split_series(sid) == ("MASS|T2", "GROCERY")


def test_requirements_rules():
    root = config.ROOT
    runtime = (root / "requirements.txt").read_text()
    assert "mlflow" not in runtime.split("#")[0] and "download.pytorch.org" not in runtime
    lines = [ln.strip() for ln in runtime.splitlines() if ln.strip() and not ln.startswith("#")]
    assert not any(ln.startswith("mlflow") for ln in lines)
    assert "download.pytorch.org/whl/cpu" in (root / "requirements-train-ci.txt").read_text()
    for f in ("requirements-ci.txt", "requirements-app.txt"):
        body = [ln for ln in (root / f).read_text().splitlines() if not ln.startswith("#")]
        assert not any(ln.startswith(("torch", "sentence-transformers", "chronos")) for ln in body), f


def test_config_module_reloads_cleanly():
    importlib.reload(config)
    assert config.table("dq_results").endswith("_ref.dq_results")
