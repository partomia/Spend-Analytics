"""The three daily stages end to end on a small parquet store, stub models (no torch)."""

from datetime import date, timedelta

import pandas as pd
import pytest

from spend import pipeline
from spend import storage as st_mod
from spend.schema import OUTPUT_TABLES
from tests.test_categorise import _mcc, _seeds, _taxonomy
from tests.test_models import calendar, synthetic_features, synthetic_series
from tests.test_storage import FakeImpala

D1 = date(2026, 8, 21)
D2 = D1 + timedelta(weeks=1)


@pytest.fixture
def store(tmp_path):
    feats = pd.concat([synthetic_features(snapshot=d, seed=i) for i, d in
                       enumerate([D1 - timedelta(weeks=4), D1, D2])], ignore_index=True)
    feats.to_parquet(tmp_path / "spend_features.parquet", index=False)
    cohort = synthetic_series(weeks=75)            # 2025-04-11 .. 2026-09-11, past D2
    cohort.to_parquet(tmp_path / "spend_cohort_weekly.parquet", index=False)
    calendar(140).assign(week_start=lambda c: pd.to_datetime(c["week_end"]) - pd.Timedelta(days=6)).to_parquet(
        tmp_path / "calendar_weekly.parquet", index=False)
    pd.DataFrame({"merchant_key": ["sai medicals", "hp petrol pump", "bharat petroleum"], "category": ["HEALTH", "FUEL", "FUEL"],
                  "method": ["MCC", "MCC", "MCC"], "first_run_date": [D1, D1, D2 + timedelta(days=30)]}
                 ).to_parquet(tmp_path / "merchant_category_map.parquet", index=False)
    for name, df in {"merchant_seed": _seeds(), "category_taxonomy": _taxonomy(), "mcc_category": _mcc()}.items():
        df.to_parquet(tmp_path / f"{name}.parquet", index=False)
    pd.DataFrame({"cif": feats["cif"].unique(), "archetype": [int(c[1:]) % 3 for c in feats["cif"].unique()]}
                 ).to_parquet(tmp_path / "archetype_truth.parquet", index=False)
    return st_mod.ParquetStorage(tmp_path), tmp_path / "artifacts"


def test_failed_gate_keeps_only_the_evidence(store):
    st, art = store
    r = pipeline.run_daily(st, D1, stub=True, root=art)
    assert not r["gate_passed"] and not r["outputs_published"]
    assert any(line.startswith("FAIL wape_ratio") for line in r["gate_detail"])
    assert st.exists("spend_model_run") and st.exists("spend_backtest") and not st.exists("customer_persona")
    run = st.read("spend_model_run").iloc[0]
    assert not run["gate_passed"] and run["forecaster_id"] == "stub-seasonal-naive"
    assert run["category_accuracy_mcc"] == 1.0 and run["map_keys"] == 3
    assert pipeline.previous_run(st, D2) is None                               # unpublished runs do not count


def test_published_runs_chain_ids_stability_and_flow(store):
    st, art = store
    r1 = pipeline.run_daily(st, D1, stub=True, root=art, publish_on_fail=True)
    assert r1["model_run"]["persona_stability"] is None and r1["outputs"]["persona_flow"].empty
    m = r1["manifest"]
    assert m["prior_snapshot_date"] == (D1 - timedelta(weeks=4)).isoformat()
    assert m["cohort_weeks"] == len([w for w in synthetic_series(weeks=75)["week_end"].unique() if w <= D1])
    for key in OUTPUT_TABLES:
        if key != "merchant_category_map":
            assert st.exists(key), key
            assert list(st.read(key).columns) == [c for c, _ in OUTPUT_TABLES[key]], key

    r2 = pipeline.run_daily(st, D2, stub=True, root=art, publish_on_fail=True)
    assert r2["manifest"]["prev_run_date"] == D1.isoformat()
    assert r2["model_run"]["persona_stability"] is not None
    flow = st.read("persona_flow", where_run_date=D2)
    assert len(flow) and (pd.to_datetime(flow["prev_run_date"]).dt.date == D1).all()
    cp = st.read("customer_persona", where_run_date=D2)
    assert cp["prev_persona_id"].notna().all() and cp["persona_name"].notna().all()
    assert st.run_dates("spend_model_run") == [D1, D2]

    pipeline.run_daily(st, D2, stub=True, root=art, publish_on_fail=True)       # rerun replaces the day
    assert len(st.read("spend_model_run", where_run_date=D2)) == 1
    lines = pipeline.summary_lines(r2)
    assert lines[0].startswith(f"run {r2['model_run']['run_id']}") and "signals" in lines[-1]


def test_stages_run_separately_and_need_their_inputs(store):
    st, art = store
    with pytest.raises(FileNotFoundError, match="feature engineering"):
        pipeline.train_models(D1, stub=True, root=art)
    manifest = pipeline.feature_engineering(st, D1, "airflow", art)
    assert (art / D1.isoformat() / "features.parquet").exists() and manifest["triggered_by"] == "airflow"
    with pytest.raises(FileNotFoundError, match="training"):
        pipeline.evaluate_models(st, D1, root=art)
    metrics = pipeline.train_models(D1, stub=True, root=art)
    assert metrics["n_series"] == 3 and metrics["k"] >= 2
    out = pipeline.evaluate_models(st, D1, write=False, root=art)
    assert out["model_run"]["run_id"] == manifest["run_id"] and not st.exists("spend_model_run")


def test_champion_and_feature_hash_travel_with_the_run(store, monkeypatch):
    st, art = store
    r1 = pipeline.run_daily(st, D1, stub=True, root=art, publish_on_fail=True)
    assert r1["manifest"]["champion_kpis"] is None
    assert any("non-regression skipped" in line for line in r1["gate_detail"])
    assert r1["model_run"]["feature_hash"] == pipeline.feature_hash()
    assert st.read("spend_model_run").iloc[0]["feature_hash"] == pipeline.feature_hash()

    again = pipeline.feature_engineering(st, D1, root=art)       # a rerun of D1 is judged against D1's run
    assert again["champion_run_id"] == r1["model_run"]["run_id"] and again["prev_run_date"] is None
    m2 = pipeline.feature_engineering(st, D2, root=art)
    assert m2["champion_run_id"] == r1["model_run"]["run_id"] and m2["prev_run_date"] == D1.isoformat()
    assert m2["champion_kpis"]["wape_ratio"] == pytest.approx(r1["model_run"]["wape_ratio"])
    # artifacts prepared by other feature code are refused, not trained on
    monkeypatch.setattr(pipeline, "feature_hash", lambda: "0" * 16)
    with pytest.raises(ValueError, match="feature definition"):
        pipeline.train_models(D2, stub=True, root=art)
    monkeypatch.undo()
    pipeline.train_models(D2, stub=True, root=art)
    out = pipeline.evaluate_models(st, D2, write=False, root=art)
    assert sum("champion" in line for line in out["gate_detail"]) == 3


def test_run_date_must_be_a_snapshot(store):
    st, art = store
    with pytest.raises(ValueError, match="not a snapshot date"):
        pipeline.feature_engineering(st, D1 + timedelta(days=1), root=art)
    assert pipeline.feature_engineering(st, None, root=art)["run_date"] == D2.isoformat()


def test_impala_reads_are_pinned_to_the_snapshot():
    st = FakeImpala({"SHOW TABLES": pd.DataFrame({"name": ["spend_features"]})})
    st.read_where("spend_features", "snapshot_date", [D1], columns=["cif"], snapshot_id="42")
    assert st.sql[-1] == ("SELECT `cif` FROM rsingh_spend_anl_gold.spend_features FOR SYSTEM_VERSION AS OF 42 "
                          "WHERE `snapshot_date` IN (DATE '2026-08-21')")
    st.distinct_dates("spend_features", "snapshot_date", "42")
    assert st.sql[-1].startswith("SELECT DISTINCT `snapshot_date` FROM rsingh_spend_anl_gold.spend_features "
                                 "FOR SYSTEM_VERSION AS OF 42")
