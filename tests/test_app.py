"""Headless render of every tab of the Spend Insights Console against a small parquet
store with two published runs (stub models, no torch), plus the cold starts."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from spend import client, config, pipeline, scoring  # noqa: E402
from spend.config import ROOT  # noqa: E402
from spend.features import CATEGORIES, CHANNELS, MIX_DIMENSIONS  # noqa: E402
from tests.test_pipeline import D1, D2, store  # noqa: E402,F401  (fixture)
from tests.test_scoring import CIF, _events  # noqa: E402

APP = str(ROOT / "app" / "streamlit_app.py")
TABS = ["Portfolio", "Where", "Channels", "Personas", "Customer 360", "Forecast", "Categoriser", "Opportunities",
        "Data quality", "Lineage"]
VALUES = {"category": [*CATEGORIES[:4], "NON_MERCHANT"], "city_tier": ["1", "2"], "state": ["Maharashtra", "Goa"],
          "city": ["Pune", "Panaji"], "away_city": ["Panaji"], "age_band": ["26-35", "36-45"],
          "segment": ["MASS", "AFFLUENT"], "location_source": ["card_city", "home_fallback"]}


def _mix() -> pd.DataFrame:
    rows = []
    for w in range(20):
        week = D2 - timedelta(weeks=w)
        for dim in MIX_DIMENSIONS:
            for k, v in enumerate(VALUES[dim]):
                for j, ch in enumerate(CHANNELS):
                    rows.append({"week_end": week, "dim": dim, "dim_value": v, "channel": ch,
                                 "amount_inr": 1000.0 * (k + 1) * (j + 1) * (0.2 if dim == "away_city" else 1),
                                 "n_txn": 10 + k, "n_customers": 5})
    return pd.DataFrame(rows)


def _dq() -> pd.DataFrame:
    ts = datetime(2026, 8, 28, 21, 0)
    return pd.DataFrame([{"pipeline_run": "scheduled-2026-08-28", "run_id": f"r-{layer}", "run_ts": ts, "as_of": D2,
                          "layer": layer, "table_name": f"{layer}.t", "check_name": f"{layer}_check_{k}",
                          "expectation_type": "expect_x", "column_name": "cif", "severity": sev, "success": ok,
                          "observed_value": "0", "element_count": 100, "unexpected_count": 0 if ok else 3,
                          "unexpected_pct": 0.0 if ok else 3.0, "kwargs": "{}", "table_snapshot_id": 123,
                          "gx_version": "1.23.2"}
                         for layer in ("bronze", "silver", "gold")
                         for k, (sev, ok) in enumerate([("critical", True), ("warning", layer != "silver")])])


def _geo() -> pd.DataFrame:
    return pd.DataFrame({"pincode": ["411001", "403001"], "city": ["Pune", "Panaji"], "state": ["Maharashtra", "Goa"],
                         "city_tier": [1, 2], "lat": [18.52, 15.49], "lon": [73.86, 73.83]})


def _use(monkeypatch, directory):
    monkeypatch.setenv("SPEND_STORAGE_BACKEND", "parquet")
    monkeypatch.setenv("SPEND_STORAGE_PARQUET_DIR", str(directory))
    monkeypatch.setenv("SPEND_ENDPOINT_URL", "")
    config.settings.cache_clear()
    st.cache_data.clear()
    st.cache_resource.clear()  # app/data.py's storage() is cache_resource; a stale one leaks across tests


@pytest.fixture
def lake(store, monkeypatch):  # noqa: F811
    s, art = store
    for name, df in {"spend_event": _events(D2), "spend_mix_weekly": _mix(), "dq_results": _dq(),
                     "pincode_geo": _geo()}.items():
        df.to_parquet(s.dir / f"{name}.parquet", index=False)
    for d in (D1, D2):
        pipeline.run_daily(s, d, stub=True, root=art, publish_on_fail=True)
    _use(monkeypatch, s.dir)
    monkeypatch.setattr(client, "_local", scoring.load_context(stub=True, storage=s))
    yield s, art
    config.settings.cache_clear()
    st.cache_data.clear()
    st.cache_resource.clear()


def _run() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_every_tab_renders(lake):
    at = _run()
    assert [t.label for t in at.tabs] == TABS
    labels = {m.label for m in at.metric}
    assert {"Spend, last 13 weeks", "Customers profiled", "Customers", "WAPE, Chronos-2",
            "Seed holdout accuracy (gate >= 0.85)"} <= labels
    assert any("silver: as of 2026-08-28, 2 checks, 0 critical / 1 warning failed" in w.value
               for w in at.sidebar.warning)
    assert any(e.value == "scheduled-2026-08-28: 1 of 6 checks failed." for e in at.error)
    assert any("customers changed persona" in (c.proto.spec or "") for c in at.get("plotly_chart"))   # Sankey
    assert not at.warning.values or all("KPI gate failed" not in w for w in at.warning.values)


def test_customer_360_profile_forecast_and_what_if(lake):
    at = _run()
    at.text_input(key="c360_cif").set_value(CIF).run()
    assert not at.exception, at.exception
    m = {x.label: x.value for x in at.metric}
    assert "Persona" in m and any(k.endswith("expected next 12 weeks") for k in m)
    at.button(key="wi_run").click().run()
    assert not at.exception, at.exception
    assert any("this customer would" in x.value for x in [*at.success, *at.info])


def test_customer_360_unknown_cif(lake):
    at = _run()
    at.text_input(key="c360_cif").set_value("C99999").run()
    assert not at.exception, at.exception
    assert any("no spend features" in i.value for i in at.info)


def test_categoriser_live(lake):
    at = _run()
    at.text_area(key="cat_texts").set_value("SAI MEDICALS 4411\nHP PETROL PUMP").run()
    at.button(key="cat_run").click().run()
    assert not at.exception, at.exception
    res = next(df.value for df in at.dataframe if "merchant_key" in df.value.columns)
    assert res["merchant_key"].tolist() == ["sai medicals", "hp petrol pump"]


def test_failed_gate_run_says_so(lake):
    s, art = lake
    pipeline.run_daily(s, D2, stub=True, root=art)            # rerun D2 without publish: gate fails
    st.cache_data.clear()
    at = _run()
    assert any("KPI gate failed on this run" in w.value for w in at.warning)
    assert any("No published personas" in i.value for i in at.info)
    at.sidebar.selectbox[0].set_value(D1).run()
    assert not at.exception, at.exception
    assert not any("KPI gate failed on this run" in w.value for w in at.warning)


def test_cold_start_without_tables(tmp_path, monkeypatch):
    _use(monkeypatch, tmp_path)
    try:
        at = AppTest.from_file(APP, default_timeout=60)
        at.run()
        assert not at.exception, at.exception
        assert at.error and "spend_model_run" in at.error[0].value
    finally:
        config.settings.cache_clear()
        st.cache_data.clear()
        st.cache_resource.clear()


def test_cold_start_with_only_a_run(lake, monkeypatch, tmp_path):
    s, _ = lake
    empty = tmp_path / "only_run"
    empty.mkdir()
    s.read("spend_model_run").to_parquet(empty / "spend_model_run.parquet", index=False)
    _use(monkeypatch, empty)
    at = _run()
    assert [t.label for t in at.tabs] == TABS
    infos = " ".join(at.info.values)
    for msg in ("No rows in", "No published personas", "No published forecast", "No published signals",
                "No data-quality results", "No category map"):
        assert msg in infos, msg
