"""Endpoint request handling on a small parquet store with published runs (stub models), and the
Impala read-back of spend_model_run through a fake Impala."""

from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from spend import pipeline, scoring
from spend.features import CAT_SHARES, CATEGORIES, CH_SHARES
from tests.test_pipeline import D1, D2, store  # noqa: F401  (fixture)
from tests.test_storage import FakeImpala

CIF = "C00000"


def _events(snapshot: date) -> pd.DataFrame:
    rows, weeks = [], [snapshot - timedelta(weeks=w) for w in range(60)]
    for i, w in enumerate(weeks):
        for key, channel, amount in (("sai medicals", "UPI_P2M", 400.0 + i), ("hp petrol pump", "CARD_POS", 1500.0),
                                     (None, "ATM_CASH", 2000.0)):
            away = i < 2 and key == "hp petrol pump"
            rows.append({"cif": CIF, "txn_date": w - timedelta(days=2), "week_end": w, "amount": amount,
                         "channel": channel, "merchant_key": key, "event_city": "Goa" if away else "Pune",
                         "event_city_tier": 2 if away else 1,
                         "location_source": "card_city" if key == "hp petrol pump" else "home_fallback",
                         "is_home": 0 if away else 1})
    future = dict(rows[0], txn_date=snapshot + timedelta(days=3), week_end=snapshot + timedelta(weeks=1))
    return pd.DataFrame([*rows, future])


@pytest.fixture
def ctx(store):  # noqa: F811
    st, art = store
    _events(D2).to_parquet(st.dir / "spend_event.parquet", index=False)
    pipeline.run_daily(st, D1, stub=True, root=art, publish_on_fail=True)
    pipeline.run_daily(st, D2, stub=True, root=art, publish_on_fail=True)
    return scoring.load_context(stub=True, storage=st)


@pytest.mark.parametrize("req, msg", [
    ([1], "JSON object"),
    ({"action": "score"}, "action must be one of"),
    ({"action": "profile"}, "send a 'cif'"),
    ({"action": "forecast", "cif": CIF, "categories": ["TRAVELS"]}, "unknown ['TRAVELS']"),
    ({"action": "what_if", "cif": CIF}, "send a 'shift'"),
    ({"action": "what_if", "cif": CIF, "shift": {"TRAVEL": "2"}}, "finite numbers"),
    ({"action": "what_if", "cif": CIF, "shift": {"TRAVEL": 20}}, "between 0 and 10"),
    ({"action": "what_if", "cif": CIF, "shift": {"channel": {"UPI": 2}}}, "channel.UPI"),
    ({"action": "categorise", "merchant_texts": []}, "merchant_texts"),
    ({"action": "categorise", "merchant_texts": ["ok", " "]}, "merchant_texts[1]"),
    ({"action": "categorise", "merchant_texts": ["x"] * 201}, "at most 200"),
])
def test_validate_gives_a_clear_message(req, msg):
    assert msg in scoring.validate(req)


def test_context_is_the_latest_published_run(ctx):
    assert ctx.run["run_date"] == D2 and ctx.meta["run_id"].startswith("20260828-")
    assert ctx.meta["forecaster_id"] == "stub-seasonal-naive" and ctx.meta["source"] == "parquet"
    assert set(ctx.centroids) == set(ctx.names) and all(len(c) == len(ctx.scaler.mean) for c in ctx.centroids.values())


def test_context_refuses_a_run_built_with_other_features(ctx, monkeypatch):
    assert ctx.meta["feature_hash"] == scoring.feature_hash()
    monkeypatch.setattr(scoring, "feature_hash", lambda: "0" * 16)
    with pytest.raises(RuntimeError, match="feature definition"):
        scoring.load_context(stub=True, storage=ctx.storage)


def test_profile_reads_one_cif(ctx):
    r = scoring.score({"action": "profile", "cif": CIF}, ctx)
    assert r["cif"] == CIF and r["as_of"] == D2.isoformat() and r["persona"]["source"] == "daily run"
    s = r["spend_13w"]
    assert abs(sum(s["by_channel"].values()) - s["total"]) < 1 and list(s["by_category"]) == sorted(
        s["by_category"], key=lambda c: -s["by_category"][c])
    hva = r["home_vs_away"]
    assert hva["top_away_cities"] == [{"city": "Goa", "amount_inr": 3000.0, "n_txn": 2}]
    assert hva["home_inr"] + hva["away_inr"] == pytest.approx(13 * (1500 + 2000) + sum(400 + i for i in range(13)))
    assert r["model"]["run_id"] == ctx.meta["run_id"]
    assert "no spend features" in scoring.score({"action": "profile", "cif": "C99999"}, ctx)["error"]


def test_customer_series_is_dense_and_stops_at_the_snapshot(ctx):
    s = scoring.customer_series(ctx, CIF, ["HEALTH", "FUEL", "TRAVEL"])
    assert s.groupby("series_id").size().tolist() == [60, 60, 60]
    assert s["week_end"].max() == pd.Timestamp(D2)
    assert s[s["series_id"] == "TRAVEL"]["amount_inr"].sum() == 0
    assert s[s["series_id"] == "FUEL"]["amount_inr"].sum() == 60 * 1500.0


def test_forecast_12_weeks_per_category(ctx):
    r = scoring.score({"action": "forecast", "cif": CIF, "categories": ["HEALTH", "FUEL"]}, ctx)
    fc = pd.DataFrame(r["forecast"])
    assert fc.groupby("category").size().to_dict() == {"FUEL": 12, "HEALTH": 12}
    assert fc["week_end"].min() == (D2 + timedelta(weeks=1)).isoformat()
    assert (fc["p10"] <= fc["p50"]).all() and (fc["p50"] <= fc["p90"]).all()
    assert fc[fc["category"] == "FUEL"]["p50"].eq(1500).all()                  # seasonal naive, flat history
    assert r["history_weeks"] == 60 and len(r["history"]) == 2 * 13
    fuel = next(t for t in r["totals"] if t["category"] == "FUEL")
    assert fuel == {"category": "FUEL", "expected_12w": 18000.0, "last_13w": 19500.0, "run_rate_12w": 18000.0}
    default = scoring.score({"action": "forecast", "cif": CIF}, ctx)
    assert len(default["categories"]) == 3


def test_shifted_keeps_the_mix_consistent(ctx):
    f = scoring._features(ctx, CIF)
    cats, chans = scoring._amounts(f)
    g = scoring.shifted(f, {"GROCERY": 2.0, "channel": {"ATM_CASH": 0.5}})
    assert g[CAT_SHARES].sum() == pytest.approx(1) and g[CH_SHARES].sum() == pytest.approx(1)
    assert g["spend_13w"] == pytest.approx(f["spend_13w"] + cats["GROCERY"] - 0.5 * chans["ATM_CASH"])
    same = scoring.shifted(f, {"TRAVEL": 1.0})
    assert np.allclose(same[CAT_SHARES].astype(float), f[CAT_SHARES].astype(float))


def test_what_if_reports_both_personas(ctx):
    r = scoring.score({"action": "what_if", "cif": CIF, "shift": {"GROCERY": 0.0}}, ctx)
    assert {"persona_before", "persona_after", "moved", "summary"} <= set(r)
    assert len(r["distances"]) == len(ctx.centroids) and "cat_grocery_share" in r["features_changed"]
    assert r["summary"].startswith("with GROCERY x0, this customer would")


def test_categorise_texts(ctx):
    r = scoring.score({"action": "categorise", "merchant_texts": ["Apollo Pharmacy 2231", "APOLLO PHARMACY*55812", "4411"]},
                      ctx)["results"]
    assert r[0]["merchant_key"] == r[1]["merchant_key"] == "apollo pharmacy"
    assert r[0]["method"] == "SEED" and r[0]["category"] == "HEALTH"
    assert r[2]["merchant_key"] is None and r[2]["method"] == "FALLBACK"


def test_score_never_raises(ctx, monkeypatch):
    assert scoring.score("{not json", ctx) == {"error": "request is not valid JSON"}
    assert scoring.score('{"action": "profile", "cif": "C00000"}', ctx)["cif"] == CIF   # PBJ string body

    def boom(ctx, req):
        raise KeyError("x")
    monkeypatch.setitem(scoring.HANDLERS, "profile", boom)
    assert scoring.score({"action": "profile", "cif": CIF}, ctx) == {"error": "profile failed: KeyError: 'x'"}


def test_latest_published_run_by_run_date_keeps_numbers_numeric():
    runs = pd.DataFrame({
        "run_date": ["2026-09-25", "2026-09-18", "2026-09-26"], "run_ts": ["2026-09-25 21:10:00",
                                                                           "2026-09-28 09:00:00",
                                                                           "2026-09-26 21:10:00"],
        "run_id": ["a", "b", "c"], "outputs_published": [True, True, False], "snapshot_date": ["2026-09-25"] * 3,
        "features_snapshot_id": [7_000_000_000_000_000_001, 2, 3], "wape_ratio": [Decimal("0.7750"), None, None],
        "k": [8, 8, 8]})
    st = FakeImpala({"SHOW TABLES": pd.DataFrame({"name": ["spend_model_run"]}), "spend_model_run": runs})
    run = scoring.latest_published_run(st)
    assert run["run_id"] == "a"                                  # not the later-written backfill, not unpublished
    assert run["features_snapshot_id"] == "7000000000000000001" and run["run_date"] == date(2026, 9, 25)
    assert isinstance(run["wape_ratio"], float) and run["wape_ratio"] == 0.775
