"""Personas, forecast, signals and the KPI gate on small synthetic inputs (no torch)."""

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from spend import forecast as fc
from spend import gate
from spend import personas as ps
from spend import signals as sg
from spend.config import policy
from spend.features import (CAT_SHARES, CATEGORIES, CH_SHARES, COVARIATES, FEATURE_COLUMNS, PERSONA_FEATURES,
                            TIER_SHARES)

AS_OF = date(2026, 9, 25)
PCFG = {"k_min": 2, "k_max": 4, "sample_rows": 500, "min_txn_13w": 5, "keep_k_within": 0.01}


def synthetic_features(n: int = 240, seed: int = 0, snapshot: date = AS_OF, groups: int = 3) -> pd.DataFrame:
    """n customers in `groups` well-separated spend patterns."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        g = i % groups
        cat = rng.dirichlet(np.ones(len(CAT_SHARES)) * 0.3)
        cat[g * 3] += 3.0
        cat /= cat.sum()
        ch = rng.dirichlet(np.ones(len(CH_SHARES)))
        ch[g] += 2.0
        ch /= ch.sum()
        tier = np.zeros(4)
        tier[g % 4] = 1.0
        spend = float(rng.lognormal(10 + g, 0.3))
        r = {"cif": f"C{i:05d}", "snapshot_date": snapshot, "segment": "MASS", "age_band": "26-35",
             "income_band": "MID", "home_city": "Pune", "home_state": "Maharashtra", "city_tier": 1,
             "cohort": "MASS|T1", **dict(zip(CAT_SHARES, cat)), **dict(zip(CH_SHARES, ch)),
             **dict(zip(TIER_SHARES, tier)), "home_share": 0.9, "digital_share": float(ch[1:5].sum()),
             "uncategorised_share": 0.0, "salary_week_share": 0.3, "spend_13w": spend, "txn_13w": 40,
             "avg_ticket": spend / 40, "spend_trend_4v13": 1.0, "merchant_spend_13w": spend * 0.8,
             "spend_4w": spend * 4 / 13, "txn_4w": 12, "away_share_4w": 0.0, "away_weeks_4w": 0}
        for c in CATEGORIES:
            r[f"cat_{c.lower()}_4w"] = r[f"cat_{c.lower()}_share"] * spend * 0.8 * 4 / 13
        rows.append(r)
    return pd.DataFrame(rows)[FEATURE_COLUMNS]


# ---------------------------------------------------------------- personas

def test_scaler_clips_and_round_trips():
    raw = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [100.0, 1.0]])
    s = ps.Scaler.fit(raw)
    x = s.transform(raw)
    assert x.max() <= ps.CLIP and (x[:, 1] == 0).all()          # zero variance column -> scale 1
    s2 = ps.Scaler.from_json(json.dumps({"mean": s.mean.tolist(), "scale": s.scale.tolist()}))
    np.testing.assert_allclose(s2.transform(raw), x)
    near = ps.Scaler.fit(np.full((3, 1), 0.9) + np.array([[0.0], [1e-16], [0.0]]))   # float-noise std
    assert ps.Scaler.from_json(near.to_json()).scale[0] == 1.0


def test_fit_personas_finds_the_groups_and_names_are_unique():
    m = ps.persona_matrix(synthetic_features(), 5)
    fit = ps.fit_personas(m, PCFG)
    assert fit.k == 3 and fit.silhouette > 0.2
    assert len(set(fit.names.values())) == len(fit.names) == 3
    assert all(n.startswith("Tier-") for n in fit.names.values())
    truth = pd.DataFrame({"cif": m["cif"], "archetype": [int(c[1:]) % 3 for c in m["cif"]]})
    out = ps.outputs(m, fit, None, None)
    assert ps.ari_hidden(out["customer_persona"], truth) == pytest.approx(1.0)


def test_persona_matrix_filters_low_activity():
    f = synthetic_features(10)
    f.loc[:4, "txn_13w"] = 2
    m = ps.persona_matrix(f, 5)
    assert len(m) == 5 and (m["log_spend_13w"] == np.log1p(m["spend_13w"])).all()


def test_ids_are_stable_across_runs_even_when_kmeans_relabels():
    a = ps.persona_matrix(synthetic_features(seed=1), 5)
    fa = ps.fit_personas(a, PCFG)
    prof = ps.outputs(a, fa, None, None)["persona_profile"]
    b = ps.persona_matrix(synthetic_features(seed=2), 5)       # same groups, new noise, same CIFs
    fb = ps.fit_personas(b, PCFG, ps.centroids_from_profile(prof), prev_k=fa.k)
    ca = ps.outputs(a, fa, None, None)["customer_persona"]
    cb = ps.outputs(b, fb, ca[["cif", "persona_id"]], AS_OF - timedelta(days=7), fa.names)
    assert ps.stability(ca, cb["customer_persona"]) == pytest.approx(1.0)
    assert fb.names == fa.names and all(v is not None for v in fb.match_distance.values())
    flow = cb["persona_flow"]
    assert set(flow["from_persona_id"]) == set(flow["to_persona_id"]) and flow["n_customers"].sum() == len(b)


def test_match_ids_gives_new_ids_to_extra_clusters():
    prev = {0: np.zeros(2), 5: np.ones(2)}
    new = np.array([[1.0, 1.0], [0.1, 0.0], [9.0, 9.0]])
    mapping, dist = ps.match_ids(new, prev)
    assert mapping == {0: 5, 1: 0, 2: 6} and dist[2] is None


def test_keep_previous_k_within_tolerance():
    x = ps.Scaler.fit(ps.persona_matrix(synthetic_features(), 5)[PERSONA_FEATURES].to_numpy()).transform(
        ps.persona_matrix(synthetic_features(), 5)[PERSONA_FEATURES].to_numpy())
    k, scores, _ = ps.choose_k(x, {**PCFG, "keep_k_within": 10.0}, prev_k=4)
    assert k == 4 and set(scores) == {2, 3, 4}


def test_assign_nearest_persona():
    m = ps.persona_matrix(synthetic_features(), 5)
    fit = ps.fit_personas(m, PCFG)
    ids, _ = ps.assign(m[PERSONA_FEATURES].to_numpy(), fit.scaler, fit.centroids_raw)
    assert (ids == fit.labels).mean() > 0.95


# ---------------------------------------------------------------- forecast

def synthetic_series(weeks: int = 70, n_series: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2025, 4, 11)
    rows = []
    for s in range(n_series):
        for w in range(weeks):
            rows.append({"cohort": f"MASS|T{s + 1}", "segment": "MASS", "city_tier": s + 1, "category": "GROCERY",
                         "week_end": start + timedelta(weeks=w), "amount_inr": 1000.0 * (1 + s) * (1 + 0.01 * w)
                         * (1 + 0.3 * np.sin(w * 2 * np.pi / 52)) * rng.lognormal(0, 0.02), "n_txn": 10,
                         "n_customers": 5, **{c: 0 for c in COVARIATES}})
    return pd.DataFrame(rows)


def calendar(weeks: int = 120) -> pd.DataFrame:
    start = date(2025, 4, 4)
    return pd.DataFrame({"week_end": [start + timedelta(weeks=i) for i in range(weeks)],
                         **{c: 0 for c in COVARIATES}, "festival_week": 0})


def test_seasonal_naive_uses_last_year_times_trend():
    s = fc.series_frame(synthetic_series(weeks=66, n_series=1))
    y = s.set_index("week_end")["amount_inr"]
    last = y.index.max()
    nxt = last + pd.Timedelta(weeks=1)
    got = fc.seasonal_naive(s, [nxt]).iloc[0]["naive"]
    recent = [w for w in y.index if w > last - pd.Timedelta(weeks=13)]
    trend = sum(y[w] for w in recent) / sum(y[w - pd.Timedelta(weeks=52)] for w in recent)
    assert got == pytest.approx(y[nxt - pd.Timedelta(weeks=52)] * trend)


def test_seasonal_naive_trend_needs_enough_overlap():
    s = fc.series_frame(synthetic_series(weeks=54, n_series=1))       # 2 weeks with a year-ago value
    y = s.set_index("week_end")["amount_inr"]
    nxt = y.index.max() + pd.Timedelta(weeks=1)
    assert fc.seasonal_naive(s, [nxt]).iloc[0]["naive"] == pytest.approx(y[nxt - pd.Timedelta(weeks=52)])


def test_backtest_cutoffs_and_stub_ratio_is_one():
    cfg = dict(policy()["forecast"])
    s = fc.series_frame(synthetic_series())
    rows, k = fc.backtest(s, fc.StubForecaster(), cfg)
    weeks = sorted(s["week_end"].unique())
    assert k["n_cutoffs"] == len(fc.cutoffs(weeks, cfg)) == 2           # 70 weeks: cut-offs at 58 and 54 weeks
    assert set(rows["horizon_week"]) == set(range(1, 13)) and len(rows) == 2 * 3 * 12
    assert k["wape_ratio"] == pytest.approx(1.0) and 0 < k["wape_model"] < 0.2
    assert 0 <= k["coverage"] <= 1


def test_forecast_next_weeks_with_info():
    cfg = dict(policy()["forecast"])
    s = fc.series_frame(synthetic_series())
    out = fc.forecast(s, calendar(), fc.StubForecaster(), cfg)
    assert len(out) == 3 * 12 and out["horizon_week"].tolist()[:12] == list(range(1, 13))
    assert (out["week_end"] > s["week_end"].max()).all() and {"cohort", "category", "naive"} <= set(out.columns)
    with pytest.raises(ValueError, match="calendar_weekly"):
        fc.forecast(s, calendar(60), fc.StubForecaster(), cfg)


def test_kpis_wape_and_coverage():
    rows = pd.DataFrame({"actual": [100.0, 100.0], "p50": [90.0, 120.0], "naive": [50.0, 150.0],
                         "p10": [80.0, 101.0], "p90": [110.0, 130.0], "cutoff": [1, 1]})
    k = fc.kpis(rows)
    assert (k["wape_model"], k["wape_naive"], k["coverage"]) == (0.15, 0.5, 0.5)
    assert k["wape_ratio"] == pytest.approx(0.3)


# ---------------------------------------------------------------- signals

RULES = policy()["signals"]


def _one(**over):
    f = synthetic_features(1)
    for k, v in over.items():
        f[k] = v
    return f


def test_travel_surge_against_the_prior_nine_weeks():
    f = _one(merchant_spend_13w=100_000.0, uncategorised_share=0.0, cat_travel_share=0.5, cat_travel_4w=45_000.0)
    s = sg.evaluate(f, RULES)
    hit = s[s["signal_id"] == "TRAVEL_SURGE"].iloc[0]
    assert hit["metric_value"] == pytest.approx(45_000 / (5_000 * 4 / 9), rel=1e-3)
    f2 = _one(merchant_spend_13w=100_000.0, cat_travel_share=0.5, cat_travel_4w=15_000.0)   # prior 35k in 9 weeks
    assert "TRAVEL_SURGE" not in set(sg.evaluate(f2, RULES)["signal_id"])


def test_festive_electronics_needs_a_festival_week():
    f = _one(cat_electronics_4w=60_000.0, snapshot_date=date(2025, 10, 24))
    assert sg.evaluate(f, RULES, festival_weeks=set()).empty or "FESTIVE_ELECTRONICS" not in set(
        sg.evaluate(f, RULES, festival_weeks=set())["signal_id"])
    s = sg.evaluate(f, RULES, festival_weeks={date(2025, 10, 10)})
    assert "FESTIVE_ELECTRONICS" in set(s["signal_id"])


def test_relocation_needs_persistence_and_a_home_prior():
    f = _one(away_share_4w=0.95, away_weeks_4w=4, txn_4w=10)
    prior = _one(away_share_4w=0.1)
    assert "RELOCATION" in set(sg.evaluate(f, RULES, prior)["signal_id"])
    assert "RELOCATION" not in set(sg.evaluate(_one(away_share_4w=0.95, away_weeks_4w=2, txn_4w=10), RULES,
                                               prior)["signal_id"])            # a trip
    assert "RELOCATION" not in set(sg.evaluate(f, RULES, _one(away_share_4w=0.8))["signal_id"])  # already away
    assert "RELOCATION" not in set(sg.evaluate(f, RULES, None)["signal_id"])   # no prior snapshot


def test_signal_columns():
    assert list(sg.evaluate(_one(), RULES).columns) == sg.COLUMNS


# ---------------------------------------------------------------- gate

GOOD = {"category_accuracy_seed": 0.93, "category_accuracy_mcc": 0.9, "wape_ratio": 0.8, "wape_model": 0.15,
        "wape_naive": 0.19, "coverage": 0.77, "silhouette": 0.17, "persona_stability": 0.95}


def test_gate_passes_and_each_check_can_fail():
    cfg = policy()["gate"]
    ok, lines = gate.evaluate(GOOD, cfg)
    assert ok and len(lines) == 6 and all(line.startswith("PASS") for line in lines)
    for key, bad in [("category_accuracy_seed", 0.8), ("category_accuracy_mcc", None), ("wape_ratio", 0.95),
                     ("coverage", 0.97), ("coverage", 0.5), ("silhouette", 0.1), ("persona_stability", 0.7)]:
        ok, lines = gate.evaluate({**GOOD, key: bad}, cfg)
        assert not ok and any(line.startswith("FAIL") and key in line for line in lines), key


def test_gate_skips_stability_on_the_first_run():
    ok, lines = gate.evaluate({**GOOD, "persona_stability": None}, policy()["gate"])
    assert ok and any("skipped" in line for line in lines)
