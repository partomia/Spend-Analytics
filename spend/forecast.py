"""12-week spend forecast per cohort x category (224 series) from gold.spend_cohort_weekly.

Chronos-2 runs zero-shot on each series' weekly history with the calendar
covariates (festival, salary, month-end, holiday count, school-fee weeks) as
known future inputs, and returns P10 / P50 / P90.

Baseline (the gate compares against it): seasonal naive, the same week last
year x the recent trend. The trend is the last 13 weeks against the same weeks
a year earlier, over the weeks that have a year-ago value. With fewer than
MIN_TREND_WEEKS of them (early backtest cut-offs, 53 weeks of history) the trend
is 1.0: a trend from one or two weeks swings the baseline by tens of percent
(PLAN decision 15).

Backtest: forecast.backtest_cutoffs rolling cut-offs, forecast.cutoff_step_weeks
apart, each hiding the next horizon_weeks. WAPE = sum |actual - P50| / sum actual
over every series, week and cut-off; coverage = share of actuals in [P10, P90].

The stub forecaster (CI, no torch) *is* the seasonal-naive baseline with a
+-20% band, so its WAPE ratio is 1.0 by construction.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from spend.features import COVARIATES, series_id

logger = logging.getLogger(__name__)

MIN_TREND_WEEKS = 4
TREND_WEEKS = 13
SEASON = 52
STUB_MODEL_ID = "stub-seasonal-naive"
STUB_BAND = 0.2
KEY = dict(id_column="series_id", timestamp_column="week_end", target="amount_inr")


def series_frame(cohort_weekly: pd.DataFrame) -> pd.DataFrame:
    """Long frame: series_id, cohort, segment, city_tier, category, week_end, amount_inr, covariates."""
    df = cohort_weekly.copy()
    df["week_end"] = pd.to_datetime(df["week_end"])
    df["series_id"] = [series_id(c, k) for c, k in zip(df["cohort"], df["category"])]
    df["amount_inr"] = df["amount_inr"].astype(float)
    for c in COVARIATES:
        df[c] = df[c].astype(float)
    return df.sort_values(["series_id", "week_end"]).reset_index(drop=True)


def future_covariates(series: pd.DataFrame, calendar: pd.DataFrame, after, horizon: int) -> pd.DataFrame:
    """series_id x the next `horizon` weeks after `after`, with the calendar covariates."""
    cal = calendar.copy()
    cal["week_end"] = pd.to_datetime(cal["week_end"])
    weeks = cal[cal["week_end"] > pd.Timestamp(after)].sort_values("week_end").head(horizon)
    if len(weeks) < horizon:
        raise ValueError(f"ref.calendar_weekly has only {len(weeks)} weeks after {after}, need {horizon}")
    ids = series[["series_id"]].drop_duplicates()
    out = ids.merge(weeks[["week_end", *COVARIATES]], how="cross")
    for c in COVARIATES:
        out[c] = out[c].astype(float)
    return out.sort_values(["series_id", "week_end"]).reset_index(drop=True)


def seasonal_naive(hist: pd.DataFrame, future_weeks: list, min_trend_weeks: int = MIN_TREND_WEEKS) -> pd.DataFrame:
    """Same week last year x trend, per series: series_id, week_end, naive."""
    rows = []
    for sid, g in hist.groupby("series_id", sort=True):
        y = g.set_index("week_end")["amount_inr"]
        last = y.index.max()
        recent = [w for w in y.index[y.index > last - pd.Timedelta(weeks=TREND_WEEKS)]
                  if w - pd.Timedelta(weeks=SEASON) in y.index]
        trend = 1.0
        if len(recent) >= min_trend_weeks:
            then = sum(y[w - pd.Timedelta(weeks=SEASON)] for w in recent)
            trend = float(sum(y[w] for w in recent) / then) if then > 0 else 1.0
        fallback = float(y.tail(TREND_WEEKS).mean())
        for w in future_weeks:
            ago = w - pd.Timedelta(weeks=SEASON)
            rows.append((sid, w, float(y[ago]) * trend if ago in y.index else fallback))
    return pd.DataFrame(rows, columns=["series_id", "week_end", "naive"])


class ChronosForecaster:
    def __init__(self, model_id: str | None = None, device: str | None = None):
        from chronos import Chronos2Pipeline

        from spend.config import settings
        from spend.models import model_path, resolve_device

        self.model_id = model_id or settings()["model"]["forecaster"]
        self.device = resolve_device(device)
        self.pipe = Chronos2Pipeline.from_pretrained(model_path(self.model_id), device_map=self.device)

    def predict(self, hist: pd.DataFrame, future: pd.DataFrame, horizon: int, quantiles: list[float]) -> pd.DataFrame:
        cols = ["series_id", "week_end", "amount_inr", *COVARIATES]
        pred = self.pipe.predict_df(hist[cols], future_df=future[["series_id", "week_end", *COVARIATES]],
                                    prediction_length=horizon, quantile_levels=list(quantiles), **KEY)
        lo, mid, hi = (str(q) for q in quantiles)
        out = pred[["series_id", "week_end"]].copy()
        out["week_end"] = pd.to_datetime(out["week_end"])
        for name, q in (("p10", lo), ("p50", mid), ("p90", hi)):
            out[name] = pred[q].clip(lower=0).astype(float).to_numpy()
        return out


class StubForecaster:
    model_id = STUB_MODEL_ID
    device = "cpu"

    def predict(self, hist: pd.DataFrame, future: pd.DataFrame, horizon: int, quantiles: list[float]) -> pd.DataFrame:
        weeks = sorted(pd.to_datetime(future["week_end"]).unique())
        out = seasonal_naive(hist, weeks).rename(columns={"naive": "p50"})
        out["p10"], out["p90"] = out["p50"] * (1 - STUB_BAND), out["p50"] * (1 + STUB_BAND)
        return out[["series_id", "week_end", "p10", "p50", "p90"]]


def load_forecaster(stub: bool = False):
    if stub:
        return StubForecaster()
    try:
        return ChronosForecaster()
    except ImportError:
        logger.warning("chronos-forecasting not installed: using the seasonal-naive stand-in")
        return StubForecaster()


def _with_info(df: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    info = series[["series_id", "cohort", "segment", "city_tier", "category"]].drop_duplicates("series_id")
    return df.merge(info, on="series_id", how="left")


def forecast(series: pd.DataFrame, calendar: pd.DataFrame, forecaster, cfg: dict) -> pd.DataFrame:
    """The next horizon_weeks after the last observed week, with the naive baseline alongside."""
    h = int(cfg["horizon_weeks"])
    last = series["week_end"].max()
    future = future_covariates(series, calendar, last, h)
    pred = forecaster.predict(series, future, h, cfg["quantiles"])
    naive = seasonal_naive(series, sorted(future["week_end"].unique()))
    out = pred.merge(naive, on=["series_id", "week_end"])
    out["horizon_week"] = ((out["week_end"] - last).dt.days // 7).astype(int)
    return _with_info(out, series)


def cutoffs(weeks: list, cfg: dict) -> list:
    h, step, n = int(cfg["horizon_weeks"]), int(cfg["cutoff_step_weeks"]), int(cfg["backtest_cutoffs"])
    min_hist = int(cfg["min_history_weeks"])
    out = []
    for i in range(n):
        cut = len(weeks) - 1 - h - i * step
        if cut + 1 >= min_hist:
            out.append(weeks[cut])
    return sorted(out)


def backtest(series: pd.DataFrame, forecaster, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Week-grain backtest rows (cutoff, series, week, horizon_week, actual, P10/50/90, naive) and KPIs."""
    h = int(cfg["horizon_weeks"])
    weeks = sorted(series["week_end"].unique())
    parts = []
    for cut in cutoffs(weeks, cfg):
        hist = series[series["week_end"] <= cut]
        hidden = series[(series["week_end"] > cut) & (series["week_end"] <= cut + pd.Timedelta(weeks=h))]
        pred = forecaster.predict(hist, hidden, h, cfg["quantiles"])
        naive = seasonal_naive(hist, sorted(hidden["week_end"].unique()))
        m = (hidden[["series_id", "week_end", "amount_inr"]].rename(columns={"amount_inr": "actual"})
             .merge(pred, on=["series_id", "week_end"]).merge(naive, on=["series_id", "week_end"]))
        m["cutoff"] = cut
        m["horizon_week"] = ((m["week_end"] - cut).dt.days // 7).astype(int)
        parts.append(m)
    if not parts:
        return pd.DataFrame(), {"wape_model": None, "wape_naive": None, "wape_ratio": None, "coverage": None,
                                "n_cutoffs": 0}
    rows = _with_info(pd.concat(parts, ignore_index=True), series)
    return rows, kpis(rows)


def wape(actual: pd.Series, pred: pd.Series) -> float | None:
    total = float(actual.sum())
    return float((actual - pred).abs().sum() / total) if total > 0 else None


def kpis(rows: pd.DataFrame) -> dict:
    wm, wn = wape(rows["actual"], rows["p50"]), wape(rows["actual"], rows["naive"])
    cover = float(((rows["actual"] >= rows["p10"]) & (rows["actual"] <= rows["p90"])).mean())
    return {"wape_model": wm, "wape_naive": wn, "wape_ratio": wm / wn if wm is not None and wn else None,
            "coverage": cover, "n_cutoffs": int(rows["cutoff"].nunique())}
