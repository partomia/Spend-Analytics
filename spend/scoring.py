"""Request handling shared by the CAI model endpoint and the app's in-process mode.

Requests (one action each):
  {"action": "profile",    "cif": "C00016807"}
  {"action": "forecast",   "cif": "C00016807", "categories": ["TRAVEL", "DINING"]}   # default: top 3
  {"action": "what_if",    "cif": "C00016807", "shift": {"TRAVEL": 2.0, "channel": {"UPI_P2M": 1.2}}}
  {"action": "categorise", "merchant_texts": ["PHONEPE*RAMESHKIRANA", "UBER TRIP 8841"]}
Responses carry "model": run_id, run_date, snapshot_date, snapshot IDs, model IDs.
A bad request returns {"error": "..."}: a clear message, never a stack trace.

The context is the latest *published* run by run date (not the latest write:
backfills write older dates later): its scaler and persona centroids from
spend_model_run / persona_profile, the gold snapshot it read (pinned), the
seed index for the categoriser and Chronos-2. Per request, only the one CIF's
rows are read from gold and silver.

  profile   persona of that run, 13-week spend by category / channel / city tier
            (gold shares x amounts), home vs away and top away cities from silver,
            and the run's signals for the CIF.
  forecast  12-week P10 / P50 / P90 and mean per category from Chronos-2 on the
            customer's own weekly history (silver events x the pinned category
            map), with the calendar covariates. One customer's weekly spend is
            intermittent, so P50 is often 0: expected_12w (the sum of the weekly
            means, spend.forecast) is the number to quote. Never computed for the
            whole book.
  what_if   multiplies 13-week category and / or channel spend, recomputes the
            shares, and re-assigns the persona: before and after, with the
            distance to every centroid. Extra category spend goes through the
            customer's merchant channels in their current proportions; where it
            happens (city tier, home share) is held.
  categorise  MCC-less text -> SEED (exact) -> EMBEDDING (MiniLM + character
            n-grams, 5-NN vote) -> FALLBACK, as the categorise job does.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from spend import forecast as fc
from spend import personas as ps
from spend.categorise import SeedIndex, assign, normalise
from spend.config import policy, settings
from spend.features import (CAT_SHARES, CATEGORIES, feature_hash, CH_SHARES, CHANNELS, COVARIATES, DIGITAL_CHANNELS,
                            FEATURE_COLUMNS, MERCHANT_CHANNELS, PERSONA_FEATURES, TIER_SHARES, TIERS)

logger = logging.getLogger(__name__)

ACTIONS = ("profile", "forecast", "what_if", "categorise")
MAX_TEXTS = 200
MAX_TEXT_LEN = 200
MAX_SHIFT = 10.0
WEEKS_13 = 13
EVENT_COLUMNS = ["cif", "txn_date", "week_end", "amount", "channel", "merchant_key", "event_city", "event_city_tier",
                 "location_source", "is_home"]
RUN_META = ("run_id", "run_date", "snapshot_date", "features_snapshot_id", "map_snapshot_id", "embedder_id",
            "forecaster_id", "feature_hash")


@dataclass
class Context:
    storage: object
    run: dict
    scaler: ps.Scaler
    centroids: dict[int, np.ndarray]
    names: dict[int, str]
    embedder: object
    forecaster: object
    index: SeedIndex
    calendar: pd.DataFrame
    policy: dict
    meta: dict = field(default_factory=dict)

    @property
    def snapshot_date(self) -> date:
        return self.run["snapshot_date"]


def _as_date(v) -> date:
    return pd.Timestamp(v).date()


def _r(v, nd: int = 2):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), nd)


# ---------------------------------------------------------------- validation

def validate(req) -> str | None:
    if not isinstance(req, dict):
        return "send a JSON object, e.g. {'action': 'profile', 'cif': 'C00016807'}"
    action = req.get("action")
    if action not in ACTIONS:
        return f"action must be one of {list(ACTIONS)}"
    if action == "categorise":
        texts = req.get("merchant_texts")
        if not texts or not isinstance(texts, list):
            return "categorise: send {'merchant_texts': ['...', ...]}"
        if len(texts) > MAX_TEXTS:
            return f"categorise: at most {MAX_TEXTS} merchant_texts per request"
        for i, t in enumerate(texts):
            if not isinstance(t, str) or not t.strip():
                return f"merchant_texts[{i}] must be a non-empty string"
            if len(t) > MAX_TEXT_LEN:
                return f"merchant_texts[{i}] is longer than {MAX_TEXT_LEN} characters"
        return None
    cif = req.get("cif")
    if not isinstance(cif, str) or not cif.strip():
        return f"{action}: send a 'cif', e.g. 'C00016807'"
    if action == "forecast" and req.get("categories") is not None:
        cats = req["categories"]
        if not isinstance(cats, list) or not cats:
            return "categories must be a non-empty list"
        unknown = [c for c in cats if c not in CATEGORIES]
        if unknown:
            return f"categories: unknown {unknown} (one of {CATEGORIES})"
    if action == "what_if":
        shift = req.get("shift")
        if not isinstance(shift, dict) or not shift:
            return "what_if: send a 'shift', e.g. {'TRAVEL': 2.0, 'channel': {'UPI_P2M': 1.2}}"
        channel = shift.get("channel", {})
        if not isinstance(channel, dict):
            return "shift.channel must be an object of channel multipliers"
        unknown = [k for k in shift if k != "channel" and k not in CATEGORIES]
        unknown += [f"channel.{k}" for k in channel if k not in CHANNELS]
        if unknown:
            return f"shift: unknown keys {unknown} (categories {CATEGORIES}, channels {CHANNELS})"
        values = [v for k, v in shift.items() if k != "channel"] + list(channel.values())
        if not values:
            return "shift: no multipliers given"
        for v in values:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                return "shift: multipliers must be finite numbers"
            if not 0 <= v <= MAX_SHIFT:
                return f"shift: multipliers must be between 0 and {MAX_SHIFT:g}"
    return None


# ---------------------------------------------------------------- per-CIF reads

def _features(ctx: Context, cif: str) -> pd.Series | None:
    f = ctx.storage.read_where("spend_features", "cif", [cif], columns=FEATURE_COLUMNS,
                               snapshot_id=ctx.run.get("features_snapshot_id") or None)
    f = f[pd.to_datetime(f["snapshot_date"]).dt.date == ctx.snapshot_date]
    return f.iloc[0] if len(f) else None


def _run_rows(ctx: Context, key: str, cif: str) -> pd.DataFrame:
    if not ctx.storage.exists(key):
        return pd.DataFrame()
    df = ctx.storage.read_where(key, "cif", [cif])
    return df[pd.to_datetime(df["run_date"]).dt.date == ctx.run["run_date"]].reset_index(drop=True)


def _events(ctx: Context, cif: str) -> pd.DataFrame:
    ev = ctx.storage.read_where("spend_event", "cif", [cif], columns=EVENT_COLUMNS)
    ev["week_end"] = pd.to_datetime(ev["week_end"])
    ev["txn_date"] = pd.to_datetime(ev["txn_date"]).dt.date
    ev["amount"] = ev["amount"].astype(float)
    return ev[ev["week_end"] <= pd.Timestamp(ctx.snapshot_date)].reset_index(drop=True)


def _categorised(ctx: Context, ev: pd.DataFrame) -> pd.DataFrame:
    """Merchant events with their category from the pinned map (UNCATEGORISED if missing)."""
    m = ev[ev["channel"].isin(MERCHANT_CHANNELS) & ev["merchant_key"].notna()].copy()
    keys = sorted(m["merchant_key"].unique())
    cmap = ctx.storage.read_where("merchant_category_map", "merchant_key", keys, columns=["merchant_key", "category"],
                                  snapshot_id=ctx.run.get("map_snapshot_id") or None)
    m["category"] = m["merchant_key"].map(cmap.drop_duplicates("merchant_key").set_index("merchant_key")["category"])
    m["category"] = m["category"].fillna("UNCATEGORISED")
    return m


# ---------------------------------------------------------------- persona helpers

def _amounts(f: pd.Series) -> tuple[dict, dict]:
    """13-week INR by category (of categorised merchant spend) and by channel."""
    categorised = float(f["merchant_spend_13w"]) * (1 - float(f["uncategorised_share"]))
    cats = {c: float(f[s]) * categorised for c, s in zip(CATEGORIES, CAT_SHARES)}
    chans = {c: float(f[s]) * float(f["spend_13w"]) for c, s in zip(CHANNELS, CH_SHARES)}
    return cats, chans


def persona_row(f: pd.Series) -> np.ndarray:
    return np.array([np.log1p(float(f["spend_13w"])) if c == ps.LOG_SPEND else float(f[c])
                     for c in PERSONA_FEATURES], dtype=float)


def shifted(f: pd.Series, shift: dict) -> pd.Series:
    """The feature row after multiplying category / channel spend (see module docstring)."""
    g = f.copy()
    cats, chans = _amounts(f)
    new_cats = {c: v * float(shift.get(c, 1.0)) for c, v in cats.items()}
    delta = sum(new_cats.values()) - sum(cats.values())
    merchant = {c: chans[c] for c in MERCHANT_CHANNELS}
    base = sum(merchant.values())
    for c in MERCHANT_CHANNELS:
        weight = merchant[c] / base if base > 0 else (1.0 if c == "UPI_P2M" else 0.0)
        chans[c] = max(chans[c] + delta * weight, 0.0)
    for c, m in (shift.get("channel") or {}).items():
        chans[c] *= float(m)
    total = sum(chans.values())
    cat_total = sum(new_cats.values())
    for c, s in zip(CATEGORIES, CAT_SHARES):
        g[s] = new_cats[c] / cat_total if cat_total > 0 else 0.0
    for c, s in zip(CHANNELS, CH_SHARES):
        g[s] = chans[c] / total if total > 0 else 0.0
    g["digital_share"] = sum(chans[c] for c in DIGITAL_CHANNELS) / total if total > 0 else 0.0
    uncat = float(f["merchant_spend_13w"]) * float(f["uncategorised_share"])
    g["merchant_spend_13w"] = cat_total + uncat
    g["uncategorised_share"] = uncat / (cat_total + uncat) if cat_total + uncat > 0 else 0.0
    g["spend_13w"] = total
    return g


def _distances(ctx: Context, raw: np.ndarray) -> dict[int, float]:
    ids = sorted(ctx.centroids)
    c = ctx.scaler.transform(np.stack([ctx.centroids[i] for i in ids]))
    d = np.linalg.norm(ctx.scaler.transform(raw[None])[0] - c, axis=1)
    return {pid: float(x) for pid, x in zip(ids, d)}


def _nearest(ctx: Context, dist: dict[int, float]) -> dict:
    pid = min(dist, key=dist.get)
    return {"id": pid, "name": ctx.names[pid], "distance": round(dist[pid], 4)}


# ---------------------------------------------------------------- actions

def _profile(ctx: Context, req: dict) -> dict:
    cif = req["cif"].strip()
    f = _features(ctx, cif)
    if f is None:
        return {"error": f"{cif}: no spend features on snapshot {ctx.snapshot_date} (unknown CIF or not onboarded)"}
    cp = _run_rows(ctx, "customer_persona", cif)
    if len(cp):
        r = cp.iloc[0]
        persona = {"id": int(r["persona_id"]), "name": r["persona_name"], "distance": _r(r["distance"], 4),
                   "prev_id": None if pd.isna(r["prev_persona_id"]) else int(r["prev_persona_id"]),
                   "source": "daily run"}
    else:
        persona = {**_nearest(ctx, _distances(ctx, persona_row(f))), "source": "nearest centroid",
                   "note": f"fewer than {policy()['personas']['min_txn_13w']} transactions in 13 weeks: "
                           "not profiled by the daily run"}
    cats, chans = _amounts(f)
    tiers = {f"T{t}": float(f[s]) * float(f["spend_13w"]) for t, s in zip(TIERS, TIER_SHARES)}

    ev = _events(ctx, cif)
    ev13 = ev[ev["week_end"] > pd.Timestamp(ctx.snapshot_date - timedelta(weeks=WEEKS_13))]
    home = ev13[ev13["is_home"] == 1]["amount"].sum()
    away = ev13[ev13["is_home"] == 0]
    top_away = (away.groupby("event_city")["amount"].agg(["sum", "size"]).sort_values("sum", ascending=False).head(5)
                .reset_index())
    by_source = ev13.groupby("location_source")["amount"].sum()

    sig = _run_rows(ctx, "spend_signals", cif)
    signals = [{k: (_r(r[k]) if k in ("metric_value", "threshold") else r[k])
                for k in ("signal_id", "action", "metric", "metric_value", "threshold", "detail")}
               for _, r in sig.iterrows()]
    return {
        "cif": cif, "as_of": ctx.snapshot_date.isoformat(),
        "customer": {k: (int(f[k]) if k == "city_tier" else f[k])
                     for k in ("segment", "age_band", "income_band", "home_city", "home_state", "city_tier", "cohort")},
        "persona": persona,
        "spend_13w": {
            "total": _r(f["spend_13w"]), "n_txn": int(f["txn_13w"]), "avg_ticket": _r(f["avg_ticket"]),
            "trend_4v13": _r(f["spend_trend_4v13"], 3), "digital_share": _r(f["digital_share"], 3),
            "uncategorised_share": _r(f["uncategorised_share"], 3),
            "by_category": {c: _r(v) for c, v in sorted(cats.items(), key=lambda kv: -kv[1]) if v > 0},
            "by_channel": {c: _r(v) for c, v in chans.items() if v > 0},
            "by_city_tier": {t: _r(v) for t, v in tiers.items() if v > 0},
        },
        "home_vs_away": {
            "home_inr": _r(home), "away_inr": _r(away["amount"].sum()), "home_share": _r(f["home_share"], 3),
            "away_share_4w": _r(f["away_share_4w"], 3), "away_weeks_4w": int(f["away_weeks_4w"]),
            "top_away_cities": [{"city": r["event_city"], "amount_inr": _r(r["sum"]), "n_txn": int(r["size"])}
                                for _, r in top_away.iterrows()],
            "by_location_source": {k: _r(v) for k, v in by_source.items()},
        },
        "signals": signals,
    }


def customer_series(ctx: Context, cif: str, categories: list[str]) -> pd.DataFrame:
    """Dense weekly history per category for one CIF, from its first week to the snapshot."""
    m = _categorised(ctx, _events(ctx, cif))
    m = m[m["category"].isin(categories)]
    cal = ctx.calendar.copy()
    cal["week_end"] = pd.to_datetime(cal["week_end"])
    first = m["week_end"].min() if len(m) else pd.Timestamp(ctx.snapshot_date)
    weeks = cal[(cal["week_end"] >= first) & (cal["week_end"] <= pd.Timestamp(ctx.snapshot_date))]
    grid = pd.MultiIndex.from_product([categories, weeks["week_end"]], names=["category", "week_end"])
    amt = m.groupby(["category", "week_end"])["amount"].sum().reindex(grid, fill_value=0.0).reset_index()
    out = amt.rename(columns={"amount": "amount_inr"}).merge(weeks[["week_end", *COVARIATES]], on="week_end")
    out["series_id"] = out["category"]
    for c in COVARIATES:
        out[c] = out[c].astype(float)
    return out.sort_values(["series_id", "week_end"]).reset_index(drop=True)


def _forecast(ctx: Context, req: dict) -> dict:
    cif = req["cif"].strip()
    f = _features(ctx, cif)
    if f is None:
        return {"error": f"{cif}: no spend features on snapshot {ctx.snapshot_date} (unknown CIF or not onboarded)"}
    cats = req.get("categories")
    if not cats:
        amounts, _ = _amounts(f)
        n = int(ctx.policy["forecast"].get("customer_categories", 3))
        cats = [c for c, v in sorted(amounts.items(), key=lambda kv: -kv[1]) if v > 0][:n]
        if not cats:
            return {"error": f"{cif}: no categorised merchant spend in the 13 weeks to {ctx.snapshot_date}"}
    series = customer_series(ctx, cif, cats)
    cfg = ctx.policy["forecast"]
    h = int(cfg["horizon_weeks"])
    future = fc.future_covariates(series, ctx.calendar, ctx.snapshot_date, h)
    pred = ctx.forecaster.predict(series, future, h, cfg["quantiles"])
    snap = pd.Timestamp(ctx.snapshot_date)
    hist = series[series["week_end"] > snap - pd.Timedelta(weeks=WEEKS_13)]
    last52 = series[series["week_end"] > snap - pd.Timedelta(weeks=52)]
    totals = [{"category": c, "expected_12w": _r(pred.loc[pred["series_id"] == c, "mean"].sum(), 0),
               "last_13w": _r(hist.loc[hist["series_id"] == c, "amount_inr"].sum(), 0),
               "run_rate_12w": _r(last52.loc[last52["series_id"] == c, "amount_inr"].mean() * h, 0)} for c in cats]
    return {
        "cif": cif, "as_of": ctx.snapshot_date.isoformat(), "categories": cats,
        "history_weeks": int(series["week_end"].nunique()), "totals": totals,
        "history": [{"category": r.series_id, "week_end": r.week_end.date().isoformat(), "amount": _r(r.amount_inr, 0)}
                    for r in hist.itertuples()],
        "forecast": [{"category": r.series_id, "week_end": pd.Timestamp(r.week_end).date().isoformat(),
                      "p10": _r(r.p10, 0), "p50": _r(r.p50, 0), "p90": _r(r.p90, 0), "mean": _r(r.mean, 0)}
                     for r in pred.sort_values(["series_id", "week_end"]).itertuples()],
    }


def _what_if(ctx: Context, req: dict) -> dict:
    cif, shift = req["cif"].strip(), req["shift"]
    f = _features(ctx, cif)
    if f is None:
        return {"error": f"{cif}: no spend features on snapshot {ctx.snapshot_date} (unknown CIF or not onboarded)"}
    g = shifted(f, shift)
    before, after = _distances(ctx, persona_row(f)), _distances(ctx, persona_row(g))
    b, a = _nearest(ctx, before), _nearest(ctx, after)
    what = ", ".join([f"{k} x{v:g}" for k, v in shift.items() if k != "channel"] +
                     [f"{k} x{v:g}" for k, v in (shift.get("channel") or {}).items()])
    sentence = (f"with {what}, this customer would move from {b['name']} to {a['name']}" if a["id"] != b["id"] else
                f"with {what}, this customer would stay {b['name']}")
    changed = {c: {"before": _r(f[c], 4), "after": _r(g[c], 4)}
               for c in [*CAT_SHARES, *CH_SHARES, "digital_share", "spend_13w"] if abs(float(f[c]) - float(g[c])) > 1e-6}
    return {
        "cif": cif, "as_of": ctx.snapshot_date.isoformat(), "shift": shift,
        "persona_before": b, "persona_after": a, "moved": a["id"] != b["id"], "summary": sentence,
        "distances": [{"persona_id": pid, "name": ctx.names[pid], "before": round(before[pid], 4),
                       "after": round(after[pid], 4)} for pid in sorted(before)],
        "features_changed": changed,
    }


def _categorise(ctx: Context, req: dict) -> dict:
    texts = req["merchant_texts"]
    keys = [normalise(t) for t in texts]
    todo = sorted({k for k in keys if k})
    res = assign(todo, ctx.index, ctx.embedder, ctx.policy["categorise"]).set_index("merchant_key") if todo else None
    out = []
    for t, k in zip(texts, keys):
        if not k:
            out.append({"text": t, "merchant_key": None, "category": "OTHER", "method": "FALLBACK", "similarity": None,
                        "nearest_seed": None})
            continue
        r = res.loc[k]
        out.append({"text": t, "merchant_key": k, "category": r["category"], "method": r["method"],
                    "similarity": _r(r["similarity"], 4), "nearest_seed": r["nearest_seed"]})
    return {"results": out}


HANDLERS = {"profile": _profile, "forecast": _forecast, "what_if": _what_if, "categorise": _categorise}


def score(req, ctx: Context) -> dict:
    if isinstance(req, str):  # the PBJ runtime can hand over the JSON body as a string
        try:
            req = json.loads(req)
        except ValueError:
            return {"error": "request is not valid JSON"}
    err = validate(req)
    if err:
        return {"error": err}
    try:
        resp = HANDLERS[req["action"]](ctx, req)
    except Exception as e:  # the caller gets a message, the log gets the trace
        logger.exception("request %s failed", req.get("action"))
        return {"error": f"{req['action']} failed: {type(e).__name__}: {e}"}
    resp["model"] = ctx.meta
    return resp


# ---------------------------------------------------------------- context

def latest_published_run(storage) -> dict:
    if not storage.exists("spend_model_run"):
        raise RuntimeError("no daily run in spend_model_run yet: run cai/jobs/daily_run.py first")
    runs = storage.read("spend_model_run")
    runs = runs[runs["outputs_published"].astype(bool)]
    if runs.empty:
        raise RuntimeError("no published daily run in spend_model_run yet (every run failed the KPI gate)")
    runs = runs.assign(_d=pd.to_datetime(runs["run_date"]), _ts=pd.to_datetime(runs["run_ts"]))
    r = runs.sort_values(["_d", "_ts"]).iloc[-1]   # by run date: backfills write older dates later
    run = {}
    for k, v in r.drop(["_d", "_ts"]).items():
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            run[k] = None
        elif k in ("run_date", "snapshot_date"):
            run[k] = _as_date(v)
        elif k in ("features_snapshot_id", "cohort_snapshot_id", "map_snapshot_id", "run_id"):
            run[k] = str(v)          # IDs and dates as strings; rates and counts stay numeric
        else:
            run[k] = v
    return run


def load_context(source: str = "auto", stub: bool = False, embedder=None, forecaster=None, storage=None) -> Context:
    """source: impala | file (the local parquet store) | auto (impala if credentials are set)."""
    from spend.categorise import load_embedder
    from spend.storage import get_storage

    if storage is None:
        imp = settings()["impala"]
        use_impala = source == "impala" or (source == "auto" and imp["user"] and imp["password"])
        storage = get_storage("impala" if use_impala else "parquet")
    backend = storage.name
    if backend == "impala":
        for key in ("spend_model_run", "persona_profile", "customer_persona", "spend_signals", "spend_features",
                    "merchant_category_map", "spend_event"):
            storage.refresh(key)
    run = latest_published_run(storage)
    if run.get("feature_hash") != feature_hash():
        raise RuntimeError(f"run {run['run_id']} was built with feature definition {run.get('feature_hash')}, this "
                           f"code has {feature_hash()}: its scaler and centroids do not apply; rerun the daily job "
                           f"with this code, then restart the endpoint")
    profile = storage.read("persona_profile", where_run_date=run["run_date"])
    if profile.empty:
        raise RuntimeError(f"persona_profile has no rows for run {run['run_id']} ({run['run_date']})")
    embedder = embedder or load_embedder(stub)
    index = SeedIndex.build(storage.read("merchant_seed"), storage.read("category_taxonomy"), embedder)
    forecaster = forecaster or fc.load_forecaster(stub)
    meta = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in run.items() if k in RUN_META}
    meta.update(endpoint_embedder=embedder.model_id, endpoint_forecaster=forecaster.model_id, source=backend)
    ctx = Context(storage=storage, run=run, scaler=ps.Scaler.from_json(run["scaler_json"]),
                  centroids=ps.centroids_from_profile(profile),
                  names=dict(zip(profile["persona_id"].astype(int), profile["persona_name"])),
                  embedder=embedder, forecaster=forecaster, index=index, calendar=storage.read("calendar_weekly"),
                  policy=policy(), meta=meta)
    logger.info("endpoint context: run %s (%s, snapshot %s), %d personas, from %s", run["run_id"], run["run_date"],
                run["snapshot_date"], len(ctx.centroids), backend)
    return ctx
