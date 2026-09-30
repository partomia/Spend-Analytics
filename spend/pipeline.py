"""The daily CAI run as three stages (PLAN decision 1), handing off through
artifacts/<run_date>/ on the CAI project filesystem:

  feature_engineering  REFRESH gold, pin the Iceberg snapshots, read the run's
                       snapshot of gold.spend_features (+ the one 4 weeks earlier,
                       for RELOCATION), the cohort series up to the run date, the
                       calendar, the categoriser holdout inputs and the previous
                       published run's personas. Writes manifest.json + inputs.
  train_models         KMeans personas with stable IDs, the categoriser holdouts
                       (MiniLM + character n-grams), the Chronos-2 backtest and
                       12-week forecast, signals. Writes outputs + metrics.json.
  evaluate_models      KPI gate, spend_model_run, MLflow (tracker, with the persona
                       model), gold writes, then a passed run's persona model
                       becomes the champion version in the registry (registrar).
                       A failed gate writes spend_model_run (gate_passed = false)
                       and the backtest as evidence but not today's personas,
                       forecast or signals: yesterday's stay live. Exit 1.

run_daily() runs the three in one process (a session, CI, backfill). A run for
run_date D reads only what was known on D: the D snapshot, cohort weeks ending
on or before D and runs before
D, so a backfill reproduces what the daily run would have said. The one
exception is the categoriser's hidden-MCC holdout, which measures the
categoriser on every MCC-mapped key of the pinned map snapshot: it is a model
quality check, not a dated output.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from spend import forecast as fc
from spend import gate as kpi_gate
from spend import personas as ps
from spend import signals as sg
from spend.categorise import SeedIndex, accuracy, load_embedder, mcc_holdout_predictions, seed_holdout_predictions
from spend.config import ROOT, settings, table
from spend.config import policy as load_policy
from spend.features import FEATURE_COLUMNS, feature_hash
from spend.models import package_versions
from spend.schema import OUTPUT_TABLES, RUN_OUTPUTS

logger = logging.getLogger(__name__)

PRIOR_WEEKS = 4
VERSION_PACKAGES = ("sentence-transformers", "chronos-forecasting", "torch", "scikit-learn")
REF_INPUTS = ("merchant_seed", "category_taxonomy", "mcc_category")
CHAMPION_KPIS = ("category_accuracy_seed", "category_accuracy_mcc", "wape_ratio", "coverage", "silhouette")


def artifacts_dir(run_date: date, root: Path | str | None = None) -> Path:
    base = Path(root or settings()["storage"]["artifacts_dir"])
    base = base if base.is_absolute() else ROOT / base
    return base / run_date.isoformat()


def latest_prepared_date(root: Path | str | None = None) -> date:
    """Run date of the manifest feature engineering wrote last: the default for the
    training and evaluation jobs when neither --run-date nor SPEND_RUN_DATE is set."""
    base = artifacts_dir(date.today(), root).parent
    manifests = sorted(base.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime)
    if not manifests:
        raise FileNotFoundError(f"no manifest under {base}: run the feature engineering stage first")
    return date.fromisoformat(_read_json(manifests[-1])["run_date"])


def _dates(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.date
    return df


def _save(d: Path, name: str, df: pd.DataFrame | None) -> None:
    if df is not None:
        df.to_parquet(d / f"{name}.parquet", index=False)


def _load(d: Path, name: str) -> pd.DataFrame | None:
    p = d / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def _published(storage, run_date: date, same_day: bool) -> pd.Series | None:
    if not storage.exists("spend_model_run"):
        return None
    runs = _dates(storage.read("spend_model_run"), "run_date")
    before = runs["run_date"] <= run_date if same_day else runs["run_date"] < run_date
    runs = runs[before & runs["outputs_published"].astype(bool)]
    return None if runs.empty else runs.sort_values(["run_date", "run_ts"]).iloc[-1]


def previous_run(storage, run_date: date) -> dict | None:
    """The last published run before run_date, which persona IDs, stability and flow
    chain to: its customer personas and profile."""
    last = _published(storage, run_date, same_day=False)
    if last is None:
        return None
    prev_date = last["run_date"]
    assign = storage.read("customer_persona", where_run_date=prev_date, columns=["cif", "persona_id"])
    profile = storage.read("persona_profile", where_run_date=prev_date)
    return {"run_date": prev_date, "run_id": last["run_id"], "k": int(last["k"]), "assign": assign,
            "profile": profile}


def champion_run(storage, run_date: date) -> dict | None:
    """The published run a candidate for run_date would replace (the same day's, else the
    last before it): what stays live if the candidate fails the gate, and the KPIs it must
    not regress from."""
    last = _published(storage, run_date, same_day=True)
    if last is None:
        return None
    return {"run_date": last["run_date"].isoformat(), "run_id": last["run_id"],
            "kpis": {k: (None if pd.isna(last.get(k)) else float(last[k])) for k in CHAMPION_KPIS}}


# ---------------------------------------------------------------- stage 1

def feature_engineering(storage, run_date: date | None = None, triggered_by: str = "manual",
                        root: Path | str | None = None) -> dict:
    t0 = time.time()
    if storage.name == "impala":
        for key in ("spend_features", "spend_cohort_weekly", "merchant_category_map", "calendar_weekly",
                    *REF_INPUTS):
            storage.refresh(key)
    snaps = {k: storage.snapshot_id(k) for k in ("spend_features", "spend_cohort_weekly", "merchant_category_map")}
    dates = storage.distinct_dates("spend_features", "snapshot_date", snaps["spend_features"])
    if not dates:
        raise ValueError(f"{table('spend_features')} is empty: run the CDE gold job first")
    run_date = run_date or dates[-1]
    if run_date not in dates:
        raise ValueError(f"{run_date} is not a snapshot date of {table('spend_features')} "
                         f"(Fridays and the load's as-of date; latest {dates[-1]})")
    prior_date = max((d for d in dates if d <= run_date - timedelta(weeks=PRIOR_WEEKS)), default=None)
    feats = _dates(storage.read_where("spend_features", "snapshot_date", [d for d in (run_date, prior_date) if d],
                                      columns=FEATURE_COLUMNS, snapshot_id=snaps["spend_features"]), "snapshot_date")
    today, prior = feats[feats["snapshot_date"] == run_date], feats[feats["snapshot_date"] == prior_date]

    cohort = _dates(storage.read("spend_cohort_weekly", snapshot_id=snaps["spend_cohort_weekly"]), "week_end")
    cohort = cohort[cohort["week_end"] <= run_date]
    calendar = _dates(storage.read("calendar_weekly"), "week_end", "week_start")
    cmap = storage.read("merchant_category_map", snapshot_id=snaps["merchant_category_map"],
                        columns=["merchant_key", "category", "method"])
    card_keys = cmap.loc[cmap["method"] == "MCC", ["merchant_key", "category"]].rename(columns={"category": "truth"})
    ref = {k: storage.read(k) for k in REF_INPUTS}
    truth = storage.read("archetype_truth", columns=["cif", "archetype"]) if storage.exists("archetype_truth") else None

    merchant = today["merchant_spend_13w"].sum()
    uncat = float((today["uncategorised_share"] * today["merchant_spend_13w"]).sum() / merchant) if merchant else None
    prev, champ = previous_run(storage, run_date), champion_run(storage, run_date)
    run_id = f"{run_date:%Y%m%d}-{uuid.uuid4().hex[:8]}"
    d = artifacts_dir(run_date, root)
    d.mkdir(parents=True, exist_ok=True)
    for name, df in {"features": today, "features_prior": prior, "cohort_weekly": cohort, "calendar": calendar,
                     "card_keys": card_keys, "archetype_truth": truth, **ref,
                     "prev_assign": prev and prev["assign"], "prev_profile": prev and prev["profile"]}.items():
        _save(d, name, df)
    manifest = {
        "run_id": run_id, "run_date": run_date.isoformat(), "snapshot_date": run_date.isoformat(),
        "prior_snapshot_date": prior_date and prior_date.isoformat(), "triggered_by": triggered_by,
        "features_snapshot_id": snaps["spend_features"], "cohort_snapshot_id": snaps["spend_cohort_weekly"],
        "map_snapshot_id": snaps["merchant_category_map"], "n_customers": len(today),
        "n_series": int(cohort[["cohort", "category"]].drop_duplicates().shape[0]),
        "cohort_weeks": int(cohort["week_end"].nunique()), "map_keys": len(cmap), "uncategorised_share": uncat,
        "prev_run_date": prev and prev["run_date"].isoformat(), "prev_run_id": prev and prev["run_id"],
        "prev_k": prev and prev["k"], "champion_run_id": champ and champ["run_id"],
        "champion_kpis": champ and champ["kpis"], "feature_hash": feature_hash(),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage_durations": {"feature_engineering": round(time.time() - t0, 1)},
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("feature engineering %s: snapshot %s (%d customers, prior %s), %d series x %d weeks, %d map keys, "
                "previous run %s -> %s", run_id, run_date, len(today), prior_date, manifest["n_series"],
                manifest["cohort_weeks"], len(cmap), manifest["prev_run_date"], d)
    return manifest


# ---------------------------------------------------------------- stage 2

def train_models(run_date: date, stub: bool = False, root: Path | str | None = None, embedder=None,
                 forecaster=None) -> dict:
    d = artifacts_dir(run_date, root)
    if not (d / "manifest.json").exists():
        raise FileNotFoundError(f"{d / 'manifest.json'} not found: run the feature engineering stage first")
    manifest = _read_json(d / "manifest.json")
    if manifest.get("feature_hash") != feature_hash():
        raise ValueError(f"{d} was prepared with feature definition {manifest.get('feature_hash')}, this code has "
                         f"{feature_hash()}: rerun the feature engineering stage")
    pol = load_policy()
    timings = {}

    t = time.time()
    feats = _dates(_load(d, "features"), "snapshot_date")
    matrix = ps.persona_matrix(feats, int(pol["personas"]["min_txn_13w"]))
    prev_profile, prev_assign = _load(d, "prev_profile"), _load(d, "prev_assign")
    fit = ps.fit_personas(matrix, pol["personas"], ps.centroids_from_profile(prev_profile), manifest["prev_k"])
    prev_names = (dict(zip(prev_profile["persona_id"].astype(int), prev_profile["persona_name"]))
                  if prev_profile is not None and len(prev_profile) else None)
    persona_out = ps.outputs(matrix, fit, prev_assign, manifest["prev_run_date"], prev_names)
    stab = ps.stability(prev_assign, persona_out["customer_persona"])
    ari = ps.ari_hidden(persona_out["customer_persona"], _load(d, "archetype_truth"))
    timings["personas"] = round(time.time() - t, 1)

    t = time.time()
    embedder = embedder or load_embedder(stub)
    ref = {k: _load(d, k) for k in REF_INPUTS}
    cat_cfg = pol["categorise"]
    seed_pred = seed_holdout_predictions(ref["merchant_seed"], ref["category_taxonomy"], embedder, cat_cfg)
    index = SeedIndex.build(ref["merchant_seed"], ref["category_taxonomy"], embedder)
    mcc_pred = mcc_holdout_predictions(_load(d, "card_keys"), index, embedder, cat_cfg)
    timings["categoriser_holdouts"] = round(time.time() - t, 1)

    t = time.time()
    forecaster = forecaster or fc.load_forecaster(stub)
    series = fc.series_frame(_load(d, "cohort_weekly"))
    backtest, fkpi = fc.backtest(series, forecaster, pol["forecast"])
    fcast = fc.forecast(series, _dates(_load(d, "calendar"), "week_end"), forecaster, pol["forecast"])
    fcast["model_id"] = forecaster.model_id
    timings["forecast"] = round(time.time() - t, 1)

    t = time.time()
    cal = _load(d, "calendar")
    festival = set(pd.to_datetime(cal.loc[cal["festival_week"] == 1, "week_end"]).dt.date)
    prior = _load(d, "features_prior")
    signals = sg.evaluate(feats, pol["signals"], _dates(prior, "snapshot_date") if prior is not None else None,
                          festival)
    names = persona_out["customer_persona"].set_index("cif")["persona_name"]
    signals["persona_name"] = signals["cif"].map(names)
    timings["signals"] = round(time.time() - t, 1)

    for name, df in {**persona_out, "spend_forecast": fcast, "spend_backtest": backtest, "spend_signals": signals,
                     "seed_holdout": seed_pred, "mcc_holdout": mcc_pred}.items():
        _save(d, name, df)
    metrics = {
        "embedder_id": embedder.model_id, "forecaster_id": forecaster.model_id,
        "device": getattr(forecaster, "device", "cpu"), "model_versions": package_versions(*VERSION_PACKAGES),
        "n_customers_profiled": len(matrix), "k": fit.k, "silhouette": fit.silhouette,
        "silhouette_by_k": fit.silhouette_by_k, "persona_stability": stab, "ari_hidden": ari,
        "persona_names": {int(k): v for k, v in fit.names.items()}, "scaler_json": fit.scaler.to_json(),
        "category_accuracy_seed": accuracy(seed_pred), "category_accuracy_mcc": accuracy(mcc_pred),
        "n_seed_holdout": len(seed_pred), "n_mcc_holdout": len(mcc_pred),
        **{k: fkpi[k] for k in ("wape_model", "wape_naive", "wape_ratio", "coverage")},
        "n_cutoffs": fkpi["n_cutoffs"], "n_series": int(series["series_id"].nunique()),
        "n_signals": len(signals), "signals_by_rule": signals["signal_id"].value_counts().to_dict(),
        "stage_durations": timings,
    }
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    r3 = {k: None if v is None else round(v, 3) for k, v in {**metrics, **fkpi}.items()
          if isinstance(v, float) or v is None}
    logger.info("trained %s: k=%d silhouette %s stability %s ARI %s | accuracy seed %s mcc %s | WAPE %s vs naive %s "
                "(ratio %s), coverage %s | %d signals %s | %s", manifest["run_id"], fit.k, r3.get("silhouette"),
                r3.get("persona_stability"), r3.get("ari_hidden"), r3.get("category_accuracy_seed"),
                r3.get("category_accuracy_mcc"), r3.get("wape_model"), r3.get("wape_naive"), r3.get("wape_ratio"),
                r3.get("coverage"), len(signals), metrics["signals_by_rule"], timings)
    return metrics


# ---------------------------------------------------------------- stage 3

def evaluate_models(storage, run_date: date, write: bool = True, publish_on_fail: bool = False, tracker=None,
                    root: Path | str | None = None, registrar=None) -> dict:
    t0 = time.time()
    d = artifacts_dir(run_date, root)
    if not (d / "metrics.json").exists():
        raise FileNotFoundError(f"{d / 'metrics.json'} not found: run the training stage first")
    manifest, metrics = _read_json(d / "manifest.json"), _read_json(d / "metrics.json")
    pol = load_policy()
    passed, lines = kpi_gate.evaluate({**metrics, "feature_hash": manifest.get("feature_hash")}, pol["gate"],
                                      code_hash=feature_hash(), champion=manifest.get("champion_kpis"))
    publish = passed or publish_on_fail
    run_id = manifest["run_id"]
    durations = {**manifest["stage_durations"], **metrics["stage_durations"]}
    now = datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None)
    started = datetime.fromisoformat(manifest["started_at"]).replace(tzinfo=None)
    model_run = {
        "run_date": run_date, "run_id": run_id, "run_ts": now, "triggered_by": manifest["triggered_by"],
        **{k: metrics[k] for k in ("embedder_id", "forecaster_id", "model_versions", "device", "n_customers_profiled",
                                   "k", "silhouette", "persona_stability", "ari_hidden", "category_accuracy_seed",
                                   "category_accuracy_mcc", "n_seed_holdout", "n_mcc_holdout", "n_series",
                                   "wape_model", "wape_naive", "wape_ratio", "coverage", "n_signals", "scaler_json")},
        "snapshot_date": date.fromisoformat(manifest["snapshot_date"]),
        **{k: manifest[k] for k in ("features_snapshot_id", "cohort_snapshot_id", "map_snapshot_id", "map_keys",
                                    "uncategorised_share", "feature_hash")},
        "gate_passed": passed, "gate_detail": " | ".join(lines), "outputs_published": publish,
        "policy_json": json.dumps(pol, sort_keys=True), "stage_durations_json": json.dumps(durations),
        "mlflow_run_id": None, "duration_s": None,
    }
    for line in lines:
        logger.info("gate: %s", line)
    profile = _load(d, "persona_profile")
    names = dict(zip(profile["persona_id"].astype(int), profile["persona_name"]))
    if tracker is not None:
        model_run["mlflow_run_id"] = tracker({**model_run, "silhouette_by_k": metrics["silhouette_by_k"],
                                              "signals_by_rule": metrics["signals_by_rule"],
                                              **persona_model(model_run["scaler_json"], profile)})
    model_run["duration_s"] = round((now - started).total_seconds() + time.time() - t0, 1)

    out = {"spend_model_run": pd.DataFrame([model_run])}
    keys = RUN_OUTPUTS[:-1] if publish else ["spend_backtest"]
    for key in keys:
        df = _load(d, key)
        if df is None:
            continue
        df = df.assign(run_date=run_date, run_id=run_id)
        for col in ("snapshot_date", "week_end", "cutoff", "prev_run_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.date
        out[key] = df[[c for c, _ in OUTPUT_TABLES[key]]]
    if not publish:
        logger.warning("KPI gate failed: today's personas, forecast and signals are NOT published "
                       "(the previous run stays live)")
    if write:
        for key in sorted(out, key=lambda k: k == "spend_model_run"):
            storage.replace_run(key, out[key], run_date)
    registered = None
    if write and passed and registrar is not None:
        # a backfilled date is not what the endpoint serves (the latest run date is), so not a champion
        latest = _published(storage, date.max, same_day=True)
        if latest is not None and latest["run_date"] > run_date:
            logger.info("registry: %s is older than the published %s, not registered", run_date, latest["run_date"])
        else:
            registered = registrar(model_run, names)
    return {"model_run": model_run, "outputs": out, "gate_passed": passed, "gate_detail": lines,
            "outputs_published": publish, "registered": registered}


def persona_model(scaler_json: str, profile: pd.DataFrame) -> dict:
    """The tracker's persona_model / _example / _names: what the registry versions."""
    from spend.registry import persona_pipeline

    centroids = ps.centroids_from_profile(profile)
    example = np.stack([centroids[i] for i in sorted(centroids)[:3]])
    return {"persona_model": persona_pipeline(ps.Scaler.from_json(scaler_json), centroids),
            "persona_example": example,
            "persona_names": dict(zip(profile["persona_id"].astype(int), profile["persona_name"]))}


def run_daily(storage, run_date: date | None = None, stub: bool = False, triggered_by: str = "manual",
              write: bool = True, publish_on_fail: bool = False, tracker=None, root: Path | str | None = None,
              embedder=None, forecaster=None, registrar=None) -> dict:
    manifest = feature_engineering(storage, run_date, triggered_by, root)
    run_date = date.fromisoformat(manifest["run_date"])
    metrics = train_models(run_date, stub, root, embedder=embedder, forecaster=forecaster)
    result = evaluate_models(storage, run_date, write, publish_on_fail, tracker, root, registrar)
    return {"manifest": manifest, "metrics": metrics, **result}


def summary_lines(result: dict) -> list[str]:
    m = result["model_run"]

    def f(v, spec=".3f"):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)
    return [f"run {m['run_id']} ({m['run_date']}): {m['n_customers_profiled']} customers in {m['k']} personas "
            f"(silhouette {f(m['silhouette'])}, stability {f(m['persona_stability'])}, ARI vs hidden "
            f"{f(m['ari_hidden'])})",
            f"categoriser accuracy: seed holdout {f(m['category_accuracy_seed'])} (n={m['n_seed_holdout']}), "
            f"hidden MCC {f(m['category_accuracy_mcc'])} (n={m['n_mcc_holdout']})",
            f"forecast {m['n_series']} series: WAPE {f(m['wape_model'])} vs naive {f(m['wape_naive'])} "
            f"(ratio {f(m['wape_ratio'])}), P10-P90 coverage {f(m['coverage'])}",
            f"{m['n_signals']} signals; outputs published: {m['outputs_published']}"]
