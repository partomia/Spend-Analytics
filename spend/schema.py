"""Column definitions (Impala types) for the tables written from CAI: one source for
storage, tests and sql/reports.sql.

Gold outputs of the daily run are keyed by run_date: a rerun for the same day
replaces that day only (DELETE + INSERT), so yesterday's insights stay
queryable. ref.merchant_category_map is appended to by the categorise job: a
merchant key is categorised once, the first day it is seen.

No column is named after an Impala reserved word (rows, location, ...): counts
are n_customers, n_txn, n_rows.
"""

OUTPUT_TABLES = {
    "merchant_category_map": [
        ("merchant_key", "STRING"), ("category", "STRING"), ("method", "STRING"), ("similarity", "DOUBLE"),
        ("nearest_seed", "STRING"), ("model_id", "STRING"), ("first_run_date", "DATE"), ("run_id", "STRING"),
    ],
    "customer_persona": [
        ("run_date", "DATE"), ("run_id", "STRING"), ("snapshot_date", "DATE"), ("cif", "STRING"),
        ("persona_id", "INT"), ("persona_name", "STRING"), ("distance", "DOUBLE"), ("prev_persona_id", "INT"),
        ("top_category", "STRING"), ("top_channel", "STRING"), ("home_city", "STRING"), ("city_tier", "INT"),
        ("segment", "STRING"), ("age_band", "STRING"), ("spend_13w", "DOUBLE"), ("txn_13w", "INT"),
        ("digital_share", "DOUBLE"), ("home_share", "DOUBLE"),
    ],
    "persona_profile": [
        ("run_date", "DATE"), ("run_id", "STRING"), ("persona_id", "INT"), ("persona_name", "STRING"),
        ("n_customers", "INT"), ("customer_share", "DOUBLE"), ("avg_spend_13w", "DOUBLE"),
        ("avg_txn_13w", "DOUBLE"), ("top_category", "STRING"), ("top_channel", "STRING"),
        ("dominant_tier", "INT"), ("digital_share", "DOUBLE"), ("home_share", "DOUBLE"),
        ("prev_match_distance", "DOUBLE"), ("centroid_json", "STRING"),
    ],
    "persona_flow": [
        ("run_date", "DATE"), ("run_id", "STRING"), ("prev_run_date", "DATE"), ("from_persona_id", "INT"),
        ("from_persona_name", "STRING"), ("to_persona_id", "INT"), ("to_persona_name", "STRING"),
        ("n_customers", "INT"),
    ],
    "spend_forecast": [
        ("run_date", "DATE"), ("run_id", "STRING"), ("series_id", "STRING"), ("cohort", "STRING"),
        ("segment", "STRING"), ("city_tier", "INT"), ("category", "STRING"), ("week_end", "DATE"),
        ("horizon_week", "INT"), ("p10", "DOUBLE"), ("p50", "DOUBLE"), ("p90", "DOUBLE"), ("naive", "DOUBLE"),
        ("model_id", "STRING"),
    ],
    # week grain, so the app can draw each backtest window; WAPE / coverage aggregate these rows
    "spend_backtest": [
        ("run_date", "DATE"), ("run_id", "STRING"), ("cutoff", "DATE"), ("series_id", "STRING"),
        ("cohort", "STRING"), ("category", "STRING"), ("week_end", "DATE"), ("horizon_week", "INT"),
        ("actual", "DOUBLE"), ("p10", "DOUBLE"), ("p50", "DOUBLE"), ("p90", "DOUBLE"), ("naive", "DOUBLE"),
    ],
    "spend_signals": [
        ("run_date", "DATE"), ("run_id", "STRING"), ("snapshot_date", "DATE"), ("cif", "STRING"),
        ("signal_id", "STRING"), ("action", "STRING"), ("metric", "STRING"), ("metric_value", "DOUBLE"),
        ("threshold", "DOUBLE"), ("detail", "STRING"), ("segment", "STRING"), ("home_city", "STRING"),
        ("persona_name", "STRING"),
    ],
    "spend_model_run": [
        ("run_date", "DATE"), ("run_id", "STRING"), ("run_ts", "TIMESTAMP"), ("triggered_by", "STRING"),
        ("embedder_id", "STRING"), ("forecaster_id", "STRING"), ("model_versions", "STRING"), ("device", "STRING"),
        ("snapshot_date", "DATE"), ("features_snapshot_id", "STRING"), ("cohort_snapshot_id", "STRING"),
        ("map_snapshot_id", "STRING"), ("feature_hash", "STRING"),
        ("n_customers_profiled", "INT"), ("k", "INT"), ("silhouette", "DOUBLE"), ("persona_stability", "DOUBLE"),
        ("ari_hidden", "DOUBLE"),
        ("category_accuracy_seed", "DOUBLE"), ("category_accuracy_mcc", "DOUBLE"), ("n_seed_holdout", "INT"),
        ("n_mcc_holdout", "INT"), ("uncategorised_share", "DOUBLE"), ("map_keys", "INT"),
        ("n_series", "INT"), ("wape_model", "DOUBLE"), ("wape_naive", "DOUBLE"), ("wape_ratio", "DOUBLE"),
        ("coverage", "DOUBLE"), ("n_signals", "INT"),
        ("gate_passed", "BOOLEAN"), ("gate_detail", "STRING"), ("outputs_published", "BOOLEAN"),
        ("policy_json", "STRING"), ("scaler_json", "STRING"), ("stage_durations_json", "STRING"),
        ("mlflow_run_id", "STRING"), ("duration_s", "DOUBLE"),
    ],
}

PARTITION = {"merchant_category_map": None, **{k: "run_date" for k in OUTPUT_TABLES if k != "merchant_category_map"}}
# written by the daily run, in this order; spend_model_run last, so a published run is never partial
RUN_OUTPUTS = ["customer_persona", "persona_profile", "persona_flow", "spend_forecast", "spend_backtest",
               "spend_signals", "spend_model_run"]
# FALLBACK below min_similarity on the taxonomy too gives category OTHER
METHODS = ["MCC", "SEED", "EMBEDDING", "FALLBACK"]
