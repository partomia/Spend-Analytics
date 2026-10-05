-- Reporting views for the Cloudera Data Visualization dashboard "Spend Analytics - Insights"
-- (docs/CDV_DASHBOARD.md, dataviz/build_dashboard.py). One flat view per dataset, so every
-- visual reads one table. Views only: nothing is copied, each visual reads the latest pipeline
-- output, and nothing in the pipeline reads them.
-- Impala has no CREATE OR REPLACE VIEW, so each is dropped and recreated. Re-runnable:
--   python scripts/run_impala_sql.py sql/dataviz_views.sql

CREATE DATABASE IF NOT EXISTS rsingh_spend_anl_report;

-- Dataset "Spend mix": weekly spend by one dimension and channel, last 26 complete weeks.
-- Each row belongs to one dim; every visual filters one dim (each dim sums to the book).
DROP VIEW IF EXISTS rsingh_spend_anl_report.v_spend_mix;
CREATE VIEW rsingh_spend_anl_report.v_spend_mix AS
SELECT w.week_end, w.dim,
       CASE WHEN w.dim = 'city_tier' THEN CONCAT('Tier ', w.dim_value) ELSE w.dim_value END AS dim_value,
       w.channel,
       CASE WHEN w.channel IN ('UPI_P2M', 'UPI_P2P', 'CARD_ECOM', 'NETBANKING') THEN 'Digital'
            ELSE 'Physical' END                                          AS channel_group,
       CASE WHEN w.week_end > DATE_SUB(m.last_week, 91) THEN 1 ELSE 0 END AS is_last_13w,
       w.amount_inr, w.amount_inr / 10000000                             AS amount_crore,
       w.n_txn, w.n_customers
FROM rsingh_spend_anl_gold.spend_mix_weekly w
CROSS JOIN (SELECT MAX(week_end) AS last_week FROM rsingh_spend_anl_gold.spend_mix_weekly) m
WHERE w.week_end > DATE_SUB(m.last_week, 182);

-- Dataset "Personas": every profiled customer in the latest run, with the persona of the
-- previous run (persona IDs are stable across runs, so its name comes from today's profile).
DROP VIEW IF EXISTS rsingh_spend_anl_report.v_personas;
CREATE VIEW rsingh_spend_anl_report.v_personas AS
SELECT c.run_date, c.run_id, c.cif, c.persona_id, c.persona_name, c.segment,
       CONCAT('Tier ', CAST(c.city_tier AS STRING))                       AS city_tier,
       c.home_city, c.age_band, c.top_category, c.top_channel,
       c.spend_13w, c.txn_13w, c.digital_share, c.home_share,
       COALESCE(p.persona_name, 'New this run')                           AS prev_persona_name,
       CASE WHEN c.prev_persona_id IS NULL THEN 'New'
            WHEN c.prev_persona_id = c.persona_id THEN 'Stayed' ELSE 'Moved' END AS persona_change
FROM rsingh_spend_anl_gold.customer_persona c
LEFT JOIN rsingh_spend_anl_gold.persona_profile p
       ON p.run_date = c.run_date AND p.persona_id = c.prev_persona_id
WHERE c.run_date = (SELECT MAX(run_date) FROM rsingh_spend_anl_gold.customer_persona);

-- Dataset "Persona runs": each persona's size and value in every run.
DROP VIEW IF EXISTS rsingh_spend_anl_report.v_persona_runs;
CREATE VIEW rsingh_spend_anl_report.v_persona_runs AS
SELECT p.run_date, CASE WHEN p.run_date = m.latest THEN 1 ELSE 0 END    AS is_latest,
       p.run_id, p.persona_id, p.persona_name, p.n_customers, p.customer_share,
       p.avg_spend_13w, p.avg_txn_13w, p.top_category, p.top_channel,
       CONCAT('Tier ', CAST(p.dominant_tier AS STRING))                   AS dominant_tier,
       p.digital_share, p.home_share
FROM rsingh_spend_anl_gold.persona_profile p
CROSS JOIN (SELECT MAX(run_date) AS latest FROM rsingh_spend_anl_gold.persona_profile) m;

-- Dataset "Signals": the next-best-action leads of every run, with the customer's city tier.
DROP VIEW IF EXISTS rsingh_spend_anl_report.v_signals;
CREATE VIEW rsingh_spend_anl_report.v_signals AS
SELECT s.run_date, CASE WHEN s.run_date = m.latest THEN 1 ELSE 0 END    AS is_latest,
       s.run_id, s.signal_id, s.action, s.cif, s.segment, s.home_city,
       CONCAT('Tier ', CAST(c.city_tier AS STRING))                       AS city_tier,
       c.age_band, s.persona_name, s.metric, s.metric_value, s.threshold, s.detail
FROM rsingh_spend_anl_gold.spend_signals s
CROSS JOIN (SELECT MAX(run_date) AS latest FROM rsingh_spend_anl_gold.spend_signals) m
JOIN rsingh_spend_anl_silver.customer c ON c.cif = s.cif;

-- Dataset "Forecast": the latest run's 12-week cohort x category forecast (P10 / P50 / P90)
-- against the seasonal-naive baseline.
DROP VIEW IF EXISTS rsingh_spend_anl_report.v_forecast;
CREATE VIEW rsingh_spend_anl_report.v_forecast AS
SELECT f.run_date, f.run_id, f.cohort, f.segment,
       CONCAT('Tier ', CAST(f.city_tier AS STRING))                       AS city_tier,
       f.category, f.week_end, f.horizon_week,
       f.p10 / 10000000 AS p10_crore, f.p50 / 10000000 AS p50_crore, f.p90 / 10000000 AS p90_crore,
       f.naive / 10000000 AS naive_crore
FROM rsingh_spend_anl_gold.spend_forecast f
WHERE f.run_date = (SELECT MAX(run_date) FROM rsingh_spend_anl_gold.spend_forecast);

-- Dataset "Model runs": the KPI scorecard of every published run, with the gate floors.
DROP VIEW IF EXISTS rsingh_spend_anl_report.v_model_run;
CREATE VIEW rsingh_spend_anl_report.v_model_run AS
SELECT r.run_date, CASE WHEN r.run_date = m.latest THEN 1 ELSE 0 END    AS is_latest,
       r.run_id, r.triggered_by,
       CASE WHEN r.gate_passed THEN 'PASS' ELSE 'FAIL' END                AS gate,
       r.category_accuracy_mcc AS accuracy_mcc, r.category_accuracy_seed AS accuracy_seed,
       r.silhouette, r.k, r.persona_stability, r.wape_model, r.wape_naive, r.wape_ratio, r.coverage,
       r.n_customers_profiled, r.n_signals, r.map_keys, r.uncategorised_share, r.feature_hash,
       r.duration_s, CAST(0.85 AS DOUBLE) AS min_accuracy, CAST(0.15 AS DOUBLE) AS min_silhouette,
       CAST(0.80 AS DOUBLE) AS min_stability, CAST(0.90 AS DOUBLE) AS max_wape_ratio
FROM rsingh_spend_anl_gold.spend_model_run r
CROSS JOIN (SELECT MAX(run_date) AS latest FROM rsingh_spend_anl_gold.spend_model_run
            WHERE outputs_published) m
WHERE r.outputs_published;

-- Dataset "Data quality": every Great Expectations result of the last 30 days, with the
-- latest run of each layer flagged.
DROP VIEW IF EXISTS rsingh_spend_anl_report.v_dq;
CREATE VIEW rsingh_spend_anl_report.v_dq AS
SELECT d.as_of, CASE WHEN d.run_id = l.run_id THEN 1 ELSE 0 END          AS is_latest,
       d.pipeline_run, d.run_id, d.run_ts, d.layer,
       CASE d.layer WHEN 'bronze' THEN 1 WHEN 'silver' THEN 2 ELSE 3 END  AS layer_order,
       d.table_name, d.check_name, d.expectation_type, d.column_name, d.severity,
       CASE WHEN d.success THEN 1 ELSE 0 END                              AS passed,
       CASE WHEN d.success THEN 0 ELSE 1 END                              AS failed,
       d.element_count, d.unexpected_count, d.unexpected_pct, d.table_snapshot_id
FROM rsingh_spend_anl_ref.dq_results d
JOIN (SELECT DISTINCT x.layer, x.run_id
      FROM rsingh_spend_anl_ref.dq_results x
      JOIN (SELECT layer, MAX(run_ts) AS last_ts FROM rsingh_spend_anl_ref.dq_results GROUP BY layer) t
        ON t.layer = x.layer AND t.last_ts = x.run_ts) l
  ON l.layer = d.layer
WHERE d.as_of > DATE_SUB(NOW(), 30);

-- Checks (the numbers the dashboard's tiles should show).
SELECT COUNT(*) AS rows_26w, SUM(is_last_13w) AS rows_13w,
       ROUND(SUM(CASE WHEN dim = 'category' AND is_last_13w = 1 THEN amount_crore END), 1) AS spend_13w_crore
FROM rsingh_spend_anl_report.v_spend_mix;
SELECT run_date, COUNT(*) AS customers, SUM(CASE WHEN persona_change = 'Moved' THEN 1 ELSE 0 END) AS moved
FROM rsingh_spend_anl_report.v_personas GROUP BY run_date;
SELECT signal_id, COUNT(DISTINCT cif) AS leads FROM rsingh_spend_anl_report.v_signals
WHERE is_latest = 1 GROUP BY signal_id ORDER BY leads DESC;
SELECT run_date, gate, accuracy_mcc, silhouette, wape_ratio, n_signals FROM rsingh_spend_anl_report.v_model_run
ORDER BY run_date DESC LIMIT 3;
SELECT layer, COUNT(*) AS checks, SUM(failed) AS failed FROM rsingh_spend_anl_report.v_dq
WHERE is_latest = 1 GROUP BY layer ORDER BY layer;
SELECT category, ROUND(SUM(p50_crore), 2) AS p50_12w_crore, ROUND(SUM(naive_crore), 2) AS naive_12w_crore
FROM rsingh_spend_anl_report.v_forecast GROUP BY category ORDER BY p50_12w_crore DESC LIMIT 5;
