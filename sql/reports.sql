-- Spend analytics: report queries for Hue (CDW Impala).
-- Tables written by Spark (CDE) or from CAI may need a metadata refresh first.
INVALIDATE METADATA rsingh_spend_anl_gold.spend_features;
REFRESH rsingh_spend_anl_gold.spend_cohort_weekly;
REFRESH rsingh_spend_anl_gold.spend_mix_weekly;
REFRESH rsingh_spend_anl_gold.spend_model_run;
REFRESH rsingh_spend_anl_ref.merchant_category_map;
REFRESH rsingh_spend_anl_ref.dq_results;

-- 1. Trust numbers and lineage for every daily run (a failed gate keeps the last published outputs live)
SELECT run_date, run_id, run_ts, triggered_by, device, snapshot_date, n_customers_profiled, k,
       ROUND(silhouette, 3) AS silhouette, ROUND(persona_stability, 3) AS stability,
       ROUND(category_accuracy_seed, 3) AS acc_seed, ROUND(category_accuracy_mcc, 3) AS acc_mcc,
       ROUND(wape_model, 3) AS wape_model, ROUND(wape_naive, 3) AS wape_naive, ROUND(wape_ratio, 3) AS wape_ratio,
       ROUND(coverage, 3) AS coverage, n_signals, gate_passed, outputs_published,
       features_snapshot_id, cohort_snapshot_id, map_snapshot_id, mlflow_run_id
FROM rsingh_spend_anl_gold.spend_model_run
ORDER BY run_date DESC, run_ts DESC;

-- 2. Book spend mix by category over the last 13 weeks, and its share
WITH latest AS (SELECT MAX(week_end) AS w FROM rsingh_spend_anl_gold.spend_cohort_weekly),
     cat AS (
  SELECT c.category, SUM(c.amount_inr) AS amount_inr, SUM(c.n_txn) AS n_txn
  FROM rsingh_spend_anl_gold.spend_cohort_weekly c, latest
  WHERE c.week_end > DATE_SUB(latest.w, 91)
  GROUP BY c.category)
SELECT category,
       ROUND(amount_inr / 1e7, 2)                            AS spend_cr,
       n_txn,
       ROUND(amount_inr / SUM(amount_inr) OVER (), 4)        AS spend_share
FROM cat
ORDER BY amount_inr DESC;

-- 3. Channel mix by week (UPI vs card vs cash): each dim partitions the spend, segment is one of them
SELECT week_end, channel,
       ROUND(SUM(amount_inr) / 1e5, 1)                                                   AS spend_lakh,
       ROUND(SUM(amount_inr) / SUM(SUM(amount_inr)) OVER (PARTITION BY week_end), 4)    AS channel_share
FROM rsingh_spend_anl_gold.spend_mix_weekly
WHERE dim = 'segment'
GROUP BY week_end, channel
ORDER BY week_end DESC, spend_lakh DESC
LIMIT 60;

-- 4. Where customers spend in the last 4 weeks: home vs away, and the top away cities
WITH latest AS (SELECT MAX(week_end) AS w FROM rsingh_spend_anl_silver.spend_event)
SELECT CASE WHEN e.is_home = 1 THEN 'home' ELSE e.event_city END AS spend_city,
       COUNT(DISTINCT e.cif)                                 AS customers,
       ROUND(SUM(e.amount) / 1e5, 1)                         AS spend_lakh
FROM rsingh_spend_anl_silver.spend_event e, latest
WHERE e.week_end > DATE_SUB(latest.w, 28)
GROUP BY 1
ORDER BY spend_lakh DESC
LIMIT 15;

-- 5. Today's personas (the latest published run)
WITH pub AS (SELECT MAX(run_date) AS d FROM rsingh_spend_anl_gold.spend_model_run WHERE outputs_published)
SELECT p.persona_id, p.persona_name, p.n_customers, ROUND(p.customer_share, 3) AS customer_share,
       ROUND(p.avg_spend_13w, 0) AS avg_spend_13w_inr, p.top_category, p.top_channel, p.dominant_tier,
       ROUND(p.digital_share, 3) AS digital_share, ROUND(p.home_share, 3) AS home_share
FROM rsingh_spend_anl_gold.persona_profile p, pub
WHERE p.run_date = pub.d
ORDER BY p.n_customers DESC;

-- 6. Persona migration since the previous run (the Sankey in the app)
WITH pub AS (SELECT MAX(run_date) AS d FROM rsingh_spend_anl_gold.spend_model_run WHERE outputs_published)
SELECT f.prev_run_date, f.from_persona_name, f.to_persona_name, f.n_customers
FROM rsingh_spend_anl_gold.persona_flow f, pub
WHERE f.run_date = pub.d AND f.from_persona_id <> f.to_persona_id
ORDER BY f.n_customers DESC
LIMIT 20;

-- 7. Next 12 weeks by category: the forecast band vs the seasonal-naive baseline
WITH pub AS (SELECT MAX(run_date) AS d FROM rsingh_spend_anl_gold.spend_model_run WHERE outputs_published)
SELECT s.category,
       ROUND(SUM(s.p10) / 1e7, 2)    AS p10_cr,
       ROUND(SUM(s.p50) / 1e7, 2)    AS p50_cr,
       ROUND(SUM(s.p90) / 1e7, 2)    AS p90_cr,
       ROUND(SUM(s.naive) / 1e7, 2)  AS naive_cr
FROM rsingh_spend_anl_gold.spend_forecast s, pub
WHERE s.run_date = pub.d
GROUP BY s.category
ORDER BY p50_cr DESC;

-- 8. Backtest accuracy by category: WAPE of P50 and of the naive baseline, P10-P90 coverage
WITH pub AS (SELECT MAX(run_date) AS d FROM rsingh_spend_anl_gold.spend_model_run WHERE outputs_published)
SELECT b.category,
       COUNT(*)                                                                    AS series_weeks,
       ROUND(SUM(ABS(b.actual - b.p50)) / SUM(b.actual), 3)                        AS wape_model,
       ROUND(SUM(ABS(b.actual - b.naive)) / SUM(b.actual), 3)                      AS wape_naive,
       ROUND(AVG(CASE WHEN b.actual BETWEEN b.p10 AND b.p90 THEN 1 ELSE 0 END), 3) AS coverage
FROM rsingh_spend_anl_gold.spend_backtest b, pub
WHERE b.run_date = pub.d
GROUP BY b.category
ORDER BY wape_model DESC;

-- 9. Opportunities: signals by rule and action, then the top 25 by metric
WITH pub AS (SELECT MAX(run_date) AS d FROM rsingh_spend_anl_gold.spend_model_run WHERE outputs_published)
SELECT s.signal_id, s.action, COUNT(*) AS customers, COUNT(DISTINCT s.home_city) AS cities
FROM rsingh_spend_anl_gold.spend_signals s, pub
WHERE s.run_date = pub.d
GROUP BY s.signal_id, s.action
ORDER BY customers DESC;

WITH pub AS (SELECT MAX(run_date) AS d FROM rsingh_spend_anl_gold.spend_model_run WHERE outputs_published)
SELECT s.signal_id, s.cif, s.segment, s.home_city, s.persona_name, s.metric,
       ROUND(s.metric_value, 3) AS metric_value, s.threshold, s.detail
FROM rsingh_spend_anl_gold.spend_signals s, pub
WHERE s.run_date = pub.d
ORDER BY s.signal_id, s.metric_value DESC
LIMIT 25;

-- 10. The categoriser: how merchant keys were mapped, and the newest keys
SELECT method, category, COUNT(*) AS merchant_keys, ROUND(AVG(similarity), 3) AS avg_similarity
FROM rsingh_spend_anl_ref.merchant_category_map
GROUP BY method, category
ORDER BY method, merchant_keys DESC;

SELECT first_run_date, merchant_key, category, method, ROUND(similarity, 3) AS similarity, nearest_seed
FROM rsingh_spend_anl_ref.merchant_category_map
ORDER BY first_run_date DESC, similarity
LIMIT 25;

-- 11. Data quality: every gate for the latest load, by layer
WITH latest AS (SELECT MAX(as_of) AS d FROM rsingh_spend_anl_ref.dq_results)
SELECT r.layer, r.severity,
       COUNT(*)                                            AS checks,
       SUM(CASE WHEN r.success THEN 0 ELSE 1 END)          AS failed,
       ROUND(AVG(CASE WHEN r.success THEN 1 ELSE 0 END), 3) AS pass_rate,
       MAX(r.run_ts)                                       AS last_run_ts
FROM rsingh_spend_anl_ref.dq_results r, latest
WHERE r.as_of = latest.d
GROUP BY r.layer, r.severity
ORDER BY r.layer, r.severity;

-- 12. Iceberg time travel: one snapshot per daily load. Take a snapshot_id from
-- spend_model_run.features_snapshot_id (query 1) to read exactly what a run read.
DESCRIBE HISTORY rsingh_spend_anl_gold.spend_features;

-- SELECT cif, spend_13w, cat_travel_share, away_share_4w
-- FROM rsingh_spend_anl_gold.spend_features FOR SYSTEM_VERSION AS OF <features_snapshot_id>
-- WHERE snapshot_date = DATE '2026-09-25'
-- ORDER BY spend_13w DESC LIMIT 20;

-- SELECT snapshot_date, COUNT(*) AS customers
-- FROM rsingh_spend_anl_gold.spend_features FOR SYSTEM_TIME AS OF '2026-09-19 23:59:59'
-- GROUP BY snapshot_date ORDER BY snapshot_date;
