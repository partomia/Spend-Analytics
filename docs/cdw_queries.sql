-- Spend Analytics: analytics queries for CDW Impala (Hue / Data Explorer).
-- Every query reads gold or ref only. "Latest run" means the latest run whose
-- outputs were published (it passed the KPI gate).

-- 1. Daily run scorecard: what ran, when, and whether it passed the gate.
SELECT run_date, run_id, triggered_by, n_customers_profiled, k,
       ROUND(silhouette, 3) AS silhouette, ROUND(ari_hidden, 3) AS ari_hidden,
       ROUND(category_accuracy_seed, 3) AS acc_seed, ROUND(category_accuracy_mcc, 3) AS acc_mcc,
       ROUND(wape_ratio, 3) AS wape_vs_naive, ROUND(coverage, 3) AS p10_p90_coverage,
       n_signals, gate_passed, outputs_published, ROUND(duration_s / 60, 1) AS minutes
FROM rsingh_spend_anl_gold.spend_model_run
ORDER BY run_date DESC, run_ts DESC;

-- 2. Channel mix, last 12 weeks: the shift between UPI, cards, net banking and cash.
--    dim = 'segment' counts each rupee once (every dim splits the same total).
SELECT week_end,
       ROUND(SUM(amount_inr) / 1e7, 2) AS spend_cr,
       ROUND(100 * SUM(CASE WHEN channel LIKE 'UPI%' THEN amount_inr END) / SUM(amount_inr), 1) AS upi_pct,
       ROUND(100 * SUM(CASE WHEN channel LIKE 'CARD%' THEN amount_inr END) / SUM(amount_inr), 1) AS card_pct,
       ROUND(100 * SUM(CASE WHEN channel = 'NETBANKING' THEN amount_inr END) / SUM(amount_inr), 1) AS netbanking_pct,
       ROUND(100 * SUM(CASE WHEN channel = 'ATM_CASH' THEN amount_inr END) / SUM(amount_inr), 1) AS cash_pct
FROM rsingh_spend_anl_gold.spend_mix_weekly
WHERE dim = 'segment'
  AND week_end > (SELECT DATE_SUB(MAX(week_end), 84) FROM rsingh_spend_anl_gold.spend_mix_weekly)
GROUP BY week_end
ORDER BY week_end;

-- 3. Digital adoption by city tier, last 13 weeks: how far tier 3 and 4 lag the metros.
SELECT dim_value AS city_tier,
       ROUND(SUM(amount_inr) / 1e7, 2) AS spend_cr,
       ROUND(100 * SUM(CASE WHEN channel IN ('CARD_ECOM', 'UPI_P2M', 'UPI_P2P', 'NETBANKING') THEN amount_inr END)
             / SUM(amount_inr), 1) AS digital_pct,
       ROUND(100 * SUM(CASE WHEN channel LIKE 'UPI%' THEN amount_inr END) / SUM(amount_inr), 1) AS upi_pct,
       ROUND(100 * SUM(CASE WHEN channel = 'ATM_CASH' THEN amount_inr END) / SUM(amount_inr), 1) AS cash_pct
FROM rsingh_spend_anl_gold.spend_mix_weekly
WHERE dim = 'city_tier'
  AND week_end > (SELECT DATE_SUB(MAX(week_end), 91) FROM rsingh_spend_anl_gold.spend_mix_weekly)
GROUP BY dim_value
ORDER BY city_tier;

-- 4. Wallet share by segment, last 13 weeks: which categories each segment spends on.
SELECT segment, category,
       ROUND(SUM(amount_inr) / 1e7, 2) AS spend_cr,
       ROUND(100 * SUM(amount_inr) / SUM(SUM(amount_inr)) OVER (PARTITION BY segment), 1) AS pct_of_segment,
       ROUND(SUM(amount_inr) / SUM(n_txn), 0) AS avg_ticket_inr
FROM rsingh_spend_anl_gold.spend_cohort_weekly
WHERE week_end > (SELECT DATE_SUB(MAX(week_end), 91) FROM rsingh_spend_anl_gold.spend_cohort_weekly)
GROUP BY segment, category
ORDER BY segment, spend_cr DESC;

-- 5. Festival uplift: average weekly spend in festival weeks vs other weeks, by category.
SELECT category,
       ROUND(AVG(CASE WHEN festival_week = 1 THEN wk END) / 1e5, 1) AS festival_week_lakh,
       ROUND(AVG(CASE WHEN festival_week = 0 THEN wk END) / 1e5, 1) AS normal_week_lakh,
       ROUND(100 * (AVG(CASE WHEN festival_week = 1 THEN wk END)
                    / AVG(CASE WHEN festival_week = 0 THEN wk END) - 1), 1) AS uplift_pct
FROM (SELECT category, week_end, MAX(festival_week) AS festival_week, SUM(amount_inr) AS wk
      FROM rsingh_spend_anl_gold.spend_cohort_weekly
      GROUP BY category, week_end) w
GROUP BY category
ORDER BY uplift_pct DESC;

-- 6. Salary-week effect: how much spend concentrates in the week salaries land.
SELECT category,
       ROUND(100 * (AVG(CASE WHEN salary_week = 1 THEN wk END)
                    / AVG(CASE WHEN salary_week = 0 AND festival_week = 0 THEN wk END) - 1), 1) AS salary_week_uplift_pct
FROM (SELECT category, week_end, MAX(salary_week) AS salary_week, MAX(festival_week) AS festival_week,
             SUM(amount_inr) AS wk
      FROM rsingh_spend_anl_gold.spend_cohort_weekly
      GROUP BY category, week_end) w
GROUP BY category
ORDER BY salary_week_uplift_pct DESC;

-- 7. Where customers travel, last 13 weeks: spend outside the home city.
SELECT dim_value AS away_city,
       ROUND(SUM(amount_inr) / 1e5, 1) AS spend_lakh,
       SUM(n_txn) AS txns,
       ROUND(SUM(amount_inr) / SUM(n_txn), 0) AS avg_ticket_inr
FROM rsingh_spend_anl_gold.spend_mix_weekly
WHERE dim = 'away_city'
  AND week_end > (SELECT DATE_SUB(MAX(week_end), 91) FROM rsingh_spend_anl_gold.spend_mix_weekly)
GROUP BY dim_value
ORDER BY spend_lakh DESC
LIMIT 15;

-- 8. Personas in the latest run: size, value and behaviour.
SELECT persona_id, persona_name, n_customers,
       ROUND(100 * customer_share, 1) AS pct_customers,
       ROUND(avg_spend_13w, 0) AS avg_spend_13w_inr,
       ROUND(avg_txn_13w, 1) AS avg_txn_13w,
       top_category, top_channel, dominant_tier,
       ROUND(100 * digital_share, 1) AS digital_pct, ROUND(100 * home_share, 1) AS home_pct
FROM rsingh_spend_anl_gold.persona_profile
WHERE run_date = (SELECT MAX(run_date) FROM rsingh_spend_anl_gold.spend_model_run WHERE outputs_published)
ORDER BY n_customers DESC;

-- 9. Personas by segment and tier: who is in each persona.
SELECT persona_name, segment, city_tier, COUNT(*) AS customers,
       ROUND(AVG(spend_13w), 0) AS avg_spend_13w_inr
FROM rsingh_spend_anl_gold.customer_persona
WHERE run_date = (SELECT MAX(run_date) FROM rsingh_spend_anl_gold.spend_model_run WHERE outputs_published)
GROUP BY persona_name, segment, city_tier
ORDER BY persona_name, customers DESC;

-- 10. Persona migration since the previous run: who changed behaviour.
SELECT prev_run_date, from_persona_name, to_persona_name, n_customers
FROM rsingh_spend_anl_gold.persona_flow
WHERE run_date = (SELECT MAX(run_date) FROM rsingh_spend_anl_gold.spend_model_run WHERE outputs_published)
  AND from_persona_id <> to_persona_id
ORDER BY n_customers DESC
LIMIT 20;

-- 11. Next-best-action signals in the latest run: campaign sizing by segment.
SELECT signal_id, action, segment, COUNT(DISTINCT cif) AS customers,
       ROUND(AVG(metric_value), 2) AS avg_metric
FROM rsingh_spend_anl_gold.spend_signals
WHERE run_date = (SELECT MAX(run_date) FROM rsingh_spend_anl_gold.spend_model_run WHERE outputs_published)
GROUP BY signal_id, action, segment
ORDER BY signal_id, customers DESC;

-- 12. Campaign list: the top 25 EMI-conversion targets by festive electronics spend.
SELECT cif, segment, home_city, persona_name, ROUND(metric_value, 0) AS electronics_4w_inr, detail
FROM rsingh_spend_anl_gold.spend_signals
WHERE run_date = (SELECT MAX(run_date) FROM rsingh_spend_anl_gold.spend_model_run WHERE outputs_published)
  AND signal_id = 'FESTIVE_ELECTRONICS'
ORDER BY metric_value DESC
LIMIT 25;

-- 13. 12-week forecast by category, book-wide: P10 / P50 / P90 vs the seasonal-naive baseline.
SELECT category,
       ROUND(SUM(p10) / 1e7, 2) AS p10_cr, ROUND(SUM(p50) / 1e7, 2) AS p50_cr, ROUND(SUM(p90) / 1e7, 2) AS p90_cr,
       ROUND(SUM(naive) / 1e7, 2) AS naive_cr,
       ROUND(100 * (SUM(p50) / SUM(naive) - 1), 1) AS p50_vs_naive_pct
FROM rsingh_spend_anl_gold.spend_forecast
WHERE run_date = (SELECT MAX(run_date) FROM rsingh_spend_anl_gold.spend_model_run WHERE outputs_published)
GROUP BY category
ORDER BY p50_cr DESC;

-- 14. Forecast trust: backtest WAPE of Chronos-2 vs the seasonal-naive baseline, by category
--     (4 cutoffs x 12 weeks). Lower is better; the gate uses the book-wide ratio.
SELECT category,
       ROUND(SUM(actual) / 1e7, 1) AS actual_cr,
       ROUND(SUM(ABS(actual - p50)) / SUM(actual), 3) AS wape_model,
       ROUND(SUM(ABS(actual - naive)) / SUM(actual), 3) AS wape_naive,
       CASE WHEN SUM(ABS(actual - p50)) < SUM(ABS(actual - naive)) THEN 'model' ELSE 'naive' END AS better,
       ROUND(100 * AVG(CASE WHEN actual BETWEEN p10 AND p90 THEN 1 ELSE 0 END), 1) AS coverage_pct
FROM rsingh_spend_anl_gold.spend_backtest
WHERE run_date = (SELECT MAX(run_date) FROM rsingh_spend_anl_gold.spend_model_run WHERE outputs_published)
GROUP BY category
ORDER BY actual_cr DESC;

-- 15. Customer 360 for one customer: persona, recent spend and any signals.
--     Replace the CIF with any from query 12.
SELECT p.cif, p.segment, p.age_band, p.home_city, p.persona_name, ROUND(p.spend_13w, 0) AS spend_13w_inr,
       p.txn_13w, p.top_category, p.top_channel, ROUND(100 * p.home_share, 1) AS home_pct,
       s.signal_id, s.action, s.detail
FROM rsingh_spend_anl_gold.customer_persona p
LEFT JOIN rsingh_spend_anl_gold.spend_signals s ON s.cif = p.cif AND s.run_date = p.run_date
WHERE p.run_date = (SELECT MAX(run_date) FROM rsingh_spend_anl_gold.spend_model_run WHERE outputs_published)
  AND p.cif = 'C98089416';

-- 16. How much of the spend has a measured location (vs the home-city fallback).
SELECT dim_value AS location_source,
       ROUND(100 * SUM(amount_inr) / SUM(SUM(amount_inr)) OVER (), 1) AS pct_of_spend,
       ROUND(100 * SUM(n_txn) / SUM(SUM(n_txn)) OVER (), 1) AS pct_of_txns
FROM rsingh_spend_anl_gold.spend_mix_weekly
WHERE dim = 'location_source'
GROUP BY dim_value
ORDER BY pct_of_spend DESC;

-- 17. Data quality: checks and failures per pipeline run and layer, newest first.
SELECT as_of, pipeline_run, layer, COUNT(*) AS checks,
       SUM(CASE WHEN success THEN 0 ELSE 1 END) AS failed,
       SUM(CASE WHEN NOT success AND severity = 'critical' THEN 1 ELSE 0 END) AS critical_failed,
       MIN(run_ts) AS checked_at
FROM rsingh_spend_anl_ref.dq_results
GROUP BY as_of, pipeline_run, layer
ORDER BY checked_at DESC
LIMIT 12;

-- 18. Merchant categorisation: how each merchant key got its category.
SELECT `method`, COUNT(*) AS merchant_keys,
       ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_keys,
       ROUND(AVG(similarity), 3) AS avg_similarity
FROM rsingh_spend_anl_ref.merchant_category_map
GROUP BY `method`
ORDER BY merchant_keys DESC;

-- 19. Time travel: the Iceberg snapshots of the feature table.
DESCRIBE HISTORY rsingh_spend_anl_gold.spend_features;

-- 20. Reproduce a run's input: the feature table exactly as the 2026-09-28 run read it
--     (features_snapshot_id from spend_model_run), compared with the table today.
SELECT 'as the run read it' AS version, COUNT(*) AS customers, ROUND(AVG(spend_13w), 0) AS avg_spend_13w_inr
FROM rsingh_spend_anl_gold.spend_features FOR SYSTEM_VERSION AS OF 7814666828235292593
WHERE snapshot_date = '2026-09-28'
UNION ALL
SELECT 'current', COUNT(*), ROUND(AVG(spend_13w), 0)
FROM rsingh_spend_anl_gold.spend_features
WHERE snapshot_date = '2026-09-28';
