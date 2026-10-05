# Spend Analytics dashboard in Cloudera Data Visualization (CDV)

A business view on the gold tables, built in the CDV UI on the existing data:
no pipeline change, no copy, no extract. Five SQL datasets, one dashboard with
five sheets (the deck's Portfolio, Where, Personas, Opportunities, plus Trust).
Every dataset reads the latest run, so the dashboard moves with the nightly DAG.
About 45 minutes to build.

- CDV instance: `IndianBank-Spend-analytics` (CDW, environment
  `federal-cdp-env`). Open it from CDW > Data Visualization.
- Data is synthetic. Ranger policies apply in CDV as in Hue: CDV queries
  Impala as the signed-in user.

## 1. Connection

Data > New connection:

- Type: CDW Impala; pick the virtual warehouse `federal-impala-1` (CDV fills
  in the host and uses your SSO login).
- Name: `spend-impala`. Test, then Connect.

## 2. Datasets

Data > `spend-impala` > New dataset > From SQL, one per block below. Name
each as in the heading. After creating it, open Fields: set the types listed
under the query (CDV guesses most of them), then Save.

### ds_spend_mix: weekly spend by dimension and channel

```sql
SELECT week_end, dim, dim_value, channel,
       CASE WHEN channel IN ('UPI_P2M', 'UPI_P2P', 'CARD_ECOM', 'NETBANKING') THEN 'Digital'
            ELSE 'Physical' END AS channel_group,
       amount_inr, n_txn, n_customers
FROM rsingh_spend_anl_gold.spend_mix_weekly
WHERE week_end > DATE_SUB((SELECT MAX(week_end) FROM rsingh_spend_anl_gold.spend_mix_weekly), 182)
```

- Dimensions: `week_end` (date), `dim`, `dim_value`, `channel`, `channel_group`.
- Measures: `amount_inr`, `n_txn` (sum).
- Each row belongs to one `dim` (category, city_tier, city, away_city, state,
  segment, age_band, location_source). **Every visual on this dataset needs a
  filter `dim = '<one value>'`**, or spend is counted once per dimension.
  `n_customers` is per row: don't sum it across channels or weeks.
- Last 26 weeks (about 14k rows). Each `dim` sums to the same book total
  (INR 3,297 crore on 2026-10-02), except `away_city` (spend outside home only).

### ds_personas: every profiled customer in the latest run

```sql
SELECT c.run_date, c.cif, c.persona_name, c.segment, CAST(c.city_tier AS STRING) AS city_tier,
       c.home_city, c.age_band, c.top_category, c.top_channel,
       c.spend_13w, c.txn_13w, c.digital_share, c.home_share,
       COALESCE(p.persona_name, 'New this run') AS prev_persona_name,
       CASE WHEN c.prev_persona_id IS NULL THEN 'New'
            WHEN c.prev_persona_id = c.persona_id THEN 'Stayed' ELSE 'Moved' END AS persona_change
FROM rsingh_spend_anl_gold.customer_persona c
LEFT JOIN rsingh_spend_anl_gold.persona_profile p
       ON p.run_date = c.run_date AND p.persona_id = c.prev_persona_id
WHERE c.run_date = (SELECT MAX(run_date) FROM rsingh_spend_anl_gold.customer_persona)
```

- Dimensions: `persona_name`, `prev_persona_name`, `persona_change`,
  `segment`, `city_tier`, `home_city`, `age_band`, `top_category`, `top_channel`.
- Measures: `cif` (count distinct), `spend_13w` (sum, avg), `digital_share`
  and `home_share` (avg).
- About 194k rows, one per customer. Persona IDs are stable across runs, so the
  previous persona's name comes from today's profile.

### ds_signals: the latest run's next-best-action list

```sql
SELECT s.run_date, s.signal_id, s.action, s.cif, s.segment, s.home_city,
       CAST(c.city_tier AS STRING) AS city_tier, c.age_band, s.persona_name,
       s.metric, s.metric_value, s.threshold, s.detail
FROM rsingh_spend_anl_gold.spend_signals s
JOIN rsingh_spend_anl_silver.customer c ON c.cif = s.cif
WHERE s.run_date = (SELECT MAX(run_date) FROM rsingh_spend_anl_gold.spend_signals)
```

- Dimensions: `signal_id`, `action`, `segment`, `home_city`, `city_tier`,
  `age_band`, `persona_name`, `cif`, `detail`.
- Measures: `cif` (count distinct), `metric_value` (avg, max).
- About 11k rows: TRAVEL_SURGE, FESTIVE_ELECTRONICS, RELOCATION.

### ds_model_runs: the KPI scorecard of every published run

```sql
SELECT run_date, run_id, triggered_by, gate_passed,
       ROUND(category_accuracy_mcc, 3) AS accuracy_mcc,
       ROUND(category_accuracy_seed, 3) AS accuracy_seed,
       ROUND(silhouette, 3) AS silhouette, k,
       ROUND(persona_stability, 3) AS persona_stability,
       ROUND(wape_ratio, 3) AS wape_ratio, ROUND(coverage, 3) AS coverage,
       n_customers_profiled, n_signals, uncategorised_share, feature_hash
FROM rsingh_spend_anl_gold.spend_model_run
WHERE outputs_published
```

- Dimensions: `run_date` (date), `run_id`, `triggered_by`, `gate_passed`, `feature_hash`.
- Measures: the KPIs (avg: one row per run date) and `n_signals`.
- Gate floors for reference lines (`config/policy.yaml`): accuracy 0.85,
  silhouette 0.15, persona stability 0.80, WAPE ratio 0.90, coverage 0.60 to 0.95.

### ds_dq: data-quality results per run and layer

```sql
SELECT as_of, pipeline_run, layer, table_name, check_name, severity,
       CASE WHEN success THEN 'Pass' ELSE 'Fail' END AS result,
       unexpected_pct, table_snapshot_id, run_ts
FROM rsingh_spend_anl_ref.dq_results
WHERE as_of > DATE_SUB(NOW(), 30)
```

- Dimensions: `as_of` (date), `pipeline_run`, `layer`, `table_name`,
  `check_name`, `severity`, `result`, `table_snapshot_id`.
- Measures: record count, `unexpected_pct` (max).
- 118 checks per nightly run (68 bronze, 23 silver, 27 gold).

## 3. Dashboard `Spend Analytics`

New dashboard, then one sheet per heading (Sheets > Add). Each visual: pick
the dataset, the type, then drag the fields onto the shelves listed. Visual
filters go on the Filters shelf.

### Sheet 1. Portfolio (`ds_spend_mix`)

| Visual | Type | Shelves | Filter |
|---|---|---|---|
| Spend by category, 26 weeks | Bars (horizontal) | Y `dim_value`, X sum `amount_inr`, sort descending | `dim = 'category'` |
| Channel mix by week | Stacked area | X `week_end`, Y sum `amount_inr`, colour `channel` | `dim = 'category'` |
| Digital vs physical by city tier | Stacked bars, 100% | X `dim_value`, Y sum `amount_inr`, colour `channel_group` | `dim = 'city_tier'` |
| Spend by segment | Pie / donut | Dimension `dim_value`, measure sum `amount_inr` | `dim = 'segment'` |

Add a dashboard filter on `channel` (applies to all four).

### Sheet 2. Where (`ds_spend_mix`)

| Visual | Type | Shelves | Filter |
|---|---|---|---|
| Spend by state | Map (India, state names) or bars | Geo `dim_value`, measure sum `amount_inr` | `dim = 'state'` |
| Top 10 away cities | Bars | Y `dim_value`, X sum `amount_inr`, top 10 | `dim = 'away_city'` |
| Away spend by week | Line | X `week_end`, Y sum `amount_inr` | `dim = 'away_city'` |
| How the location is known | Donut | `dim_value`, sum `n_txn` | `dim = 'location_source'` |

The map needs `dim_value` set to the State geo type in the dataset's Fields; if
the names do not match the built-in map, use bars.

### Sheet 3. Personas (`ds_personas`)

| Visual | Type | Shelves |
|---|---|---|
| Persona size and value | Combo: bars count distinct `cif`, line avg `spend_13w` | X `persona_name` |
| Persona by segment | Heatmap | X `persona_name`, Y `segment`, colour count distinct `cif` |
| Digital share by persona | Bars | X `persona_name`, Y avg `digital_share` |
| Who moved since the previous run | Table (or Sankey if enabled) | `prev_persona_name`, `persona_name`, count distinct `cif`; filter `persona_change = 'Moved'` |
| KPI tiles | KPI | count distinct `cif`; sum `spend_13w` |

Dashboard filters: `city_tier`, `segment`, `home_city`.

### Sheet 4. Opportunities (`ds_signals`)

| Visual | Type | Shelves |
|---|---|---|
| Leads by signal | KPI tiles, one per `signal_id` | count distinct `cif` |
| Leads by action and segment | Stacked bars | X `action`, Y count distinct `cif`, colour `segment` |
| Leads by city | Bars, top 15 | Y `home_city`, X count distinct `cif` |
| Campaign list | Table | `cif`, `signal_id`, `action`, `persona_name`, `home_city`, `metric_value`, `detail`; sort `metric_value` descending |

Dashboard filters: `signal_id`, `city_tier`, `persona_name`. The table is
the list a branch or campaign team would export (Download as CSV).

### Sheet 5. Trust (`ds_model_runs`, `ds_dq`)

| Visual | Dataset | Type | Shelves |
|---|---|---|---|
| Model KPIs over time | `ds_model_runs` | Lines | X `run_date`; Y `accuracy_mcc`, `silhouette`, `wape_ratio`, `persona_stability`; reference lines at the gate floors |
| Latest scorecard | `ds_model_runs` | Table | `run_date`, `run_id`, `gate_passed`, the KPIs, `n_signals`, `feature_hash`; sort `run_date` descending |
| Checks per night and layer | `ds_dq` | Stacked bars | X `as_of`, Y record count, colour `result`; trellis `layer` |
| Failed checks | `ds_dq` | Table | `as_of`, `layer`, `table_name`, `check_name`, `severity`, `unexpected_pct`; filter `result = 'Fail'` |

## 4. Talking points

- One gold layer serves the app, the model endpoint and these dashboards: one
  definition of every number, no extract.
- The Trust sheet shows the KPI gate and the data-quality gates as business
  charts: every published number came through both.
- Before the demo: run one query in Hue 5 minutes ahead (the Impala warehouse
  auto-suspends), then open the dashboard.
