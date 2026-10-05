# Spend Analytics dashboard in Cloudera Data Visualization (CDV)

A business view on the gold tables: no pipeline change, no copy, no extract.
One dashboard, **Spend Analytics - Insights**, six sheets (the deck's
Portfolio, Where, Personas, Opportunities, plus Forecast and Trust), 40
visuals on 7 datasets. Every dataset is a view on the latest pipeline output,
so the dashboard moves with the nightly DAG.

- CDV instance: `IndianBank-Spend-analytics` (CDW, environment
  `federal-cdp-env`), URL and API key in `.env` (`SPEND_CDV_URL`,
  `SPEND_CDV_API_KEY`).
- Shared workspace **`rsingh-spend-anl`**: the workload user manages it,
  Everyone can view. Open the instance > Visuals > `rsingh-spend-anl`.
- Connection `rsingh-spend-anl-impala`: Impala `federal-impala-1` over the
  public endpoint, as the workload user (LDAP). The instance is shared with
  other projects; the build creates or updates only the names above and the
  `Spend - *` datasets.
- Data is synthetic.

## Build, as code

```bash
set -a; source .env; set +a
python scripts/run_impala_sql.py sql/dataviz_views.sql  # report views (re-runnable)
python dataviz/build_dashboard.py --check   # every visual's query on Impala
python dataviz/build_dashboard.py           # connection, workspace (if missing), import, move
python dataviz/build_dashboard.py --verify  # every visual's query through CDV
```

- `sql/dataviz_views.sql` creates `rsingh_spend_anl_report` with one flat view
  per dataset: `v_spend_mix`, `v_personas`, `v_persona_runs`, `v_signals`,
  `v_forecast`, `v_model_run`, `v_dq`. Nothing in the pipeline reads them.
- `dataviz/build_dashboard.py` declares datasets, visuals and sheets, writes
  the CDV export file `dataviz/spend_dashboards.json` and imports it through
  the migration API. UUIDs are fixed, so a re-run updates the dashboard in
  place. The import lands in the importer's Private workspace; the script then
  moves the dashboard into `rsingh-spend-anl`.
- The connection caches results for 30 minutes: after changing a view, CDV
  shows the new values within 30 minutes.
- `tests/test_dataviz.py` checks the declaration offline: datasets are the
  report views, columns exist, spend-mix visuals filter one dimension, the
  grid has no overlap, the export has no credentials.

## What is on it (2026-10-03 run, measured through CDV on 2026-10-05)

| Sheet | Visuals | Headline |
|---|---|---|
| Portfolio | 3 KPIs; spend by category; weekly spend by channel; digital vs physical by city tier; segment by channel | INR 1,609.4 crore and 4.84 M transactions in the last 13 weeks; 68.4% digital |
| Where | top 15 states; top 10 away cities; weekly away spend; how location is known | Telangana leads (INR 94.8 crore) |
| Personas | 3 KPIs; size by segment; avg spend; digital share; size in every run; who moved | 193,682 customers profiled, 11,748 moved persona |
| Opportunities | 3 KPIs (one per signal); leads by action and segment, city, persona; leads per run; EMI campaign list (top 50) | TRAVEL_SURGE 7,373, FESTIVE_ELECTRONICS 3,845, RELOCATION 53 |
| Forecast | book-wide P10/P50/P90 vs seasonal naive; by category; by segment | ELECTRONICS P50 INR 307.85 crore vs naive 388.7 |
| Trust | accuracy, silhouette, WAPE, failed DQ checks; KPIs per run against the gate; DQ checks per layer and night; every run; failed checks | accuracy 0.934, silhouette 0.161, WAPE ratio 0.855, gate PASS; 127 DQ checks (76 bronze, 23 silver, 28 gold), 0 failed |

Rules the declaration keeps:

- Each `v_spend_mix` row belongs to one `dim` (category, city_tier, city,
  away_city, state, segment, age_band, location_source); every visual on it
  filters one `dim`, or spend is counted once per dimension. `away_city`
  covers spend outside the home city only.
- Latest-run visuals filter `is_latest = 1` (signals, runs, DQ); `v_personas`
  and `v_forecast` hold the latest run only.
- Gate floors on the Trust sheet come from `config/policy.yaml`: accuracy
  0.85, silhouette 0.15, persona stability 0.80, WAPE ratio 0.90.

## Talking points

- One gold layer serves the app, the model endpoint and this dashboard: one
  definition of every number, no extract.
- The Trust sheet shows the KPI gate and the data-quality gates as business
  charts: every published number came through both.
- The campaign list is what a branch or campaign team would export
  (Download as CSV).
- Before the demo: open the dashboard 5 minutes ahead (the Impala warehouse
  auto-suspends; the first query wakes it).
