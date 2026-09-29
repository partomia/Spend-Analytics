# Project Log

Purpose: recover full context in a new session without replaying any prior
conversation. Chronological, most-recent-day-last. Times are IST (UTC+5:30)
unless marked UTC; the platform (CDE CLI, Impala) reports UTC. Cross-refs:
`PLAN.md` (names, access, decisions, phase checklist), `README.md`
(architecture and how to run), `docs/DEMO_RUNBOOK.md` (setup and demo script).

## 2026-09-29: Day 1

### Spec review

Reviewed the build spec against the four sibling repos (Mule, Collections, CASA,
Churn; all local, none modified) and the Cloudera AI MLOps workshop. Adopted as
`PLAN.md` decisions:

- The CAI daily run becomes three chained jobs, Feature Engineering -> Model
  Training -> Model Evaluation, handing off through `artifacts/<run_date>/`; the
  DAG grows to 10 tasks and 7 Airflow Variables (decision 1).
- CI gains a `train-validate` job that runs the real models on CPU with the KPI
  gate enforced: GitHub -> Train & Validate -> KPI Gate (decision 2).
- The spec's "14 category shares sum to 1" conflicts with cash, P2P and
  uncategorised spend existing; shares use categorised merchant spend and
  `uncategorised_share` is separate (decision 3).

### Phase 0: scaffold + spike

- venv (Python 3.11): sentence-transformers 6.1.0, chronos-forecasting 2.3.2,
  torch 2.14.0, transformers 5.17.0, pyspark 4.0.1, great_expectations 1.23.2,
  pandas 2.3.3 (pip first resolved pandas 3.0.6; every requirements file now pins
  `pandas>=2.2,<3` like the siblings), streamlit 1.64.0, scikit-learn 1.9.1.
  `HF_HOME=.hf_cache` (gitignored) holds both models.
- Spike (`scripts/spike_models.py`): MiniLM 2,250 texts/s CPU, 5,321 MPS;
  Chronos-2 224 series x 12 weeks 0.88 s CPU, 0.76 s MPS (decision 7).
- Copied from Churn and adapted: `spend/config.py` (`SPEND_` overrides),
  `scripts/run_cde_local.py` (adds a `categorise` stage and `--stub`, pins
  `TZ=UTC`), the `commit-msg` hook, `.gitignore` (+ `artifacts/`), Dockerfile
  (app only, from Mule), `cdsw-build.sh`, `.env.example`.
- New: `spend/features.py` (column contract), `spend/schema.py` (CAI outputs, no
  reserved words), `spend/storage.py` (parquet + Impala; Decimal -> float;
  5,000-row chunks for `customer_persona`).

### Phase 1: bronze generator

- One customer is one pure-Python simulation (`simulate_unit(idx, seed, until)`):
  profile, products and trips are drawn for the whole world first, then a day-by-day
  loop runs to the as-of date. Every draw for a day depends only on earlier days, so a
  later as-of adds rows and never changes one (tested for 7 units, including idx 150,000).
- 7 hidden archetypes set category mix, channel preferences, spend level, ATM / P2P
  habits and trips; ~1.5% relocate for good while the CRM keeps the old pincode.
  Seasonality: Diwali electronics / apparel, summer and year-end travel, school fees
  in April and June, salary-week spend.
- Tuning: first pass gave 187 events per customer (~37M at 200k, spec 25-30M);
  `PURCHASE_SCALE = 0.68` brings it to ~151.
- Fixes found while testing:
  - 395 UPI payments in the 20k book used a payer VPA the registry did not have yet
    (a second VPA or credit card linked up to 60-90 days after onboarding, so recent
    customers transacted with it before the as-of load listed it). The primary VPA and
    debit card now start on the onboarding date; the second is used only once linked.
    0 unresolvable events in 3,000 simulated customers.
  - VPAs could collide across customers (name + 4 random digits: ~1,500 expected
    collisions at 200k); the suffix is now a base-36 bijection of the unit index.
  - The vowel-drop name variant never fired (lower-case pattern on an upper-case
    name); 153 merchants had a single variant. Every merchant now has 2-5.
  - Card e-commerce at a state utility biller (no head office) raised a KeyError on
    `CITY[None]`; the merchant city falls back to where the customer is.
  - `functools.lru_cache` wrappers cannot be shipped to Spark workers when the job
    module is not importable there (cloudpickle pickles them by reference); replaced
    with a plain-dict `_memo`.
- Spark path: `createDataFrame(rdd)` of nested rows took 280 s at 20k; `mapInArrow`,
  one pass per table, takes 47-50 s (decision 9). Duplicates 0.20% of UPI rows.
- Demo customers are planted purchases (decision 10) and tested: unit 0 travel in the
  4 weeks to 2026-09-25 >= 3x its prior norm; unit 1 all located UPI after 2026-08-24
  in Nagpur (440xxx) with the CRM still in Hyderabad; unit 2 >= INR 50k electronics
  in the 4 weeks to 2025-10-24.
- `pytest -q`: 40 passed.

### Phase 2: DQ bronze + silver

- `build_silver.py`: dedupe (latest ingested wins), CIF resolution (card_hash ->
  card_master -> account; payer VPA -> vpa_registry -> account; NB and CRM carry the
  CIF), one `spend_event` stream with channel, location, `merchant_key` and the
  Friday `week_end`. 20k book: 19,416 customers, 22,987 accounts, 2,801,881 events in 15 s.
- The first silver run put 57% of events on `home_fallback`. Two generator causes:
  utility and EMI purchases were drawn both as random purchases and from the monthly
  bill schedule (billers were 40% of UPI P2M), and every customer paid both bills
  from this bank. Now bills and EMIs come from the schedule only, each customer pays
  utility / phone bills here with 80% (heavy users) or 40% probability, local and chain
  QR codes carry a pincode 95% / 85% of the time, and 72% of UPI P2M goes to local shops.
  Result 48.2%; the rest is structural (decision 4). ~146 events per customer.
- `dq_check.py` copied from Churn with only the constants and the check functions
  replaced: 68 bronze checks (non-empty, keys, dates, batch, duplicates, amounts, MCC /
  entry mode / txn type / biller type sets, 9 orphan checks incl. home pincode, raw
  16-digit card numbers in 8 text columns), 23 silver checks (keys, dedupe ratios,
  unresolved share, channels, tiers, fallback share, events per day, Friday week_end,
  merchant_key presence, reconciliation per source against bronze recomputed
  independently). Gold checks are written against the `spend/features.py` contract and
  verified in Phase 4.
- `--inject-bad-data` (3,000 customers, scratch warehouse): one critical failure,
  `upi_txn.payer_vpa in vpa_registry` at 1.999% orphans; exit 1.
- `pytest -q`: 49 passed (the Spark `merchant_key` equals `spend.categorise.normalise`
  on ~1,000 payee names and variants).

### Phase 3: Categorise

- `spend/categorise.py`: MCC (modal MCC of a card key) -> SEED (exact normalised
  name) -> EMBEDDING (5-NN vote over the seed names) -> FALLBACK (nearest category
  description, else OTHER). `categorise_new` returns map rows with `model_id`,
  `first_run_date` and `run_id`. The map is append-only, so a key keeps its first
  category. The seed holdout (20% of variant names, by hash, never a canonical
  name) and the hidden-MCC holdout (5,000 card keys) are the functions the Phase 5
  evaluate stage gates on.
- First measurement, MiniLM alone: 0.819 on the seed holdout, a fail. A grid over k
  (1, 3, 5, 10) and a category-description prior (0 to 2) topped out at 0.82 on the
  seed holdout and 0.85 on card keys the seed index misses. Most errors are
  spelling-mangled payee names and shared owner names. A TF-IDF over character
  2-4 grams scored 0.93-0.94 on the same splits. Blending the two 50/50 with the
  prior gives 0.926 / 0.934 (decision 11), and the character half is the stub.
- `cai/jobs/categorise.py` (CAI job spend-categorise), in the Churn job shape:
  SPEND_RUN_DATE / SPEND_TRIGGERED_BY, `parse_known_args`, and exit only on
  failure. It reads new keys with `new_merchant_keys()` (Impala LEFT ANTI JOIN
  against the map, or pandas on parquet), appends them, and exits 1 if any key is
  still unmapped.
- Local stage on the 20k book: 21,466 keys in 29 s; 0 of 1,982,830 merchant events
  unmapped; a second run is a no-op. OTHER holds 6.2% of merchant spend.
- `pytest -q`: 60 passed.

### Phase 4: Gold + DQ gold

- `build_gold_features.py` builds a dense CIF x week panel (from the first extract
  week to the as-of week) with 13- and 4-week window sums. It then keeps the
  Friday snapshots from 2025-07-04 plus the as-of date; a mid-week as-of date gets
  a partial last week. Customers are included from their onboarding date. Events
  join the merchant map (broadcast); a merchant event whose key is not in the map
  is UNCATEGORISED. Cohort series are dense (16 cohorts x 14 categories x
  complete weeks, 0 when a week has no spend) and carry the calendar covariates.
  The mix table explodes each event into its 8 dimensions. Values are rounded, so
  a rebuild of the same inputs is bit-identical.
- 20k book: 40-50 s per load. spend_features 1,208,205 rows, cohort 17,248, mix
  37,669. `dq_gold`: 27 checks, 0 failed, including shares summing to 1,
  uncategorised 0.0%, at least 52 weeks for all 224 series, the week vs the same
  week last year within 30%, and the mix reconciling to the cohort series in all 77
  weeks.
- Rerun: 0 rows differ between the two Iceberg snapshots, but the copy-on-write
  MERGE rewrote every file (decision 13).
- First cut of `away_share_4w` counted fallback events as home, which put the
  relocated demo customer at 0.67. Now it is measured on located spend only, plus
  `away_weeks_4w` to tell a move from a trip (decision 12). Adding the column
  meant dropping the local gold tables: MERGE `INSERT *` does not evolve the
  schema, so a column change on the cluster also needs a drop and rebuild before
  the first live load.
- Tests: the gold constants match `spend/features.py`, snapshot dates, and exact
  features for one customer with 6 events (13-week window edge, the four share
  families, uncategorised, salary weeks, trend, located away share and away
  weeks). Also the dense cohort grid over complete weeks only, and the mix
  category totals. `pytest -q`: 64 passed.

### Phase 5: Core ML

- Three stages in `spend/pipeline.py`, handing off through `artifacts/<run_date>/`:
  - `feature_engineering` pins the snapshots of features, cohort and map, reads
    the run's snapshot, the snapshot 4 weeks earlier, cohort weeks up to the
    run date, the calendar, the categoriser inputs and the previous published
    run. It writes `manifest.json` and the inputs.
  - `train_models` fits the personas, runs the categoriser holdouts, the
    backtest and forecast, and the signals, writing outputs and `metrics.json`.
  - `evaluate_models` applies the gate, builds `spend_model_run`, calls the
    tracker hook and writes the tables; `spend_model_run` goes last.
  - A failed gate writes only `spend_model_run` and `spend_backtest`, as
    evidence.
- Measured before building:
  - Chronos-2 vs seasonal naive on the 20k cohort series: 0.151 vs 0.195 WAPE,
    coverage 0.765, about 2-3 s per cut-off on CPU.
  - A single-week trend wrecks the earliest naive cut-off (0.28), which
    became decision 15.
  - KMeans silhouette topped out at 0.149. Clipping at +-3 sd gives 0.166
    with better ARI (decision 14). Stability with warm start or Hungarian
    matching is about 0.96 either way; Hungarian is kept, as the spec asks.
- Signals: the TRAVEL_SURGE norm is the prior 9 weeks (decision 16). RELOCATION
  needs `away_weeks_4w >= 3` (new `min_away_weeks` in the policy), a prior
  away share <= 0.3 and 4 or more transactions. All three planted customers
  fire: unit 0 TRAVEL_SURGE at 88x, unit 1 RELOCATION 100% for 4 of 4 weeks,
  unit 2 FESTIVE_ELECTRONICS INR 85k on 2025-10-24.
- Bug found by the first real two-date run: the 2026-09-18 run saw 0 map keys,
  because the one local categorise run stamped every key 2026-09-25. The
  hidden-MCC holdout was empty and the gate (rightly) failed an unmeasured KPI.
  The holdout now uses the whole pinned map.
- Real models, `data/parquet`, 2026-09-18 then 2026-09-25: both published. On
  2026-09-25 every gate check passes (stability 0.958 vs 2026-09-18), in 35 s.
  Stub run: 13 s, the gate fails on WAPE ratio 1.000 and coverage 0.500 by
  construction.
- Tests: `test_models.py` (scaler, fit, stable IDs across relabelled runs, new
  IDs for extra clusters, keep-k, naive trend and its guard, backtest cut-offs,
  KPIs, each signal rule and its negatives, each gate check failing) and
  `test_pipeline.py` (failed gate keeps only the evidence, two published runs
  chain IDs / stability / flow, reruns replace the day, stages need their
  inputs, run date must be a snapshot, Impala reads pinned with
  FOR SYSTEM_VERSION AS OF). `pytest -q`: 87 passed.

### Phase 6: CAI jobs + endpoint

- Jobs, each a thin wrapper over one `spend.pipeline` stage, in the Churn
  `daily_score.py` shape (`SPEND_RUN_DATE` / `SPEND_TRIGGERED_BY`,
  `parse_known_args`, exit only on failure):
  - `feature_engineering.py`
  - `train_models.py`
  - `evaluate_models.py`, which exits 1 on a failed gate unless
    `--ignore-gate`, and takes `--publish-on-fail` and `--dry-run`.
  - `daily_run.py`, all three stages in-process.
  - `backfill_history.py`, which loads the models once for the last N
    Fridays.

  With no run date, training and evaluation take the folder that feature
  engineering prepared last. The MLflow tracker lives in `spend/tracking.py`,
  shared by evaluation and `daily_run`.
- Real models, `data/parquet`, 2026-09-25, the three jobs one after another:
  - Timing: 1.7 s, 44 s and 0.8 s.
  - KPIs identical to Phase 5, and every gate check passes: 0.926 / 0.934,
    WAPE ratio 0.775, coverage 0.765, silhouette 0.166, stability 0.958.
  - Exit code 0.
- On a scratch copy of the store:
  - The stub run exits 1 (WAPE ratio 1.000, coverage 0.500), and 0 with
    `--ignore-gate`.
  - A real `--dry-run` exits 0 and writes nothing.
  - `backfill_history.py --weeks 3` runs 2026-09-11/18/25 in 90 s. All three
    publish, with stability 0.955 and 0.958 against the previous Friday.
- Endpoint (`spend/scoring.py`, `cai/model/predict.py`): four actions,
  validated like Collections' `validate()`. Unexpected errors come back as
  `{"error": ...}`, never a stack trace. The context is the latest *published*
  run by run date, with its scaler, centroids and pinned snapshot IDs, plus
  MiniLM, Chronos-2 and the seed index, loaded once per replica. Timing on the
  laptop CPU:
  - Context: 17 s.
  - Unit 0 profile: 1.6-2.3 s. He is a Tier-1 traveller with a TRAVEL_SURGE
    signal and INR 49k in Panaji.
  - Forecast: 1-1.9 s.
  - What-if: 0.1-0.2 s. TRAVEL x0 moves him to Tier-1 digital foodie.
  - Categorise: 0.02 s for 6 texts.
- Per-customer forecasts: the weekly P50 of an intermittent series is near 0,
  so the endpoint also returns a quantile-average mean and its 12-week sum
  (decision 17). Adding the 19 extra quantile levels left the cohort backtest
  unchanged.
- The endpoint's nearest-centroid persona agrees with the daily label for 98.6%
  of profiled customers (decision 18). What-if shifts that move people, sampled
  on 3,000 customers:
  - TRAVEL x2 moves 67, mostly into Tier-1 traveller.
  - ELECTRONICS x3 moves 235 into Tier-1 digital gadget buyer.
  - ATM_CASH x0.2 moves 228, mostly out of the cash-first personas.
- Categoriser limitation seen live: "PHONEPE*RAMESHKIRANA" comes out as DINING,
  because the run-together name's nearest seed is "phonepe rameshcafe", while
  "RAMESH KIRANA STORE" comes out as GROCERY. The demo request uses texts that
  were checked with the real model.
- Bug found by the scoring tests: a constant feature's standard deviation is
  about 1e-16, not 0. It escaped the zero-variance guard and was then rounded to
  0 in `scaler_json`, which gave NaN distances. Any scale at or below 1e-6 now
  becomes 1, both when the scaler is fitted and when it is loaded. Real runs
  were unaffected, since no persona feature is constant on the book.
- Tests:
  - `test_scoring.py`: validation messages, context from the published run,
    profile, dense customer series, forecast and totals, what-if mix
    consistency, categorise, never-raise, and the Impala read-back (by run
    date; IDs as strings, Decimal as float).
  - `test_cai_jobs.py`: MLflow tracker, three jobs chained through artifacts
    with env-driven run date and Jupyter `-f` args, default run date, exit
    codes, backfill Fridays, `predict.py`.
  - `pytest -q`: 114 passed.

### Phase 7: Spend Insights Console

- `app/streamlit_app.py` (10 tabs), `app/data.py` (cached gold reads, empty frames
  on a cold start, endpoint requests cached by their JSON), and `app/run.py`
  (the CAI Application launcher, copied from Churn).
- Sidebar: the run date (every run, including gate failures), the gate status
  and published flag, models and device, freshness as the last DQ run per layer,
  and whether Customer 360 goes to the endpoint or runs in-process.
- The tabs read gold and never scan silver:
  - Portfolio, Where and Channels read `spend_mix_weekly`. Where puts a
    caption on the home_fallback share.
  - Personas and Channels read the persona centroids and `persona_flow`
    (Sankey).
  - Forecast reads `spend_forecast`, `spend_backtest` and the cohort series.
  - Opportunities reads `spend_signals`, with filters, reach and a CSV
    download.
  - Data quality reads `dq_results`.
  - Lineage reads `spend_model_run`, with Hue queries for the snapshot and
    time travel.
- Customer 360 and the Categoriser call the scorer: profile, forecast with a fan
  chart and expected 12-week totals, what-if, and categorise.
- `data/parquet`, 20k book, run 2026-09-25:
  - All ten tabs render in 2.9 s with the stubs. With MiniLM and Chronos-2
    in-process, the first render takes 22 s (model load) and is cached after.
  - Portfolio shows INR 162.9 cr over 13 weeks, 488,446 transactions and 68.5%
    digital.
  - Unit 0's expected next 12 weeks: TRAVEL INR 19.8k, DINING INR 4.7k.
  - `python app/run.py` serves, and `/_stcore/health` returns ok.
- Tests (`test_app.py`, AppTest):
  - Every tab renders, including the sidebar DQ state, the failed-check list
    and the Sankey.
  - Customer 360 profile, forecast and what-if; an unknown CIF.
  - The live categoriser.
  - A failed-gate run says so and shows no personas; switching back to the
    earlier run clears the warning.
  - Cold start with no tables at all.
  - Cold start with only `spend_model_run`, where every tab shows its
    "not yet" message.
  - `pytest -q`: 121 passed.

### Phase 8: Orchestration + CI + docs

- `cde/dags/spend_dag.py`, adapted from Churn's DAG:
  - 10 tasks: `generate >> dq_bronze >> silver >> dq_silver >> cai_categorise
    >> gold >> dq_gold >> cai_feature_engineering >> cai_model_training >>
    cai_model_evaluation`.
  - The four `cai_*` tasks share one `trigger_cai_job(as_of, job_var)`. It
    posts `{"SPEND_RUN_DATE": as_of, "SPEND_TRIGGERED_BY": "airflow"}` as the
    run's environment and polls every 30 s, with a 90-minute deadline and
    retries=0.
  - Seven Airflow Variables. Without `SPEND_CAI_HOST` the CAI steps are
    skipped.
  - Daily at 21:00 UTC, `start_date` 27 Sep 21:00, created unpaused like the
    siblings.
- `cde/scripts/`:
  - `deploy_jobs.sh` and `deploy_dag.sh` are Churn's, with Spend's names.
  - `backfill_drill.sh` also starts `spend-categorise` over the API v2
    between silver and gold for each date, using stdlib Python and the four
    values from the environment. `SKIP_CATEGORISE=1` opts out (decision 20).
- CI (`ci.yml`):
  - `test` runs pytest, then generate / dq_bronze / silver / dq_silver at
    3,000 customers, stub categorise, gold / dq_gold / export, the stub daily
    run with `--ignore-gate`, and the DQ gate failing on cue. It uploads the
    parquet book.
  - `train-validate` (needs `test`) installs CPU torch, caches
    `~/.cache/huggingface`, downloads the book and runs the real daily run
    with the gate enforced.
  - The same sequence in scratch folders on the laptop took 174 s end to end.
    Stub run: WAPE ratio 1.000 and coverage 0.384 (both fail, as expected).
    Real run: 2,914 customers in 8 personas, accuracy 0.926 / 0.935, WAPE
    0.271 vs 0.390 naive (ratio 0.696), coverage 0.771, silhouette 0.166,
    214 signals. Gate PASS, exit 0.
- `sql/reports.sql`: 12 reports.
  - Covered: run lineage and KPIs, category mix, channel mix by week, home vs
    away, personas, persona moves, forecast by category, backtest WAPE and
    coverage, signals (summary and top 25), the categoriser's methods and
    newest keys, DQ by layer, and time travel.
  - All 13 SELECTs ran in local Spark SQL over the CI book.
  - Two fixes from that run: `is_home` is an int in silver (`= 1`), and an
    empty first-run `persona_flow` export had no types. The second is a
    scratch artifact; Impala has the DDL.
- Tests:
  - `test_orchestration.py` covers prefixes, deployed jobs pointing at real
    files, the DAG and drill running only deployed jobs, the 10-task chain,
    the CAI task to Variable pairs, `--db-prefix` repeated on every override,
    every triggered job reading `SPEND_RUN_DATE`, the schedule and
    registration names, and the CI shape (train-validate needs test, no
    `--stub` or `--ignore-gate`).
  - `test_reports_sql.py` checks every identifier in every statement: it
    must be a keyword, an alias, or a column of a table the statement reads
    (from `schema.py`, `features.py`, silver's `EVENT_COLUMNS` and dq_check's
    `RESULT_SCHEMA`). A misspelled column fails it.
- Demo texts checked with the real categoriser:
  - SWIGGY*BLR 5521 is FOOD_DELIVERY (0.81) and "ola cabs pvt" is TRAVEL
    (SEED).
  - SHARMA MEDICOS is HEALTH (0.79).
  - TANISHQ JEWELLERS falls back to OTHER, a brand with no type word.
- README and `docs/DEMO_RUNBOOK.md`. Cluster numbers and IDs are left blank
  for Phase 9.
- `pytest -q`: 132 passed.

## Phase 8b: CAI MLOps hardening (2026-09-29)

Ideas taken from the CXR Triage CAI build guide and adapted (PLAN decisions 21-23).

- `tests/test_cai_contract.py`: every script under `cai/` plus `app/run.py`
  follows the CAI job runtime rules. No module-level `__file__` (and an
  `except NameError` fallback), `parse_known_args`, and no `sys.exit` on
  success. A mutation to `sys.exit(main())` makes it fail.
- Feature hash (`spend.features.feature_hash()`, `aa21012976534ed4` at logic
  version 1.0.0):
  - It is written to the manifest, `spend_model_run.feature_hash` and MLflow.
  - Training refuses artifacts that carry another hash.
  - The gate checks it against the code.
  - The endpoint and app refuse a run with a different hash.
- Non-regression against the champion, meaning the published run the candidate
  would replace:
  - 8-Friday backfill with real models on a scratch copy of the 20k book
    (2026-08-07 to 2026-09-25): 235 s, every gate PASS.
  - Accuracy was 0.926 / 0.934 every week. The WAPE ratio was 0.799, 0.801,
    0.783, 0.788, 0.786, 0.791, 0.765, 0.775, so the largest weekly rise was
    +0.010.
  - Silhouette ranged 0.166-0.170 and stability 0.945-0.961.
  - A same-day rerun of 2026-09-25 as a dry run reproduced a WAPE ratio of
    0.775 against the champion's 0.775.
  - Tolerances set from these numbers: accuracy may drop at most 0.01 and the
    WAPE ratio may rise at most 0.03.
- GitHub -> CAI:
  - New `cai/jobs/sync_code.py` (job `spend-sync-code`) and `ci/cai_jobs.py`.
  - `ci/trigger_cai_pipeline.py` uses the standard library only.
  - `cai-pipeline` in `ci.yml` needs train-validate, runs on push only, is in
    concurrency group `spend-analytics-cai`, and is a no-op until `CAI_URL` is
    set.
  - Evaluation reads `SPEND_DRY_RUN` from the environment.
  - Runbook step 10 covers the three secrets.
  - Tested against a fake API v2 (order, per-job environment, gate rejection
    named, a failed job stops the chain, timeout, missing jobs) and a
    throwaway git repo (reset to the pushed commit, untracked `artifacts/`
    kept, a moved branch stops the chain).
- `pytest -q`: 178 passed. Not yet run against the live workbench (Phase 9).

## Phase 9: live on Cloudera (in progress, 2026-09-29)

- CDE (`deploy_jobs.sh`, 6 min): repository `rsingh-spend-anl-pipeline`, python-env
  `rsingh-spend-anl-python-env`, and the four Spark jobs, nothing else under
  `spend-anl`. Nothing of the sibling projects was touched.
- generate-bronze failed twice: run 2375 (200k) and run 2376 (20k).
  - Both wrote the six ref tables and then the driver shut down, with no
    traceback in `cde run logs` (it truncates on this vcluster, as Collections
    recorded).
  - Reproduced on local PySpark 3.5.4 (the cluster's runtime): `mapInArrow`
    checks for pyarrow on the driver and raises ImportError, and the env had
    only Great Expectations.
  - Fixed by pinning `pyarrow==25.0.1` in `cde/resources/requirements.txt`,
    with a test. The env rebuild took about 7.5 min.
- Chain for as-of 2026-09-25 at 200k customers, all verified in Impala rather
  than from logs:

  | Run | Job | Result | Wall |
  |---|---|---|---|
  | 2377 | generate-bronze | succeeded | ~4 min |
  | 2378 | dq-check bronze | succeeded, 68 checks, 0 failed | 131 s |
  | 2379 | build-silver | succeeded | 91 s |
  | 2380 | dq-check silver | succeeded, 23 checks, 0 failed | 134 s |

  - Bronze: crm_customer 194,490; cbs_accounts 228,835; card_master 218,463;
    vpa_registry 261,593.
  - Bronze transactions: upi_txn 16,138,066, card_txn 5,813,837, atm_txn
    4,264,583, nb_txn 1,895,604 (28.1M).
  - Silver: customer 194,076, account 228,835, spend_event 28,056,197.
  - Location sources: home_fallback 48.3%, upi_pincode 25.1%, atm 15.2%,
    card_city 11.4%.
  - 21,551 merchant keys await categorisation.
- Impala from the laptop: `.env` (gitignored) holds `SPEND_IMPALA_USER` /
  `SPEND_IMPALA_PASSWORD`, the same workload credentials the sibling projects
  use against the same warehouse.
- Categorise was the first CDW write, run from the laptop against Impala
  because there's no CAI API key yet (decision 24).
  - The first attempt failed on `CREATE TABLE` because `method` is reserved
    since Impala 3.0, and nothing was written.
  - `ImpalaStorage` now backtick-quotes every identifier it generates.
    `reports.sql` quotes `method`, and a test checks our columns against
    Impala's 409 reserved words.
  - Retry: 21,551 keys in 561 s (EMBEDDING 11,816, SEED 6,796, MCC 2,937,
    FALLBACK 2). Every silver key is mapped, with no duplicates.
- build-gold-features, run 2381: succeeded in 466 s.
  - spend_features has 12,075,228 rows over 65 snapshots, with 194,076
    customers on 2026-09-25.
  - Cohort series: 224 x 77 weeks. spend_mix_weekly: 39,717 rows.
    Uncategorised share 0.0%.
- dq-check gold, run 2387: 27 checks, 0 failed. It waited about 9 min in the
  shared queue, and `cde job run --wait` returned while the run was still
  "starting", so poll `cde run describe` instead.
- Daily run on the 200k book, from the laptop against Impala:
  - Dry run took 10 min; feature engineering's Impala reads were about 9 min
    of that.
  - Results: 193,370 customers profiled in 8 personas, silhouette 0.163, ARI
    vs the hidden archetypes 0.649.
  - Categoriser accuracy 0.925 (seed holdout) / 0.934 (hidden MCC, n=2,937).
  - Forecast: WAPE 0.100 vs 0.114 naive (ratio 0.872), P10-P90 coverage
    0.779, 14,908 signals. Every gate check PASS.
  - The WAPE margin to the 0.90 limit is 0.028, against 0.125 at 20k: larger
    cohorts are smoother, so seasonal naive does better.
- Publish (evaluation from the same artifacts): 18 min in total.
  - Rows: customer_persona 193,370 (about 8 min), persona_profile 8,
    persona_flow 0 (first run), spend_forecast 2,688, spend_backtest 10,752,
    spend_signals 14,908, spend_model_run 1.
- `sql/reports.sql` on live Impala: all 20 statements OK, 6-12 s each.
  Persona moves are empty on the first run.
- CAI project `rsingh-spend-analytics` (`8ze8-2kux-eik9-8pzv`) and its five
  jobs created with `ci/setup_cai.py` (decision 25).
  - The first `spend-sync-code` run (27 s) pulled `5f99510`.
  - The second failed after about 20 s with no traceback, and so did a
    throwaway diagnostic job that ran pip and wrote its output to a project
    file. That file stopped in the middle of the 553 MB `nvidia_cudnn_cu13`
    download: at 1 vCPU / 2 GB, pip is killed while it installs torch.
  - At 2 vCPU / 8 GB the install succeeded in 893 s, and the hash marker
    matches `requirements.txt`. `setup_cai.py` now resizes existing jobs to
    `ci/cai_jobs.py`.
  - API v2 has no job-run log endpoint. Project files are read with
    `POST /files/<path>:download` (a GET returns 404).
- 2026-09-25 as CAI jobs (triggered_by `cai-job`), in order: feature
  engineering 71 s, training 321 s, evaluation 277 s, 11 min in all against
  28 min from the laptop.
  - Run `20260925-9956bba8` replaced the laptop run for the date (one
    `spend_model_run` row, 193,370 `customer_persona` rows).
  - Same KPIs as the laptop run: k=8, silhouette 0.163, ARI 0.649, accuracy
    0.925 / 0.934, WAPE 0.0998 vs 0.1144 (ratio 0.872), coverage 0.779,
    14,908 signals, gate PASS. Feature hash `aa21012976534ed4`, MLflow run
    `wqym-trq3-pxqa-wqw9`.
- Model `spend-scorer` (`30e2f08e-8fcc-4433-b803-cdb7f55e3675`) created by
  `setup_cai.py` with a build that deploys itself.
  - Build 455 s, deployed at 617 s.
  - `SPEND_ENDPOINT_URL` and `SPEND_ENDPOINT_ACCESS_KEY` are set in the
    project environment.
  - `test_endpoint.py` against the deployment, with authentication on (the
    API v2 key is also accepted as the bearer): profile 16.8 s cold, then
    2.4 s; forecast 2.2 s; what-if 1.8 s; categorise 1.2 s.
  - It serves run `20260925-9956bba8`, feature hash `aa21012976534ed4`,
    source impala.
  - The planted customer is Tier-1 traveller with TRAVEL_SURGE. The what-if
    moves him nearest to Tier-1 digital foodie (5.54 vs traveller 5.81).
  - For now `SPEND_ENDPOINT_API_KEY` is the same API v2 key (Ravi's call); a
    separate Model API key would be narrower. Rerunning `setup_cai.py` set it
    and created the app `Spend Insights Console` (`4lrw-tk0m-9lr8-8x8m`,
    subdomain `rsingh-spend-insights`), which reached APPLICATION_RUNNING.
- Airflow: `cde/scripts/set_airflow_variables.py` created the seven
  `SPEND_CAI_*` Variables through the vcluster's Airflow API (Knox token for
  the workload user). The 16 other projects' variables are untouched.
- DAG deploy: `cde job create --schedule-paused` is rejected for Airflow jobs
  ("does not support setting schedule from outside of DAG"), and nothing was
  created.
  - With `catchup=False` and `start_date` 27 Sep, an unpaused registration
    would at once run the latest closed interval.
  - The DAG now sets `is_paused_upon_creation=True` (decision 26).
- After the push of `602e3f8`, `deploy_dag.sh` registered
  `rsingh-spend-anl-orchestration`. The Airflow API shows `is_paused` true
  and 0 DAG runs; the next logical date is 2026-09-27 21:00 UTC, so
  unpausing runs as_of 2026-09-27.
- First DAG run (unpaused 14:52 UTC, run 2390, as_of 2026-09-27): stopped
  in `generate_spend_bronze` (run 2391) by cluster capacity, not by the code.
  - The driver pod waited 17 min (14:55 to 15:12). Two executor pod requests
    were withdrawn as "excess" at 15:12:41.
  - After that, Spark logged "Initial job has not accepted any resources"
    every 15 s for over an hour. No other run on the vcluster was active.
  - Same job size as run 2377, which finished in about 4 min on 25 Sep:
    driver 4 cores / 8 GB, executors 8 cores / 12 GB, 2 minimum / 4 initial
    / 16 maximum. Larger executors would be harder to place, not easier.
  - An Impala `DESCRIBE HISTORY` from the laptop hung for 13 min at the same
    time.
  - Ravi moved the project to another cluster. On this one, at 16:17 UTC,
    runs 2390 and 2391 were killed and the DAG was paused again
    (`is_paused` true), so the nightly 21:00 UTC run does not fire.
- GitHub -> CAI: the secrets `CAI_URL`, `CAI_API_KEY` and `CAI_PROJECT_ID`
  were set with `gh secret set` from `.env` (values never printed).
  - Not exercised yet: the re-run of `cai-pipeline` was held while the DAG's
    CAI tasks could still run, so that sync-code could not reset code under
    them.
  - Chest-X-ray-Triage, set up the same way today, could not connect from
    GitHub's hosted runners to its workbench. Such workbenches are often not
    reachable from the public internet, which calls for a self-hosted runner
    (docs/DEMO_RUNBOOK.md step 10).
  - While the secrets are set, every push to main runs the chain against
    this workbench. If it is unreachable, the `cai-pipeline` check fails at
    the first API call (listing jobs), before any job starts.

### Phase 9b: the federal environment (from 2026-09-29)

- Ravi moved all projects to a new environment. The committed hosts were
  repointed in `523299c`.
  - The `cde` CLI logs in with the workload user and password
    (`auth-pass-file`): the environment is in another CDP account, so the
    old access key fails with "Environment ... not found".
  - `CAI_URL` was deleted from GitHub before that push, so `cai-pipeline`
    skipped.
- CDE: `deploy_jobs.sh` created the repository (cloned from GitHub), the
  python-env (built from PyPI in about 6 min) and four jobs.
- CAI: `setup_cai.py --skip-serving` created the project
  (`gq4g-0gpd-etm4-0j7n`) and five jobs. The model and app wait for a
  published run.
  - `spend-sync-code` installed `requirements.txt` in 751 s, so the
    workbench reaches PyPI.
  - The runtime `python3.11-standard:2026.08.1-b5` exists here as well.
- Airflow: the seven Variables were created. The DAG was registered paused
  (`is_paused` true, 0 runs).
- Bronze, run 1: rejected in 37 s. "queue root.dex-app-bxjjm2cr cannot fit
  application: task group request memory 84 GB, vcore 36.1, pods 5 larger
  than max queue allocation".
  - The vcluster's quota is below the go01 sizing (driver plus 4 initial
    8-core / 12 GB executors).
  - The jobs now use 4-core / 8 GB executors, 1 minimum / 2 initial / 16
    maximum (the new default in `deploy_jobs.sh`).
  - Run 2 was admitted after about 2.5 min and succeeded in about 21 min
    (about 4 min on go01 at the larger size).
- The rest of the 2026-09-28 chain succeeded:
  - dq bronze: 182 s, 68 checks.
  - silver: 153 s.
  - dq silver: 214 s, 23 checks.
  - CAI categorise: 165 s, 21,551 keys.
  - gold: 395 s.
  - dq gold: 152 s, 27 checks.
  - CAI feature engineering: 63 s.
  - All 118 checks passed.
- CAI training at 8 vCPU / 32 GB stayed in ENGINE_SCHEDULING for 20 min: the
  workbench had no node that size free.
  - Stopped and resized to 4 vCPU / 16 GB (now in `ci/cai_jobs.py`). It
    started at once and succeeded in 184 s (321 s at 8 / 32 on go01).
  - Evaluation: 205 s.
- Published run `20260928-d0b1059b` (triggered_by `cai-job`, feature hash
  `aa21012976534ed4`, MLflow `9jep-ssq2-t3wf-w8ql`):
  - 193,456 customers, k=8, silhouette 0.163, ARI 0.635.
  - Accuracy 0.926 / 0.934 (n=2,937).
  - WAPE 0.0998 vs 0.1144 (ratio 0.872), coverage 0.779.
  - 12,518 signals, gate PASS.
  - Gold rows: customer_persona 193,456, spend_forecast 2,688.
- Serving: model `spend-scorer` (`a9f47a25-97e5-4f99-a9f4-c91503d76028`) and
  app `Spend Insights Console` (`801g-3mb1-mub1-c9t7`) created by
  `setup_cai.py`. The endpoint key is the API v2 key, as on go01.
  - Model build 550 s, deployed at 703 s. The app reached APPLICATION_RUNNING
    in 31 s.
  - `test_endpoint.py`: all four actions answer (categorise 0.20 s) and serve
    `20260928-d0b1059b`, hash `aa21012976534ed4`.
  - The planted customer is Tier-1 traveller with TRAVEL_SURGE; the what-if
    moves him to Tier-1 digital foodie.

