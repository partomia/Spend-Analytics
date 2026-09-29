# Build plan — Spend Analytics on Cloudera

Link six source systems (CRM, core banking, card switch, UPI switch, net
banking / bill pay, ATM switch) on the customer key (CIF), give every spend
event a category (UPI payments carry no MCC, so a MiniLM embedding model
categorises the free-text payee names), and turn the result into ML personas
(KMeans, stable IDs across runs), a 12-week cohort x category spend forecast
(Chronos-2, gated against a seasonal-naive baseline) and rule-based
opportunity signals for marketing.

Demo of a governed data + ML pipeline on synthetic data. Personas and signals
suggest who to talk to; nothing is offered to a customer automatically.

## Platform mapping

Same CDP environment as `partomia/Mule-Account-Identifier`,
`partomia/Collections-Delinquency-Roll-Forward-Prediction`,
`partomia/ALM-IRRBB-CASA-Behavioural-Forecasting` and
`partomia/Customer-Churn-Prediction`.

| Layer | Service | What runs there |
|---|---|---|
| Ingest + medallion | CDE (Spark 3, Iceberg v2) | generate 8 bronze + 7 ref tables, silver (dedupe, CIF resolution, `spend_event`), gold (CIF x weekly snapshot, cohort x category x week, mix x week; MERGE) |
| Data quality | CDE + Great Expectations 1.23.2 | `dq_check.py --layer bronze/silver/gold` after each layer, results in `ref.dq_results` |
| Orchestration | CDE Airflow | daily DAG, 10 tasks: 4 CDE jobs, 3 DQ gates, 4 CAI jobs (API v2) |
| Categorisation | CAI Job `spend-categorise` | new merchant keys: MCC -> SEED -> MiniLM kNN -> FALLBACK, appended to `ref.merchant_category_map` |
| Daily ML | CAI Jobs `spend-feature-engineering` -> `spend-model-training` -> `spend-model-evaluation` | features + covariates; KMeans personas, MiniLM holdouts, Chronos-2 backtest + forecast, signals; KPIs, KPI gate, MLflow, publish |
| Experiment tracking | CAI Experiments (MLflow) | params, KPIs, gate detail per daily run |
| On-demand | CAI Model Deployment `spend-scorer` | `predict.py`: profile, per-customer forecast, what-if, live categorise |
| SQL + reports | CDW Impala (Hue) | insight reports, DQ summaries, Iceberg time travel |
| Spend Insights Console | CAI Application (Streamlit) | 10 tabs |
| CI | GitHub Actions | `test` (pytest, CDE jobs + GX small, stub CAI stages, DQ fails on cue) then `train-validate` (real MiniLM + Chronos-2 on CPU, **KPI gate enforced**) |

Names:

- Databases `rsingh_spend_anl_{bronze,silver,gold,ref}`. The prefix lives on the
  database only; tables are never prefixed (`rsingh_spend_anl_gold.spend_features`).
- Python package `spend/`, env var prefix `SPEND_<SECTION>_<KEY>`.
- CDE: repository `rsingh-spend-anl-pipeline`, python-env `rsingh-spend-anl-python-env`,
  jobs `rsingh-spend-anl-{generate-bronze,dq-check,build-silver,build-gold-features}`,
  DAG job `rsingh-spend-anl-orchestration` (dag_id `spend_analytics_pipeline`).
- CAI: project `spend-analytics`; jobs `spend-categorise`, `spend-feature-engineering`,
  `spend-model-training`, `spend-model-evaluation`; model `spend-scorer`; application
  `Spend Insights Console`; MLflow experiment `spend-analytics`.
- Schedule: daily 21:00 UTC (02:30 IST). Live neighbours on the vcluster: Mule
  20:30 UTC (~27 min), Churn 22:00 UTC, Collections 00:30 UTC (~21 min), CASA monthly
  06:00 UTC on the 1st.
- Nothing here touches the sibling repos, their databases, CDE resources or CAI
  projects: every name above carries `spend`.

## Access and environment (as recorded by Churn, 2026-09-28; re-checked live in Phase 9)

| What | Where / how |
|---|---|
| CDE vcluster | `https://2dbvrngw.cde-gvjmd7zs.go01-dem.ylcu-atmi.cloudera.site/dex/api/v1`; the `cde` CLI on this laptop is already authenticated (`~/.cde/config.yaml`) |
| CDE Git credential | reuse `rsingh-github-pat`; the repo is public, a credential only avoids anonymous sync limits |
| CDW Impala | `coordinator-default-impala-aws.dw-go01-demo-aws.ylcu-atmi.cloudera.site:443`, HTTP transport, LDAP workload user; `SPEND_IMPALA_USER` / `_PASSWORD` in `.env` (gitignored) |
| Iceberg storage | `s3a://go01-demo/warehouse/tablespace/external/hive/<db>.db/<table>` |
| CAI workbench | `https://ml-dbfc64d1-783.go01-dem.ylcu-atmi.cloudera.site`. No CAI CLI here: Ravi runs UI steps and pastes results back |
| CAI keys | the app needs a **Model API key**; the Airflow trigger needs a separate **CAI API v2 key** (User Settings > API Keys) |
| GitHub | `partomia/Spend-Analytics`, public; `gh` authenticated; commits as `Ravi <partomia@users.noreply.github.com>` |

## How we work

1. Read the sibling repo's file before writing its counterpart; copy, then adapt.
2. One commit per phase, committed at the end of the phase, with a detailed message.
3. No attribution trailers of any kind in commits or docs. A local
   `commit-msg` hook strips them; check `git log --format=%B` before each push.
4. `pytest -q` green before moving on. "Works locally" is not "works on Cloudera":
   verify each resource live, one at a time, and record what ran in `docs/PROJECT_LOG.md`.
5. Judge DQ from `ref.dq_results` in Impala, not from `cde run logs`.
6. Record measured numbers only; a change from the spec is written down below as a decision.
7. Experiments run in scratch folders (`--warehouse`, `--parquet-dir`), never in
   `data/warehouse` / `data/parquet`.
8. Ask Ravi before `git push`, before the first CDW write and before the first DAG
   trigger / schedule unpause. Nothing secret or unredacted goes into the public repo:
   `.env` stays gitignored; check every image before committing it.
9. Never modify the sibling repos, their venvs or their cluster resources.

## Decisions (where this plan differs from the build spec, and why)

1. **The daily CAI run is three chained jobs: Feature Engineering -> Model Training
   -> Model Evaluation** (the Cloudera AI MLOps workshop pattern), instead of the
   spec's single `spend-daily-run`. Each is a thin wrapper over one
   `spend.pipeline` stage; the DAG triggers them in order, so a failure shows
   which stage broke and a stage can be rerun alone.
   - `spend-feature-engineering` (`cai/jobs/feature_engineering.py`): REFRESH gold,
     record snapshot IDs, build the persona matrix, the 224 cohort x category series
     with calendar covariates, the categoriser holdout sets and the previous run's
     centroids / assignments.
   - `spend-model-training` (`cai/jobs/train_models.py`): KMeans + stable IDs, MiniLM
     seed index and holdout predictions, Chronos-2 backtest (4 cut-offs, 4-week step)
     and the 12-week forecast, the seasonal-naive baseline, signals.
   - `spend-model-evaluation` (`cai/jobs/evaluate_models.py`): KPIs, the KPI gate,
     MLflow, and the gold writes. A failed gate writes `spend_model_run` with
     `gate_passed=false`, keeps yesterday's outputs and exits 1.
   - Hand-off: `artifacts/<run_date>/` (the CAI project filesystem, which every job
     of the project mounts; `storage.artifacts_dir`), with a manifest carrying the
     run_id and snapshot IDs, so training and evaluation read exactly what feature
     engineering pinned. `cai/jobs/daily_run.py` runs the three stages in-process
     (sessions, CI, backfill).
   - DAG: `generate >> dq_bronze >> silver >> dq_silver >> cai_categorise >> gold >>
     dq_gold >> cai_feature_engineering >> cai_model_training >> cai_model_evaluation`.
     Seven Airflow Variables: `SPEND_CAI_HOST`, `SPEND_CAI_PROJECT_ID`,
     `SPEND_CAI_API_KEY`, `SPEND_CAI_CATEGORISE_JOB_ID`, `SPEND_CAI_FEATURES_JOB_ID`,
     `SPEND_CAI_TRAIN_JOB_ID`, `SPEND_CAI_EVALUATE_JOB_ID`.
2. **GitHub -> Train & Validate -> KPI Gate.** CI has two jobs. `test` is the
   spec's CI (pytest, CDE stages small with GX, stub categorise, stub daily run with
   `--ignore-gate`, DQ fails on cue). `train-validate` (needs `test`) installs
   `requirements-train-ci.txt` (CPU torch), caches the Hugging Face models, and runs
   the real MiniLM + Chronos-2 on a small book **with the gate enforced**: a
   change that makes the models worse than the gate turns the commit red. The stub
   run keeps `--ignore-gate` because the stub forecaster *is* the naive baseline
   (WAPE ratio 1.0 by construction).
3. **Category shares use categorised merchant spend as the denominator.** The spec
   asks for 14 category shares summing to 1, but cash, UPI P2P and not-yet
   categorised payments also exist. The 14 shares are a mix of card, UPI P2M and net
   banking spend with a category; `uncategorised_share` (of merchant spend) is its own
   column and is gated below 5%. `OTHER` is one of the 14 categories.
4. **Location falls back to home, and says so; the fallback warning is at 55%, not
   40%.** Card POS uses the merchant city, UPI P2M the QR pincode, ATM the ATM's
   pincode; card e-commerce, net banking, UPI P2P and UPI P2M without a pincode (bills,
   EMIs, online merchants) get the home city with `location_source = 'home_fallback'`.
   Measured on the 20k book: 48.2% of events. Bills and EMIs alone are ~1 UPI or net
   banking payment per customer-month and have no place, so the spec's 40% warning
   would fire on every load and teach people to ignore it. The warning sits at 55%
   and still catches the failure it is for: losing the QR pincodes would push the
   share above 70%.
5. **Two gold tables beyond the spec.** `gold.spend_mix_weekly` (week x dimension x
   value x channel: amount, n_txn, n_customers; dimensions category, city_tier,
   state, city, away_city, age_band, segment, location_source) so the app never
   scans silver; `gold.persona_flow` (run_date, prev_run_date, from / to persona,
   n_customers) for the Sankey.
6. **Weeks run Saturday to Friday**, `week_end` is the Friday. Gold snapshots are
   every Friday from 2025-07-04 (13 weeks of history) plus the as-of date.
7. **Spike timings (Phase 0, this laptop, M-series CPU / MPS).** MiniLM: 2,250
   texts/s on CPU (5,321 on MPS), model load 37 s cold. Chronos-2: 224 series x 12
   weeks in 0.88 s on CPU (0.76 s on MPS), load 27 s cold. Both are fast on CPU, so
   `model.device: ""` (CUDA if present, else CPU) and CPU CAI jobs are enough; the
   CAI GPU is not needed.
8. **`spend_backtest` is stored at week grain** (cut-off x series x week: actual,
   P10 / P50 / P90, naive) rather than one row per cut-off x series, so the app can
   draw each backtest window; WAPE and coverage are aggregates of those rows and the
   run's KPIs are stored in `spend_model_run`.
9. **The generator ships Arrow batches, one pass per table.** Pickled nested rows
   through `createDataFrame(rdd)` took 280 s at 20k customers; `mapInArrow` with the
   master tables skipping the day-by-day simulation takes 47 s. Re-sent duplicates
   are emitted in Python (a deterministic hash of the key) so no plan is scanned
   twice, and Arrow timestamps carry the simulated wall-clock time as UTC, whatever
   the worker's time zone (`run_cde_local.py` also pins `TZ=UTC`).
10. **Demo customers are planted, not hoped for.** Units 0-2 are fixed: a Bengaluru
    affluent foodie with a Goa trip and travel bookings in Sep 2026 (TRAVEL_SURGE on
    the 2026-09-25 run), a Hyderabad bill payer who relocates to Nagpur on
    2026-08-24 (RELOCATION), and a Pune online shopper buying ~INR 67k of electronics
    over Diwali 2025 (FESTIVE_ELECTRONICS on the Oct 2025 backfill runs).

## Method (demo policy, see `config/policy.yaml`)

- **Categorise**: MCC (card) -> SEED (exact normalised match) -> EMBEDDING (MiniLM,
  5-NN similarity-weighted vote, similarity >= 0.45) -> FALLBACK (nearest category
  description, else OTHER). Gate: accuracy >= 0.85 on a 20% seed-variant holdout and
  on 5,000 card merchants with the MCC hidden.
- **Personas**: KMeans on 27 standardised features for customers with >= 5
  transactions in 13 weeks, k in 5..9 by silhouette on a 20k sample, IDs matched to
  the previous run (Hungarian), rule-based names. Gate: silhouette >= 0.15,
  stability >= 0.80 (skipped on the first run); ARI vs the hidden archetype logged.
- **Forecast**: Chronos-2 on 16 cohorts x 14 categories, 12 weeks, calendar
  covariates. Gate: WAPE <= 0.90 x seasonal-naive WAPE, P10-P90 coverage in [0.60, 0.95].
- **Signals**: TRAVEL_SURGE, FESTIVE_ELECTRONICS, RELOCATION (rules on gold features).

## Phases

- [x] **0. Scaffold + spike**: layout, config, requirement files, `spend/{config,
  features,schema,storage}.py`, `scripts/run_cde_local.py`, commit hook, model spike
  (decision 7). 15 tests.
- [x] **1. Bronze**: generator (8 bronze + 7 ref tables), `tests/test_generator.py`.
  20k customers locally in 50 s (19,465 onboarded by 2026-09-25): 2.95M events,
  ~151 per customer (~30M at 200k), 58% UPI (23% of UPI is P2P), 19% card, 8% net
  banking, 14% ATM; 8,170 merchants with 2-5 variants, 31% of them seeds. 40 tests.
- [x] **2. DQ bronze + silver**: `dq_check.py`, `build_silver.py`, `tests/test_cde_jobs.py`.
  68 bronze + 23 silver checks, 0 failed on the 20k book (silver 15 s: 19,416 customers,
  2.80M events, 0.0% unresolved, 48.2% home fallback); `--inject-bad-data` fails
  exactly one critical check (2.0% unknown payer VPAs) and `dq_bronze` exits 1. 49 tests.
- [ ] **3. Categorise**: `spend/categorise.py`, `cai/jobs/categorise.py`, local stage, stub.
- [ ] **4. Gold + DQ gold**: `build_gold_features.py` (three gold tables, MERGE), gold checks.
- [ ] **5. Core ML**: personas, forecast, signals, gate, three-stage pipeline, storage.
- [ ] **6. CAI jobs + endpoint**: the four jobs + `daily_run.py`, backfill, `predict.py`, `test_endpoint.py`.
- [ ] **7. Spend Insights Console**: 10 tabs + AppTest.
- [ ] **8. Orchestration + CI + docs**: DAG, deploy / backfill scripts, CI (test + train-validate), SQL, README, runbook.
- [ ] **9. Live on Cloudera** (with Ravi).
