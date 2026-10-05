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
| Orchestration | CDE Airflow | daily DAG, 12 tasks: 4 CDE jobs, 3 DQ gates, 4 CAI jobs, the endpoint refresh and the registry deployment (API v2) |
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
| CDE vcluster | `https://bxjjm2cr.cde-lzjl69mv.federal.dp5i-5vkq.cloudera.site/dex/api/v1` (the `federal` environment, from 2026-09-29; the go01 cluster before). The `cde` CLI logs in with the workload user and password |
| CDE Git credential | reuse `rsingh-github-pat`; the repo is public, a credential only avoids anonymous sync limits |
| CDW Impala | `coordinator-federal-impala-1.dw-federal-cdp-env.dp5i-5vkq.cloudera.site:443`, HTTP transport, LDAP workload user; `SPEND_IMPALA_USER` / `_PASSWORD` in `.env` (gitignored) |
| Iceberg storage | the environment's Hive warehouse, `<db>.db/<table>` (on go01: `s3a://go01-demo/warehouse/tablespace/external/hive`) |
| CAI workbench | `https://federal-cml.federal.dp5i-5vkq.cloudera.site`, driven over the API v2 (`ci/setup_cai.py`) |
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
11. **EMBEDDING blends MiniLM with character n-grams, plus a category-description
    prior.** MiniLM alone does not pass the 0.85 gate: best 0.82 on the seed holdout
    (k in 1..10, with or without the prior) and at most 0.85 on card merchants that
    miss the seed index. UPI payee names are truncated, have their vowels dropped or
    are run together ("PHONEPE*RAMESHKIRANA"). MiniLM's word pieces lose those, and
    owner names shared across shop types ("vivaan clinic" vs "vivaan diagnostics") then
    dominate. Similarity is now 0.5 x MiniLM cosine + 0.5 x TF-IDF character 2-4 gram
    cosine (`model.char_weight`). The 5-NN vote adds 1.0 x the similarity to each
    category description (`categorise.taxonomy_prior`). On the 20k book: 0.926 on the
    seed holdout (1,412 names), 0.934 on 2,932 card merchants with the MCC hidden
    (0.901 on the 1,392 that miss the seed index). Character n-grams alone score about
    the same here (0.938 / 0.933), because synthetic names carry their type in the
    spelling. MiniLM stays in for what n-grams cannot do on real data: brand names with
    no type word, and synonyms. The n-gram half alone is the stub (CI, no torch).
12. **Away-from-home is measured on located spend only, and over weeks.**
    `away_share_4w` leaves out home_fallback events: a bill or P2P transfer placed at
    home only because it has no location diluted unit 1's move to Nagpur to 0.67.
    Measured on located spend it is 1.0. A trip looks the same over 4 weeks, so
    `away_weeks_4w` (a feature the spec does not list) counts the weeks in which at
    least half the located spend was away. On 2026-09-25, unit 0's Goa trip peaks at
    2 and unit 1 has 4. 142 of about 19,300 active customers have 3 or more.
    RELOCATION uses both (Phase 5).
13. **Gold MERGE is copy-on-write, as in the four earlier projects.** A rebuild from
    the same inputs changes 0 rows, but `WHEN NOT MATCHED BY SOURCE` scans the whole
    target, so Iceberg rewrites every data file (1.2M rows at 20k customers). It is
    still one snapshot per load, and CDW Impala reads it without delete files.
    Merge-on-read (`write.merge.mode`) would write only the changes; it is
    worth trying live at 200k in Phase 9 only if gold load time matters.
14. **Persona inputs are clipped at +-3 standard deviations.** On the spec's 27
    standardised features, silhouette peaks at 0.149 (k=8), below the 0.15 gate:
    a few customers with one huge ticket pull clusters of their own. Clipping the
    standardised values at +-3 gives 0.166 (k=8) and improves recovery of the
    hidden archetypes (ARI 0.60 -> 0.65 at k=8, 0.64 -> 0.68 at k=7), so it is a
    better clustering, not just a better score. PCA would score higher (0.23-0.27),
    but only because silhouette is then measured in fewer dimensions, so it is not
    used. Four consecutive Fridays: silhouette 0.166-0.168, stability 0.945-0.958,
    the same 8 IDs and names each week.
15. **The seasonal-naive trend needs 4 weeks with a year-ago value.** Otherwise the
    trend is 1.0. The earliest backtest cut-off has 53 weeks of history, so its
    13-week trend would rest on one week and swing the baseline by tens of percent
    (WAPE 0.28 against 0.19). 20k book, 4 cut-offs x 12 weeks x 224 series:
    Chronos-2 with the 5 calendar covariates has WAPE 0.151 against 0.195 for
    naive (ratio 0.775), and P10-P90 coverage 0.765. Without covariates WAPE is
    0.177, so the covariates earn their place.
16. **TRAVEL_SURGE compares the last 4 weeks with the 9 weeks before them.** A
    13-week norm includes the surge, so 4 weeks against it can never exceed
    13/4 = 3.25, and a 3x threshold would mean "almost all travel was recent".
    On 2026-09-25: 1,004 TRAVEL_SURGE (unit 0 at 88x his norm), 512
    FESTIVE_ELECTRONICS and 8 RELOCATION (unit 1). The volumes are a policy
    setting, not a model output: `min_amount_4w` is the knob. The hidden-MCC
    categoriser holdout uses every MCC-mapped key of the pinned map snapshot,
    whatever its first_run_date: it measures the model, not a dated output, and
    a backfill date would otherwise see an empty map.
17. **The per-customer forecast also returns a mean, and quotes its 12-week sum.**
    One customer's weekly spend in a category is intermittent (unit 0 dines out
    in 29% of weeks), so the weekly P50 is near 0: his DINING P50 summed over
    12 weeks is INR 365 against INR 8,837 in the last 13 weeks. Quantiles do not
    add across weeks, means do. Chronos-2 is asked for the 19 quantiles
    0.05..0.95 as well, and their average approximates the mean (the integral of
    the quantile function). The endpoint's `expected_12w` is its sum: unit 0's
    DINING is INR 4,873 against a 52-week run rate of INR 5,386, and unit 2's
    ELECTRONICS is INR 16,188 (Diwali 2026 falls in the horizon). P10 / P50 / P90
    are unchanged (native levels), so the cohort backtest and the gate are too.
18. **The endpoint's what-if re-assigns personas by nearest raw centroid.** The
    run stores each persona's raw centroid (member mean) and the scaler, not the
    KMeans centroid in clipped standardised space, so the endpoint's nearest
    centroid is an approximation of the daily label: it agrees for 98.6% of the
    19,333 customers profiled on 2026-09-25. The profile action reports the
    daily run's label; what-if compares before and after with the same rule,
    so a move is never an artefact of mixing the two.
19. **The console's state map is a bubble map, and what-if sits in Customer 360.**
    The repo ships no Indian state boundaries (and must not fetch them at run
    time on CAI), so Portfolio draws one bubble per state at the mean of its
    `ref.pincode_geo` coordinates, with area proportional to spend. The spec's
    ten tabs have no what-if tab, so the endpoint's what-if is a panel under the
    customer's profile and forecast in Customer 360.
20. **CI's `train-validate` reuses the book `test` built, and the drill runs the
    categorise job itself.** `test` uploads `data/parquet` (3,000 customers,
    ~46 MB) as an artifact, and `train-validate` downloads it, so the second
    job needs neither Java nor Spark nor GX, only CPU torch. Its map comes from
    the stub categoriser, but the gate's accuracy checks run the real MiniLM on
    the holdouts, so the gate still measures the real models. On that book,
    locally: accuracy 0.926 / 0.935, WAPE ratio 0.696, coverage 0.771,
    silhouette 0.166, all PASS. `backfill_drill.sh` starts `spend-categorise`
    over the API v2 between silver and gold for each date (the same four values
    as the DAG, from the environment), because gold joins the map and a fresh
    map would leave every key UNCATEGORISED and fail `dq_gold`.
21. **A feature hash ties a run to the feature code that built it.** Not in the
    spec; taken from the CXR Triage build guide. `spend.features.feature_hash()`
    is 16 hex digits of SHA-256 over the feature columns, persona features and
    clip, covariates, categories, channels and `FEATURE_LOGIC_VERSION` (bump it
    when a formula changes but no name does). Feature engineering writes it to
    the manifest; training refuses artifacts with another hash; the gate checks
    it against the code; it is stored in `spend_model_run.feature_hash` and in
    MLflow; the endpoint and app refuse to load a run with a different hash
    (the scaler and centroids would be applied to other features). The first
    run after an upgrade therefore has to be a fresh daily run, not a replay.
22. **The gate adds non-regression against the champion, with tolerances from
    a backfill.** The champion is the published run the candidate would
    replace: the same run date's if published (a GitHub candidate, or a rerun,
    is judged on the same data), else the last one before. Persona IDs and
    stability still chain to the run strictly before. Each accuracy may drop at
    most 0.01 and the WAPE ratio rise at most 0.03; skipped when nothing is
    published. Measured with real models over 8 Fridays (2026-08-07 to
    2026-09-25, 20k book, scratch copy): accuracy stayed 0.926 / 0.934 every
    week (0.01 is 14 of 1,412 seed texts); the WAPE ratio ranged 0.765-0.801,
    the largest weekly rise +0.010; a same-day rerun of 2026-09-25 reproduced
    0.775 exactly. Coverage and silhouette keep only their absolute floors.
23. **GitHub -> Train & Validate -> KPI Gate also runs on Cloudera AI.** A third
    CI job, `cai-pipeline` (after `train-validate`, pushes to main only, one at
    a time), calls the CAI API v2 from `ci/trigger_cai_pipeline.py`. Unlike CXR
    Triage the jobs have no CAI dependencies (Airflow sequences them nightly),
    so the trigger starts each job itself: `spend-sync-code` (new; git reset to
    origin/main, stops if main moved past the pushed SHA), feature engineering
    and training into `artifacts/github/`, and evaluation with
    `SPEND_DRY_RUN=1`, i.e. the gate only. It is a candidate check, not a
    publish: the nightly DAG publishes, with the synced code. `spend-categorise`
    is left out because it writes the map. Jobs are looked up by name
    (`ci/cai_jobs.py`, which the runbook table must match), so GitHub needs only
    three secrets; without `CAI_URL` the job prints a notice and passes. Not
    adopted from CXR: a deploy-champion job (the endpoint loads the latest
    published run at start-up) and PSI drift.
24. **The first categorise and daily run were run from the laptop against CDW.**
    The spec runs them as CAI jobs. There was no CAI API key when Phase 9
    started, so `cai/jobs/categorise.py` and `daily_run.py` ran locally with
    `SPEND_STORAGE_BACKEND=impala` (triggered_by `manual-laptop`): the same code
    and the same tables. The CAI jobs take over from the next run date, and a
    CAI rerun of 2026-09-25 replaces that day (DELETE + INSERT on run_date).
    At 200k the laptop path spends about 9 min on Impala reads and about 18 min
    on chunked INSERTs; inside CAI both should be closer to the warehouse.

25. **The CAI project is `rsingh-spend-analytics`, set up over the API, and sync-code
    installs the requirements.** The spec names the project `spend-analytics`; the
    shared workbench holds 39 projects and the siblings are all `rsingh-*`, so the
    prefix keeps it findable and collision-free (job names are per project and stay
    `spend-*`). `ci/setup_cai.py` creates the project from Git, its environment
    variables (Impala user and password from `.env`, never printed) and the jobs of
    `ci/cai_jobs.py` on the runtime the siblings' jobs use (JupyterLab, Python
    3.11, Standard, 2026.08.1-b5). There is no API to run a command in a session,
    so `spend-sync-code` pip-installs `requirements.txt` whenever its hash changes;
    packages persist in the project's `/home/cdsw/.local`, and a changed
    requirements file pushed from GitHub is installed before the chain uses it.
    Sync-code runs at 2 vCPU / 8 GB because pip is killed at 2 GB while it
    installs torch. The same script creates the `spend-scorer` model (built and
    deployed once, authentication on), puts its URL and access key in the
    project environment, and creates the application once a Model API key is
    supplied. `cde/scripts/set_airflow_variables.py` upserts only the seven
    `SPEND_CAI_*` Airflow Variables.

26. **The DAG is registered paused (`is_paused_upon_creation=True`).** The spec
    deploys it live. But CDE takes an Airflow job's schedule only from the DAG
    (`--schedule-paused` is rejected), and with `catchup=False` a DAG whose
    `start_date` is more than a day back runs the latest closed interval as soon
    as it is registered. That would be an unasked first trigger. Unpausing is
    the deliberate first run, and it needs no manual trigger.
27. **MLOps on what exists: the registry versions the persona model, the DAG and CI
    redeploy.** No retraining or serving redesign.
    - The persona model is logged to MLflow as a stock scikit-learn pipeline
      (`spend/registry.py`). It is not a custom pyfunc, which registry
      deployments don't load.
    - A passed, latest-dated run registers it as the next `spend-persona`
      version, so the highest version is the champion. Its run ID comes from the
      version's MLflow parameters, because tag updates through the API v2 never
      returned (18 minutes).
    - The DAG's `refresh_endpoint` restarts `spend-scorer` and checks it serves
      the champion's run (560 s).
    - A push touching the model code rebuilds `spend-scorer` after the CAI gate
      (build 8 min, deploy 2 min 48 s).
    - The signature is float64: MLflow rejects JSON numbers against a float32
      tensor (HTTP 400 on version 1, reproduced locally).
    - `ci/deploy_registered_model.py` deploys the champion version as its own
      model, `spend-persona-registry` (version 2 in 185 s; 0.3 s calls). After
      that, the DAG task `deploy_registry_model` builds each night's new version
      into it, beside `refresh_endpoint`.
    - The AI Registry has public internet access: from the private one, every
      call to a registry-deployed model timed out after 30 s.
28. **The CDV dashboard is code, on reporting views, in its own shared workspace.**
    - `sql/dataviz_views.sql` adds 7 views in `rsingh_spend_anl_report`.
      Nothing is copied, and the pipeline doesn't read them.
    - `dataviz/build_dashboard.py` imports "Spend Analytics - Insights"
      (7 datasets, 40 visuals, 6 sheets) through the migration API into the
      workspace `rsingh-spend-anl` (the workload user manages it, Everyone views it),
      with fixed UUIDs, the same pattern as the Collections project on this instance.
    - It is not built in the UI, so it can be re-created and reviewed.
    - All 40 visual queries answered through CDV on 2026-10-05.
## Method (demo policy, see `config/policy.yaml`)

- **Categorise**: MCC (card) -> SEED (exact normalised match) -> EMBEDDING (MiniLM +
  character n-grams, 5-NN similarity-weighted vote + category-description prior,
  similarity >= 0.45, decision 11) -> FALLBACK (nearest category
  description, else OTHER). Gate: accuracy >= 0.85 on a 20% seed-variant holdout and
  on 5,000 card merchants with the MCC hidden.
- **Personas**: KMeans on 27 standardised features for customers with >= 5
  transactions in 13 weeks, k in 5..9 by silhouette on a 20k sample, IDs matched to
  the previous run (Hungarian), rule-based names. Gate: silhouette >= 0.15,
  stability >= 0.80 (skipped on the first run); ARI vs the hidden archetype logged.
- **Forecast**: Chronos-2 on 16 cohorts x 14 categories, 12 weeks, calendar
  covariates. Gate: WAPE <= 0.90 x seasonal-naive WAPE, P10-P90 coverage in [0.60, 0.95].
- **Signals**: TRAVEL_SURGE, FESTIVE_ELECTRONICS, RELOCATION (rules on gold features).
- **Gate, besides the floors**: accuracy down <= 0.01 and WAPE ratio up <= 0.03 against the
  champion (decision 22); the run's feature hash equals the code's (decision 21).

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
- [x] **3. Categorise**: `spend/categorise.py`, `spend/models.py`, `cai/jobs/categorise.py`,
  `new_merchant_keys()` on both storages, local stage, stub. 20k book: 21,466 keys in
  29 s (SEED 6,776, MCC 2,932, EMBEDDING 11,756, FALLBACK 2). Every one of 1.98M
  merchant events is mapped, and a rerun adds nothing. Accuracy 0.926 on the seed
  holdout and 0.934 with the MCC hidden (decision 11). 60 tests.
- [x] **4. Gold + DQ gold**: `build_gold_features.py` (three gold tables, MERGE), gold checks.
  20k book in 40-50 s: spend_features 1,208,205 rows (63 snapshots), cohort series
  17,248 rows (224 series x 77 complete weeks), mix 37,669 rows. 27 gold checks, 0
  failed; uncategorised 0.0%; the category mix reconciles with the cohort series in
  every week. Planted customers visible: unit 0 TRAVEL INR 91k in 4 weeks, unit 1
  away 4 of 4 weeks, unit 2 ELECTRONICS INR 85k in the 4 weeks to 2025-10-24. 64 tests.
- [x] **5. Core ML**: `spend/{personas,forecast,signals,gate,pipeline}.py`, snapshot-pinned
  storage reads. Real models on the 20k book, 2026-09-18 then 2026-09-25, 35-45 s a
  run on the laptop CPU. On the 2026-09-25 run every gate check passes: category
  accuracy 0.926 / 0.934, WAPE ratio 0.775, coverage 0.765, silhouette 0.166 (k=8),
  stability 0.958; ARI vs the hidden archetypes 0.654; 1,524 signals. 87 tests.
- [x] **6. CAI jobs + endpoint**: `cai/jobs/{feature_engineering,train_models,evaluate_models,
  daily_run,backfill_history}.py`, `spend/{tracking,scoring,client}.py`, `cai/model/{predict,
  test_endpoint}.py`. Real models on the 20k book: the three jobs for 2026-09-25 in 2 s /
  44 s / 1 s, the same KPIs as Phase 5, exit 0; stub exit 1 (0 with `--ignore-gate`);
  a 3-Friday backfill in 90 s. Endpoint: context 17 s, profile 1.6-2.3 s, forecast
  1-1.9 s, what-if and categorise < 0.25 s (decisions 17, 18). 114 tests.
- [x] **7. Spend Insights Console**: `app/{streamlit_app,data,run}.py`, 10 tabs (Portfolio, Where,
  Channels, Personas, Customer 360, Forecast, Categoriser, Opportunities, Data quality, Lineage).
  Book-wide views read gold only (`spend_mix_weekly`, run outputs); one customer comes from the
  endpoint, or in-process with the stub fallback. 20k book: all tabs in 2.9 s with the stubs,
  22 s on first load with the real models in-process (then cached). AppTest on every tab,
  the live categoriser, what-if, a failed-gate run and two cold starts (decision 19). 121 tests.
- [x] **8. Orchestration + CI + docs**: `cde/dags/spend_dag.py` (10 tasks, 7 Variables),
  `cde/scripts/{deploy_jobs,deploy_dag,backfill_drill}.sh`, `ci.yml` (test, then
  train-validate with the gate enforced), `sql/reports.sql` (12 reports; its 13 SELECTs all run
  in Spark SQL against the CI book), README, `docs/DEMO_RUNBOOK.md` (decision 20).
  `tests/test_orchestration.py` cross-checks job names, prefixes, the chain, the Variables
  and the CI shape; `tests/test_reports_sql.py` checks every SQL column against the table
  definitions. 132 tests.
- [x] **8b. CAI MLOps hardening** (from the CXR Triage build guide): feature hash (decision 21),
  non-regression against the champion (decision 22), `cai-pipeline` with `spend-sync-code`
  and `ci/trigger_cai_pipeline.py` (decision 23), `tests/test_cai_contract.py` (the CAI job
  runtime rules on every CAI script). 178 tests.
- [ ] **9. Live on Cloudera** (with Ravi).
