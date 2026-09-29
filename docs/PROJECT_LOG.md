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
