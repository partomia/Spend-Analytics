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
