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
