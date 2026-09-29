# Demo runbook

Two parts: **Setup** (one-time, to get the pipeline live on Cloudera) and
**Demo** (the ~12-minute walkthrough). Numbers quoted below are from the
20,000-customer book on a laptop (run date 2026-09-25) unless marked
otherwise; replace them with the cluster run's numbers as each step is done
live (Phase 9), and never quote a cluster number before you have seen it.
Secrets (`SPEND_IMPALA_PASSWORD`, `HF_TOKEN`, the model access key, any API
key) are never written out; take them from your own User Settings / `.env`
each time. Check every screenshot for keys and hostnames before committing it.

| What | Name | ID |
|---|---|---|
| CAI project | `spend-analytics` | (fill in) |
| CAI jobs | `spend-categorise`, `spend-feature-engineering`, `spend-model-training`, `spend-model-evaluation` | (fill in) |
| Model | `spend-scorer` | (fill in) |
| Application | `Spend Insights Console` | (fill in) |
| CDE DAG job | `rsingh-spend-anl-orchestration` | dagID `spend_analytics_pipeline` |

## Setup (one-time)

Ask Ravi before the `git push`, before the first write to CDW (the first real
categorise or evaluation run) and before the first DAG trigger or unpause.

### 1. CDE: repository, four Spark jobs

After the `git push` (Ravi):

```bash
./cde/scripts/deploy_jobs.sh
```

It creates the repository `rsingh-spend-anl-pipeline`, the python-env
`rsingh-spend-anl-python-env` with Great Expectations (wait for `ready`,
about 6 minutes) and the four jobs. Verify:

```bash
cde repository describe --name rsingh-spend-anl-pipeline
cde resource describe --name rsingh-spend-anl-python-env      # status ready
cde job list | python3 -c "import json,sys; [print(j['name'], j.get('type')) for j in json.load(sys.stdin) if 'spend-anl' in j['name']]"
```

Expected: `rsingh-spend-anl-{generate-bronze,dq-check,build-silver,build-gold-features}`
(type `spark`) and nothing else; delete orphaned jobs by hand if any remain.

Run the chain once by hand for one as-of date, as far as silver (the
categorise job needs the CAI project of step 2):

```bash
AS_OF=2026-09-25
run() { cde job run --name "rsingh-spend-anl-$1" --arg=--db-prefix --arg=rsingh_spend_anl "${@:2}" --wait; }
run generate-bronze --arg=--as-of --arg=$AS_OF
run dq-check --arg=--layer --arg=bronze --arg=--as-of --arg=$AS_OF --arg=--pipeline-run --arg=manual-$AS_OF
run build-silver
run dq-check --arg=--layer --arg=silver --arg=--as-of --arg=$AS_OF --arg=--pipeline-run --arg=manual-$AS_OF
```

Judge each layer from `ref.dq_results` in Hue, not from `cde run logs`:

```sql
SELECT layer, COUNT(*) AS checks,
       SUM(CASE WHEN NOT success AND severity = 'critical' THEN 1 ELSE 0 END) AS critical_failed
FROM rsingh_spend_anl_ref.dq_results GROUP BY layer;
```

20k book: 68 bronze and 23 silver checks, 0 failed.

### 2. CAI project

Projects > New Project: name `spend-analytics`, Git URL
`https://github.com/partomia/Spend-Analytics`, Python 3.11 runtime. A GPU is
not needed (MiniLM and Chronos-2 are fast on CPU, PLAN decision 7). Project
Settings > Advanced > Environment Variables:

| Variable | Value |
|---|---|
| `HF_HOME` | `/home/cdsw/.hf_cache` |
| `SPEND_IMPALA_USER` | your workload username |
| `SPEND_IMPALA_PASSWORD` | your workload password, typed into the UI, never into a committed file |
| `HF_TOKEN` | optional: a Hugging Face read token, avoids anonymous rate limits on the downloads |

The three `SPEND_ENDPOINT_*` variables are added in step 6.

### 3. Session: categorise, then finish the CDE chain

Open a Session (Python 3.11, 4 vCPU / 16 GB):

```bash
git pull
pip3 install -r requirements.txt
python cai/jobs/categorise.py --dry-run     # new keys from silver via Impala; writes nothing
```

Then, **after asking**, the first CDW write: `python cai/jobs/categorise.py`
appends the keys to `ref.merchant_category_map`. 20k book: 21,466 keys (SEED
6,776, MCC 2,932, EMBEDDING 11,756, FALLBACK 2), every merchant event
mapped. Back on the laptop, finish the chain:

```bash
run build-gold-features --arg=--as-of --arg=$AS_OF
run dq-check --arg=--layer --arg=gold --arg=--as-of --arg=$AS_OF --arg=--pipeline-run --arg=manual-$AS_OF
```

20k book: 27 gold checks, 0 failed, uncategorised share 0.0%. Back in the
session:

```bash
python cai/jobs/daily_run.py --dry-run                        # downloads both models once; gate; writes nothing
python cai/model/test_endpoint.py --local --context impala    # profile, forecast, what-if, categorise
```

Expected: every gate check PASS (20k book: accuracy 0.926 / 0.934, WAPE
ratio 0.775, coverage 0.765, silhouette 0.166; stability is skipped on the
first run).

### 4. CAI Jobs

Jobs > New Job, four times, Python 3.11 runtime, arguments empty, schedule
**Manual** (Airflow triggers them):

| Job | Script | Resources |
|---|---|---|
| `spend-categorise` | `cai/jobs/categorise.py` | 4 vCPU / 16 GB |
| `spend-feature-engineering` | `cai/jobs/feature_engineering.py` | 2 vCPU / 8 GB |
| `spend-model-training` | `cai/jobs/train_models.py` | 8 vCPU / 32 GB |
| `spend-model-evaluation` | `cai/jobs/evaluate_models.py` | 2 vCPU / 8 GB |

Run feature engineering, training and evaluation once, in order, and confirm
each shows **succeeded** in the UI. Evaluation writes the gold outputs and an
MLflow run in the experiment `spend-analytics`; `spend_model_run.mlflow_run_id`
names it. Laptop timings for the three: 2 s, 44 s and 1 s.

IDs for the Airflow Variables, from a session terminal (one line):

```bash
python3 -c "import os, cmlapi; c = cmlapi.default_client(); pid = os.environ['CDSW_PROJECT_ID']; print('SPEND_CAI_HOST       =', 'https://' + os.environ['CDSW_DOMAIN']); print('SPEND_CAI_PROJECT_ID =', pid); [print(j.id, j.name) for j in c.list_jobs(pid).jobs]"
```

### 5. Model Deployment: `spend-scorer`

Model Deployments > New Model: name `spend-scorer`, file
`cai/model/predict.py`, function `predict`, Python 3.11, 4 vCPU / 16 GB, 1
replica, authentication **on**. Example input for each action for the Test
tab:

```bash
python cai/model/test_endpoint.py --print-request --action profile      # also forecast, what_if, categorise
```

The response's `model` block names the run it serves. Restart the model after
each new daily run to serve it. Laptop timings: context 17 s at start-up,
profile 1.6-2.3 s, forecast 1-1.9 s, what-if and categorise under 0.25 s.

### 6. Application: `Spend Insights Console`

Applications > New Application: name `Spend Insights Console`, script
`app/run.py`, Python 3.11, 2 vCPU / 4 GB. Set at project level, from the
`spend-scorer` Overview page's sample code:

| Variable | Where to find it |
|---|---|
| `SPEND_ENDPOINT_URL` | the `https://modelservice.../model` URL |
| `SPEND_ENDPOINT_ACCESS_KEY` | the `accessKey` value |
| `SPEND_ENDPOINT_API_KEY` | User Settings > API Keys > a **Model API key** (not the API v2 key of step 7) |

The sidebar says whether Customer 360 and the Categoriser went to the
endpoint or ran in-process.

### 7. Airflow Variables and the DAG

CDE Airflow UI > Admin > Variables, the seven values (step 4 for the IDs):
`SPEND_CAI_HOST`, `SPEND_CAI_PROJECT_ID`, `SPEND_CAI_API_KEY` (a CAI **API
v2** key), `SPEND_CAI_CATEGORISE_JOB_ID`, `SPEND_CAI_FEATURES_JOB_ID`,
`SPEND_CAI_TRAIN_JOB_ID`, `SPEND_CAI_EVALUATE_JOB_ID`. The other projects'
`*_CAI_*` variables stay in the same list untouched. Then:

```bash
./cde/scripts/deploy_dag.sh
cde job schedule pause --name rsingh-spend-anl-orchestration     # a new DAG job is live; pause until ready
```

The DAG's `start_date` (27 Sep 21:00 UTC) is in the past and its first
interval closes after the deploy, so registering it starts no run by itself.
With a future `start_date` a manual trigger "succeeds" in seconds and runs
no task.

### 8. First end-to-end run

**After asking**: unpause (which also arms the daily 21:00 UTC schedule) and
trigger for yesterday:

```bash
cde job schedule unpause --name rsingh-spend-anl-orchestration
cde job run --name rsingh-spend-anl-orchestration --config-json '{"as_of": "<yesterday>"}' --wait
cde run logs --id <run-id> --type cai_model_evaluation/attempt_1 --follow=false
```

Verify from `ref.dq_results` (one `pipeline_run`, three layers) and from
each CDE job's run history. The four `cai_*` task logs show `CAI run <id>:
running` every 30 s until the job ends. The Lineage tab shows the run with
`triggered_by = airflow`.

### 9. History

```bash
export SPEND_CAI_HOST=... SPEND_CAI_PROJECT_ID=... SPEND_CAI_API_KEY=... SPEND_CAI_CATEGORISE_JOB_ID=...
./cde/scripts/backfill_drill.sh 2026-08-07 2026-08-14 ... <yesterday>    # 8 Fridays, then yesterday
python cai/jobs/backfill_history.py --weeks 8                             # from a CAI session
```

Don't trigger the DAG while either runs. Laptop: a 3-Friday backfill in 90 s,
stability 0.955 and 0.958 against the previous Friday.

## Before the demo (30 minutes ahead)

- CDE Airflow grid: last night's run green, all ten tasks, including the three
  `dq_*` gates and the four `cai_*` jobs.
- App Lineage tab: today's run with `triggered_by = airflow`, gate PASS, and
  the backfilled run dates.
- `spend-scorer` restarted after today's run; its Test tab shows today's run id.
- Run one Hue query 5 minutes before: the Impala warehouse auto-suspends.
- Hue open on `sql/reports.sql`; Airflow UI open on the DAG grid; CAI
  Experiments open on `spend-analytics`.
- Demoing a DQ failure live? Have a scratch prefix ready:
  `cde job run --name rsingh-spend-anl-generate-bronze --arg=--db-prefix --arg=rsingh_spend_drill --arg=--inject-bad-data --wait`,
  then the bronze gate on the same prefix fails with exit 1 (2.0% of UPI
  payments from unknown payer VPAs).

## 0-1 min. The question (slide)

Six systems, six keys, and UPI, most of today's transactions, has no
category. A card swipe carries an MCC; a UPI payment to "RAMESH KIRANA STORE"
carries only a name. One view of spend needs linking plus enrichment.

## 1-3 min. The pipeline: CDE Airflow UI

Jobs > `rsingh-spend-anl-orchestration` > Airflow UI.

- Ten tasks, three Great Expectations gates: six source extracts → gate →
  silver (one CIF per card, VPA and account; every event located or marked
  home_fallback) → gate → **the CAI categorise job** → gold (Iceberg MERGE,
  one snapshot per load) → gate → feature engineering → model training →
  model evaluation with the KPI gate.
- A critical failure stops the DAG at that layer; yesterday's insights stand.
- Hue: `sql/reports.sql` query 11 over `ref.dq_results`: every check, every
  run, and the snapshot it saw. Mention the `--inject-bad-data` run that
  stopped at `dq_bronze`.
- The same gate runs on every push: GitHub Actions `train-validate` runs the
  real MiniLM and Chronos-2 on a small book and fails the commit if a KPI
  drops below the gate.

## 3-4 min. App: Categoriser

- Type `SWIGGY*BLR 5521` (FOOD_DELIVERY, embedding, similarity 0.81, nearest
  seed "swiggy"), then `ola cabs pvt` (TRAVEL, exact seed match), then a new
  merchant: `SHARMA MEDICOS` (HEALTH, 0.79, nearest "sharma medicals").
  Category, method and similarity, live.
- Honest miss: `TANISHQ JEWELLERS` falls back to OTHER (a brand with no type
  word; this is what MiniLM is kept for on real data).
- Accuracy against hidden MCCs: 0.934 on 2,932 card merchants (seed holdout
  0.926). MiniLM alone reached 0.82; blended with character n-grams it passes
  the 0.85 gate (PLAN decision 11).

## 4-6 min. App: Portfolio, Where, Channels

- Spend by category, tier and state across all six sources (20k book: INR
  162.9 cr over 13 weeks, 488,446 transactions, 68.5% digital).
- The UPI vs card vs cash trend by week.
- Where: home vs away and the top away cities, with the honest
  `home_fallback` share (48% of events: bills, EMIs and online payments have
  no place, PLAN decision 4).

## 6-8 min. App: Personas

- Eight personas from KMeans (k chosen by silhouette), named by rule, IDs
  stable across runs: the Sankey shows who moved since last week (stability
  0.958 on the 20k book).
- Say it plainly: the data is synthetic and the generator planted seven
  archetypes; "the model recovered the planted archetypes, ARI X" (20k
  book: 0.654; fill in X from the cluster run).

## 8-10 min. App: Customer 360 + Forecast

- Type `C00016807`: a Bengaluru affluent foodie with a Goa trip. Persona,
  mix, the TRAVEL_SURGE signal (88x his norm), INR 49k spent in Panaji, and a
  12-week fan chart per category from Chronos-2, with the expected 12-week
  total (TRAVEL INR 19.8k, DINING INR 4.7k).
- What-if: set TRAVEL to 0 and he moves to "Tier-1 digital foodie".
- Forecast tab: cohort forecasts beat seasonal-naive by Y% (20k book: WAPE
  0.151 vs 0.195, a 22.5% lower WAPE, P10-P90 coverage 0.765; fill in Y from
  the cluster run).

## 10-11 min. App: Opportunities

- Travel-surge and festive-electronics lists ready for campaigns, by persona
  and city, with a CSV download (20k book, 2026-09-25: 1,004 TRAVEL_SURGE,
  512 FESTIVE_ELECTRONICS, 8 RELOCATION). The volumes are a policy setting
  (`min_amount_4w`), not a model output.

## 11-12 min. CAI Experiments + Hue

- CAI Experiments: one MLflow run per day with the KPIs and the gate detail.
- Lineage tab / query 1: models, snapshot ids and KPIs per run date. Query
  12: `FOR SYSTEM_VERSION AS OF <features_snapshot_id>` reads exactly what
  yesterday's run read.

## Honest framing (close)

Synthetic data with planted archetypes and seasonality, so recovery metrics
partly measure the generator. UPI location is only as good as the merchant
pincode; the fallback share is shown on purpose. MiniLM is an English model:
Hindi or regional-language payee names will categorise less well. Personas
are descriptive segments, not a credit or pricing model, and offers from
signals need the bank's consent and marketing rules.
