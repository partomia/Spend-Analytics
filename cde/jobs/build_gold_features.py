"""
Stage 4 - Gold (spend features, cohort series, spend mix)

Every merchant event (card, UPI P2M, net banking) takes the category of its
merchant_key in ref.merchant_category_map; a key the map has not seen yet is
UNCATEGORISED (the spend-categorise CAI job runs before this stage, so that is
rare and dq_gold fails above 5%). UPI P2P and ATM cash have no merchant.

gold.spend_features, one row per CIF per snapshot: every Friday from
FIRST_SNAPSHOT (13 complete weeks after the extracts start) to the load's
as-of date, plus the as-of date itself. Windows are whole Saturday-Friday weeks
ending with the snapshot's week (13 and 4 weeks); on a mid-week as-of date the
last week is partial, up to the as-of date. Population: customers onboarded on
or before the snapshot. Shares (PLAN decision 3):
  cat_<c>_share        of categorised merchant spend, 14 categories, sum to 1
  uncategorised_share  of merchant spend
  ch_<c>_share, tier<n>_share, home_share, digital_share, salary_week_share
                       of all spend; channel and tier shares sum to 1
  away_share_4w        of located spend in the last 4 weeks (card city, UPI pincode,
                       ATM) outside the home city; an event placed at home only
                       because it carries no location does not dilute a move
  away_weeks_4w        of the last 4 weeks, those where at least half the located
                       spend was away: a move persists, a trip does not
A customer with nothing to divide by gets 0 shares (spend_13w = 0).

gold.spend_cohort_weekly: cohort (segment x home city tier, 16) x category (14)
x complete week, dense (a week with no spend is 0), with the ref.calendar_weekly
covariates: the Chronos-2 series and their context.

gold.spend_mix_weekly: complete week x dimension x value x channel, the
aggregates the app charts (PLAN decision 5). Dimensions: category (NON_MERCHANT
for P2P and cash), city_tier, state, city, away_city (events outside the home
city only), age_band, segment, location_source.

Each table is rebuilt in full and written with MERGE INTO once it exists:
changed rows updated, new rows inserted, rows no longer produced (yesterday's
mid-week as-of snapshot) deleted, so a load is one Iceberg snapshot per table
and an unchanged week is not rewritten. Values are rounded so a rebuild of the
same inputs matches exactly.

The column lists mirror spend/features.py; tests/test_cde_jobs.py checks they agree.

Reads:
  <prefix>_bronze.crm_customer (batch_as_of), <prefix>_silver.{customer, spend_event},
  <prefix>_ref.{merchant_category_map, calendar_weekly}
Writes:
  <prefix>_gold.{spend_features, spend_cohort_weekly, spend_mix_weekly}

Usage:
  spark-submit build_gold_features.py [--db-prefix P] [--as-of YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DB_PREFIX = "rsingh_spend_anl"
HISTORY_START = date(2025, 4, 1)      # first day of the transaction extracts (a Tuesday)
FIRST_WEEK = date(2025, 4, 11)        # first complete Saturday-Friday week
FIRST_SNAPSHOT = date(2025, 7, 4)     # 13 complete weeks in
WINDOW_WEEKS, RECENT_WEEKS = 13, 4
RATIO_CAP = 5.0

CATEGORIES = ["GROCERY", "FOOD_DELIVERY", "DINING", "FUEL", "TRAVEL", "APPAREL", "ELECTRONICS", "UTILITIES",
              "TELECOM", "EMI_LOAN", "EDUCATION", "HEALTH", "ENTERTAINMENT", "OTHER"]
UNCATEGORISED = "UNCATEGORISED"
NON_MERCHANT = "NON_MERCHANT"
HOME_FALLBACK = "home_fallback"
CHANNELS = ["CARD_POS", "CARD_ECOM", "UPI_P2M", "UPI_P2P", "NETBANKING", "ATM_CASH"]
MERCHANT_CHANNELS = ["CARD_POS", "CARD_ECOM", "UPI_P2M", "NETBANKING"]
DIGITAL_CHANNELS = ["CARD_ECOM", "UPI_P2M", "UPI_P2P", "NETBANKING"]
TIERS = [1, 2, 3, 4]
SEGMENTS = ["MASS", "AFFLUENT", "NRI", "SME_OWNER"]

CAT_SHARES = [f"cat_{c.lower()}_share" for c in CATEGORIES]
CH_SHARES = [f"ch_{c.lower()}_share" for c in CHANNELS]
TIER_SHARES = [f"tier{t}_share" for t in TIERS]
SHARE_COLUMNS = CAT_SHARES + CH_SHARES + TIER_SHARES
CAT_4W = [f"cat_{c.lower()}_4w" for c in CATEGORIES]
KEY = ["cif", "snapshot_date"]
INFO = ["segment", "age_band", "income_band", "home_city", "home_state", "city_tier", "cohort"]
NUMERIC = [*SHARE_COLUMNS, "home_share", "digital_share", "uncategorised_share", "salary_week_share",
           "spend_13w", "txn_13w", "avg_ticket", "spend_trend_4v13", "merchant_spend_13w",
           "spend_4w", "txn_4w", "away_share_4w", "away_weeks_4w", *CAT_4W]
FEATURE_COLUMNS = [*KEY, *INFO, *NUMERIC]

COHORT_KEY = ["cohort", "category", "week_end"]
COVARIATES = ["festival_week", "salary_week", "month_end_week", "holiday_count", "school_fee_week"]
COHORT_COLUMNS = ["cohort", "segment", "city_tier", "category", "week_end", "amount_inr", "n_txn", "n_customers",
                  *COVARIATES]
MIX_DIMENSIONS = ["category", "city_tier", "state", "city", "away_city", "age_band", "segment", "location_source"]
MIX_KEY = ["week_end", "dim", "dim_value", "channel"]
MIX_COLUMNS = [*MIX_KEY, "amount_inr", "n_txn", "n_customers"]

TABLES = {  # name: (key, columns, partition)
    "spend_features": (KEY, FEATURE_COLUMNS, "months(snapshot_date)"),
    "spend_cohort_weekly": (COHORT_KEY, COHORT_COLUMNS, None),
    "spend_mix_weekly": (MIX_KEY, MIX_COLUMNS, None),
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db-prefix", default=DEFAULT_DB_PREFIX)
    p.add_argument("--as-of", default=None, help="default: the bronze CRM batch date")
    args, _ = p.parse_known_args(argv)
    return args


def week_end_of(d: date) -> date:
    """Friday closing d's Saturday-Friday week."""
    return d + timedelta(days=(4 - d.weekday()) % 7)


def snapshot_dates(as_of: date) -> list[date]:
    out, d = [], FIRST_SNAPSHOT
    while d <= as_of:
        out.append(d)
        d += timedelta(days=7)
    if not out or out[-1] != as_of:
        out.append(as_of)
    return out


def _share(num: Column, den: Column) -> Column:
    return F.round(F.when(den > 0, num / den).otherwise(F.lit(0.0)), 6)


def _col(name: str) -> str:
    return name.lower()


def categorised_events(spark: SparkSession, db_prefix: str, as_of: date) -> DataFrame:
    """spend_event up to as_of with its category (NULL off the merchant channels)."""
    cmap = spark.table(f"{db_prefix}_ref.merchant_category_map").select("merchant_key", "category")
    ev = spark.table(f"{db_prefix}_silver.spend_event").where(F.col("txn_date") <= F.lit(as_of))
    merchant = F.col("channel").isin(*MERCHANT_CHANNELS)
    return (ev.join(F.broadcast(cmap), "merchant_key", "left")
            .withColumn("category", F.when(merchant, F.coalesce("category", F.lit(UNCATEGORISED))))
            .withColumn("is_merchant", merchant))


def weekly_panel(spark: SparkSession, ev: DataFrame, customer: DataFrame, calendar: DataFrame,
                 as_of: date) -> DataFrame:
    """CIF x week (dense, first extract week to as_of's week) with 13- and 4-week window sums."""
    amt = F.col("amount")
    located = F.col("location_source") != HOME_FALLBACK
    sums = {"amt": amt, "merchant": F.when(F.col("is_merchant"), amt),
            "uncat": F.when(F.col("category") == UNCATEGORISED, amt),
            "home": F.when(F.col("is_home") == 1, amt),
            "located": F.when(located, amt), "located_away": F.when(located & (F.col("is_home") == 0), amt),
            "digital": F.when(F.col("channel").isin(*DIGITAL_CHANNELS), amt)}
    sums |= {f"cat_{_col(c)}": F.when(F.col("category") == c, amt) for c in CATEGORIES}
    sums |= {f"ch_{_col(c)}": F.when(F.col("channel") == c, amt) for c in CHANNELS}
    sums |= {f"tier{t}": F.when(F.col("event_city_tier") == t, amt) for t in TIERS}
    weekly = ev.groupBy("cif", "week_end").agg(F.count("*").alias("n"),
                                               *[F.sum(v).alias(k) for k, v in sums.items()])
    first, last = week_end_of(HISTORY_START), week_end_of(as_of)
    weeks = F.broadcast(spark.createDataFrame(
        [(first + timedelta(weeks=i),) for i in range((last - first).days // 7 + 1)], "week_end date")
        .join(calendar.select("week_end", "salary_week"), "week_end", "left"))
    grid = customer.select("cif").crossJoin(weeks)
    p = grid.join(weekly, ["cif", "week_end"], "left").fillna(0, subset=["n", *sums])
    p = (p.withColumn("salary", F.col("amt") * F.coalesce("salary_week", F.lit(0)))
         .withColumn("away_week", ((F.col("located") > 0) & (F.col("located_away") >= 0.5 * F.col("located")))
                     .cast("int")))

    w = Window.partitionBy("cif").orderBy("week_end")
    w13, w4 = w.rowsBetween(-(WINDOW_WEEKS - 1), 0), w.rowsBetween(-(RECENT_WEEKS - 1), 0)
    cols13 = ["n", "amt", "merchant", "uncat", "home", "digital", "salary",
              *[f"ch_{_col(c)}" for c in CHANNELS], *[f"tier{t}" for t in TIERS],
              *[f"cat_{_col(c)}" for c in CATEGORIES]]
    cols4 = ["n", "amt", "located", "located_away", "away_week", *[f"cat_{_col(c)}" for c in CATEGORIES]]
    return p.select("cif", "week_end",
                    *[F.sum(c).over(w13).alias(f"{c}_13") for c in cols13],
                    *[F.sum(c).over(w4).alias(f"{c}_4") for c in cols4])


def features(spark: SparkSession, panel: DataFrame, customer: DataFrame, as_of: date) -> DataFrame:
    snaps = snapshot_dates(as_of)
    snap_df = F.broadcast(spark.createDataFrame([(s, week_end_of(s)) for s in snaps],
                                                "snapshot_date date, week_end date"))
    f = (panel.join(snap_df, "week_end").join(customer, "cif")
         .where(F.col("onboarded_on") <= F.col("snapshot_date")))
    amt13, amt4 = F.col("amt_13"), F.col("amt_4")
    categorised = F.col("merchant_13") - F.col("uncat_13")
    trend = (F.when(amt13 > 0, F.least(F.lit(RATIO_CAP), (amt4 / RECENT_WEEKS) / (amt13 / WINDOW_WEEKS)))
             .when(amt4 > 0, F.lit(RATIO_CAP)).otherwise(F.lit(1.0)))
    out = f.select(
        "cif", "snapshot_date", *INFO,
        *[_share(F.col(f"cat_{_col(c)}_13"), categorised).alias(s) for c, s in zip(CATEGORIES, CAT_SHARES)],
        *[_share(F.col(f"ch_{_col(c)}_13"), amt13).alias(s) for c, s in zip(CHANNELS, CH_SHARES)],
        *[_share(F.col(f"tier{t}_13"), amt13).alias(s) for t, s in zip(TIERS, TIER_SHARES)],
        _share(F.col("home_13"), amt13).alias("home_share"),
        _share(F.col("digital_13"), amt13).alias("digital_share"),
        _share(F.col("uncat_13"), F.col("merchant_13")).alias("uncategorised_share"),
        _share(F.col("salary_13"), amt13).alias("salary_week_share"),
        F.round(amt13, 2).alias("spend_13w"),
        F.col("n_13").cast("int").alias("txn_13w"),
        F.round(F.when(F.col("n_13") > 0, amt13 / F.col("n_13")).otherwise(0.0), 2).alias("avg_ticket"),
        F.round(trend, 4).alias("spend_trend_4v13"),
        F.round(F.col("merchant_13"), 2).alias("merchant_spend_13w"),
        F.round(amt4, 2).alias("spend_4w"),
        F.col("n_4").cast("int").alias("txn_4w"),
        _share(F.col("located_away_4"), F.col("located_4")).alias("away_share_4w"),
        F.col("away_week_4").cast("int").alias("away_weeks_4w"),
        *[F.round(F.col(f"cat_{_col(c)}_4"), 2).alias(s) for c, s in zip(CATEGORIES, CAT_4W)])
    return out.select(*FEATURE_COLUMNS)


def complete_weeks(calendar: DataFrame, as_of: date) -> DataFrame:
    last = as_of if as_of.weekday() == 4 else week_end_of(as_of) - timedelta(weeks=1)
    return calendar.where(F.col("week_end").between(F.lit(FIRST_WEEK), F.lit(last)))


def cohort_weekly(spark: SparkSession, ev: DataFrame, customer: DataFrame, weeks: DataFrame) -> DataFrame:
    spend = (ev.where(F.col("category").isin(*CATEGORIES)).join(customer.select("cif", "cohort"), "cif")
             .groupBy("cohort", "category", "week_end")
             .agg(F.round(F.sum("amount"), 2).alias("amount_inr"), F.count("*").alias("n_txn"),
                  F.countDistinct("cif").alias("n_customers")))
    series = spark.createDataFrame([(f"{s}|T{t}", s, t, c) for s in SEGMENTS for t in TIERS for c in CATEGORIES],
                                   "cohort string, segment string, city_tier int, category string")
    grid = F.broadcast(series).crossJoin(F.broadcast(weeks.select("week_end", *COVARIATES)))
    out = (grid.join(spend, COHORT_KEY, "left").fillna(0.0, subset=["amount_inr"])
           .fillna(0, subset=["n_txn", "n_customers"]))
    return out.select(*[F.col(c).cast("int") if c in COVARIATES or c in ("n_txn", "n_customers") else F.col(c)
                        for c in COHORT_COLUMNS])


def mix_weekly(ev: DataFrame, customer: DataFrame, weeks: DataFrame) -> DataFrame:
    e = (ev.join(F.broadcast(weeks.select("week_end")), "week_end")
         .join(customer.select("cif", "age_band", "segment"), "cif"))
    unknown = F.lit("UNKNOWN")
    values = {"category": F.coalesce("category", F.lit(NON_MERCHANT)),
              "city_tier": F.coalesce(F.col("event_city_tier").cast("string"), unknown),
              "state": F.coalesce("event_state", unknown), "city": F.coalesce("event_city", unknown),
              "away_city": F.when(F.col("is_home") == 0, F.coalesce("event_city", unknown)),
              "age_band": F.coalesce("age_band", unknown), "segment": F.coalesce("segment", unknown),
              "location_source": F.coalesce("location_source", unknown)}
    pairs = F.array(*[F.struct(F.lit(d).alias("dim"), values[d].alias("dim_value")) for d in MIX_DIMENSIONS])
    return (e.select("cif", "week_end", "channel", "amount", F.explode(pairs).alias("p"))
            .where(F.col("p.dim_value").isNotNull())
            .groupBy("week_end", F.col("p.dim").alias("dim"), F.col("p.dim_value").alias("dim_value"), "channel")
            .agg(F.round(F.sum("amount"), 2).alias("amount_inr"), F.count("*").alias("n_txn"),
                 F.countDistinct("cif").alias("n_customers"))
            .select(*MIX_COLUMNS))


def build(spark: SparkSession, db_prefix: str, as_of: date | None = None) -> tuple[dict[str, DataFrame], date]:
    as_of = as_of or spark.table(f"{db_prefix}_bronze.crm_customer").agg(F.max("batch_as_of")).first()[0]
    customer = spark.table(f"{db_prefix}_silver.customer")
    calendar = spark.table(f"{db_prefix}_ref.calendar_weekly")
    ev = categorised_events(spark, db_prefix, as_of).cache()
    panel = weekly_panel(spark, ev, customer, calendar, as_of)
    weeks = complete_weeks(calendar, as_of)
    logger.info("as of %s: %d snapshots from %s", as_of, len(snapshot_dates(as_of)), FIRST_SNAPSHOT)
    return {"spend_features": features(spark, panel, customer, as_of),
            "spend_cohort_weekly": cohort_weekly(spark, ev, customer, weeks),
            "spend_mix_weekly": mix_weekly(ev, customer, weeks)}, as_of


def bootstrap_or_merge(spark: SparkSession, target: str, source_view: str, key: list[str], columns: list[str],
                       partition: str | None) -> None:
    if spark.catalog.tableExists(target):
        on = " AND ".join(f"t.{c} = s.{c}" for c in key)
        changed = " OR ".join(f"NOT (t.{c} <=> s.{c})" for c in columns if c not in key)
        spark.sql(f"""
            MERGE INTO {target} t
            USING {source_view} s
            ON {on}
            WHEN MATCHED AND ({changed}) THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
            WHEN NOT MATCHED BY SOURCE THEN DELETE
        """)
        logger.info("Merged into %s (new snapshot)", target)
    else:
        spark.sql(f"""
            CREATE TABLE {target}
            USING iceberg
            {f"PARTITIONED BY ({partition})" if partition else ""}
            TBLPROPERTIES ('format-version'='2')
            AS SELECT * FROM {source_view}
        """)
        logger.info("Bootstrapped %s (first run)", target)


def main(argv=None, spark: SparkSession | None = None) -> None:
    args = parse_args(argv)
    own = spark is None
    spark = spark or SparkSession.builder.appName("spend-build-gold-features").getOrCreate()
    try:
        gold_db = f"{args.db_prefix}_gold"
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {gold_db}")
        as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else None
        frames, as_of = build(spark, args.db_prefix, as_of)
        for name, df in frames.items():
            key, columns, partition = TABLES[name]
            df.withColumn("loaded_at", F.current_timestamp()).createOrReplaceTempView(f"{name}_src")
            bootstrap_or_merge(spark, f"{gold_db}.{name}", f"{name}_src", key, columns, partition)

        feats = spark.table(f"{gold_db}.spend_features")
        (feats.where(F.col("snapshot_date") >= F.lit(as_of - timedelta(weeks=4)))
         .groupBy("snapshot_date")
         .agg(F.count("*").alias("customers"), F.round(F.sum("spend_13w") / 1e7, 2).alias("spend_13w_cr"),
              F.round(F.avg("digital_share"), 4).alias("digital_share"),
              F.round(F.sum(F.col("uncategorised_share") * F.col("merchant_spend_13w"))
                      / F.sum("merchant_spend_13w"), 5).alias("uncategorised"))
         .orderBy(F.col("snapshot_date").desc()).show(truncate=False))
        for name in TABLES:
            logger.info("%s rows in %s.%s (as of %s)", spark.table(f"{gold_db}.{name}").count(), gold_db, name, as_of)
    finally:
        if own:
            spark.stop()


if __name__ == "__main__":
    main()
