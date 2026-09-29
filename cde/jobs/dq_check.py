"""
Data quality gate (Great Expectations) for one layer of the lakehouse.

Runs after each CDE stage in the DAG:

  generate -> dq bronze -> silver -> dq silver -> [CAI categorise] -> gold -> dq gold
           -> [CAI feature engineering -> model training -> model evaluation]

Every expectation carries a severity:
  critical  the load is wrong (null keys, future dates, orphans, unresolved
            events, shares that do not add up, raw card numbers...): the job
            exits 1 after recording results, so Airflow stops the DAG and
            yesterday's insights stay in place.
  warning   worth a look (unexpected MCC or channel code, volume swing, more
            spend than usual without a measured location): recorded, the
            pipeline continues.

All results, pass or fail, are appended to <prefix>_ref.dq_results (Iceberg),
one row per expectation, with the Iceberg snapshot id of the table checked, so
Impala / Hue / Cloudera Data Visualization can show quality per load and link
it to the exact data version.

Some checks span tables or compare with history (orphans, dedupe ratio,
reconciliation, share sums, volume change vs the previous load). Those are
computed in Spark as a small derived frame and validated with a standard
expectation, so the results table stays uniform.

Usage:
  spark-submit dq_check.py --layer bronze|silver|gold [--as-of YYYY-MM-DD]
                           [--db-prefix P] [--pipeline-run ID]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")
os.environ.setdefault("TQDM_DISABLE", "1")

import great_expectations as gx  # noqa: E402
import great_expectations.expectations as gxe  # noqa: E402
from pyspark.sql import DataFrame, SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dq_check")
for noisy in ("great_expectations", "py4j"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

DEFAULT_DB_PREFIX = "rsingh_spend_anl"
LAYERS = ("bronze", "silver", "gold")
CRITICAL, WARNING = "critical", "warning"
MAX_ORPHAN_RATE = 0.005
MAX_UNRESOLVED = 0.005
RECONCILE_TOLERANCE = 0.001
SHARE_TOLERANCE = 0.001
# Card e-commerce, net banking, UPI P2P and bill payments have no place, so ~48% of events carry the home
# city by design (PLAN decision 4); the warning fires when measured locations start to disappear.
MAX_HOME_FALLBACK = 0.55
MAX_UNCATEGORISED = 0.05
MAX_EVENTS_PER_DAY = 199
MIN_HISTORY_WEEKS = 52
CARD_NUMBER_REGEX = r"[0-9]{16}"
CHANNELS = ["CARD_POS", "CARD_ECOM", "UPI_P2M", "UPI_P2P", "NETBANKING", "ATM_CASH"]
MERCHANT_CHANNELS = ["CARD_POS", "CARD_ECOM", "UPI_P2M", "NETBANKING"]
ENTRY_MODES = ["CHIP", "CONTACTLESS", "MAGSTRIPE", "ECOM"]
BILLER_TYPES = ["UTILITY", "TELECOM", "LOAN", "EDUCATION", "MERCHANT"]
SEGMENTS = ["MASS", "AFFLUENT", "NRI", "SME_OWNER"]
CATEGORIES = ["GROCERY", "FOOD_DELIVERY", "DINING", "FUEL", "TRAVEL", "APPAREL", "ELECTRONICS", "UTILITIES",
              "TELECOM", "EMI_LOAN", "EDUCATION", "HEALTH", "ENTERTAINMENT", "OTHER"]
CAT_SHARES = [f"cat_{c.lower()}_share" for c in CATEGORIES]
CH_SHARES = [f"ch_{c.lower()}_share" for c in CHANNELS]
TIER_SHARES = [f"tier{t}_share" for t in (1, 2, 3, 4)]
OTHER_SHARES = ["home_share", "digital_share", "uncategorised_share", "salary_week_share", "away_share_4w"]

RESULT_SCHEMA = (
    "pipeline_run string, run_id string, run_ts timestamp, as_of date, layer string, "
    "table_name string, check_name string, expectation_type string, column_name string, "
    "severity string, success boolean, observed_value string, element_count bigint, "
    "unexpected_count bigint, unexpected_pct double, kwargs string, table_snapshot_id bigint, "
    "gx_version string"
)

# table -> (key columns, event date column)
BRONZE = {
    "crm_customer": (["cif"], "onboarded_on"),
    "cbs_accounts": (["account_id"], "opened_on"),
    "card_master": (["card_hash"], "issued_on"),
    "vpa_registry": (["vpa"], "linked_on"),
    "card_txn": (["txn_id"], "txn_ts"),
    "upi_txn": (["txn_id"], "txn_ts"),
    "nb_txn": (["txn_id"], "txn_ts"),
    "atm_txn": (["txn_id"], "txn_ts"),
}
TXN_TABLES = ["card_txn", "upi_txn", "nb_txn", "atm_txn"]
# free-text columns scanned for a raw 16-digit card number (card_hash is a hash and is not scanned)
TEXT_COLUMNS = {"crm_customer": ["customer_name"], "card_txn": ["merchant_name", "merchant_city"],
                "upi_txn": ["payer_vpa", "payee_vpa", "payee_name", "remarks"], "nb_txn": ["biller_name"]}
# silver table -> (key columns, bronze sources it deduplicates)
SILVER = {
    "customer": (["cif"], ["crm_customer"]),
    "account": (["account_id"], ["cbs_accounts"]),
    "spend_event": (["event_id"], TXN_TABLES),
}


@dataclass
class Check:
    table: str          # table the check is about (as shown in results)
    frame: str          # key of the Spark frame validated (a table or a derived frame)
    name: str           # human-readable check name
    expectation: object
    severity: str


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--layer", choices=LAYERS, required=True)
    p.add_argument("--as-of", default=None, help="load date (default: max batch_as_of in bronze.crm_customer)")
    p.add_argument("--db-prefix", default=DEFAULT_DB_PREFIX)
    p.add_argument("--pipeline-run", default=None, help="groups the layers of one DAG run (Airflow run_id)")
    args, _ = p.parse_known_args(argv)
    return args


# ---------------------------------------------------------------- checks

def _days_after(col, as_of: date):
    col = F.col(col) if isinstance(col, str) else col
    return F.datediff(F.to_date(col), F.lit(as_of))


def _ratio_frame(spark: SparkSession, value: float | None, column: str) -> DataFrame:
    return spark.createDataFrame([(float(value) if value is not None else None,)], f"{column} double")


def _orphans(child: DataFrame, key: str, parent: DataFrame, parent_key: str | None = None) -> DataFrame:
    ids = parent.select(F.col(parent_key or key).alias(key)).distinct().withColumn("known", F.lit(1))
    return (child.select(key).join(ids, key, "left")
            .select(F.when(F.col("known").isNull(), 1).otherwise(0).alias("orphan")))


def _unique(keys: list[str], **kw):
    return (gxe.ExpectCompoundColumnsToBeUnique(column_list=keys, **kw) if len(keys) > 1
            else gxe.ExpectColumnValuesToBeUnique(column=keys[0], **kw))


def _share_sum(df: DataFrame, columns: list[str], where) -> DataFrame:
    total = sum((F.coalesce(F.col(c), F.lit(0.0)) for c in columns), F.lit(0.0))
    return df.where(where).select(F.abs(total - F.lit(1.0)).alias("share_sum_error"))


def _sum_within(frame: str, table: str, name: str, column: str = "share_sum_error") -> Check:
    return Check(table, frame, name, gxe.ExpectColumnValuesToBeBetween(column=column, min_value=0,
                                                                      max_value=SHARE_TOLERANCE), CRITICAL)


def bronze_checks(spark: SparkSession, db: str, as_of: date, previous: dict) -> tuple[dict, list[Check]]:
    frames, checks = {}, []
    mccs = [r.mcc for r in spark.table(f"{db}_ref.mcc_category").select("mcc").collect()]
    for name, (keys, date_col) in BRONZE.items():
        df = (spark.table(f"{db}_bronze.{name}")
              .withColumn("days_after_as_of", _days_after(date_col, as_of))
              .withColumn("batch_days_from_as_of", _days_after("batch_as_of", as_of)))
        frames[name] = df
        checks += [
            Check(name, name, "row count", gxe.ExpectTableRowCountToBeBetween(min_value=1), CRITICAL),
            Check(name, name, "latest event not after the load date",
                  gxe.ExpectColumnMaxToBeBetween(column="days_after_as_of", max_value=0), CRITICAL),
            Check(name, name, "batch is for the load date",
                  gxe.ExpectColumnValuesToBeBetween(column="batch_days_from_as_of", min_value=0, max_value=0),
                  CRITICAL),
            Check(name, name, "re-sent duplicates below 1% (removed in silver)", _unique(keys, mostly=0.99),
                  WARNING),
        ]
        checks += [Check(name, name, f"{k} not null", gxe.ExpectColumnValuesToNotBeNull(column=k), CRITICAL)
                   for k in keys]
        prev = previous.get((name, "row count"))
        if prev:
            frames[f"{name}[volume]"] = _ratio_frame(spark, df.count() / prev, "rows_vs_previous_load")
            checks.append(Check(name, f"{name}[volume]", "volume within -20% / +25% of the previous load",
                                gxe.ExpectColumnValuesToBeBetween(column="rows_vs_previous_load",
                                                                  min_value=0.8, max_value=1.25), WARNING))
    for name in TXN_TABLES:
        checks.append(Check(name, name, "amount positive",
                            gxe.ExpectColumnValuesToBeBetween(column="amount", min_value=0.01), CRITICAL))
    for name, cols in TEXT_COLUMNS.items():
        checks += [Check(name, name, f"no raw 16-digit card number in {c}",
                         gxe.ExpectColumnValuesToNotMatchRegex(column=c, regex=CARD_NUMBER_REGEX), CRITICAL)
                   for c in cols]

    crm, acct, cards, vpas = (frames[t] for t in ("crm_customer", "cbs_accounts", "card_master", "vpa_registry"))
    card, upi, nb, atm = (frames[t] for t in TXN_TABLES)
    frames["atm_txn[hundreds]"] = atm.select(((F.col("amount") % 100) == 0).alias("multiple_of_100"))
    checks += [
        Check("atm_txn", "atm_txn[hundreds]", "ATM amounts are multiples of 100",
              gxe.ExpectColumnValuesToBeInSet(column="multiple_of_100", value_set=[True]), WARNING),
        Check("card_txn", "card_txn", "mcc in ref.mcc_category",
              gxe.ExpectColumnValuesToBeInSet(column="mcc", value_set=mccs, mostly=0.99), WARNING),
        Check("card_txn", "card_txn", "entry_mode in the allowed set",
              gxe.ExpectColumnValuesToBeInSet(column="entry_mode", value_set=ENTRY_MODES), WARNING),
        Check("upi_txn", "upi_txn", "txn_type is P2P or P2M",
              gxe.ExpectColumnValuesToBeInSet(column="txn_type", value_set=["P2P", "P2M"]), WARNING),
        Check("nb_txn", "nb_txn", "biller_type known",
              gxe.ExpectColumnValuesToBeInSet(column="biller_type", value_set=BILLER_TYPES), WARNING),
        Check("crm_customer", "crm_customer", "segment known",
              gxe.ExpectColumnValuesToBeInSet(column="segment", value_set=SEGMENTS), WARNING),
        Check("cbs_accounts", "cbs_accounts", "status is ACTIVE or CLOSED",
              gxe.ExpectColumnValuesToBeInSet(column="status", value_set=["ACTIVE", "CLOSED"]), CRITICAL),
    ]

    geo = spark.table(f"{db}_ref.pincode_geo")
    orphan_checks = [
        ("crm_customer", crm, "home_pincode", geo, "pincode", "home pincode in ref.pincode_geo", 0.0),
        ("cbs_accounts", acct, "cif", crm, None, "accounts for unknown CIFs below 0.5%", MAX_ORPHAN_RATE),
        ("card_master", cards, "account_id", acct, None, "cards for unknown accounts below 0.5%", MAX_ORPHAN_RATE),
        ("vpa_registry", vpas, "account_id", acct, None, "VPAs for unknown accounts below 0.5%", MAX_ORPHAN_RATE),
        ("card_txn", card, "card_hash", cards, None, "card_hash in card_master: orphans below 0.5%", MAX_ORPHAN_RATE),
        ("upi_txn", upi, "payer_vpa", vpas, "vpa", "payer_vpa in vpa_registry: orphans below 0.5%", MAX_ORPHAN_RATE),
        ("nb_txn", nb, "cif", crm, None, "net banking CIF in crm_customer: orphans below 0.5%", MAX_ORPHAN_RATE),
        ("atm_txn", atm, "card_hash", cards, None, "card_hash in card_master: orphans below 0.5%", MAX_ORPHAN_RATE),
        ("atm_txn", atm, "atm_id", spark.table(f"{db}_ref.atm_location"), None,
         "atm_id in ref.atm_location: orphans below 0.5%", MAX_ORPHAN_RATE),
    ]
    for table, child, key, parent, parent_key, name, limit in orphan_checks:
        frame = f"{table}[orphans {key}]"
        frames[frame] = _orphans(child, key, parent, parent_key)
        checks.append(Check(table, frame, name,
                            gxe.ExpectColumnMeanToBeBetween(column="orphan", max_value=limit), CRITICAL))
    return frames, checks


def _bronze_resolved_totals(spark: SparkSession, db: str) -> DataFrame:
    """Per source, the bronze amount that resolves to a known CIF, recomputed from bronze independently of
    build_silver.py (duplicates dropped on the transaction id)."""
    b = f"{db}_bronze"
    acct = spark.table(f"{b}.cbs_accounts").select("account_id", "cif").dropDuplicates(["account_id"])
    cards = spark.table(f"{b}.card_master").select("card_hash", "account_id").dropDuplicates(["card_hash"])
    vpas = spark.table(f"{b}.vpa_registry").select(F.col("vpa").alias("payer_vpa"), "account_id") \
        .dropDuplicates(["payer_vpa"])
    crm = spark.table(f"{b}.crm_customer").select("cif").distinct()
    card_cif = cards.join(acct, "account_id").select("card_hash", "cif")

    def total(name: str, joined: DataFrame, source: str) -> DataFrame:
        return joined.agg(F.sum("amount").alias("bronze_amount")).select(F.lit(source).alias("source"),
                                                                         "bronze_amount")
    txn = {t: spark.table(f"{b}.{t}").dropDuplicates(["txn_id"]) for t in TXN_TABLES}
    return (total("card_txn", txn["card_txn"].join(card_cif, "card_hash"), "CARD")
            .unionByName(total("upi_txn", txn["upi_txn"].join(vpas.join(acct, "account_id"), "payer_vpa"), "UPI"))
            .unionByName(total("nb_txn", txn["nb_txn"].join(crm, "cif"), "NB"))
            .unionByName(total("atm_txn", txn["atm_txn"].join(card_cif, "card_hash"), "ATM")))


def silver_checks(spark: SparkSession, db: str, as_of: date, previous: dict) -> tuple[dict, list[Check]]:
    frames, checks = {}, []
    for name, (keys, sources) in SILVER.items():
        df = spark.table(f"{db}_silver.{name}")
        frames[name] = df
        checks.append(Check(name, name, "row count", gxe.ExpectTableRowCountToBeBetween(min_value=1), CRITICAL))
        checks.append(Check(name, name, "key unique after dedupe", _unique(keys), CRITICAL))
        bronze_rows = sum(spark.table(f"{db}_bronze.{s}").count() for s in sources)
        frames[f"{name}[dedupe]"] = _ratio_frame(spark, df.count() / max(bronze_rows, 1), "silver_to_bronze_rows")
        checks.append(Check(name, f"{name}[dedupe]", "dedupe removed less than 2% of bronze rows",
                            gxe.ExpectColumnValuesToBeBetween(column="silver_to_bronze_rows",
                                                              min_value=0.98, max_value=1.0), CRITICAL))

    cust, ev = frames["customer"], frames["spend_event"]
    ev = ev.withColumn("days_after_as_of", _days_after("txn_date", as_of))
    frames["spend_event"] = ev
    frames["spend_event[resolved]"] = ev.select(F.col("cif").isNull().cast("int").alias("unresolved"))
    frames["spend_event[fallback]"] = ev.select(
        (F.col("location_source") == "home_fallback").cast("int").alias("home_fallback"))
    frames["spend_event[merchant]"] = ev.where(F.col("channel").isin(*MERCHANT_CHANNELS))
    frames["spend_event[no merchant]"] = ev.where(~F.col("channel").isin(*MERCHANT_CHANNELS))
    frames["spend_event[per day]"] = (ev.where(F.col("cif").isNotNull()).groupBy("cif", "txn_date")
                                      .agg(F.count("*").alias("events_per_day")))
    frames["spend_event[week_end]"] = ev.select(
        ((F.dayofweek("week_end") == 6) & (F.datediff("week_end", "txn_date").between(0, 6))).alias("friday_ok"))
    frames["account[orphans]"] = _orphans(frames["account"], "cif", cust)
    checks += [
        Check("spend_event", "spend_event[resolved]", "spend_event.cif resolved: unresolved below 0.5%",
              gxe.ExpectColumnMeanToBeBetween(column="unresolved", max_value=MAX_UNRESOLVED), CRITICAL),
        Check("spend_event", "spend_event", "channel in the six allowed values",
              gxe.ExpectColumnValuesToBeInSet(column="channel", value_set=CHANNELS), CRITICAL),
        Check("spend_event", "spend_event", "event city tier in 1-4",
              gxe.ExpectColumnValuesToBeInSet(column="event_city_tier", value_set=[1, 2, 3, 4], mostly=0.995),
              CRITICAL),
        Check("spend_event", "spend_event[fallback]",
              f"home_fallback location share below {MAX_HOME_FALLBACK:.0%} (PLAN decision 4)",
              gxe.ExpectColumnMeanToBeBetween(column="home_fallback", max_value=MAX_HOME_FALLBACK), WARNING),
        Check("spend_event", "spend_event[per day]", "events per CIF per day below 200 (bot / duplicate guard)",
              gxe.ExpectColumnValuesToBeBetween(column="events_per_day", max_value=MAX_EVENTS_PER_DAY), WARNING),
        Check("spend_event", "spend_event", "amount positive",
              gxe.ExpectColumnValuesToBeBetween(column="amount", min_value=0.01), CRITICAL),
        Check("spend_event", "spend_event", "no event after the load date",
              gxe.ExpectColumnMaxToBeBetween(column="days_after_as_of", max_value=0), CRITICAL),
        Check("spend_event", "spend_event[week_end]", "week_end is the Friday closing the event's week",
              gxe.ExpectColumnValuesToBeInSet(column="friday_ok", value_set=[True]), CRITICAL),
        Check("spend_event", "spend_event[merchant]", "merchant_key present for card, UPI P2M and net banking",
              gxe.ExpectColumnValuesToNotBeNull(column="merchant_key", mostly=0.999), CRITICAL),
        Check("spend_event", "spend_event[no merchant]", "no merchant_key for ATM and UPI P2P",
              gxe.ExpectColumnValuesToBeNull(column="merchant_key"), CRITICAL),
        Check("customer", "customer", "home city tier in 1-4",
              gxe.ExpectColumnValuesToBeInSet(column="city_tier", value_set=[1, 2, 3, 4]), CRITICAL),
        Check("customer", "customer", "cohort not null", gxe.ExpectColumnValuesToNotBeNull(column="cohort"),
              CRITICAL),
        Check("account", "account[orphans]", "accounts for unknown CIFs below 0.5%",
              gxe.ExpectColumnMeanToBeBetween(column="orphan", max_value=MAX_ORPHAN_RATE), CRITICAL),
    ]

    silver_total = (ev.where(F.col("cif").isNotNull()).groupBy("source")
                    .agg(F.sum("amount").alias("silver_amount")))
    frames["spend_event[reconcile]"] = _bronze_resolved_totals(spark, db).join(silver_total, "source", "full") \
        .select((F.abs(F.coalesce("silver_amount", F.lit(0.0)) - F.coalesce("bronze_amount", F.lit(0.0)))
                 / F.greatest(F.abs("bronze_amount"), F.lit(1.0))).alias("relative_difference"))
    checks.append(Check("spend_event", "spend_event[reconcile]",
                        "resolved silver amounts match resolved bronze amounts within 0.1% per source",
                        gxe.ExpectColumnValuesToBeBetween(column="relative_difference", min_value=0,
                                                          max_value=RECONCILE_TOLERANCE), CRITICAL))
    return frames, checks


def gold_checks(spark: SparkSession, db: str, as_of: date, previous: dict) -> tuple[dict, list[Check]]:
    t, c, m = "spend_features", "spend_cohort_weekly", "spend_mix_weekly"
    feat = spark.table(f"{db}_gold.{t}").withColumn("days_after_as_of", _days_after("snapshot_date", as_of))
    today = feat.where(F.col("snapshot_date") == F.lit(as_of))
    cohort = spark.table(f"{db}_gold.{c}")
    mix = spark.table(f"{db}_gold.{m}")
    shares = CAT_SHARES + CH_SHARES + TIER_SHARES + OTHER_SHARES
    frames = {
        t: feat, f"{t}[today]": today,
        f"{t}[category sum]": _share_sum(feat, CAT_SHARES, F.col("merchant_spend_13w") * (
            1 - F.col("uncategorised_share")) > 0),
        f"{t}[channel sum]": _share_sum(feat, CH_SHARES, F.col("spend_13w") > 0),
        f"{t}[tier sum]": _share_sum(feat, TIER_SHARES, F.col("spend_13w") > 0),
        f"{t}[share range]": feat.select(F.least(*[F.coalesce(F.col(s), F.lit(0.0)) for s in shares]).alias("lo"),
                                         F.greatest(*[F.coalesce(F.col(s), F.lit(0.0)) for s in shares]).alias("hi")),
        f"{t}[snapshot day]": feat.select(((F.dayofweek("snapshot_date") == 6)
                                           | (F.col("snapshot_date") == F.lit(as_of))).alias("friday_or_as_of")),
        c: cohort,
        m: mix,
    }
    uncat = today.agg((F.sum(F.col("uncategorised_share") * F.col("merchant_spend_13w"))
                       / F.sum("merchant_spend_13w")).alias("u")).first()["u"]
    frames[f"{t}[uncategorised]"] = _ratio_frame(spark, uncat, "uncategorised_share_of_spend")
    checks = [
        Check(t, t, "row count", gxe.ExpectTableRowCountToBeBetween(min_value=1), CRITICAL),
        Check(t, t, "one row per CIF per snapshot", _unique(["cif", "snapshot_date"]), CRITICAL),
        Check(t, t, "snapshot not after the load date",
              gxe.ExpectColumnMaxToBeBetween(column="days_after_as_of", max_value=0), CRITICAL),
        Check(t, f"{t}[today]", "today's snapshot present", gxe.ExpectTableRowCountToBeBetween(min_value=1),
              CRITICAL),
        Check(t, f"{t}[snapshot day]", "snapshots are Fridays plus the as-of date",
              gxe.ExpectColumnValuesToBeInSet(column="friday_or_as_of", value_set=[True]), CRITICAL),
        _sum_within(f"{t}[category sum]", t, "category shares sum to 1 for customers with categorised spend"),
        _sum_within(f"{t}[channel sum]", t, "channel shares sum to 1 for customers with spend"),
        _sum_within(f"{t}[tier sum]", t, "city-tier shares sum to 1 for customers with spend"),
        Check(t, f"{t}[share range]", "every share within 0-1",
              gxe.ExpectColumnValuesToBeBetween(column="lo", min_value=0), CRITICAL),
        Check(t, f"{t}[share range]", "every share at most 1",
              gxe.ExpectColumnValuesToBeBetween(column="hi", max_value=1 + SHARE_TOLERANCE), CRITICAL),
        Check(t, f"{t}[uncategorised]", f"UNCATEGORISED share of merchant spend below {MAX_UNCATEGORISED:.0%}",
              gxe.ExpectColumnValuesToBeBetween(column="uncategorised_share_of_spend", min_value=0,
                                                max_value=MAX_UNCATEGORISED), CRITICAL),
        Check(t, t, "spend_13w not negative", gxe.ExpectColumnValuesToBeBetween(column="spend_13w", min_value=0),
              CRITICAL),
        Check(t, t, "txn_13w not negative", gxe.ExpectColumnValuesToBeBetween(column="txn_13w", min_value=0),
              CRITICAL),
    ]
    checks += [Check(t, t, f"{col} not null", gxe.ExpectColumnValuesToNotBeNull(column=col), CRITICAL)
               for col in ("segment", "city_tier", "cohort", "home_city")]

    last_week = cohort.where(F.col("week_end") <= F.lit(as_of)).agg(F.max("week_end")).first()[0]
    series = (cohort.where(F.col("week_end") <= F.lit(as_of)).groupBy("cohort", "category")
              .agg(F.countDistinct("week_end").alias("history_weeks")))
    frames[f"{c}[history]"] = series
    frames[f"{c}[series]"] = _ratio_frame(spark, series.count(), "n_series")
    checks += [
        Check(c, c, "row count", gxe.ExpectTableRowCountToBeBetween(min_value=1), CRITICAL),
        Check(c, c, "one row per cohort x category x week", _unique(["cohort", "category", "week_end"]), CRITICAL),
        Check(c, c, "amount not negative", gxe.ExpectColumnValuesToBeBetween(column="amount_inr", min_value=0),
              CRITICAL),
        Check(c, c, "category is one of the 14", gxe.ExpectColumnValuesToBeInSet(column="category",
                                                                                 value_set=CATEGORIES), CRITICAL),
        Check(c, f"{c}[history]", f"every cohort series has at least {MIN_HISTORY_WEEKS} weeks (Chronos-2 context)",
              gxe.ExpectColumnValuesToBeBetween(column="history_weeks", min_value=MIN_HISTORY_WEEKS), WARNING),
        Check(c, f"{c}[series]", "16 cohorts x 14 categories present",
              gxe.ExpectColumnValuesToBeBetween(column="n_series", min_value=224, max_value=224), WARNING),
    ]
    if last_week is not None:
        totals = {r["week_end"]: r["s"] for r in cohort.where(F.col("week_end").isin(
            last_week, last_week - timedelta(weeks=52))).groupBy("week_end").agg(F.sum("amount_inr").alias("s"))
            .collect()}
        year_ago = totals.get(last_week - timedelta(weeks=52))
        if year_ago:
            frames[f"{c}[yoy]"] = _ratio_frame(spark, totals[last_week] / year_ago, "week_vs_same_week_last_year")
            checks.append(Check(c, f"{c}[yoy]", "total weekly spend within +-30% of the same week last year",
                                gxe.ExpectColumnValuesToBeBetween(column="week_vs_same_week_last_year",
                                                                  min_value=0.7, max_value=1.3), WARNING))

    cat_mix = (mix.where((F.col("dim") == "category") & F.col("dim_value").isin(*CATEGORIES)).groupBy("week_end")
               .agg(F.sum("amount_inr").alias("mix_amount")))
    cat_cohort = cohort.groupBy("week_end").agg(F.sum("amount_inr").alias("cohort_amount"))
    frames[f"{m}[reconcile]"] = (cat_mix.join(cat_cohort, "week_end", "inner").select(
        (F.abs(F.col("mix_amount") - F.col("cohort_amount")) / F.greatest(F.col("cohort_amount"), F.lit(1.0)))
        .alias("relative_difference")))
    checks += [
        Check(m, m, "row count", gxe.ExpectTableRowCountToBeBetween(min_value=1), CRITICAL),
        Check(m, m, "one row per week x dimension x value x channel",
              _unique(["week_end", "dim", "dim_value", "channel"]), CRITICAL),
        Check(m, f"{m}[reconcile]", "category mix matches the cohort series within 0.1% per week",
              gxe.ExpectColumnValuesToBeBetween(column="relative_difference", min_value=0,
                                                max_value=RECONCILE_TOLERANCE), CRITICAL),
    ]
    prev = previous.get((t, "row count"))
    if prev:
        frames[f"{t}[volume]"] = _ratio_frame(spark, feat.count() / prev, "rows_vs_previous_load")
        checks.append(Check(t, f"{t}[volume]", "rows within -5% / +10% of the previous load",
                            gxe.ExpectColumnValuesToBeBetween(column="rows_vs_previous_load",
                                                              min_value=0.95, max_value=1.10), WARNING))
    return frames, checks


LAYER_CHECKS = {"bronze": bronze_checks, "silver": silver_checks, "gold": gold_checks}


# ---------------------------------------------------------------- running

def _exception(result) -> str | None:
    info = result.exception_info or {}
    if isinstance(info, dict) and "raised_exception" not in info:   # keyed by metric id
        info = next((v for v in info.values() if isinstance(v, dict) and v.get("raised_exception")), {})
    return info.get("exception_message") if isinstance(info, dict) and info.get("raised_exception") else None


def _json(value) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value, default=str)


def run_checks(frames: dict[str, DataFrame], checks: list[Check]) -> list[dict]:
    """Validate each frame with its expectations; one result dict per check."""
    ctx = gx.get_context(mode="ephemeral")
    source = ctx.data_sources.add_spark("spark")
    out = []
    for frame in dict.fromkeys(c.frame for c in checks):
        mine = [c for c in checks if c.frame == frame]
        asset = source.add_dataframe_asset(f"frame_{len(out)}_{abs(hash(frame))}")
        batch = asset.add_batch_definition_whole_dataframe("all").get_batch(
            batch_parameters={"dataframe": frames[frame]})
        for c in mine:
            c.expectation.meta = {"check": c.name, "severity": c.severity, "table": c.table}
        suite = ctx.suites.add(gx.ExpectationSuite(name=f"suite_{len(out)}_{abs(hash(frame))}",
                                                   expectations=[c.expectation for c in mine]))
        res = batch.validate(suite, result_format="SUMMARY")
        by_name = {c.name: c for c in mine}
        for r in res.results:
            cfg = r.expectation_config
            c = by_name[cfg.meta["check"]]
            res_dict = r.result or {}
            error = _exception(r)
            kwargs = {k: v for k, v in cfg.kwargs.items() if k not in ("batch_id",)}
            out.append({
                "table_name": c.table, "check_name": c.name, "expectation_type": cfg.type,
                "column_name": kwargs.get("column") or ",".join(kwargs.get("column_list", []) or []) or None,
                "severity": c.severity, "success": bool(r.success) and error is None,
                "observed_value": f"ERROR: {error}"[:500] if error else _json(res_dict.get("observed_value")),
                "element_count": res_dict.get("element_count"),
                "unexpected_count": res_dict.get("unexpected_count"),
                "unexpected_pct": res_dict.get("unexpected_percent"),
                "kwargs": _json(kwargs),
            })
    return out


def _snapshot_id(spark: SparkSession, table: str) -> int | None:
    try:
        row = spark.sql(f"SELECT snapshot_id FROM {table}.snapshots ORDER BY committed_at DESC LIMIT 1").first()
        return int(row[0]) if row else None
    except Exception:  # lineage is best effort
        return None


def _previous(spark: SparkSession, results_table: str, layer: str, as_of: date) -> dict:
    """Last recorded row counts per table from earlier loads, for volume checks."""
    if not spark.catalog.tableExists(results_table):
        return {}
    rows = spark.sql(f"""
        SELECT table_name, check_name, observed_value FROM (
          SELECT table_name, check_name, observed_value,
                 ROW_NUMBER() OVER (PARTITION BY table_name, check_name ORDER BY as_of DESC, run_ts DESC) rn
          FROM {results_table}
          WHERE layer = '{layer}' AND check_name = 'row count' AND as_of < DATE '{as_of.isoformat()}'
            AND observed_value IS NOT NULL AND observed_value NOT LIKE 'ERROR%')
        WHERE rn = 1""").collect()
    return {(r.table_name, r.check_name): float(r.observed_value) for r in rows}


def write_results(spark: SparkSession, results_table: str, rows: list[dict]) -> None:
    df = spark.createDataFrame(rows, RESULT_SCHEMA)
    if spark.catalog.tableExists(results_table):
        df.writeTo(results_table).append()
    else:
        (df.writeTo(results_table).using("iceberg").tableProperty("format-version", "2")
         .partitionedBy(F.months("as_of")).create())


def resolve_as_of(spark: SparkSession, db: str, as_of: str | None) -> date:
    if as_of:
        return datetime.strptime(as_of, "%Y-%m-%d").date()
    return spark.table(f"{db}_bronze.crm_customer").agg(F.max("batch_as_of")).first()[0]


def run(spark: SparkSession, args: argparse.Namespace) -> list[dict]:
    db, layer = args.db_prefix, args.layer
    as_of = resolve_as_of(spark, db, args.as_of)
    results_table = f"{db}_ref.dq_results"
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db}_ref")
    frames, checks = LAYER_CHECKS[layer](spark, db, as_of, _previous(spark, results_table, layer, as_of))
    logger.info("%s layer as of %s: %d checks on %d frames", layer, as_of, len(checks), len(frames))

    results = run_checks(frames, checks)
    run_ts = datetime.now(timezone.utc).replace(tzinfo=None)
    run_id = f"{layer}-{as_of:%Y%m%d}-{uuid.uuid4().hex[:8]}"
    snapshot = {name: _snapshot_id(spark, f"{db}_{layer}.{name}") for name in {r["table_name"] for r in results}}
    for r in results:
        r.update(pipeline_run=args.pipeline_run or f"manual-{run_ts:%Y%m%dT%H%M%S}", run_id=run_id, run_ts=run_ts,
                 as_of=as_of, layer=layer, table_snapshot_id=snapshot.get(r["table_name"]),
                 gx_version=gx.__version__)
    write_results(spark, results_table, results)

    failed = [r for r in results if not r["success"]]
    for r in results:
        logger.info("%-4s %-8s %-20s %-55s %s", "PASS" if r["success"] else "FAIL", r["severity"], r["table_name"],
                    r["check_name"], r["observed_value"] if r["observed_value"] is not None
                    else f"{r['unexpected_count']} unexpected of {r['element_count']}")
    logger.info("%s: %d checks, %d failed (%d critical); results in %s (run %s)", layer, len(results), len(failed),
                sum(r["severity"] == CRITICAL for r in failed), results_table, run_id)
    return results


def main(argv=None, spark: SparkSession | None = None) -> None:
    args = parse_args(argv)
    own = spark is None
    spark = spark or SparkSession.builder.appName(f"spend-dq-{args.layer}").getOrCreate()
    try:
        results = run(spark, args)
    finally:
        if own:
            spark.stop()
    critical = [r for r in results if not r["success"] and r["severity"] == CRITICAL]
    if critical:
        for r in critical:
            seen = r["observed_value"] if r["observed_value"] is not None else f"{r['unexpected_count']} unexpected"
            logger.error("DATA QUALITY GATE FAILED: %s.%s: %s", r["table_name"], r["check_name"], seen)
        sys.exit(1)
    logger.info("%s data quality gate passed", args.layer)


if __name__ == "__main__":
    main()
