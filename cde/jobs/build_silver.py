"""
Stage 3 - Silver (one conformed spend event stream, keyed by CIF)

Cleans the bronze extracts: removes re-sent duplicates (latest ingested record
wins), resolves every source to the CIF, and turns the four transaction feeds
into one silver.spend_event table with a channel, a location and a merchant key.

CIF resolution:
  crm_customer, nb_txn   carry the CIF
  card_txn, atm_txn      card_hash -> card_master.account_id -> cbs_accounts.cif
  upi_txn                payer_vpa -> vpa_registry.account_id -> cbs_accounts.cif
An event that does not resolve keeps cif NULL (dq_silver gates the share).

Channels: CARD_POS, CARD_ECOM (entry_mode ECOM), UPI_P2M, UPI_P2P, NETBANKING,
ATM_CASH.

Where the money was spent (event_city_code, location_source):
  card_city      card POS: the merchant city on the card switch record
  upi_pincode    UPI P2M whose QR carried a pincode
  atm            the ATM's pincode (ref.atm_location)
  home_fallback  card e-commerce, net banking, UPI P2P and UPI P2M without a
                 pincode: the customer's home city, flagged so nobody mistakes
                 it for a measured location

merchant_key is the payee / merchant / biller text normalised exactly as
spend.categorise.normalise() does (lower case, separators to spaces, 3+ digit
reference numbers and company suffixes removed), so the CAI categorise job and
the gold join agree on the key. NULL for ATM and UPI P2P (no merchant).

week_end is the Friday closing the Saturday-Friday week of the event.

Reads:
  <prefix>_bronze.{crm_customer, cbs_accounts, card_master, card_txn, vpa_registry,
                   upi_txn, nb_txn, atm_txn}
  <prefix>_ref.{pincode_geo, atm_location}
Writes (drop + recreate every run):
  <prefix>_silver.{customer, account, spend_event}

Usage:
  spark-submit build_silver.py [--db-prefix P]
"""

from __future__ import annotations

import argparse
import logging

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DB_PREFIX = "rsingh_spend_anl"
TABLES = ["customer", "account", "spend_event"]
EVENT_COLUMNS = ["event_id", "source", "cif", "account_id", "txn_ts", "txn_date", "week_end", "amount", "channel",
                 "merchant_text", "merchant_key", "mcc", "event_city_code", "event_city", "event_state",
                 "event_city_tier", "location_source", "home_city_code", "is_home"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db-prefix", default=DEFAULT_DB_PREFIX)
    args, _ = p.parse_known_args(argv)
    return args


def dedupe(df: DataFrame, *keys: str) -> DataFrame:
    w = Window.partitionBy(*keys).orderBy(F.col("ingested_at").desc())
    return df.withColumn("_rn", F.row_number().over(w)).where("_rn = 1").drop("_rn")


def _write(df: DataFrame, name: str, partition_col=None) -> None:
    w = df.writeTo(name).using("iceberg").tableProperty("format-version", "2")
    if partition_col is not None:
        w = w.partitionedBy(partition_col)
    w.createOrReplace()


def _months(col: str):
    try:
        from pyspark.sql.functions import partitioning   # Spark 4
        return partitioning.months(col)
    except ImportError:                                  # Spark 3.5 on CDE
        return F.months(col)


def merchant_key(col: Column) -> Column:
    """Spark mirror of spend.categorise.normalise(); tests/test_cde_jobs.py checks they agree."""
    t = F.lower(col)
    t = F.regexp_replace(t, r"[*#/_.\-]", " ")
    t = F.regexp_replace(t, r"\b\d{3,}\b", " ")
    t = F.regexp_replace(t, r"\b(pvt|ltd|private|limited|india|in)\b", " ")
    t = F.trim(F.regexp_replace(t, r"\s+", " "))
    return F.when(t == "", None).otherwise(t)


def week_end(date_col: Column) -> Column:
    """Friday closing the Saturday-Friday week (dayofweek: Sunday = 1 ... Friday = 6, Saturday = 7)."""
    return F.date_add(date_col, (F.lit(13) - F.dayofweek(date_col)) % 7)


def conform_customer(crm: DataFrame, geo: DataFrame) -> DataFrame:
    home = geo.select(F.col("pincode").alias("home_pincode"), F.col("city_code").alias("home_city_code"),
                      F.col("city").alias("home_city"), F.col("state").alias("home_state"), "city_tier")
    return (dedupe(crm, "cif").join(home, "home_pincode", "left")
            .select("cif", "home_pincode", "home_city_code", "home_city", "home_state", "city_tier", "age_band",
                    "income_band", "segment", "onboarded_on",
                    F.concat_ws("|", "segment", F.concat(F.lit("T"), F.col("city_tier"))).alias("cohort")))


def spend_events(bronze: dict[str, DataFrame], account: DataFrame, customer: DataFrame, geo: DataFrame,
                 atms: DataFrame) -> DataFrame:
    acct_cif = account.select("account_id", "cif")
    card_cif = dedupe(bronze["card_master"], "card_hash").select("card_hash", "account_id").join(
        acct_cif, "account_id", "left")
    vpa_cif = dedupe(bronze["vpa_registry"], "vpa").select(F.col("vpa").alias("payer_vpa"), "account_id").join(
        acct_cif, "account_id", "left")
    city_by_name = geo.groupBy(F.upper("city").alias("merchant_city")).agg(F.first("city_code").alias("m_code"))
    city_by_pin = geo.select(F.col("pincode").alias("payee_pincode"), F.col("city_code").alias("p_code"))
    atm_city = atms.select("atm_id", "pincode").join(
        geo.select("pincode", F.col("city_code").alias("a_code")), "pincode", "left").drop("pincode")
    null_s, null_i = F.lit(None).cast("string"), F.lit(None).cast("int")

    card = (dedupe(bronze["card_txn"], "txn_id").join(card_cif, "card_hash", "left")
            .join(city_by_name, "merchant_city", "left")
            .select(F.col("txn_id").alias("event_id"), F.lit("CARD").alias("source"), "cif", "account_id",
                    "txn_ts", "amount",
                    F.when(F.col("entry_mode") == "ECOM", "CARD_ECOM").otherwise("CARD_POS").alias("channel"),
                    F.col("merchant_name").alias("merchant_text"), F.col("mcc").cast("int").alias("mcc"),
                    F.when(F.col("entry_mode") != "ECOM", F.col("m_code")).alias("loc_code"),
                    F.when(F.col("entry_mode") != "ECOM", F.lit("card_city")).alias("loc_source")))
    p2m = F.col("txn_type") == "P2M"
    upi = (dedupe(bronze["upi_txn"], "txn_id").join(vpa_cif, "payer_vpa", "left")
           .join(city_by_pin, "payee_pincode", "left")
           .select(F.col("txn_id").alias("event_id"), F.lit("UPI").alias("source"), "cif", "account_id",
                   "txn_ts", "amount", F.when(p2m, "UPI_P2M").otherwise("UPI_P2P").alias("channel"),
                   F.when(p2m, F.col("payee_name")).alias("merchant_text"), null_i.alias("mcc"),
                   F.when(p2m, F.col("p_code")).alias("loc_code"),
                   F.when(p2m & F.col("p_code").isNotNull(), F.lit("upi_pincode")).alias("loc_source")))
    known = customer.select("cif", F.lit(1).alias("_known"))
    nb = (dedupe(bronze["nb_txn"], "txn_id").join(known, "cif", "left")
          .select(F.col("txn_id").alias("event_id"), F.lit("NB").alias("source"),
                  F.when(F.col("_known") == 1, F.col("cif")).alias("cif"), null_s.alias("account_id"), "txn_ts",
                  "amount", F.lit("NETBANKING").alias("channel"), F.col("biller_name").alias("merchant_text"),
                  null_i.alias("mcc"), null_s.alias("loc_code"), null_s.alias("loc_source")))
    atm = (dedupe(bronze["atm_txn"], "txn_id").join(card_cif, "card_hash", "left").join(atm_city, "atm_id", "left")
           .select(F.col("txn_id").alias("event_id"), F.lit("ATM").alias("source"), "cif", "account_id", "txn_ts",
                   "amount", F.lit("ATM_CASH").alias("channel"), null_s.alias("merchant_text"), null_i.alias("mcc"),
                   F.col("a_code").alias("loc_code"), F.lit("atm").alias("loc_source")))

    home = customer.select("cif", "home_city_code")
    places = geo.groupBy("city_code").agg(F.first("city").alias("event_city"), F.first("state").alias("event_state"),
                                          F.first("city_tier").alias("event_city_tier"))
    located = F.col("loc_code").isNotNull()
    ev = (card.unionByName(upi).unionByName(nb).unionByName(atm).join(home, "cif", "left")
          .withColumn("event_city_code", F.when(located, F.col("loc_code")).otherwise(F.col("home_city_code")))
          .withColumn("location_source", F.when(located, F.col("loc_source")).otherwise(F.lit("home_fallback")))
          .join(places.withColumnRenamed("city_code", "event_city_code"), "event_city_code", "left")
          .withColumn("txn_date", F.to_date("txn_ts"))
          .withColumn("week_end", week_end(F.col("txn_date")))
          .withColumn("merchant_key", merchant_key(F.col("merchant_text")))
          .withColumn("is_home", (F.col("event_city_code") == F.col("home_city_code")).cast("int")))
    return ev.select(*EVENT_COLUMNS)


def run(spark: SparkSession, db_prefix: str) -> None:
    bronze_db, ref, silver = f"{db_prefix}_bronze", f"{db_prefix}_ref", f"{db_prefix}_silver"
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {silver}")
    bronze = {t: spark.table(f"{bronze_db}.{t}") for t in
              ("crm_customer", "cbs_accounts", "card_master", "card_txn", "vpa_registry", "upi_txn", "nb_txn",
               "atm_txn")}
    geo = spark.table(f"{ref}.pincode_geo")

    customer = conform_customer(bronze["crm_customer"], geo).cache()
    account = (dedupe(bronze["cbs_accounts"], "account_id")
               .select("account_id", "cif", "product", "branch_code", "opened_on", "status")).cache()
    _write(customer, f"{silver}.customer")
    _write(account, f"{silver}.account")
    events = spend_events(bronze, account, customer, geo, spark.table(f"{ref}.atm_location"))
    _write(events, f"{silver}.spend_event", _months("txn_date"))
    customer.unpersist()
    account.unpersist()

    for t in TABLES:
        logger.info("%s.%s: %d rows", silver, t, spark.table(f"{silver}.{t}").count())
    (spark.table(f"{silver}.spend_event").groupBy("channel", "location_source").count()
     .orderBy("channel", "location_source").show(truncate=False))


def main(argv=None, spark: SparkSession | None = None) -> None:
    args = parse_args(argv)
    own = spark is None
    spark = spark or SparkSession.builder.appName("spend-build-silver").getOrCreate()
    try:
        run(spark, args.db_prefix)
    finally:
        if own:
            spark.stop()


if __name__ == "__main__":
    main()
