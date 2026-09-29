"""CDE job logic on small in-memory frames (no Iceberg). The full layers run in CI through
scripts/run_cde_local.py, where the GX gates check the real tables."""

from datetime import date, datetime, timedelta

import pytest

from spend.categorise import normalise

AS_OF = date(2026, 9, 25)
NAMES = ["SWIGGY*BLR 5521", "swiggy instamart", "Swiggy Ltd", "BigBasket Pvt Ltd", "PHONEPE*RAMESHKIRANA",
         "DMART*MUM 7781", "Reliance Digital India", "Sri Venkateswara Provisions", "Tata Power-DDL",
         "JIO RECHARGE 123456789", "in n out 12 cafe", "A.B.C. Medicals", "   ", "12345", "IRCTC/eTicket#4412",
         "Café Coffee Day", "LIC_PREMIUM", "ind-oil pvt. ltd."]


def test_merchant_key_matches_python_normalise(spark, silver_job, generator):
    from pyspark.sql import functions as F

    mu = generator.merchant_universe(generator.DEFAULT_SEED)
    texts = NAMES + [v for m in mu[::40] for v in m["variants"]] + [m["name"].upper()[:24] for m in mu[::40]]
    df = spark.createDataFrame([(t,) for t in texts], "t string")
    got = [r.k for r in df.select(silver_job.merchant_key(F.col("t")).alias("k")).collect()]
    assert got == [normalise(t) for t in texts]


def test_normalise_examples():
    assert normalise("SWIGGY*BLR 5521") == "swiggy blr"
    assert normalise("Swiggy Ltd") == "swiggy"
    assert normalise("BigBasket Pvt Ltd") == "bigbasket"
    assert normalise("12345") is None and normalise(None) is None


def test_week_end_is_the_closing_friday(spark, silver_job):
    from pyspark.sql import functions as F

    days = [date(2026, 9, 19) + timedelta(days=i) for i in range(8)]     # Saturday .. next Saturday
    df = spark.createDataFrame([(d,) for d in days], "d date")
    got = [r.w for r in df.select(silver_job.week_end(F.col("d")).alias("w")).collect()]
    assert got == [date(2026, 9, 25)] * 7 + [date(2026, 10, 2)]


def test_dedupe_keeps_latest(spark, silver_job):
    df = spark.createDataFrame([("T1", 1.0, datetime(2026, 9, 25, 1)), ("T1", 2.0, datetime(2026, 9, 25, 2)),
                                ("T2", 3.0, datetime(2026, 9, 25, 1))], "txn_id string, amount double, ingested_at timestamp")
    assert sorted((r.txn_id, r.amount) for r in silver_job.dedupe(df, "txn_id").collect()) == [("T1", 2.0),
                                                                                              ("T2", 3.0)]


@pytest.fixture(scope="module")
def tiny(spark):
    ts = datetime(2026, 9, 24, 10)
    ing = datetime(2026, 9, 25, 23)
    geo = spark.createDataFrame([("560001", "Bengaluru", "Karnataka", 1, 12.9, 77.6, "BLR"),
                                 ("403001", "Panaji", "Goa", 3, 15.5, 73.8, "GOA")],
                                "pincode string, city string, state string, city_tier int, lat double, lon double, "
                                "city_code string")
    atms = spark.createDataFrame([("ATMGOA01", "403001", "Panaji", "Goa", 3)],
                                 "atm_id string, pincode string, city string, state string, city_tier int")
    crm = spark.createDataFrame([("C1", "Asha Rao", "560001", "26-35", "HIGH", "AFFLUENT", date(2020, 1, 1), ing),
                                 ("C1", "Asha Rao", "560001", "26-35", "HIGH", "AFFLUENT", date(2020, 1, 1),
                                  ing - timedelta(hours=1))],
                                "cif string, customer_name string, home_pincode string, age_band string, "
                                "income_band string, segment string, onboarded_on date, ingested_at timestamp")
    acct = spark.createDataFrame([("SA1", "C1", "SA", "BLR01", date(2020, 1, 1), "ACTIVE", ing)],
                                 "account_id string, cif string, product string, branch_code string, opened_on date, "
                                 "status string, ingested_at timestamp")
    cards = spark.createDataFrame([("h1", "SA1", ing)], "card_hash string, account_id string, ingested_at timestamp")
    vpas = spark.createDataFrame([("asha@ybl", "SA1", ing)], "vpa string, account_id string, ingested_at timestamp")
    card_txn = spark.createDataFrame([
        ("C01", "h1", ts, 500.0, 5411, "RAMESH KIRANA", "PANAJI", "CHIP", ing),
        ("C02", "h1", ts, 900.0, 5732, "CROMA*ONLINE 5521", "MUMBAI", "ECOM", ing),
        ("C03", "hX", ts, 100.0, 5411, "UNKNOWN CARD SHOP", "BENGALURU", "CHIP", ing)],
        "txn_id string, card_hash string, txn_ts timestamp, amount double, mcc int, merchant_name string, "
        "merchant_city string, entry_mode string, ingested_at timestamp")
    upi_txn = spark.createDataFrame([
        ("U01", "asha@ybl", "shop@ybl", "PHONEPE*RAMESHKIRANA", "403001", "P2M", ts, 250.0, None, ing),
        ("U02", "asha@ybl", "swiggy@icici", "Swiggy Ltd", None, "P2M", ts, 400.0, None, ing),
        ("U03", "asha@ybl", "ravi12@ybl", "Ravi Kumar", None, "P2P", ts, 1000.0, "rent", ing),
        ("U04", "unknown1@ybl", "shop@ybl", "PHONEPE*RAMESHKIRANA", "403001", "P2M", ts, 50.0, None, ing)],
        "txn_id string, payer_vpa string, payee_vpa string, payee_name string, payee_pincode string, "
        "txn_type string, txn_ts timestamp, amount double, remarks string, ingested_at timestamp")
    nb_txn = spark.createDataFrame([("N01", "C1", "BESCOM", "UTILITY", ts, 1700.0, ing),
                                    ("N02", "C9", "BESCOM", "UTILITY", ts, 1.0, ing)],
                                   "txn_id string, cif string, biller_name string, biller_type string, "
                                   "txn_ts timestamp, amount double, ingested_at timestamp")
    atm_txn = spark.createDataFrame([("A01", "h1", "ATMGOA01", ts, 2000.0, ing)],
                                    "txn_id string, card_hash string, atm_id string, txn_ts timestamp, amount double, "
                                    "ingested_at timestamp")
    return dict(geo=geo, atms=atms, crm_customer=crm, cbs_accounts=acct, card_master=cards, vpa_registry=vpas,
                card_txn=card_txn, upi_txn=upi_txn, nb_txn=nb_txn, atm_txn=atm_txn)


def test_spend_events_resolve_and_locate(tiny, silver_job):
    customer = silver_job.conform_customer(tiny["crm_customer"], tiny["geo"])
    c = customer.collect()
    assert len(c) == 1 and (c[0].home_city_code, c[0].city_tier, c[0].cohort) == ("BLR", 1, "AFFLUENT|T1")
    account = silver_job.dedupe(tiny["cbs_accounts"], "account_id")
    ev = {r.event_id: r for r in silver_job.spend_events(tiny, account, customer, tiny["geo"], tiny["atms"]).collect()}
    assert len(ev) == 10
    assert (ev["C01"].cif, ev["C01"].channel, ev["C01"].location_source, ev["C01"].event_city_code,
            ev["C01"].is_home) == ("C1", "CARD_POS", "card_city", "GOA", 0)
    assert (ev["C02"].channel, ev["C02"].location_source, ev["C02"].event_city_code) == (
        "CARD_ECOM", "home_fallback", "BLR")
    assert ev["C03"].cif is None and ev["U04"].cif is None and ev["N02"].cif is None
    assert (ev["U01"].channel, ev["U01"].location_source, ev["U01"].merchant_key) == (
        "UPI_P2M", "upi_pincode", "phonepe rameshkirana")
    assert (ev["U02"].location_source, ev["U02"].merchant_key) == ("home_fallback", "swiggy")
    assert (ev["U03"].channel, ev["U03"].merchant_key, ev["U03"].location_source) == ("UPI_P2P", None,
                                                                                      "home_fallback")
    assert (ev["N01"].channel, ev["N01"].merchant_key) == ("NETBANKING", "bescom")
    assert (ev["A01"].channel, ev["A01"].location_source, ev["A01"].event_city_tier, ev["A01"].merchant_key) == (
        "ATM_CASH", "atm", 3, None)
    assert all(r.week_end == AS_OF and r.txn_date == date(2026, 9, 24) for r in ev.values())
    assert list(next(iter(ev.values())).asDict()) == silver_job.EVENT_COLUMNS


def test_orphans_and_share_sum(spark, dq_job):
    child = spark.createDataFrame([("a",), ("b",), ("z",)], "k string")
    parent = spark.createDataFrame([("a",), ("b",)], "id string")
    assert sorted(r.orphan for r in dq_job._orphans(child, "k", parent, "id").collect()) == [0, 0, 1]
    from pyspark.sql import functions as F

    df = spark.createDataFrame([(0.5, 0.5, 10.0), (0.7, 0.2, 10.0), (0.0, 0.0, 0.0)], "a double, b double, s double")
    err = [round(r.share_sum_error, 6) for r in dq_job._share_sum(df, ["a", "b"], F.col("s") > 0).collect()]
    assert err == [0.0, 0.1]


def test_gx_run_checks_records_severity(spark, dq_job):
    import great_expectations.expectations as gxe

    df = spark.createDataFrame([(1, "P2M"), (2, "P2P"), (3, "BAD")], "id int, txn_type string")
    checks = [dq_job.Check("upi_txn", "upi", "txn_type known",
                           gxe.ExpectColumnValuesToBeInSet(column="txn_type", value_set=["P2P", "P2M"]),
                           dq_job.WARNING),
              dq_job.Check("upi_txn", "upi", "id not null", gxe.ExpectColumnValuesToNotBeNull(column="id"),
                           dq_job.CRITICAL)]
    out = {r["check_name"]: r for r in dq_job.run_checks({"upi": df}, checks)}
    assert out["txn_type known"]["success"] is False and out["txn_type known"]["unexpected_count"] == 1
    assert out["txn_type known"]["severity"] == "warning"
    assert out["id not null"]["success"] is True


def test_card_number_regex(spark, dq_job):
    import re

    assert re.search(dq_job.CARD_NUMBER_REGEX, "paid 4111111111111111 ok")
    assert not re.search(dq_job.CARD_NUMBER_REGEX, "SWIGGY*BLR 5521")


def test_gold_constants_match_features(gold_job):
    from spend import features

    for name in ("CATEGORIES", "CHANNELS", "MERCHANT_CHANNELS", "DIGITAL_CHANNELS", "TIERS", "SEGMENTS",
                 "SHARE_COLUMNS", "CAT_4W", "INFO", "NUMERIC", "FEATURE_COLUMNS", "COHORT_KEY", "COHORT_COLUMNS",
                 "COVARIATES", "MIX_DIMENSIONS", "MIX_COLUMNS"):
        assert getattr(gold_job, name) == getattr(features, name), name
    assert gold_job.UNCATEGORISED == features.UNCATEGORISED


def test_gold_snapshot_dates(gold_job):
    assert gold_job.week_end_of(date(2026, 9, 19)) == date(2026, 9, 25)
    assert gold_job.week_end_of(date(2026, 9, 25)) == date(2026, 9, 25)
    assert gold_job.snapshot_dates(date(2025, 7, 11)) == [date(2025, 7, 4), date(2025, 7, 11)]
    assert gold_job.snapshot_dates(date(2025, 7, 9)) == [date(2025, 7, 4), date(2025, 7, 9)]
    assert gold_job.FIRST_SNAPSHOT == gold_job.FIRST_WEEK + timedelta(weeks=gold_job.WINDOW_WEEKS - 1)


GOLD_AS_OF = date(2025, 7, 9)      # a Wednesday: snapshots 2025-07-04 and the as-of date


@pytest.fixture(scope="module")
def gold_tiny(spark, gold_job):
    customer = spark.createDataFrame(
        [("C1", "AFFLUENT", "26-35", "HIGH", "Bengaluru", "Karnataka", 1, "AFFLUENT|T1", date(2020, 1, 1)),
         ("C2", "MASS", "18-25", "LOW", "Panaji", "Goa", 3, "MASS|T3", date(2025, 8, 1))],
        "cif string, segment string, age_band string, income_band string, home_city string, home_state string, "
        "city_tier int, cohort string, onboarded_on date")
    first = date(2025, 4, 4)
    calendar = spark.createDataFrame(
        [(first + timedelta(weeks=i), int(first + timedelta(weeks=i) == date(2025, 7, 4)), 0, 0, 0, 0)
         for i in range(20)],
        "week_end date, salary_week int, festival_week int, month_end_week int, holiday_count int, "
        "school_fee_week int")
    ev = spark.createDataFrame([
        # cif, txn_date, amount, channel, category, is_home, tier, location_source, state, city
        ("C1", date(2025, 4, 2), 9999.0, "CARD_POS", "GROCERY", 1, 1, "card_city", "Karnataka", "Bengaluru"),
        ("C1", date(2025, 7, 2), 1000.0, "CARD_POS", "GROCERY", 1, 1, "card_city", "Karnataka", "Bengaluru"),
        ("C1", date(2025, 7, 1), 2000.0, "UPI_P2P", None, 1, 1, "home_fallback", "Karnataka", "Bengaluru"),
        ("C1", date(2025, 7, 8), 3000.0, "CARD_POS", "TRAVEL", 0, 3, "card_city", "Goa", "Panaji"),
        ("C1", date(2025, 7, 8), 500.0, "UPI_P2M", "UNCATEGORISED", 1, 1, "home_fallback", "Karnataka", "Bengaluru"),
        ("C1", date(2025, 7, 9), 1000.0, "ATM_CASH", None, 0, 3, "atm", "Goa", "Panaji")],
        "cif string, txn_date date, amount double, channel string, category string, is_home int, "
        "event_city_tier int, location_source string, event_state string, event_city string")
    from pyspark.sql import functions as F

    ev = (ev.withColumn("week_end", F.date_add("txn_date", (F.lit(13) - F.dayofweek("txn_date")) % 7))
          .withColumn("is_merchant", F.col("channel").isin(*gold_job.MERCHANT_CHANNELS)))
    panel = gold_job.weekly_panel(spark, ev, customer, calendar, GOLD_AS_OF)
    weeks = gold_job.complete_weeks(calendar, GOLD_AS_OF)
    return dict(features={(r.cif, r.snapshot_date): r for r in
                          gold_job.features(spark, panel, customer, GOLD_AS_OF).collect()},
                cohort=gold_job.cohort_weekly(spark, ev, customer, weeks),
                mix=gold_job.mix_weekly(ev, customer, weeks))


def test_gold_features_windows_and_shares(gold_tiny, gold_job):
    f = gold_tiny["features"]
    assert set(f) == {("C1", date(2025, 7, 4)), ("C1", GOLD_AS_OF)}      # C2 not onboarded yet
    friday = f[("C1", date(2025, 7, 4))]
    assert friday.spend_13w == 3000.0 and friday.txn_13w == 2               # the April event is outside 13 weeks
    assert friday.cat_grocery_share == 1.0 and friday.away_share_4w == 0.0

    r = f[("C1", GOLD_AS_OF)]
    assert list(r.asDict()) == gold_job.FEATURE_COLUMNS
    assert (r.spend_13w, r.txn_13w, r.merchant_spend_13w, r.avg_ticket) == (7500.0, 5, 4500.0, 1500.0)
    assert r.uncategorised_share == pytest.approx(500 / 4500, abs=1e-6)
    assert (r.cat_grocery_share, r.cat_travel_share) == (0.25, 0.75)       # of categorised merchant spend
    assert sum(getattr(r, c) for c in gold_job.CAT_SHARES) == pytest.approx(1.0)
    assert sum(getattr(r, c) for c in gold_job.CH_SHARES) == pytest.approx(1.0)
    assert (r.ch_upi_p2p_share, r.tier3_share) == (pytest.approx(2000 / 7500, abs=1e-6),
                                                   pytest.approx(4000 / 7500, abs=1e-6))
    assert r.home_share == pytest.approx(3500 / 7500, abs=1e-6)
    assert r.digital_share == pytest.approx(2500 / 7500, abs=1e-6)
    assert r.salary_week_share == pytest.approx(3000 / 7500, abs=1e-6)
    assert (r.spend_4w, r.txn_4w, r.cat_travel_4w) == (7500.0, 5, 3000.0)
    assert r.spend_trend_4v13 == 3.25
    assert r.away_share_4w == 0.8           # located spend only: 4000 away of 5000 (fallback events left out)
    assert r.away_weeks_4w == 1


def test_gold_cohort_series_dense_and_mix_reconciles(gold_tiny, gold_job):
    from pyspark.sql import functions as F

    cohort, mix = gold_tiny["cohort"], gold_tiny["mix"]
    weeks = [r.week_end for r in cohort.select("week_end").distinct().orderBy("week_end").collect()]
    assert weeks[0] == gold_job.FIRST_WEEK and weeks[-1] == date(2025, 7, 4)   # complete weeks only
    assert cohort.count() == 224 * len(weeks)
    assert list(cohort.columns) == gold_job.COHORT_COLUMNS
    got = cohort.where(F.col("amount_inr") > 0).collect()
    assert [(r.cohort, r.category, r.week_end, r.amount_inr, r.n_customers, r.salary_week) for r in got] == [
        ("AFFLUENT|T1", "GROCERY", date(2025, 7, 4), 1000.0, 1, 1)]
    cat = {(r.dim_value, r.channel): r.amount_inr for r in mix.where("dim = 'category'").collect()}
    assert cat == {("GROCERY", "CARD_POS"): 1000.0, ("NON_MERCHANT", "UPI_P2P"): 2000.0}
    assert {r.dim for r in mix.select("dim").distinct().collect()} == set(gold_job.MIX_DIMENSIONS) - {"away_city"}


def test_dq_constants_match_features(dq_job, silver_job):
    from spend import features

    assert dq_job.CATEGORIES == features.CATEGORIES
    assert dq_job.CHANNELS == features.CHANNELS
    assert dq_job.MERCHANT_CHANNELS == features.MERCHANT_CHANNELS
    assert dq_job.CAT_SHARES + dq_job.CH_SHARES + dq_job.TIER_SHARES == features.SHARE_COLUMNS
