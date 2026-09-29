"""Names shared by the gold job's contract, the ML stages, the endpoint and the app.

cde/jobs/build_gold_features.py is self-contained (PySpark + stdlib) and keeps
its own copy of these lists; tests/test_cde_jobs.py checks the two agree.

Shares, per CIF over the 13 weeks up to the snapshot:
  cat_<c>_share   share of categorised merchant spend (card, UPI P2M, net banking) in
                  each of the 14 categories; sums to 1 for customers with merchant spend.
                  Cash and UPI P2P have no merchant, and UNCATEGORISED merchant spend is
                  reported on its own (uncategorised_share), so the 14 shares stay a mix.
  ch_<c>_share    share of all spend by channel (6); sums to 1.
  tier<n>_share   share of all spend by the city tier where it happened (4); sums to 1.
  home_share      share of all spend in the customer's home city.
  digital_share   share of all spend through UPI, card e-commerce and net banking.
  away_share_4w   share of located spend (card_city, upi_pincode, atm) outside the home
                  city in the last 4 weeks; home_fallback events are left out.
  away_weeks_4w   weeks of the last 4 where at least half the located spend was away
                  (a move persists, a trip does not).
"""

from __future__ import annotations

import hashlib
import json

# Bump when the gold feature code or persona_matrix changes behaviour without
# changing a list below; it is folded into feature_hash().
FEATURE_LOGIC_VERSION = "1.0.0"

CATEGORIES = ["GROCERY", "FOOD_DELIVERY", "DINING", "FUEL", "TRAVEL", "APPAREL", "ELECTRONICS", "UTILITIES",
              "TELECOM", "EMI_LOAN", "EDUCATION", "HEALTH", "ENTERTAINMENT", "OTHER"]
UNCATEGORISED = "UNCATEGORISED"
CHANNELS = ["CARD_POS", "CARD_ECOM", "UPI_P2M", "UPI_P2P", "NETBANKING", "ATM_CASH"]
MERCHANT_CHANNELS = ["CARD_POS", "CARD_ECOM", "UPI_P2M", "NETBANKING"]
DIGITAL_CHANNELS = ["CARD_ECOM", "UPI_P2M", "UPI_P2P", "NETBANKING"]
TIERS = [1, 2, 3, 4]
SEGMENTS = ["MASS", "AFFLUENT", "NRI", "SME_OWNER"]
AGE_BANDS = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
LOCATION_SOURCES = ["card_city", "upi_pincode", "atm", "home_fallback"]

CAT_SHARES = [f"cat_{c.lower()}_share" for c in CATEGORIES]
CH_SHARES = [f"ch_{c.lower()}_share" for c in CHANNELS]
TIER_SHARES = [f"tier{t}_share" for t in TIERS]
SHARE_COLUMNS = CAT_SHARES + CH_SHARES + TIER_SHARES                     # the 24 share columns
CAT_4W = [f"cat_{c.lower()}_4w" for c in CATEGORIES]                      # INR per category, last 4 weeks

KEY = ["cif", "snapshot_date"]
INFO = ["segment", "age_band", "income_band", "home_city", "home_state", "city_tier", "cohort"]
NUMERIC = [*SHARE_COLUMNS, "home_share", "digital_share", "uncategorised_share", "salary_week_share",
           "spend_13w", "txn_13w", "avg_ticket", "spend_trend_4v13", "merchant_spend_13w",
           "spend_4w", "txn_4w", "away_share_4w", "away_weeks_4w", *CAT_4W]
FEATURE_COLUMNS = [*KEY, *INFO, *NUMERIC]

# KMeans input, standardised: the 24 shares + log1p(spend_13w) + digital and home share,
# clipped at +-PERSONA_CLIP standard deviations (PLAN decision 14).
LOG_SPEND = "log_spend_13w"
PERSONA_FEATURES = [*SHARE_COLUMNS, LOG_SPEND, "digital_share", "home_share"]
PERSONA_CLIP = 3.0

COHORT_KEY = ["cohort", "category", "week_end"]
COHORT_COLUMNS = ["cohort", "segment", "city_tier", "category", "week_end", "amount_inr", "n_txn", "n_customers",
                  "festival_week", "salary_week", "month_end_week", "holiday_count", "school_fee_week"]
COVARIATES = ["festival_week", "salary_week", "month_end_week", "holiday_count", "school_fee_week"]

# gold.spend_mix_weekly: week x dimension x value x channel, so the app never scans silver.
MIX_DIMENSIONS = ["category", "city_tier", "state", "city", "away_city", "age_band", "segment", "location_source"]
MIX_COLUMNS = ["week_end", "dim", "dim_value", "channel", "amount_inr", "n_txn", "n_customers"]


def cohort_name(segment: str, city_tier: int) -> str:
    return f"{segment}|T{int(city_tier)}"


def series_id(cohort: str, category: str) -> str:
    return f"{cohort}|{category}"


def feature_hash() -> str:
    """Fingerprint of the feature definition the models are trained on and the endpoint
    re-applies (the stored scaler and centroids are only valid for the same one). Recorded
    by feature engineering, checked by training, the gate and the endpoint."""
    payload = {"logic": FEATURE_LOGIC_VERSION, "features": FEATURE_COLUMNS, "persona": PERSONA_FEATURES,
               "clip": PERSONA_CLIP, "covariates": COVARIATES, "categories": CATEGORIES, "channels": CHANNELS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def split_series(sid: str) -> tuple[str, str]:
    segment, tier, category = sid.split("|")
    return f"{segment}|{tier}", category
