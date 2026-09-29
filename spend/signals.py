"""Opportunity signals: plain rules on gold features, like the reason codes in
Collections and Mule. Each rule lives in config/policy.yaml `signals`; a row in
gold.spend_signals carries the rule, the number that triggered it and the
threshold, so marketing can see why a customer is listed.

  TRAVEL_SURGE         category spend in the last 4 weeks >= ratio_4w_vs_norm x
                       its norm, and >= min_amount_4w. The norm is the 4-weekly
                       rate over the 9 weeks before those 4 (PLAN decision 16: a
                       13-week norm includes the surge, so the ratio could never
                       exceed 13/4 = 3.25).
  FESTIVE_ELECTRONICS  category spend in the last 4 weeks >= min_amount_4w with
                       a festival week among those 4 weeks.
  RELOCATION           away_share_4w >= threshold for at least min_away_weeks of
                       the 4 weeks, the customer was mostly at home 4 weeks
                       earlier (away share <= max_prior_away_share), and at least
                       min_txn_4w transactions (PLAN decision 12).
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

WINDOW, RECENT = 13, 4
RATIO_CAP = 99.0
COLUMNS = ["snapshot_date", "cif", "signal_id", "action", "metric", "metric_value", "threshold", "detail",
           "segment", "home_city"]


def _inr(x: float) -> str:
    return f"INR {x:,.0f}"


def category_13w(f: pd.DataFrame, category: str) -> pd.Series:
    """13-week spend in one category: its share of categorised merchant spend x that spend."""
    categorised = f["merchant_spend_13w"] * (1 - f["uncategorised_share"])
    return f[f"cat_{category.lower()}_share"] * categorised


def _surge(f: pd.DataFrame, rule: dict) -> pd.DataFrame:
    cat = rule["category"]
    recent = f[f"cat_{cat.lower()}_4w"]
    prior = (category_13w(f, cat) - recent).clip(lower=0)
    norm = prior * RECENT / (WINDOW - RECENT)
    ratio = np.where(norm > 0, np.minimum(recent / norm.where(norm > 0, 1), RATIO_CAP), RATIO_CAP)
    hit = (recent >= float(rule["min_amount_4w"])) & (ratio >= float(rule["ratio_4w_vs_norm"]))
    out = f[hit].copy()
    out["metric"], out["metric_value"] = f"{cat.lower()}_4w_vs_norm", np.round(ratio[hit.to_numpy()], 3)
    out["threshold"] = float(rule["ratio_4w_vs_norm"])
    out["detail"] = [f"{cat.lower()} {_inr(r)} in 4 weeks vs a norm of {_inr(n)} per 4 weeks"
                     for r, n in zip(recent[hit], norm[hit])]
    return out


def _festive(f: pd.DataFrame, rule: dict, festival_weeks: set) -> pd.DataFrame:
    cat = rule["category"]
    amount = f[f"cat_{cat.lower()}_4w"]
    in_festival = f["snapshot_date"].map(
        lambda d: any(_week_end(d) - timedelta(weeks=i) in festival_weeks for i in range(RECENT)))
    hit = (amount >= float(rule["min_amount_4w"]))
    if rule.get("festival_weeks"):
        hit &= in_festival
    out = f[hit].copy()
    out["metric"], out["metric_value"] = f"{cat.lower()}_4w", amount[hit].round(2)
    out["threshold"] = float(rule["min_amount_4w"])
    out["detail"] = [f"{cat.lower()} {_inr(a)} in the 4 weeks to {d} (festival season)"
                     for a, d in zip(amount[hit], f.loc[hit, "snapshot_date"])]
    return out


def _relocation(f: pd.DataFrame, rule: dict, prior: pd.DataFrame | None) -> pd.DataFrame:
    before = (prior.set_index("cif")["away_share_4w"] if prior is not None and len(prior)
              else pd.Series(dtype=float))
    prev = f["cif"].map(before)
    hit = ((f["away_share_4w"] >= float(rule["away_share_4w"]))
           & (f["away_weeks_4w"] >= int(rule["min_away_weeks"]))
           & (f["txn_4w"] >= int(rule["min_txn_4w"]))
           & prev.notna() & (prev <= float(rule["max_prior_away_share"])))
    out = f[hit].copy()
    out["metric"], out["metric_value"] = "away_share_4w", f.loc[hit, "away_share_4w"].round(4)
    out["threshold"] = float(rule["away_share_4w"])
    out["detail"] = [f"{a:.0%} of located spend away from {h} in 4 weeks ({w} of 4 weeks), {p:.0%} four weeks "
                     f"earlier" for a, h, w, p in zip(f.loc[hit, "away_share_4w"], f.loc[hit, "home_city"],
                                                      f.loc[hit, "away_weeks_4w"], prev[hit])]
    return out


def _week_end(d):
    return d + timedelta(days=(4 - d.weekday()) % 7)


def evaluate(features: pd.DataFrame, rules: list[dict], prior: pd.DataFrame | None = None,
             festival_weeks: set | None = None) -> pd.DataFrame:
    """features: one snapshot of gold.spend_features; prior: the snapshot 4 weeks earlier."""
    f = features.copy()
    f["snapshot_date"] = pd.to_datetime(f["snapshot_date"]).dt.date
    parts = []
    for rule in rules:
        rid = rule["id"]
        if rid == "RELOCATION":
            hit = _relocation(f, rule, prior)
        elif rid == "FESTIVE_ELECTRONICS":
            hit = _festive(f, rule, festival_weeks or set())
        else:
            hit = _surge(f, rule)
        parts.append(hit.assign(signal_id=rid, action=rule["action"]))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if out.empty:
        return pd.DataFrame(columns=COLUMNS)
    return out[COLUMNS].sort_values(["signal_id", "metric_value", "cif"], ascending=[True, False, True]
                                    ).reset_index(drop=True)
