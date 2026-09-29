"""Cached reads of the gold / ref tables for the Streamlit app. Every reader
returns an empty frame when its table does not exist yet (cold start). The app
never scans silver: book-wide views read gold.spend_mix_weekly, and one
customer's detail comes from the endpoint (spend/scoring.py)."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from spend.client import spend_request
from spend.schema import OUTPUT_TABLES
from spend.storage import get_storage

DATE_COLS = ("run_date", "snapshot_date", "prev_run_date", "week_end", "week_start", "cutoff", "as_of",
             "first_run_date")


@st.cache_resource
def storage():
    return get_storage()


def _dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.date
    return df


@st.cache_data(ttl=300, show_spinner="Reading gold tables...")
def read(key: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    return _dates(storage().read(key, columns=list(columns) if columns else None))


def read_or_empty(key: str, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return read(key, tuple(columns) if columns and key not in OUTPUT_TABLES else None)
    except Exception:  # not created yet: no run, a gate that never passed, no DQ run
        cols = [c for c, _ in OUTPUT_TABLES[key]] if key in OUTPUT_TABLES else (columns or [])
        return pd.DataFrame(columns=cols)


def for_run(key: str, run_date) -> pd.DataFrame:
    df = read_or_empty(key)
    return df[df["run_date"] == run_date].reset_index(drop=True) if not df.empty else df


def mix() -> pd.DataFrame:
    df = read_or_empty("spend_mix_weekly", ["week_end", "dim", "dim_value", "channel", "amount_inr", "n_txn",
                                            "n_customers"])
    if not df.empty:
        df["amount_inr"] = df["amount_inr"].astype(float)
    return df


def cohort_weekly() -> pd.DataFrame:
    df = read_or_empty("spend_cohort_weekly", ["cohort", "segment", "city_tier", "category", "week_end",
                                               "amount_inr"])
    if not df.empty:
        df["amount_inr"] = df["amount_inr"].astype(float)
    return df


@st.cache_data(ttl=3600)
def state_centroids() -> pd.DataFrame:
    geo = read_or_empty("pincode_geo", ["state", "lat", "lon"])
    if geo.empty:
        return geo
    return geo.groupby("state")[["lat", "lon"]].mean().reset_index()


@st.cache_data(ttl=300, show_spinner="Reading the category map...")
def map_methods() -> pd.DataFrame:
    try:
        m = storage().read("merchant_category_map", columns=["method", "category"])
    except Exception:
        return pd.DataFrame(columns=["method", "category", "keys"])
    return m.groupby(["method", "category"]).size().reset_index(name="keys")


def dq_results() -> pd.DataFrame:
    df = read_or_empty("dq_results")
    if not df.empty:
        df["run_ts"] = pd.to_datetime(df["run_ts"])
        df["success"] = df["success"].astype(bool)
    return df


@st.cache_data(ttl=300, show_spinner="Asking the spend scorer...")
def ask(req_json: str) -> tuple[dict, str]:
    """One endpoint request (profile / forecast / what_if / categorise), cached by its JSON."""
    try:
        return spend_request(json.loads(req_json))
    except Exception as e:  # endpoint down, no published run for the in-process context, ...
        return {"error": f"{type(e).__name__}: {e}"}, ""
