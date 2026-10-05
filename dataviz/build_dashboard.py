#!/usr/bin/env python3
"""
The Cloudera Data Visualization dashboard of this project, as code: "Spend Analytics -
Insights" (Portfolio, Where, Personas, Opportunities, Forecast, Trust).

Datasets, visuals and sheets are declared below; this script turns them into a Data
Visualization export file (dataviz/spend_dashboards.json) and imports it through the
migration REST API of the CDW Data Visualization instance at SPEND_CDV_URL, into this
project's own shared workspace (WORKSPACE). The instance is shared with other projects: only
the connection, workspace, datasets and dashboard named here are created or updated. UUIDs
are fixed per artefact, so an import updates the dashboard in place. Datasets read the
rsingh_spend_anl_report views (sql/dataviz_views.sql); see docs/CDV_DASHBOARD.md.

  set -a; source .env; set +a
  python dataviz/build_dashboard.py              # connection, workspace (if missing), file, import, move
  python dataviz/build_dashboard.py --no-import  # write the file only
  python dataviz/build_dashboard.py --check      # every visual's query directly on Impala
  python dataviz/build_dashboard.py --verify     # every visual's query through the Data API

Needs SPEND_CDV_URL, SPEND_CDV_API_KEY and, for the column types and a new connection,
SPEND_IMPALA_USER / SPEND_IMPALA_PASSWORD. The password goes to the Data Visualization
connection only; it is never printed or written to the file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import uuid
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "dataviz" / "spend_dashboards.json"
DB = "rsingh_spend_anl_report"
CONNECTION = "rsingh-spend-anl-impala"
WORKSPACE = "rsingh-spend-anl"
NS = uuid.UUID("5d1f0c8e-2a47-4b9e-8f63-1c9a7e40b2d5")
DASHBOARD_PK = 8000
DATASET_PK0 = 8100
VISUAL_PK0 = 8200

DATASETS = {                          # key: (name, view, integer columns that are dimensions)
    "mix": ("Spend - Weekly mix", "v_spend_mix", {"is_last_13w"}),
    "personas": ("Spend - Personas", "v_personas", {"persona_id"}),
    "persona_runs": ("Spend - Persona runs", "v_persona_runs", {"is_latest", "persona_id"}),
    "signals": ("Spend - Signals", "v_signals", {"is_latest"}),
    "forecast": ("Spend - Forecast", "v_forecast", {"horizon_week"}),
    "runs": ("Spend - Model runs", "v_model_run", {"is_latest", "k"}),
    "dq": ("Spend - Data quality", "v_dq", {"is_latest", "layer_order", "table_snapshot_id"}),
}

LATEST = "[is_latest] = 1"
L13 = "[is_last_13w] = 1"
CAT = "[dim] = 'category'"
CRORE = "round(sum([amount_crore]), 1)"
DIGITAL_PCT = ("round(100 * sum(case when [channel_group] = 'Digital' then [amount_inr] else 0 end)"
               " / sum([amount_inr]), 1)")

# A visual: dims are (column, alias); measures are (expression, alias); filters are expressions;
# pos is (column, row, width, height) on a 64-column grid.
SHEETS = [
    ("Portfolio", [
        dict(type="kpi", ds="mix", title="Spend, last 13 weeks (INR crore)", measures=[(CRORE, "Spend (crore)")],
             filters=[CAT, L13], pos=(1, 1, 21, 10)),
        dict(type="kpi", ds="mix", title="Transactions, last 13 weeks (millions)",
             measures=[("round(sum([n_txn]) / 1000000, 2)", "Transactions (M)")], filters=[CAT, L13],
             pos=(22, 1, 21, 10)),
        dict(type="kpi", ds="mix", title="Digital share of spend (%)", measures=[(DIGITAL_PCT, "Digital %")],
             filters=[CAT, L13], pos=(43, 1, 22, 10)),
        dict(type="trellis-bars", ds="mix", title="Spend by category, last 13 weeks (INR crore)",
             x=[("dim_value", "Category")], measures=[(CRORE, "Spend (crore)")],
             filters=[CAT, L13], sort_desc=True, pos=(1, 11, 32, 22)),
        dict(type="trellis-lines", ds="mix", title="Weekly spend by channel, 26 weeks (INR crore)",
             x=[("week_end", "Week ending")], measures=[(CRORE, "Spend (crore)")], color=[("channel", "Channel")],
             filters=[CAT], pos=(33, 11, 32, 22)),
        dict(type="trellis-bars", ds="mix", title="Digital and physical spend by city tier, last 13 weeks",
             x=[("dim_value", "City tier")], measures=[(CRORE, "Spend (crore)")],
             color=[("channel_group", "Channel group")], filters=["[dim] = 'city_tier'", L13], pos=(1, 33, 32, 22)),
        dict(type="trellis-bars", ds="mix", title="Spend by segment and channel, last 13 weeks",
             x=[("dim_value", "Segment")], measures=[(CRORE, "Spend (crore)")], color=[("channel", "Channel")],
             filters=["[dim] = 'segment'", L13], pos=(33, 33, 32, 22)),
    ]),
    ("Where", [
        dict(type="trellis-bars", ds="mix", title="Top 15 states by spend, last 13 weeks (INR crore)",
             x=[("dim_value", "State")], measures=[(CRORE, "Spend (crore)")],
             filters=["[dim] = 'state'", L13], sort_desc=True, limit=15, pos=(1, 1, 32, 24)),
        dict(type="trellis-bars", ds="mix", title="Top 10 cities customers travel to (spend away from home)",
             x=[("dim_value", "Away city")], measures=[(CRORE, "Spend (crore)")],
             filters=["[dim] = 'away_city'", L13], sort_desc=True, limit=10, pos=(33, 1, 32, 24)),
        dict(type="trellis-lines", ds="mix", title="Weekly spend away from the home city (INR crore)",
             x=[("week_end", "Week ending")], measures=[(CRORE, "Spend (crore)")],
             filters=["[dim] = 'away_city'"], pos=(1, 25, 32, 22)),
        dict(type="trellis-bars", ds="mix", title="How the location of a transaction is known (millions)",
             x=[("dim_value", "Location source")], measures=[("round(sum([n_txn]) / 1000000, 2)", "Transactions (M)")],
             filters=["[dim] = 'location_source'", L13], sort_desc=True, pos=(33, 25, 32, 22)),
    ]),
    ("Personas", [
        dict(type="kpi", ds="personas", title="Customers profiled (latest run)", measures=[("sum(1)", "Customers")],
             pos=(1, 1, 21, 10)),
        dict(type="kpi", ds="personas", title="Moved persona since the previous run",
             measures=[("sum(case when [persona_change] = 'Moved' then 1 else 0 end)", "Moved")],
             pos=(22, 1, 21, 10)),
        dict(type="kpi", ds="personas", title="Average 13-week spend per customer (INR)",
             measures=[("round(avg([spend_13w]), 0)", "Avg 13-week spend")], pos=(43, 1, 22, 10)),
        dict(type="trellis-bars", ds="personas", title="Persona size by segment",
             x=[("persona_name", "Persona")], measures=[("sum(1)", "Customers")], color=[("segment", "Segment")],
             sort_desc=True, pos=(1, 11, 32, 24)),
        dict(type="trellis-bars", ds="personas", title="Average 13-week spend by persona (INR)",
             x=[("persona_name", "Persona")], measures=[("round(avg([spend_13w]), 0)", "Avg spend")],
             sort_desc=True, pos=(33, 11, 32, 24)),
        dict(type="trellis-bars", ds="personas", title="Digital share of spend by persona (%)",
             x=[("persona_name", "Persona")], measures=[("round(100 * avg([digital_share]), 1)", "Digital %")],
             sort_desc=True, pos=(1, 35, 32, 22)),
        dict(type="trellis-lines", ds="persona_runs", title="Persona size in every run",
             x=[("run_date", "Run date")], measures=[("sum([n_customers])", "Customers")],
             color=[("persona_name", "Persona")], pos=(33, 35, 32, 22)),
        dict(type="table", ds="personas", title="Who moved: previous persona to today's",
             dims=[("prev_persona_name", "Previous persona"), ("persona_name", "Persona today")],
             measures=[("sum(1)", "Customers"), ("round(avg([spend_13w]), 0)", "Avg 13-week spend")],
             filters=["[persona_change] = 'Moved'"], sort_dim="prev_persona_name", pos=(1, 57, 64, 24)),
    ]),
    ("Opportunities", [
        dict(type="kpi", ds="signals", title="Travel card / forex leads (TRAVEL_SURGE)", measures=[("sum(1)", "Leads")],
             filters=[LATEST, "[signal_id] = 'TRAVEL_SURGE'"], pos=(1, 1, 21, 10)),
        dict(type="kpi", ds="signals", title="EMI conversion leads (FESTIVE_ELECTRONICS)",
             measures=[("sum(1)", "Leads")], filters=[LATEST, "[signal_id] = 'FESTIVE_ELECTRONICS'"],
             pos=(22, 1, 21, 10)),
        dict(type="kpi", ds="signals", title="Home-branch change leads (RELOCATION)", measures=[("sum(1)", "Leads")],
             filters=[LATEST, "[signal_id] = 'RELOCATION'"], pos=(43, 1, 22, 10)),
        dict(type="trellis-bars", ds="signals", title="Leads by action and segment (latest run)",
             x=[("action", "Action")], measures=[("sum(1)", "Leads")], color=[("segment", "Segment")],
             filters=[LATEST], sort_desc=True, pos=(1, 11, 32, 22)),
        dict(type="trellis-bars", ds="signals", title="Top 15 home cities by leads (latest run)",
             x=[("home_city", "Home city")], measures=[("sum(1)", "Leads")], color=[("signal_id", "Signal")],
             filters=[LATEST], sort_desc=True, limit=15, pos=(33, 11, 32, 22)),
        dict(type="trellis-lines", ds="signals", title="Leads per run, by signal",
             x=[("run_date", "Run date")], measures=[("sum(1)", "Leads")], color=[("signal_id", "Signal")],
             pos=(1, 33, 32, 22)),
        dict(type="trellis-bars", ds="signals", title="Leads by persona (latest run)",
             x=[("persona_name", "Persona")], measures=[("sum(1)", "Leads")], color=[("signal_id", "Signal")],
             filters=[LATEST], sort_desc=True, pos=(33, 33, 32, 22)),
        dict(type="table", ds="signals", title="Campaign list: EMI conversion targets (latest run)",
             dims=[("cif", "Customer"), ("persona_name", "Persona"), ("segment", "Segment"),
                   ("home_city", "Home city"), ("city_tier", "Tier"), ("detail", "Why")],
             measures=[("round(max([metric_value]), 0)", "Festive electronics spend (INR)")],
             filters=[LATEST, "[signal_id] = 'FESTIVE_ELECTRONICS'"], sort_measure_desc=True, limit=50,
             pos=(1, 55, 64, 26)),
    ]),
    ("Forecast", [
        dict(type="trellis-lines", ds="forecast", title="Book-wide spend, next 12 weeks: Chronos-2 against seasonal naive (INR crore)",
             x=[("week_end", "Week ending")],
             measures=[("round(sum([p10_crore]), 2)", "P10"), ("round(sum([p50_crore]), 2)", "P50"),
                       ("round(sum([p90_crore]), 2)", "P90"), ("round(sum([naive_crore]), 2)", "Seasonal naive")],
             pos=(1, 1, 64, 24)),
        dict(type="trellis-bars", ds="forecast", title="Next 12 weeks by category: P50 against seasonal naive (INR crore)",
             x=[("category", "Category")],
             measures=[("round(sum([p50_crore]), 2)", "P50"), ("round(sum([naive_crore]), 2)", "Seasonal naive")],
             sort_desc=True, pos=(1, 25, 32, 24)),
        dict(type="trellis-lines", ds="forecast", title="Next 12 weeks by segment, P50 (INR crore)",
             x=[("week_end", "Week ending")], measures=[("round(sum([p50_crore]), 2)", "P50")],
             color=[("segment", "Segment")], pos=(33, 25, 32, 24)),
    ]),
    ("Trust", [
        dict(type="kpi", ds="runs", title="Categoriser accuracy, MCC hidden (gate >= 0.85)",
             measures=[("round(max([accuracy_mcc]), 3)", "Accuracy")], filters=[LATEST], pos=(1, 1, 16, 10)),
        dict(type="kpi", ds="runs", title="Persona silhouette (gate >= 0.15)",
             measures=[("round(max([silhouette]), 3)", "Silhouette")], filters=[LATEST], pos=(17, 1, 16, 10)),
        dict(type="kpi", ds="runs", title="Forecast WAPE vs seasonal naive (gate <= 0.90)",
             measures=[("round(max([wape_ratio]), 3)", "WAPE ratio")], filters=[LATEST], pos=(33, 1, 16, 10)),
        dict(type="kpi", ds="dq", title="Data-quality checks failed (latest gates)",
             measures=[("sum([failed])", "Failed checks")], filters=[LATEST], pos=(49, 1, 16, 10)),
        dict(type="trellis-lines", ds="runs", title="Accuracy and persona stability per run, against the gate floor",
             x=[("run_date", "Run date")],
             measures=[("max([accuracy_mcc])", "Accuracy (MCC hidden)"), ("max([accuracy_seed])", "Accuracy (seed)"),
                       ("max([persona_stability])", "Persona stability"), ("max([min_accuracy])", "Accuracy floor")],
             pos=(1, 11, 32, 22)),
        dict(type="trellis-lines", ds="runs", title="Forecast WAPE ratio and silhouette per run, against the gate",
             x=[("run_date", "Run date")],
             measures=[("max([wape_ratio])", "WAPE ratio"), ("max([max_wape_ratio])", "WAPE ceiling"),
                       ("max([silhouette])", "Silhouette"), ("max([min_silhouette])", "Silhouette floor")],
             pos=(33, 11, 32, 22)),
        dict(type="trellis-bars", ds="dq", title="Data-quality checks per layer, latest gates",
             x=[("layer", "Layer")], measures=[("sum([passed])", "Passed"), ("sum([failed])", "Failed")],
             filters=[LATEST], pos=(1, 33, 24, 22)),
        dict(type="trellis-bars", ds="dq", title="Checks per night and layer, 30 days",
             x=[("as_of", "As of")], measures=[("sum(1)", "Checks")], color=[("layer", "Layer")],
             pos=(25, 33, 40, 22)),
        dict(type="table", ds="runs", title="Every published run: KPIs and the gate",
             dims=[("run_date", "Run date"), ("run_id", "Run"), ("triggered_by", "Triggered by"), ("gate", "Gate"),
                   ("feature_hash", "Feature hash")],
             measures=[("round(max([accuracy_mcc]), 3)", "Accuracy MCC"), ("round(max([accuracy_seed]), 3)", "Accuracy seed"),
                       ("round(max([silhouette]), 3)", "Silhouette"), ("round(max([persona_stability]), 3)", "Stability"),
                       ("round(max([wape_ratio]), 3)", "WAPE ratio"), ("round(max([coverage]), 3)", "Coverage"),
                       ("sum([n_customers_profiled])", "Profiled"), ("sum([n_signals])", "Signals")],
             sort_dim="run_date", sort_asc=False, pos=(1, 55, 64, 22)),
        dict(type="table", ds="dq", title="Failed data-quality checks, 30 days (empty on a healthy pipeline)",
             dims=[("as_of", "As of"), ("layer", "Layer"), ("table_name", "Table"), ("check_name", "Check"),
                   ("severity", "Severity")],
             measures=[("max([unexpected_pct])", "Unexpected %")], filters=["[failed] = 1"], may_be_empty=True,
             sort_dim="as_of", sort_asc=False, pos=(1, 77, 64, 18)),
    ]),
]

DASHBOARD = dict(key="insights", pk=DASHBOARD_PK, ds="mix", title="Spend Analytics - Insights", sheets=SHEETS,
                 subtitle="Spend by category, channel and place; personas; next-best-action leads; forecast; "
                          "model and data trust (synthetic data, federal CDW)")

SHELVES = {
    "kpi": [("dimensions_shelf", 1, 1), ("aggregates_shelf", 1, 2), ("compare_shelf", 1, 2), ("label_shelf", 1, 2),
            ("tooltip_shelf", 1, 2), ("x_shelf", 1, 1), ("y_shelf", 1, 1), ("filters_shelf", 2, 3)],
    "table": [("dimensions_shelf", 1, 1), ("aggregates_shelf", 1, 2), ("filters_shelf", 2, 3)],
    "trellis-bars": [("x_shelf", 1, 3), ("y_shelf", 1, 3), ("color_shelf", 1, 3), ("tooltip_shelf", 1, 2),
                     ("drill_shelf", 1, 1), ("label_shelf", 1, 2), ("filters_shelf", 2, 3)],
    "trellis-lines": [("x_shelf", 1, 3), ("y_shelf", 1, 3), ("color_shelf", 1, 3), ("tooltip_shelf", 1, 2),
                      ("filters_shelf", 2, 3)],
}


def uid(*parts: str) -> str:
    return str(uuid.uuid5(NS, "/".join(parts)))


def impala():
    from spend.storage import get_storage
    return get_storage("impala")


def column_types(engine, ds_key: str) -> dict[str, str]:
    df = engine.query(f"DESCRIBE {DB}.{DATASETS[ds_key][1]}")
    return {r["name"]: r["type"].upper() for _, r in df.iterrows()}


def is_dim(ds_key: str, col: str, typ: str) -> bool:
    return col in DATASETS[ds_key][2] or not any(t in typ for t in ("INT", "DOUBLE", "FLOAT", "DECIMAL"))


def visuals():
    for sheet, items in SHEETS:
        for v in items:
            yield sheet, v


def missing_columns(types: dict[str, dict[str, str]]) -> list[str]:
    out = []
    for sheet, v in visuals():
        exprs = [e for e, _ in v["measures"]] + v.get("filters", [])
        cols = {c for c, _ in v.get("dims", []) + v.get("x", []) + v.get("color", [])}
        cols |= {c for e in exprs for c in re.findall(r"\[(\w+)\]", e)}
        out += [f"{sheet} / {v['title']}: {c}" for c in sorted(cols - set(types[v["ds"]]))]
    return out


def dataset_record(key: str, pk: int, types: dict[str, str], conn_id: int) -> dict:
    name, view, _ = DATASETS[key]
    table = f"{DB}.{view}"
    cols = [{"alias": c, "type": t, "name": c, "isdim": is_dim(key, c, t)} for c, t in types.items()]
    return {"model": "datasets.dataset", "pk": pk, "fields": {
        "dataconnection": conn_id, "dataset_name": name, "dataset_type": "singletable", "dataset_detail": table,
        "dataset_description": f"{table} (sql/dataviz_views.sql)",
        "dataset_info": json.dumps([{"tablename": table, "columns": cols}]),
        "dataset_tablenames": json.dumps([table]), "uuid": uid("dataset", key), "imported_uuid": None,
        "cache_sequence": 0, "dataset_settings": "{}", "search_enabled": False, "dashboards": [DASHBOARD_PK],
        "version_id": pk, "version_group_id": pk, "is_active_version": True,
        "version_name": "spend-analytics", "is_named_version": False}}


def dim_item(col: str, alias: str, typ: str) -> dict:
    return {"dataset_colname": col, "dataset_coltype": typ, "expression_for_trigger": f"[{col}]", "col_alias": alias}


def measure_item(expr: str, alias: str) -> dict:
    return {"custom_expr": expr, "expression_for_trigger": expr, "expr_hasagg": True, "col_alias": alias,
            "dataset_colname": alias, "dataset_coltype": "DOUBLE"}


def filter_item(expr: str) -> dict:
    return {"custom_expr": expr, "expression_for_trigger": expr, "filter_input": {}, "filter_data": [],
            "dataset_colname": "", "dataset_coltype": "STRING", "filter_column": ""}


def visual_record(v: dict, pk: int, sheet: str, types: dict[str, str], dataset_pk: int, ws_id: int) -> dict:
    kind = v["type"]
    shelves = {name: [] for name, _, _ in SHELVES[kind]}
    sources = {}

    def add_dims(shelf, pairs):
        for col, alias in pairs:
            shelves[shelf].append(dim_item(col, alias, types[col]))
            sources[f"[{col}] as 'sub:{alias}'"] = shelf

    def add_measures(shelf, pairs):
        for expr, alias in pairs:
            shelves[shelf].append(measure_item(expr, alias))
            sources[f"{expr} as 'sub:{alias}'"] = shelf

    if kind in ("kpi", "table"):
        add_dims("dimensions_shelf", v.get("dims", []))
        add_measures("aggregates_shelf", v["measures"])
    else:
        add_dims("x_shelf", v["x"])
        add_measures("y_shelf", v["measures"])
        add_dims("color_shelf", v.get("color", []))
    for expr in v.get("filters", []):
        shelves["filters_shelf"].append(filter_item(expr))
        sources[expr] = "filters_shelf"
    if v.get("sort_desc"):
        shelves["y_shelf"][0]["order"] = {"priority": 1, "ascending": False}
    if v.get("sort_measure_desc"):
        shelves["aggregates_shelf"][0]["order"] = {"priority": 1, "ascending": False}
    if v.get("sort_dim"):
        shelf = "dimensions_shelf" if kind == "table" else "x_shelf"
        item = next(i for i in shelves[shelf] if i["dataset_colname"] == v["sort_dim"])
        item["order"] = {"priority": 1, "ascending": v.get("sort_asc", True)}
    report = {
        "report_title": v["title"], "report_subtitle": "", "dashboard_id": DASHBOARD_PK,
        "limit": v.get("limit", 1000), "sample_pct": "Off", "selected_segments": [], "report_derived_data": [],
        "click_behaviors": {}, "sort_orders_asc": {}, "user_settings": {}, **shelves,
        "core": {"viz_type": kind, "saved_shelf_sources": sources,
                 "shelves": [{"name": n, "shelf_type": s, "column_type": c} for n, s, c in SHELVES[kind]]},
    }
    return {"model": "reports.report", "pk": pk, "fields": {
        "report_name": "", "report_description": f"{DASHBOARD['title']} / {sheet}", "dataset": dataset_pk,
        "workspace": ws_id, "report_type": kind, "report_mode": "", "dashboard_url_name": "",
        "report_data": json.dumps({"report_data": report, "report_type": kind}), "shared_visual_dashboards": None,
        "parent_report": None, "uuid": uid("visual", DASHBOARD["key"], sheet, v["title"]), "imported_uuid": None,
        "has_css_styles": False, "report_search_text": ""}}


def build(conn_id: int, ws_id: int, version: dict, types: dict[str, dict[str, str]]) -> dict:
    missing = missing_columns(types)
    if missing:
        raise SystemExit("columns not in the dataset's view:\n  " + "\n  ".join(missing))
    ds_pk = {k: DATASET_PK0 + i for i, k in enumerate(DATASETS)}
    out_visuals, sheets, pk = [], [], VISUAL_PK0
    for order, (sheet, items) in enumerate(SHEETS, 1):
        placed = []
        for v in items:
            pk += 1
            out_visuals.append(visual_record(v, pk, sheet, types[v["ds"]], ds_pk[v["ds"]], ws_id))
            placed.append((pk, v["pos"]))
        sheets.append({"sheet_id": order, "order": order, "sheet_handle_title": sheet, "behaviors": {},
                       "visual_widgets": [{"col": c, "row": r, "size_x": w, "size_y": h, "id": f"uri-{i}-widget-{p}"}
                                          for i, (p, (c, r, w, h)) in enumerate(placed, 1)],
                       "control_widgets": []})
    body = {"report_title": DASHBOARD["title"], "numColumns": 64, "report_subtitle": DASHBOARD["subtitle"],
            "dashboard_widgets": sheets[0]["visual_widgets"], "dashboard_sheets": sheets,
            "user_settings": {"dashboard_width": "1280", "display_filters": "true",
                              "permit_csv_download_dashboard": "true"},
            "global_control_widgets": [], "control_widgets": [], "click_behavior": {}}
    dashboard = {"model": "reports.report", "pk": DASHBOARD_PK, "fields": {
        "report_name": DASHBOARD["title"], "report_description": "docs/CDV_DASHBOARD.md",
        "dataset": ds_pk[DASHBOARD["ds"]], "workspace": ws_id, "report_type": "dashboard", "report_mode": None,
        "dashboard_url_name": "", "report_data": json.dumps(body), "shared_visual_dashboards": "[]",
        "parent_report": None, "uuid": uid("dashboard", DASHBOARD["key"]), "imported_uuid": None,
        "has_css_styles": False, "report_search_text": None}}
    return {"segments": [], "staticasset": [], "dashboards": [dashboard], "appgroupmembership": [],
            "reportannotation": [], "events": [], "customcss": [], "reportimage": [], "dateranges": [],
            "visuals": out_visuals, "colorpalette": [], "appgroups": [],
            "datasets": [dataset_record(k, ds_pk[k], types[k], conn_id) for k in DATASETS], "version": version}


def _strip(e: str) -> str:
    return re.sub(r"\[(\w+)\]", r"\1", e)


def visual_sql(v: dict) -> str:
    """The query a visual sends, as plain Impala SQL."""
    dims = [c for c, _ in v.get("dims", []) + v.get("x", []) + v.get("color", [])]
    where = " AND ".join(_strip(f) for f in v.get("filters", [])) or "TRUE"
    group = f" GROUP BY {', '.join(dims)}" if dims else ""
    return (f"SELECT {', '.join(dims + [_strip(e) for e, _ in v['measures']])} FROM {DB}.{DATASETS[v['ds']][1]} "
            f"WHERE {where}{group} LIMIT {v.get('limit', 1000)}")


def check(engine) -> int:
    """Every visual's query straight on Impala: the expressions parse and the visual has rows."""
    failed = 0
    for sheet, v in visuals():
        try:
            rows = engine.query(visual_sql(v))
            ok = len(rows) > 0 or v.get("may_be_empty", False)
            note = f"{rows.iloc[0, -1]}" if len(rows) and v["type"] == "kpi" else f"{len(rows)} rows"
        except Exception as e:  # noqa: BLE001
            ok, note = False, str(e).strip().splitlines()[-1][:200]
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {sheet} / {v['title']}: {note}", flush=True)
    return failed


class DataViz:
    """The CDW Data Visualization instance at SPEND_CDV_URL, authenticated with its API key."""

    def __init__(self):
        p = urllib.parse.urlparse(os.environ["SPEND_CDV_URL"])
        self.url = f"{p.scheme or 'https'}://{p.netloc or p.path.split('/', 1)[0]}"
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"apikey {os.environ['SPEND_CDV_API_KEY']}"

    def get(self, path: str, **params):
        r = self.s.get(self.url + path, params=params, timeout=60)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, items: list[dict]) -> list[dict]:
        r = self.s.post(self.url + path, data={"data": json.dumps(items)}, timeout=60)
        if r.status_code != 200:
            raise SystemExit(f"POST {path}: HTTP {r.status_code} {r.text[:200]}")
        return r.json()

    def connection(self) -> int:
        found = next((c for c in self.get("/arc/adminapi/v1/connections") if c["name"] == CONNECTION), None)
        if found:
            print(f"connection {CONNECTION}: exists ({found['id']})")
            return found["id"]
        from spend.config import settings
        imp = settings()["impala"]
        params = {"HOST": imp["host"], "PORT": str(imp.get("port", 443)), "USERNAME": os.environ["SPEND_IMPALA_USER"],
                  "MODE": "http", "HS2_HTTP_SQLPATH": imp.get("http_path", "cliservice"), "SOCK": "ssl",
                  "AUTH": "ldap", "SOCKET_TIMEOUT": 600, "IMPERSONATION": False, "APP_NAME": "viz",
                  "CONCURRENCY": 100, "CONCURRENCY_USER": 5, "QUERY_TIMEOUT": 120,
                  "QUERY_LOADING_WARNING_SECONDS": 20, "CACHE": {"ENABLED": 1, "RETENTION": 1800}}
        body = {"name": CONNECTION, "type": "impyla", "info": {"PARAMS": params},
                "password": os.environ["SPEND_IMPALA_PASSWORD"]}
        conn_id = self.post("/arc/adminapi/v1/connections", [body])[0]["id"]
        print(f"connection {CONNECTION}: created ({conn_id})")
        return conn_id

    def workspace(self) -> int:
        found = next((w for w in self.get("/arc/adminapi/v1/workspaces") if w["name"] == WORKSPACE), None)
        if found:
            print(f"workspace {WORKSPACE}: exists ({found['id']})")
            return found["id"]
        user = os.environ["SPEND_IMPALA_USER"]
        body = {"name": WORKSPACE, "desc": "Spend Analytics (rsingh-spend-anl): shared, everyone can view",
                "editable": True, "acl": [[1, 3, user], [2, 1, "Everyone"]]}
        ws_id = self.post("/arc/adminapi/v1/workspaces", [body])[0]["id"]
        print(f"workspace {WORKSPACE}: created ({ws_id})")
        return ws_id

    def version(self) -> dict:
        return self.get("/arc/migration/api/export/", dashboards="[]", filename="version", dry_run="False")["version"]

    def verify(self) -> int:
        """Run every visual's query through the Data API (Data Visualization -> its connection -> Impala)."""
        ids = {d["name"]: d["id"] for d in self.get("/arc/adminapi/v1/datasets")}
        failed = 0
        for sheet, v in visuals():
            dims = v.get("dims", []) + v.get("x", []) + v.get("color", [])
            dsreq = {"version": 1, "type": "SQL", "limit": v.get("limit", 1000),
                     "dimensions": [{"type": "SIMPLE", "expr": f"[{c}] as '{a}'"} for c, a in dims],
                     "aggregates": [{"expr": f"{e} as '{a}'"} for e, a in v["measures"]],
                     "filters": v.get("filters", []), "dataset_id": ids[DATASETS[v["ds"]][0]]}
            r = self.s.post(self.url + "/arc/api/data", data={"version": 1, "dsreq": json.dumps(dsreq)}, timeout=300)
            rows = json.loads(r.json()["rows"]) if r.status_code == 200 else None
            ok = rows is not None and (bool(rows) or v.get("may_be_empty", False))
            failed += not ok
            first = rows[0] if rows else (None if rows is not None else r.text[:200])
            print(f"{'ok  ' if ok else 'FAIL'} {sheet} / {v['title']}: "
                  f"{len(rows) if rows is not None else r.status_code} rows, first {first}", flush=True)
        return failed

    def import_file(self, path: Path) -> None:
        with path.open("rb") as f:
            r = self.s.post(self.url + "/arc/migration/api/import/", files={"import_file": f},
                            data={"dry_run": "False", "dataconnection_name": CONNECTION}, timeout=300)
        if r.status_code != 200:
            raise SystemExit(f"import: HTTP {r.status_code} {r.text[:500]}")
        done = r.json()
        print("import: " + ", ".join(f"{len(done.get(k, []))} {k}" for k in ("datasets", "visuals", "dashboards")))

    def move_to_workspace(self, ws_id: int) -> None:
        """The import ignores the file's workspace (it lands in the importer's Private one).
        Move every visual and dashboard of this project's datasets; POST to the visual's own
        URL updates it (POST to the collection would create a copy)."""
        names = {n for n, _, _ in DATASETS.values()}
        ours = {d["id"] for d in self.get("/arc/adminapi/v1/datasets") if d["name"] in names}
        moved = 0
        for v in self.get("/arc/adminapi/v1/visuals", detail="true"):
            if v.get("dataset_id") not in ours or v.get("workspace_id") == ws_id:
                continue
            body = {k: v[k] for k in ("id", "title", "type", "description", "dataset_id", "data")}
            body["workspace_id"] = ws_id
            r = self.s.post(f"{self.url}/arc/adminapi/v1/visuals/{v['id']}", data={"data": json.dumps([body])},
                            timeout=60)
            if r.status_code != 200:
                raise SystemExit(f"move visual {v['id']}: HTTP {r.status_code} {r.text[:200]}")
            moved += 1
        print(f"workspace {WORKSPACE}: moved {moved} (the admin API lists dashboards; their visuals go with them)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-import", action="store_true", help="write the export file only")
    p.add_argument("--check", action="store_true", help="run every visual's query directly on Impala")
    p.add_argument("--verify", action="store_true", help="only run every visual's query through the Data API")
    args = p.parse_args()
    engine = impala()
    if args.check:
        return 1 if check(engine) else 0
    viz = DataViz()
    if args.verify:
        return 1 if viz.verify() else 0
    types = {k: column_types(engine, k) for k in DATASETS}
    conn_id, ws_id = viz.connection(), viz.workspace()
    doc = build(conn_id, ws_id, viz.version(), types)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(doc['datasets'])} datasets, {len(doc['visuals'])} visuals, "
          f"{len(SHEETS)} sheets")
    if not args.no_import:
        viz.import_file(OUT)
        viz.move_to_workspace(ws_id)
        print(f"open the instance -> Visuals -> workspace {WORKSPACE}: {DASHBOARD['title']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
