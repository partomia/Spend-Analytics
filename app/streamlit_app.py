"""Spend Analytics: Spend Insights Console.

  streamlit run app/streamlit_app.py            # storage from config/spend.yaml
  SPEND_STORAGE_BACKEND=parquet streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path(os.getcwd())
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app import data  # noqa: E402
from spend.client import endpoint_configured  # noqa: E402
from spend.config import policy, table  # noqa: E402
from spend.features import CATEGORIES, CHANNELS, DIGITAL_CHANNELS  # noqa: E402

st.set_page_config(page_title="Spend Analytics", page_icon=":bar_chart:", layout="wide")

DEMO_CIFS = {"C00016807": "unit 0: Goa trip (TRAVEL_SURGE)", "C00033614": "unit 1: moved to Nagpur (RELOCATION)",
             "C00050421": "unit 2: Diwali electronics (FESTIVE_ELECTRONICS)"}
CHANNEL_LABEL = {"CARD_POS": "Card (POS)", "CARD_ECOM": "Card (online)", "UPI_P2M": "UPI to merchant",
                 "UPI_P2P": "UPI to person", "NETBANKING": "Net banking", "ATM_CASH": "ATM cash"}
CHANNEL_GROUP = {"CARD_POS": "Card", "CARD_ECOM": "Card", "UPI_P2M": "UPI", "UPI_P2P": "UPI",
                 "NETBANKING": "Net banking", "ATM_CASH": "Cash"}
GROUP_COLOR = {"UPI": "#2e86ab", "Card": "#f3a712", "Net banking": "#3bb273", "Cash": "#e4572e"}
SIGNAL_COLOR = {"TRAVEL_SURGE": "#2e86ab", "FESTIVE_ELECTRONICS": "#f3a712", "RELOCATION": "#3bb273"}
BAND = "rgba(46,134,171,0.18)"


def lbl(code: str) -> str:
    return code.replace("_", " ").capitalize()


def pct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.1%}"


def inr(x) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    x = float(x)
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:,.2f} cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:,.1f} L"
    return f"₹{x:,.0f}"


def num(x, fmt: str = "{:.3f}") -> str:
    return "n/a" if x is None or pd.isna(x) else fmt.format(x)


def fig_layout(fig: go.Figure, title: str = "", height: int = 320, **kw) -> go.Figure:
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40 if title else 10, b=10),
                      legend=dict(orientation="h", y=-0.2), **kw)
    return fig


def chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch")


def fan(fig: go.Figure, x, lo, hi, mid, name: str, color: str = "#2e86ab") -> None:
    x = list(x)
    fig.add_trace(go.Scatter(x=x + x[::-1], y=list(hi) + list(lo)[::-1], fill="toself", fillcolor=BAND,
                             line=dict(width=0), name=f"{name} P10-P90", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=list(mid), mode="lines+markers", name=f"{name} P50", line=dict(color=color)))


# ---------------------------------------------------------------- sidebar
try:
    runs = data.read("spend_model_run").sort_values(["run_date", "run_ts"], ascending=False)
except Exception as e:  # no data yet / connection problem
    st.error(f"Could not read {table('spend_model_run')} via {data.storage().name}: {e}")
    st.info("Run the daily run first (cai/jobs/daily_run.py, or the three CAI jobs), or set "
            "SPEND_STORAGE_BACKEND=parquet with exported tables in data/parquet.")
    st.stop()
if runs.empty:
    st.warning("No daily run yet. Run cai/jobs/daily_run.py first.")
    st.stop()
runs = runs.drop_duplicates("run_date")

st.sidebar.title("Daily run")
run_date = st.sidebar.selectbox("Run date", runs["run_date"].tolist(), format_func=lambda d: d.strftime("%a %d %b %Y"))
run = runs[runs["run_date"] == run_date].iloc[0]
published = bool(run.outputs_published)
st.sidebar.caption(f"Run `{run.run_id}` · {run.triggered_by}  \nEmbedder `{run.embedder_id}`  \n"
                   f"Forecaster `{run.forecaster_id}` on `{run.device}`  \nStorage: {data.storage().name}")
(st.sidebar.success if bool(run.gate_passed) else st.sidebar.error)(
    f"KPI gate: {'PASS' if bool(run.gate_passed) else 'FAIL'} · outputs published: {published}")

dq = data.dq_results()
if not dq.empty:
    st.sidebar.markdown("**Data freshness: last DQ run per layer**")
    for layer in ("bronze", "silver", "gold"):
        d = dq[dq["layer"] == layer]
        if d.empty:
            continue
        last = d[d["run_ts"] == d["run_ts"].max()]
        crit = int((~last["success"] & (last["severity"] == "critical")).sum())
        warn = int((~last["success"] & (last["severity"] != "critical")).sum())
        msg = (f"{layer}: as of {last['as_of'].iloc[0]}, {len(last)} checks, {crit} critical / {warn} warning "
               f"failed")
        (st.sidebar.error if crit else st.sidebar.warning if warn else st.sidebar.success)(msg)
st.sidebar.caption("Customer 360 and the Categoriser call the "
                   + ("**CAI model endpoint**." if endpoint_configured() else "scorer in this app (no endpoint set)."))
st.sidebar.divider()
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption("Demo on synthetic data. Personas and signals suggest who to talk to; nothing is offered to a "
                   "customer automatically.")

personas = data.for_run("customer_persona", run_date)
profile = data.for_run("persona_profile", run_date).sort_values("persona_id") if published else pd.DataFrame()
signals = data.for_run("spend_signals", run_date)
mix = data.mix()
if not mix.empty:
    mix = mix[mix["week_end"] <= run.snapshot_date]
weeks = sorted(mix["week_end"].unique()) if not mix.empty else []
last13 = weeks[-13:]

st.title("Spend Analytics: Spend Insights Console")
st.caption(f"Run {run_date:%d %b %Y}, snapshot {run.snapshot_date}. Six source systems (CRM, core banking, card "
           "switch, UPI switch, net banking / bill pay, ATM switch) linked on the CIF; every merchant payment "
           "categorised (MCC, else MiniLM on the payee text); KMeans personas, a Chronos-2 cohort forecast and "
           "rule-based opportunity signals.")
if not published:
    st.warning("The KPI gate failed on this run, so its personas, forecast and signals were not published; the "
               "previous published run stays live (see Lineage). Book-wide views below still show gold.")

(tab_port, tab_where, tab_chan, tab_pers, tab_360, tab_fc, tab_cat, tab_opp, tab_dq, tab_lin) = st.tabs(
    ["Portfolio", "Where", "Channels", "Personas", "Customer 360", "Forecast", "Categoriser", "Opportunities",
     "Data quality", "Lineage"])

# ---------------------------------------------------------------- portfolio
with tab_port:
    if mix.empty:
        st.info(f"No rows in {table('spend_mix_weekly')} yet: run the CDE gold job.")
    else:
        cat = mix[mix["dim"] == "category"]
        recent = cat[cat["week_end"].isin(last13)]
        total13 = recent["amount_inr"].sum()
        digital13 = recent[recent["channel"].isin(DIGITAL_CHANNELS)]["amount_inr"].sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Spend, last 13 weeks", inr(total13))
        c2.metric("Transactions, last 13 weeks", f"{int(recent['n_txn'].sum()):,}")
        c3.metric("Digital share", pct(digital13 / total13 if total13 else None))
        c4.metric("Customers profiled", f"{int(run.n_customers_profiled):,}")
        weekly = cat.pivot_table(index="week_end", columns="dim_value", values="amount_inr", aggfunc="sum").fillna(0)
        fig = go.Figure()
        order = [c for c in [*CATEGORIES, "UNCATEGORISED", "NON_MERCHANT"] if c in weekly.columns]
        for c in order:
            fig.add_trace(go.Scatter(x=weekly.index, y=weekly[c], stackgroup="one", mode="lines", line=dict(width=0.5),
                                     name="Cash + UPI to person" if c == "NON_MERCHANT" else lbl(c)))
        chart(fig_layout(fig, "Weekly spend by category (₹)", 380))
        left, right = st.columns(2)
        with left:
            ch = cat.assign(group=cat["channel"].map(CHANNEL_GROUP)).pivot_table(
                index="week_end", columns="group", values="amount_inr", aggfunc="sum").fillna(0)
            share = ch.div(ch.sum(axis=1), axis=0)
            fig = go.Figure()
            for g in [g for g in GROUP_COLOR if g in share.columns]:
                fig.add_trace(go.Scatter(x=share.index, y=share[g], mode="lines", name=g,
                                         line=dict(color=GROUP_COLOR[g])))
            chart(fig_layout(fig, "Share of spend: UPI vs card vs cash", 320, yaxis=dict(tickformat=".0%")))
        with right:
            tier = mix[(mix["dim"] == "city_tier") & mix["week_end"].isin(last13)].groupby("dim_value")["amount_inr"].sum()
            fig = go.Figure(go.Bar(x=[f"Tier {t}" for t in tier.index], y=tier.values, marker_color="#2e86ab",
                                   text=[inr(v) for v in tier.values], textposition="outside"))
            chart(fig_layout(fig, "Where the spend happened, by city tier (13 weeks)", 320))
        state = (mix[(mix["dim"] == "state") & mix["week_end"].isin(last13)].groupby("dim_value")
                 .agg(amount=("amount_inr", "sum"), txn=("n_txn", "sum")).reset_index()
                 .merge(data.state_centroids(), left_on="dim_value", right_on="state", how="inner"))
        if not state.empty:
            fig = go.Figure(go.Scattergeo(
                lat=state["lat"], lon=state["lon"], text=[f"{s}: {inr(a)}" for s, a in zip(state["state"], state["amount"])],
                marker=dict(size=(state["amount"] / state["amount"].max()) ** 0.5 * 45 + 5, color="#e4572e",
                            opacity=0.6, line=dict(width=0)), hoverinfo="text"))
            fig.update_geos(fitbounds="locations", showcountries=True, showland=True, landcolor="#f4f4f4",
                            showsubunits=True)
            chart(fig_layout(fig, "Spend by state (13 weeks, bubble area = amount)", 460))

# ---------------------------------------------------------------- where
with tab_where:
    if mix.empty:
        st.info("No mix data yet.")
    else:
        total = mix[mix["dim"] == "category"].groupby("week_end")["amount_inr"].sum()
        away = mix[mix["dim"] == "away_city"].groupby("week_end")["amount_inr"].sum().reindex(total.index, fill_value=0)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=total.index, y=total - away, name="At home", marker_color="#9aa5b1"))
        fig.add_trace(go.Bar(x=away.index, y=away, name="Away from home", marker_color="#2e86ab"))
        chart(fig_layout(fig, "Home vs away spend per week (₹)", 340, barmode="stack"))
        left, right = st.columns(2)
        with left:
            top = (mix[(mix["dim"] == "away_city") & mix["week_end"].isin(last13)].groupby("dim_value")
                   .agg(amount=("amount_inr", "sum"), customers=("n_customers", "max")).nlargest(12, "amount"))
            fig = go.Figure(go.Bar(y=top.index[::-1], x=top["amount"][::-1], orientation="h", marker_color="#2e86ab",
                                   text=[inr(v) for v in top["amount"][::-1]], textposition="outside"))
            chart(fig_layout(fig, "Top cities for away-from-home spend (13 weeks)", 400))
        with right:
            src = (mix[(mix["dim"] == "location_source") & mix["week_end"].isin(last13)].groupby("dim_value")
                   ["n_txn"].sum())
            fig = go.Figure(go.Pie(labels=src.index, values=src.values, hole=0.45, sort=False))
            chart(fig_layout(fig, "Where the location comes from (share of transactions, 13 weeks)", 400))
            fb = src.get("home_fallback", 0) / src.sum() if src.sum() else None
            st.caption(f"Card POS uses the merchant city, UPI to merchant the QR pincode, ATM the ATM's pincode. "
                       f"Online card, net banking, UPI to person and UPI without a pincode have no place, so they are "
                       f"placed at home and marked home_fallback: {pct(fb)} of transactions here. Away share is "
                       f"measured on located spend only.")

# ---------------------------------------------------------------- channels
with tab_chan:
    left, right = st.columns(2)
    with left:
        if profile.empty:
            st.info("No published personas for this run.")
        else:
            fig = go.Figure()
            cents = [json.loads(t) for t in profile["centroid_json"]]
            for c in CHANNELS:
                fig.add_trace(go.Bar(y=profile["persona_name"], x=[d[f"ch_{c.lower()}_share"] for d in cents],
                                     orientation="h", name=CHANNEL_LABEL[c]))
            chart(fig_layout(fig, "Channel mix by persona (share of spend)", 420, barmode="stack",
                             xaxis=dict(tickformat=".0%")))
    with right:
        if not mix.empty:
            age = mix[(mix["dim"] == "age_band") & mix["week_end"].isin(last13)].pivot_table(
                index="dim_value", columns="channel", values="amount_inr", aggfunc="sum").fillna(0)
            age = age.div(age.sum(axis=1), axis=0)
            fig = go.Figure()
            for c in [c for c in CHANNELS if c in age.columns]:
                fig.add_trace(go.Bar(x=age.index, y=age[c], name=CHANNEL_LABEL[c]))
            chart(fig_layout(fig, "Channel mix by age band (13 weeks)", 420, barmode="stack",
                             yaxis=dict(tickformat=".0%")))
    if not mix.empty:
        seg = mix[mix["dim"] == "segment"]
        fig = go.Figure()
        for s, g in seg.groupby("dim_value"):
            w = g.groupby("week_end")["amount_inr"].sum()
            dig = g[g["channel"].isin(DIGITAL_CHANNELS)].groupby("week_end")["amount_inr"].sum().reindex(w.index,
                                                                                                           fill_value=0)
            fig.add_trace(go.Scatter(x=w.index, y=dig / w, mode="lines", name=s))
        chart(fig_layout(fig, "Digital share of spend per week, by segment (cash-to-digital migration)", 340,
                         yaxis=dict(tickformat=".0%")))

# ---------------------------------------------------------------- personas
with tab_pers:
    if profile.empty:
        st.info("No published personas for this run.")
    else:
        st.markdown(f"KMeans on 27 standardised spend features for {int(run.n_customers_profiled):,} customers "
                    f"(k = {int(run.k)}, silhouette {num(run.silhouette)}). IDs are matched to the previous run, "
                    f"so a persona keeps its ID and name: stability {pct(run.persona_stability)} of customers kept "
                    "their persona since the last run.")
        for row in range(0, len(profile), 4):
            cols = st.columns(4)
            for col, (_, p) in zip(cols, profile.iloc[row:row + 4].iterrows()):
                with col.container(border=True):
                    st.markdown(f"**{p['persona_name']}**  \n`#{int(p['persona_id'])}`")
                    st.metric("Customers", f"{int(p['n_customers']):,}", pct(p["customer_share"]), delta_color="off")
                    st.caption(f"Avg 13-week spend {inr(p['avg_spend_13w'])} · {num(p['avg_txn_13w'], '{:.0f}')} txns  \n"
                               f"Top: {lbl(p['top_category'])} via {CHANNEL_LABEL.get(p['top_channel'], p['top_channel'])}"
                               f"  \nDigital {pct(p['digital_share'])} · home {pct(p['home_share'])}")
        cents = {int(p["persona_id"]): json.loads(p["centroid_json"]) for _, p in profile.iterrows()}
        weights = profile.set_index("persona_id")["n_customers"].astype(float)
        book = {c: sum(cents[i][f"cat_{c.lower()}_share"] * weights[i] for i in cents) / weights.sum()
                for c in CATEGORIES}
        names = dict(zip(profile["persona_id"].astype(int), profile["persona_name"]))
        pid = st.selectbox("Persona", list(names), format_func=names.get, key="persona_pick")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[lbl(c) for c in CATEGORIES], y=[cents[pid][f"cat_{c.lower()}_share"] for c in CATEGORIES],
                             name=names[pid], marker_color="#2e86ab"))
        fig.add_trace(go.Bar(x=[lbl(c) for c in CATEGORIES], y=[book[c] for c in CATEGORIES], name="Book average",
                             marker_color="#9aa5b1"))
        chart(fig_layout(fig, "Category mix of the persona centroid vs the book", 340, yaxis=dict(tickformat=".0%")))
        flow = data.for_run("persona_flow", run_date)
        if flow.empty:
            st.info("No previous run to compare with: persona flows appear from the second run.")
        else:
            prev = flow["prev_run_date"].iloc[0]
            ids = sorted(set(flow["from_persona_id"].astype(int)) | set(flow["to_persona_id"].astype(int)))
            fnames = dict(zip(flow["from_persona_id"].astype(int), flow["from_persona_name"]))
            tnames = dict(zip(flow["to_persona_id"].astype(int), flow["to_persona_name"]))
            idx = {i: k for k, i in enumerate(ids)}
            label = [f"{fnames.get(i, tnames.get(i))} ({prev:%d %b})" for i in ids] + \
                    [f"{tnames.get(i, fnames.get(i))} ({run_date:%d %b})" for i in ids]
            moved = flow[flow["from_persona_id"] != flow["to_persona_id"]]
            fig = go.Figure(go.Sankey(node=dict(label=label, pad=12),
                                      link=dict(source=[idx[int(i)] for i in flow["from_persona_id"]],
                                                target=[len(ids) + idx[int(i)] for i in flow["to_persona_id"]],
                                                value=flow["n_customers"].astype(int).tolist())))
            chart(fig_layout(fig, f"Persona flow {prev:%d %b} -> {run_date:%d %b} "
                                  f"({int(moved['n_customers'].sum()):,} customers changed persona)", 480))

# ---------------------------------------------------------------- customer 360
with tab_360:
    st.markdown("One customer across the six systems, resolved to one CIF: persona, 13-week mix, where they spend, "
                "signals, and a 12-week forecast from Chronos-2 on their own history.")
    options = [c for c in DEMO_CIFS if personas.empty or c in set(personas["cif"])]
    options += [c for c in signals["cif"].drop_duplicates().head(50) if c not in options]
    options += [c for c in personas["cif"].head(50) if c not in options]
    typed = st.text_input("Any CIF", key="c360_cif", placeholder="e.g. C00016807")
    cif = typed.strip() if typed.strip() else (
        st.selectbox("Customer", options, key="c360_select", format_func=lambda c: f"{c} · {DEMO_CIFS[c]}"
                     if c in DEMO_CIFS else c) if options else None)
    if cif:
        prof, src = data.ask(json.dumps({"action": "profile", "cif": cif}))
        if "error" in prof:
            st.info(prof["error"])
        else:
            cu, pe, s13, hva = prof["customer"], prof["persona"], prof["spend_13w"], prof["home_vs_away"]
            st.caption(f"`{cif}` · {cu['segment']} · age {cu['age_band']} · income {cu['income_band']} · "
                       f"{cu['home_city']}, {cu['home_state']} (tier {cu['city_tier']}) · as of {prof['as_of']} · "
                       f"answered by: {src}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Persona", pe["name"], f"was #{pe['prev_id']}" if pe.get("prev_id") not in (None, pe["id"])
                      else None, delta_color="off")
            c2.metric("Spend, 13 weeks", inr(s13["total"]), f"{s13['n_txn']} txns", delta_color="off")
            c3.metric("Last 4 weeks vs 13-week pace", f"{num(s13['trend_4v13'], '{:.2f}')}x")
            c4.metric("Away from home, 4 weeks", pct(hva["away_share_4w"]), f"{hva['away_weeks_4w']} of 4 weeks away",
                      delta_color="off")
            for sgn in prof["signals"]:
                st.success(f"**{sgn['signal_id']}** · {sgn['action']}: {sgn['detail']}")
            g1, g2 = st.columns(2)
            with g1:
                bc = s13["by_category"]
                fig = go.Figure(go.Bar(y=[lbl(c) for c in bc][::-1], x=list(bc.values())[::-1], orientation="h",
                                       marker_color="#2e86ab"))
                chart(fig_layout(fig, "13-week spend by category (₹)", 340))
            with g2:
                bch = s13["by_channel"]
                fig = go.Figure(go.Pie(labels=[CHANNEL_LABEL.get(c, c) for c in bch], values=list(bch.values()),
                                       hole=0.45))
                chart(fig_layout(fig, "13-week spend by channel", 340))
            if hva["top_away_cities"]:
                st.caption("Away from home (13 weeks): " + ", ".join(
                    f"{c['city']} {inr(c['amount_inr'])} ({c['n_txn']} txns)" for c in hva["top_away_cities"]))

            cats = [c for c in s13["by_category"] if c in CATEGORIES][:3]
            pick = st.multiselect("Forecast categories", CATEGORIES, default=cats, key="c360_fc_cats")
            if pick:
                fcr, _ = data.ask(json.dumps({"action": "forecast", "cif": cif, "categories": pick}))
                if "error" in fcr:
                    st.error(fcr["error"])
                else:
                    hist, fc = pd.DataFrame(fcr["history"]), pd.DataFrame(fcr["forecast"])
                    tot = pd.DataFrame(fcr["totals"])
                    cols = st.columns(len(tot))
                    for col, t in zip(cols, tot.itertuples()):
                        col.metric(f"{lbl(t.category)}: expected next 12 weeks", inr(t.expected_12w),
                                   f"last 13 weeks {inr(t.last_13w)}", delta_color="off")
                    for c in pick:
                        h, f = hist[hist["category"] == c], fc[fc["category"] == c]
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=h["week_end"], y=h["amount"], name="Actual", marker_color="#9aa5b1"))
                        fan(fig, f["week_end"], f["p10"], f["p90"], f["p50"], "Forecast")
                        fig.add_trace(go.Scatter(x=f["week_end"], y=f["mean"], mode="lines", name="Expected (mean)",
                                                 line=dict(color="#e4572e", dash="dot")))
                        chart(fig_layout(fig, f"{lbl(c)}: last 13 weeks and the next 12 (₹ per week)", 300))
                    st.caption(f"{fcr['model'].get('endpoint_forecaster', '')} on {fcr['history_weeks']} weeks of this "
                               "customer's history with the calendar covariates. Weekly spend in one category is often "
                               "zero, so P50 can sit at 0; the expected (mean) line adds up to the 12-week total.")

            st.markdown("**What-if: change this customer's spend and see the persona**")
            w1, w2, w3 = st.columns(3)
            wcat = w1.selectbox("Category", CATEGORIES, index=CATEGORIES.index("TRAVEL"), key="wi_cat")
            wmul = w2.slider("Multiply its 13-week spend by", 0.0, 5.0, 2.0, 0.1, key="wi_mult")
            wcash = w3.slider("Multiply ATM cash by", 0.0, 3.0, 1.0, 0.1, key="wi_cash")
            if st.button("Run what-if", type="primary", key="wi_run"):
                shift = {wcat: wmul, **({"channel": {"ATM_CASH": wcash}} if wcash != 1.0 else {})}
                wi, _ = data.ask(json.dumps({"action": "what_if", "cif": cif, "shift": shift}))
                if "error" in wi:
                    st.error(wi["error"])
                else:
                    (st.success if wi["moved"] else st.info)(wi["summary"])
                    d = pd.DataFrame(wi["distances"])
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=d["name"], y=d["before"], name="Before", marker_color="#9aa5b1"))
                    fig.add_trace(go.Bar(x=d["name"], y=d["after"], name="After", marker_color="#2e86ab"))
                    chart(fig_layout(fig, "Distance to each persona centroid (lower = closer)", 320))

# ---------------------------------------------------------------- forecast
with tab_fc:
    fcst = data.for_run("spend_forecast", run_date)
    bt = data.for_run("spend_backtest", run_date)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("WAPE, Chronos-2", num(run.wape_model), help="sum |actual - P50| / sum actual over the backtest")
    c2.metric("WAPE, seasonal naive", num(run.wape_naive), help="same week last year x recent trend")
    c3.metric("Ratio (gate <= 0.90)", num(run.wape_ratio))
    c4.metric("P10-P90 coverage (gate 60-95%)", pct(run.coverage))
    cw = data.cohort_weekly()
    if fcst.empty or cw.empty:
        st.info("No published forecast for this run.")
    else:
        f1, f2, f3 = st.columns(3)
        seg = f1.selectbox("Segment", sorted(fcst["segment"].unique()), key="fc_seg")
        tier = f2.selectbox("City tier", sorted(fcst["city_tier"].astype(int).unique()), key="fc_tier")
        catf = f3.selectbox("Category", CATEGORIES, index=CATEGORIES.index("GROCERY"), key="fc_cat")
        cohort = f"{seg}|T{tier}"
        h = cw[(cw["cohort"] == cohort) & (cw["category"] == catf) & (cw["week_end"] <= run.snapshot_date)]
        f = fcst[(fcst["cohort"] == cohort) & (fcst["category"] == catf)].sort_values("week_end")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h["week_end"], y=h["amount_inr"], mode="lines", name="Actual",
                                 line=dict(color="#555")))
        fan(fig, f["week_end"], f["p10"], f["p90"], f["p50"], "Chronos-2")
        fig.add_trace(go.Scatter(x=f["week_end"], y=f["naive"], mode="lines", name="Seasonal naive",
                                 line=dict(color="#e4572e", dash="dot")))
        chart(fig_layout(fig, f"{cohort} · {lbl(catf)}: weekly spend and the next 12 weeks (₹)", 380))
        if not bt.empty:
            cut = st.selectbox("Backtest cut-off", sorted(bt["cutoff"].unique()), key="fc_cut")
            b = bt[(bt["cohort"] == cohort) & (bt["category"] == catf) & (bt["cutoff"] == cut)].sort_values("week_end")
            fig = go.Figure()
            fan(fig, b["week_end"], b["p10"], b["p90"], b["p50"], "Chronos-2")
            fig.add_trace(go.Scatter(x=b["week_end"], y=b["actual"], mode="lines+markers", name="Actual",
                                     line=dict(color="#555")))
            fig.add_trace(go.Scatter(x=b["week_end"], y=b["naive"], mode="lines", name="Seasonal naive",
                                     line=dict(color="#e4572e", dash="dot")))
            chart(fig_layout(fig, f"Backtest from {cut}: the 12 hidden weeks", 320))
            by = bt.groupby("category").apply(lambda g: pd.Series({
                "WAPE Chronos-2": (g["actual"] - g["p50"]).abs().sum() / g["actual"].sum(),
                "WAPE naive": (g["actual"] - g["naive"]).abs().sum() / g["actual"].sum(),
                "coverage": ((g["actual"] >= g["p10"]) & (g["actual"] <= g["p90"])).mean()}), include_groups=False)
            st.dataframe(by.reset_index(), hide_index=True, width="stretch", column_config={
                c: st.column_config.NumberColumn(format="%.3f") for c in ("WAPE Chronos-2", "WAPE naive", "coverage")})

# ---------------------------------------------------------------- categoriser
with tab_cat:
    st.markdown("UPI payments carry no MCC. Payee names go MCC -> exact seed match -> MiniLM + character n-gram "
                "nearest neighbours -> fallback. Try any payee name:")
    texts = st.text_area("Payee names, one per line", "RAMESH KIRANA STORE\nUBER TRIP 8841\nPAYTM*BESCOM BILL",
                         key="cat_texts")
    lines = [t.strip() for t in texts.splitlines() if t.strip()][:50]
    if st.button("Categorise", type="primary", key="cat_run") and lines:
        resp, src = data.ask(json.dumps({"action": "categorise", "merchant_texts": lines}))
        if "error" in resp:
            st.error(resp["error"])
        else:
            st.dataframe(pd.DataFrame(resp["results"]), hide_index=True, width="stretch", column_config={
                "similarity": st.column_config.ProgressColumn("similarity", min_value=0, max_value=1, format="%.2f")})
            st.caption(f"Answered by: {src} · {resp['model'].get('endpoint_embedder', '')}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Seed holdout accuracy (gate >= 0.85)", pct(run.category_accuracy_seed), f"n={int(run.n_seed_holdout):,}",
              delta_color="off")
    c2.metric("Card merchants, MCC hidden", pct(run.category_accuracy_mcc), f"n={int(run.n_mcc_holdout):,}",
              delta_color="off")
    c3.metric("Uncategorised share of merchant spend", pct(run.uncategorised_share))
    mm = data.map_methods()
    left, right = st.columns(2)
    with left:
        if mm.empty:
            st.info("No category map yet.")
        else:
            by = mm.groupby("method")["keys"].sum().reindex(["MCC", "SEED", "EMBEDDING", "FALLBACK"]).fillna(0)
            fig = go.Figure(go.Bar(x=by.index, y=by.values, marker_color="#2e86ab",
                                   text=[f"{int(v):,}" for v in by.values], textposition="outside"))
            chart(fig_layout(fig, f"Merchant keys in the map by method ({int(by.sum()):,})", 320))
    with right:
        h = runs.sort_values("run_date")
        fig = go.Figure()
        for col, name in (("category_accuracy_seed", "Seed holdout"), ("category_accuracy_mcc", "MCC hidden"),
                          ("uncategorised_share", "Uncategorised share")):
            fig.add_trace(go.Scatter(x=h["run_date"], y=h[col], mode="lines+markers", name=name))
        chart(fig_layout(fig, "Accuracy and uncategorised share per run", 320, yaxis=dict(tickformat=".0%")))

# ---------------------------------------------------------------- opportunities
with tab_opp:
    if signals.empty:
        st.info("No published signals for this run.")
    else:
        rules = {r["id"]: r for r in policy()["signals"]}
        cols = st.columns(max(len(rules), 1))
        for col, (rid, r) in zip(cols, rules.items()):
            n = int((signals["signal_id"] == rid).sum())
            col.metric(rid.replace("_", " ").title(), f"{n:,}", r.get("action", ""), delta_color="off")
        f1, f2, f3 = st.columns(3)
        rule_f = f1.multiselect("Signal", sorted(signals["signal_id"].unique()), key="opp_rule")
        seg_f = f2.multiselect("Segment", sorted(signals["segment"].dropna().unique()), key="opp_seg")
        per_f = f3.multiselect("Persona", sorted(signals["persona_name"].dropna().unique()), key="opp_persona")
        view = signals
        for col, sel in (("signal_id", rule_f), ("segment", seg_f), ("persona_name", per_f)):
            if sel:
                view = view[view[col].isin(sel)]
        reach = view["cif"].nunique()
        base = int(run.n_customers_profiled) or 1
        st.markdown(f"**Estimated reach: {reach:,} customers** ({reach / base:.1%} of the {base:,} profiled), "
                    f"{len(view):,} signals.")
        left, right = st.columns([3, 2])
        with left:
            st.dataframe(view[["cif", "signal_id", "action", "detail", "segment", "home_city", "persona_name",
                               "metric_value", "threshold"]].sort_values(["signal_id", "metric_value"],
                                                                         ascending=[True, False]),
                         hide_index=True, width="stretch", height=420)
            st.download_button("Download list (CSV)", view.to_csv(index=False).encode(),
                               f"spend_signals_{run_date:%Y%m%d}.csv", "text/csv", key="opp_download")
        with right:
            byp = view.groupby(["persona_name", "signal_id"]).size().reset_index(name="n")
            fig = go.Figure()
            for rid, g in byp.groupby("signal_id"):
                fig.add_trace(go.Bar(y=g["persona_name"], x=g["n"], orientation="h", name=rid,
                                     marker_color=SIGNAL_COLOR.get(rid)))
            chart(fig_layout(fig, "Signals by persona", 420, barmode="stack"))
        st.caption("Signals are rules in config/policy.yaml on the gold features, not model outputs; the volumes are "
                   "a policy setting (min amounts, ratios).")

# ---------------------------------------------------------------- data quality
with tab_dq:
    st.markdown(f"Great Expectations results from the CDE pipeline, appended to `{table('dq_results')}` after each "
                "layer. A critical failure stops the pipeline before the next layer is built.")
    if dq.empty:
        st.info("No data-quality results yet.")
    else:
        flagged = dq.assign(failed=~dq["success"], critical_failed=~dq["success"] & (dq["severity"] == "critical"))
        summary = (flagged.groupby(["pipeline_run", "layer"])
                   .agg(as_of=("as_of", "first"), run_ts=("run_ts", "max"), checks=("success", "size"),
                        failed=("failed", "sum"), critical_failed=("critical_failed", "sum"))
                   .reset_index())
        summary["pass_rate"] = 1 - summary["failed"] / summary["checks"]
        st.dataframe(summary.sort_values("run_ts", ascending=False), hide_index=True, width="stretch",
                     column_config={"pass_rate": st.column_config.ProgressColumn("pass rate", min_value=0,
                                                                                 max_value=1, format="%.0%%")})
        pr = st.selectbox("Pipeline run", summary.sort_values("run_ts", ascending=False)["pipeline_run"].unique(),
                          key="dq_run")
        sel = dq[dq["pipeline_run"] == pr]
        failed = sel[~sel["success"]]
        if failed.empty:
            st.success(f"{pr}: all {len(sel)} checks passed.")
        else:
            st.error(f"{pr}: {len(failed)} of {len(sel)} checks failed.")
            st.dataframe(failed[["layer", "table_name", "check_name", "severity", "observed_value", "unexpected_count",
                                 "unexpected_pct"]], hide_index=True, width="stretch")
        st.markdown("**Iceberg snapshot of each table checked** (time-travel to the exact data the gate saw):")
        st.dataframe(sel.groupby(["layer", "table_name"])["table_snapshot_id"].first().astype(str).reset_index(),
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------- lineage
with tab_lin:
    st.markdown("**KPIs by run.** Every daily run is kept, keyed by run date, including runs the gate stopped.")
    h = runs.sort_values("run_date")
    fig = go.Figure()
    for col, name in (("silhouette", "Silhouette"), ("persona_stability", "Persona stability"),
                      ("wape_ratio", "WAPE ratio vs naive"), ("coverage", "P10-P90 coverage"),
                      ("ari_hidden", "ARI vs hidden archetypes")):
        fig.add_trace(go.Scatter(x=h["run_date"], y=h[col], name=name, mode="lines+markers"))
    chart(fig_layout(fig, "", 320))
    for line in str(run.gate_detail or "").split(" | "):
        if line:
            (st.success if line.startswith("PASS") else st.error)(line)
    st.markdown("**Lineage**: each run records the models and package versions, the Iceberg snapshots it read, the "
                "stage durations and its MLflow run (CAI Experiments).")
    cols = ["run_date", "run_id", "run_ts", "triggered_by", "gate_passed", "outputs_published", "embedder_id",
            "forecaster_id", "model_versions", "device", "features_snapshot_id", "cohort_snapshot_id",
            "map_snapshot_id", "mlflow_run_id", "k", "n_signals", "duration_s", "stage_durations_json"]
    st.dataframe(runs[[c for c in cols if c in runs]].astype({c: str for c in cols if c.endswith("snapshot_id")
                                                               and c in runs}),
                 hide_index=True, width="stretch")
    snap = run.features_snapshot_id if isinstance(run.features_snapshot_id, str) and run.features_snapshot_id \
        else "<snapshot_id>"
    st.markdown("Read exactly what this run read, in Hue (CDW Impala):")
    st.code(f"SELECT cif, spend_13w, cat_travel_share, away_share_4w\n"
            f"  FROM {table('spend_features')}\n  FOR SYSTEM_VERSION AS OF {snap}\n"
            f"  WHERE snapshot_date = DATE '{run.snapshot_date}'\n  ORDER BY spend_13w DESC LIMIT 20;", language="sql")
    prev_week = run.snapshot_date - timedelta(weeks=1)
    st.code(f"-- the same customers as the table stood a week earlier\n"
            f"SELECT COUNT(*) FROM {table('spend_features')}\n"
            f"  FOR SYSTEM_TIME AS OF '{prev_week} 23:59:59'\n  WHERE snapshot_date = DATE '{prev_week}';",
            language="sql")
