"""
Streamlit dashboard: Are Ethereum Whales Smart Money?

Visualises the event study results -- hit rates, returns, and
sentiment-conditioned analysis.

Run with:
    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src.features.feature_engineer import assign_transaction_label

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Whale Signals Dashboard",
    page_icon="whale",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading (cached so it only runs once)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    """Load and prepare all data for the dashboard."""
    whale = pd.read_csv(config.PROCESSED_DATA_DIR / "whale_txs.csv")
    prices = pd.read_csv(config.PROCESSED_DATA_DIR / "eth_prices_hourly.csv")
    funding = pd.read_csv(config.PROCESSED_DATA_DIR / "eth_funding_rate.csv")
    fng = pd.read_csv(config.PROCESSED_DATA_DIR / "fear_greed_daily.csv")

    whale["timestamp_utc"] = pd.to_datetime(whale["timestamp_utc"], utc=True)
    prices["timestamp_utc"] = pd.to_datetime(prices["timestamp_utc"], utc=True)
    funding["timestamp_utc"] = pd.to_datetime(
        funding["timestamp_utc"], utc=True, format="ISO8601"
    )
    fng["date"] = pd.to_datetime(fng["date"], utc=True)

    whale = assign_transaction_label(whale)
    whale["hour_utc"] = whale["timestamp_utc"].dt.floor("h")

    # Merge funding rate onto whale transactions
    funding_sorted = funding.sort_values("timestamp_utc")
    whale = pd.merge_asof(
        whale.sort_values("hour_utc"),
        funding_sorted[["timestamp_utc", "funding_rate"]].rename(
            columns={"timestamp_utc": "hour_utc"}
        ),
        on="hour_utc",
        direction="backward",
    )
    whale["funding_rate"] = whale["funding_rate"].fillna(0)

    # Merge FnG onto whale transactions
    whale["_date"] = whale["timestamp_utc"].dt.floor("D")
    fng_m = fng.rename(columns={"date": "_date"}).sort_values("_date")
    whale = pd.merge_asof(
        whale.sort_values("_date"),
        fng_m[["_date", "fng_value"]],
        on="_date",
        direction="backward",
    )
    whale["fng_value"] = whale["fng_value"].fillna(50)
    whale.drop(columns="_date", inplace=True)

    # Compute forward returns at multiple horizons
    price_lookup = prices.set_index("timestamp_utc")["close"].to_dict()

    for h in [1, 6, 12, 24, 36, 48, 72]:
        offset = pd.Timedelta(hours=h)
        whale[f"fwd_{h}h"] = whale["hour_utc"].apply(
            lambda ts, _o=offset: (
                price_lookup.get(ts + _o, np.nan) - price_lookup.get(ts, np.nan)
            )
            / price_lookup.get(ts, np.nan)
            if price_lookup.get(ts) and price_lookup.get(ts + _o)
            else np.nan
        )

    # Compute base rates for all hours
    all_prices = prices.copy()
    all_prices = pd.merge_asof(
        all_prices,
        funding_sorted[["timestamp_utc", "funding_rate"]],
        on="timestamp_utc",
        direction="backward",
    )
    all_prices["funding_rate"] = all_prices["funding_rate"].fillna(0)
    all_prices["_date"] = all_prices["timestamp_utc"].dt.floor("D")
    all_prices = pd.merge_asof(
        all_prices.sort_values("_date"),
        fng_m[["_date", "fng_value"]],
        on="_date",
        direction="backward",
    )
    all_prices["fng_value"] = all_prices["fng_value"].fillna(50)

    for h in [1, 6, 12, 24, 36, 48, 72]:
        offset = pd.Timedelta(hours=h)
        all_prices[f"fwd_{h}h"] = all_prices["timestamp_utc"].apply(
            lambda ts, _o=offset: (
                price_lookup.get(ts + _o, np.nan) - price_lookup.get(ts, np.nan)
            )
            / price_lookup.get(ts, np.nan)
            if price_lookup.get(ts) and price_lookup.get(ts + _o)
            else np.nan
        )

    return whale, prices, all_prices, fng, funding


whale, prices, all_prices, fng, funding = load_data()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_hit_rate(returns: pd.Series, direction: str) -> dict:
    """Compute hit rate and return stats for a set of returns."""
    returns = returns.dropna()
    if len(returns) < 10:
        return None

    if direction == "up":
        hits = (returns > 0).sum()
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
    else:
        hits = (returns < 0).sum()
        wins = returns[returns < 0]
        losses = returns[returns >= 0]

    return {
        "n": len(returns),
        "hits": int(hits),
        "hit_rate": hits / len(returns),
        "mean_return": returns.mean() * 100,
        "win_mean": wins.mean() * 100 if len(wins) > 0 else 0,
        "loss_mean": losses.mean() * 100 if len(losses) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Filters")

min_usd = st.sidebar.slider(
    "Minimum transaction size (USD)",
    min_value=1_000_000,
    max_value=50_000_000,
    value=1_000_000,
    step=1_000_000,
    format="$%d",
)

filtered = whale[whale["usd_value"] >= min_usd]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Transactions:** {len(filtered):,}")
st.sidebar.markdown(
    f"**Date range:** {filtered['timestamp_utc'].min().strftime('%Y-%m-%d')} "
    f"to {filtered['timestamp_utc'].max().strftime('%Y-%m-%d')}"
)

cat_counts = filtered["tx_category"].value_counts()
st.sidebar.markdown("**Categories:**")
for cat, count in cat_counts.items():
    st.sidebar.markdown(f"- {cat}: {count:,}")

# ---------------------------------------------------------------------------
# Title and overview
# ---------------------------------------------------------------------------

st.title("Are Ethereum Whales Smart Money?")
st.markdown(
    "Measuring whether large on-chain transactions (>$1M) predict ETH price "
    "direction, and whether market sentiment moderates this relationship."
)

# Key metrics row
col1, col2, col3, col4 = st.columns(4)

withdrawals = filtered[filtered["tx_category"] == "exchange_withdrawal"]
deposits = filtered[filtered["tx_category"] == "exchange_deposit"]

neg_fund_withdrawals = withdrawals[withdrawals["funding_rate"] < 0]["fwd_24h"].dropna()
neg_fund_hit = (neg_fund_withdrawals > 0).mean() if len(neg_fund_withdrawals) > 0 else 0

col1.metric("Total Transactions", f"{len(filtered):,}")
col2.metric("Wallet Labels", "52,768")
col3.metric(
    "Best Hit Rate (24h)",
    f"{neg_fund_hit:.1%}",
    help="Whale withdrawals during negative funding rate",
)
col4.metric("Data Span", "2.5 years")

# ---------------------------------------------------------------------------
# Section 1: Hit rate across horizons
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("Whale Edge Across Time Horizons")
st.markdown(
    "How long does the whale's informational advantage last? "
    "We compare the whale withdrawal hit rate to the base rate "
    "(any random hour during negative funding)."
)

horizons = [1, 6, 12, 24, 36, 48, 72]
neg_withdrawals = withdrawals[withdrawals["funding_rate"] < 0]
neg_all = all_prices[all_prices["funding_rate"] < 0]

whale_hrs = []
base_hrs = []
edges = []

for h in horizons:
    col = f"fwd_{h}h"
    w_returns = neg_withdrawals[col].dropna()
    b_returns = neg_all[col].dropna()

    w_hr = (w_returns > 0).mean() if len(w_returns) > 0 else 0.5
    b_hr = (b_returns > 0).mean() if len(b_returns) > 0 else 0.5

    whale_hrs.append(w_hr * 100)
    base_hrs.append(b_hr * 100)
    edges.append((w_hr - b_hr) * 100)

fig_horizon = go.Figure()

fig_horizon.add_trace(go.Bar(
    x=[f"{h}h" for h in horizons],
    y=edges,
    name="Whale edge",
    marker_color=["#2ecc71" if e > 2 else "#95a5a6" if e > 0 else "#e74c3c" for e in edges],
    text=[f"+{e:.1f}%" if e > 0 else f"{e:.1f}%" for e in edges],
    textposition="outside",
))

fig_horizon.add_hline(y=0, line_dash="dash", line_color="gray")

fig_horizon.update_layout(
    title="Whale Edge Over Base Rate (Withdrawals During Negative Funding)",
    xaxis_title="Prediction Horizon",
    yaxis_title="Edge (percentage points)",
    yaxis_range=[min(edges) - 2, max(edges) + 3],
    showlegend=False,
    height=400,
)

st.plotly_chart(fig_horizon, use_container_width=True)

st.markdown(
    "The whale edge peaks at **12-24 hours** (+4.5-5.8%), then decays. "
    "By 48 hours, whatever the whale knew is priced in."
)

# ---------------------------------------------------------------------------
# Section 2: Hit rates by sentiment condition
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("Are Whales Smarter in Certain Sentiment Regimes?")

tab_withdraw, tab_deposit = st.tabs(["Withdrawals (buy signal)", "Deposits (sell signal)"])

conditions = {
    "Negative funding": lambda df: df["funding_rate"] < 0,
    "Extreme fear": lambda df: df["fng_value"] <= 25,
    "Fear": lambda df: df["fng_value"] <= 45,
    "Neutral": lambda df: (df["fng_value"] > 45) & (df["fng_value"] <= 55),
    "Greed": lambda df: df["fng_value"] > 55,
    "Extreme greed": lambda df: df["fng_value"] > 75,
    "Positive funding": lambda df: df["funding_rate"] >= 0,
}

with tab_withdraw:
    cond_names = []
    hit_rates = []
    base_rates = []
    ns = []
    colours = []

    for cond_name, cond_fn in conditions.items():
        subset = withdrawals[cond_fn(withdrawals)]["fwd_24h"].dropna()
        base_subset = all_prices[cond_fn(all_prices)]["fwd_24h"].dropna()

        if len(subset) < 30:
            continue

        hr = (subset > 0).mean() * 100
        br = (base_subset > 0).mean() * 100

        cond_names.append(cond_name)
        hit_rates.append(hr)
        base_rates.append(br)
        ns.append(len(subset))
        colours.append("#2ecc71" if hr - br > 2 else "#e74c3c" if hr - br < -2 else "#95a5a6")

    fig_w = go.Figure()

    fig_w.add_trace(go.Bar(
        x=cond_names, y=hit_rates, name="Whale hit rate",
        marker_color=colours,
        text=[f"{h:.1f}%" for h in hit_rates],
        textposition="outside",
    ))
    fig_w.add_trace(go.Scatter(
        x=cond_names, y=base_rates, name="Base rate",
        mode="markers+lines", line=dict(color="black", dash="dash"),
        marker=dict(size=8),
    ))
    fig_w.add_hline(y=50, line_dash="dot", line_color="gray",
                    annotation_text="50% (random)")
    fig_w.update_layout(
        title="Withdrawal Hit Rate by Sentiment (24h horizon)",
        yaxis_title="Hit rate (%)",
        yaxis_range=[35, 75],
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_w, use_container_width=True)

    st.markdown(
        "**Green** = whale edge above base rate. "
        "**Red** = whale edge below base rate. "
        "**Grey** = marginal. Dashed line = base rate for that condition."
    )

with tab_deposit:
    cond_names_d = []
    hit_rates_d = []
    base_rates_d = []
    colours_d = []

    for cond_name, cond_fn in conditions.items():
        subset = deposits[cond_fn(deposits)]["fwd_24h"].dropna()
        base_subset = all_prices[cond_fn(all_prices)]["fwd_24h"].dropna()

        if len(subset) < 30:
            continue

        hr = (subset < 0).mean() * 100  # deposit hit = price dropped
        br = (base_subset < 0).mean() * 100

        cond_names_d.append(cond_name)
        hit_rates_d.append(hr)
        base_rates_d.append(br)
        colours_d.append("#2ecc71" if hr - br > 2 else "#e74c3c" if hr - br < -2 else "#95a5a6")

    fig_d = go.Figure()

    fig_d.add_trace(go.Bar(
        x=cond_names_d, y=hit_rates_d, name="Whale hit rate",
        marker_color=colours_d,
        text=[f"{h:.1f}%" for h in hit_rates_d],
        textposition="outside",
    ))
    fig_d.add_trace(go.Scatter(
        x=cond_names_d, y=base_rates_d, name="Base rate",
        mode="markers+lines", line=dict(color="black", dash="dash"),
        marker=dict(size=8),
    ))
    fig_d.add_hline(y=50, line_dash="dot", line_color="gray",
                    annotation_text="50% (random)")
    fig_d.update_layout(
        title="Deposit Hit Rate by Sentiment (24h horizon)",
        yaxis_title="Hit rate (%)",
        yaxis_range=[30, 65],
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_d, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3: Return breakdown
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("Return Breakdown: Wins vs Losses")
st.markdown(
    "For whale withdrawals during negative funding (the strongest signal), "
    "what do the winning and losing trades look like?"
)

neg_wd_returns = neg_withdrawals["fwd_24h"].dropna()
wins = neg_wd_returns[neg_wd_returns > 0]
losses = neg_wd_returns[neg_wd_returns <= 0]

col1, col2, col3 = st.columns(3)

col1.metric("Win Rate", f"{len(wins)/len(neg_wd_returns):.1%}")
col1.metric("Avg Win", f"+{wins.mean()*100:.2f}%")
col1.metric("Median Win", f"+{wins.median()*100:.2f}%")

col2.metric("Loss Rate", f"{len(losses)/len(neg_wd_returns):.1%}")
col2.metric("Avg Loss", f"{losses.mean()*100:.2f}%")
col2.metric("Median Loss", f"{losses.median()*100:.2f}%")

col3.metric("Total Signals", f"{len(neg_wd_returns):,}")
col3.metric("Avg Return (all)", f"+{neg_wd_returns.mean()*100:.2f}%")
col3.metric("After 0.1% Fees", f"+{(neg_wd_returns.mean()*100 - 0.1):.2f}%")

# Return distribution histogram
fig_hist = go.Figure()

fig_hist.add_trace(go.Histogram(
    x=wins * 100,
    name="Wins (price rose)",
    marker_color="#2ecc71",
    opacity=0.7,
    nbinsx=50,
))
fig_hist.add_trace(go.Histogram(
    x=losses * 100,
    name="Losses (price fell)",
    marker_color="#e74c3c",
    opacity=0.7,
    nbinsx=50,
))

fig_hist.update_layout(
    title="Distribution of 24h Returns (Withdrawals During Negative Funding)",
    xaxis_title="24h Forward Return (%)",
    yaxis_title="Count",
    barmode="overlay",
    height=400,
)

st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4: 100-trade simulation
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("Theoretical 100-Trade Simulation")

win_pct = len(wins) / len(neg_wd_returns)
n_wins = int(round(win_pct * 100))
n_losses = 100 - n_wins
avg_win = wins.mean() * 100
avg_loss = losses.mean() * 100
fee = 0.1

gross_gain = n_wins * avg_win + n_losses * avg_loss
net_gain = gross_gain - 100 * fee

sim_col1, sim_col2 = st.columns(2)

with sim_col1:
    st.markdown(f"""
    | | Trades | Avg Return | Total |
    |---|---|---|---|
    | Wins | {n_wins} | +{avg_win:.2f}% | +{n_wins * avg_win:.1f}% |
    | Losses | {n_losses} | {avg_loss:.2f}% | {n_losses * avg_loss:.1f}% |
    | **Gross** | | | **+{gross_gain:.1f}%** |
    | Fees | 100 x 0.1% | | -{100 * fee:.1f}% |
    | **Net** | | | **+{net_gain:.1f}%** |
    """)

with sim_col2:
    capital = 10000
    final = capital * (1 + net_gain / 100)
    st.metric("Starting Capital", f"${capital:,.0f}")
    st.metric("Final Value", f"${final:,.0f}", delta=f"+${final - capital:,.0f}")
    st.markdown(
        "*Over ~2.5 years (negative funding is rare, ~9% of hours)*"
    )

# ---------------------------------------------------------------------------
# Section 5: Limitations
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("Limitations")

st.markdown("""
1. **Backtested, not live-tested.** Historical results do not guarantee future performance.
2. **Negative funding is rare.** Only ~9% of hours — the strategy is idle most of the time.
3. **Whale-specific edge is modest.** +4.5% above the base rate at 24h.
4. **On-chain latency.** By the time you see the whale transaction and react, some edge may be gone.
5. **Bull market bias.** ETH trended up over Jan 2023 - Jun 2025. Results may differ in prolonged bear markets.
6. **No causal claim.** Whales predict direction, they do not cause price movements.
7. **Survivorship bias in labels.** Unknown wallets may include unlabelled exchanges or institutions.
""")

st.markdown("---")
st.caption(
    "Data: 392,517 whale transactions (Jan 2023 - Jun 2025) | "
    "52,768 labelled addresses | "
    "Dune Analytics, Binance API, alternative.me"
)
