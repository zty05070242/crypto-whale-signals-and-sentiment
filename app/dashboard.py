"""
Streamlit dashboard: Are Ethereum Whales Smart Money?

Terminal-research aesthetic over a rigorous event study. Renders entirely from
a pre-computed artefact (`app/dashboard_data.json`) built by
`scripts/build_dashboard_data.py`. This keeps the app fast and lets it deploy
to Streamlit Community Cloud without the 187 MB raw dataset.

Run locally:
    streamlit run app/dashboard.py

Rebuild the data after the underlying dataset changes:
    python scripts/build_dashboard_data.py
"""

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Whale Signals Terminal",
    page_icon="\U0001F4C8",
    layout="wide",
)

DATA_PATH = Path(__file__).resolve().parent / "dashboard_data.json"

# ---------------------------------------------------------------------------
# Theme constants (shared by CSS and Plotly so the whole surface is consistent)
# ---------------------------------------------------------------------------

BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
TEXT = "#c9d1d9"
MUTED = "#8b949e"
GREEN = "#26a69a"
RED = "#ef5350"
BLUE = "#58a6ff"
GREY = "#484f58"
GRID = "#21262d"
MONO = "'SFMono-Regular', 'JetBrains Mono', 'Menlo', monospace"


def style_fig(fig: go.Figure, height: int = 400) -> go.Figure:
    """Apply the terminal theme to a Plotly figure in place."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=MONO, color=TEXT, size=12),
        height=height,
        # Roomy top margin holds the title (left) and the legend (right) on one
        # band so neither collides with the plot, its labels, or each other.
        margin=dict(l=55, r=25, t=70, b=55),
        title=dict(font=dict(family=MONO, color=TEXT, size=14),
                   x=0, xanchor="left", y=0.97, yanchor="top"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED),
                    orientation="h", yanchor="bottom", y=1.02,
                    x=1, xanchor="right"),
        bargap=0.28,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER)
    return fig


def edge_colour(edge: float, hi: float = 1.5) -> str:
    """Colour a bar by whether the whale edge clears the base rate."""
    e = edge or 0
    if e > hi:
        return GREEN
    if e < -hi:
        return RED
    return GREY


def z(v) -> float:
    """Coerce a possibly-null aggregate to a plottable number."""
    return 0.0 if v is None else float(v)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> dict:
    """Load the pre-computed aggregate artefact."""
    with open(DATA_PATH) as f:
        return json.load(f)


DATA = load_data()
META = DATA["meta"]
YEARS = DATA["years"]
HORIZON_LABELS = DATA["horizon_labels"]

# ---------------------------------------------------------------------------
# Custom CSS: quant-terminal chrome
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {BG}; }}
    html, body, [class*="css"] {{ font-family: {MONO}; }}

    .term-title {{
        font-family: {MONO}; font-size: 2.0rem; font-weight: 700;
        color: {TEXT}; letter-spacing: -0.5px; margin-bottom: 0.1rem;
    }}
    .term-sub {{
        font-family: {MONO}; color: {MUTED}; font-size: 0.95rem;
        margin-bottom: 0.4rem;
    }}
    .term-rule {{
        border: none; border-top: 1px solid {BORDER};
        margin: 0.6rem 0 1.2rem 0;
    }}
    h2 {{
        font-family: {MONO} !important; color: {TEXT} !important;
        border-left: 3px solid {GREEN}; padding-left: 0.6rem;
        font-size: 1.35rem !important;
    }}
    [data-testid="stMetric"] {{
        background-color: {PANEL}; border: 1px solid {BORDER};
        border-radius: 4px; padding: 0.8rem 1rem;
    }}
    [data-testid="stMetricValue"] {{
        font-family: {MONO}; color: {GREEN}; font-size: 1.7rem;
    }}
    [data-testid="stMetricLabel"] {{
        font-family: {MONO}; color: {MUTED}; text-transform: uppercase;
        font-size: 0.72rem; letter-spacing: 0.5px;
    }}
    [data-testid="stSidebar"] {{
        background-color: {PANEL}; border-right: 1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 2px; }}
    .stTabs [data-baseweb="tab"] {{
        font-family: {MONO}; background-color: {PANEL};
        border: 1px solid {BORDER}; border-radius: 3px 3px 0 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown(f"<div style='font-family:{MONO};color:{GREEN};"
                    f"font-weight:700;font-size:1.1rem'>WHALE SIGNALS</div>",
                    unsafe_allow_html=True)
st.sidebar.markdown(f"<div style='font-family:{MONO};color:{MUTED};"
                    f"font-size:0.75rem;margin-bottom:1rem'>terminal // event study</div>",
                    unsafe_allow_html=True)

min_usd = st.sidebar.slider(
    "MIN TX SIZE (USD)",
    min_value=DATA["thresholds"][0], max_value=DATA["thresholds"][-1],
    value=DATA["thresholds"][0], step=1_000_000, format="$%d",
)

# The slider value keys straight into the pre-computed aggregates.
B = DATA["by_threshold"][str(min_usd)]

st.sidebar.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)
st.sidebar.markdown(f"**TXNS** &nbsp; `{B['n_filtered']:,}`")
st.sidebar.markdown(f"**SPAN** &nbsp; `{META['date_min']} → {META['date_max']}`")
st.sidebar.markdown("**CATEGORIES**")
for cat, count in B["category_counts"].items():
    st.sidebar.markdown(f"`{cat:<20} {count:>7,}`")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<div class="term-title">ARE ETHEREUM WHALES SMART MONEY?</div>',
            unsafe_allow_html=True)
st.markdown(
    f'<div class="term-sub">Event study // {META["n_total"]:,} whale transactions // '
    f'{META["date_min"]} → {META["date_max"]} // '
    'do large on-chain moves predict ETH direction?</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="term-rule">', unsafe_allow_html=True)

base_24h = META["base_rate_24h"]
dep_hit = z(B["deposit_hit_24h"])
greed_hit = z(B["greed_deposit_hit_24h"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions", f"{B['n_filtered']:,}")
col2.metric("Deposit Hit 24h", f"{dep_hit:.1f}%",
            delta=f"{dep_hit - base_24h:+.1f}pp",
            help="Share of deposits followed by a 24h price drop, vs base rate.")
col3.metric("Deposits in Greed", f"{greed_hit:.1f}%",
            help="Whale deposits when Fear & Greed > 75.")
col4.metric("Base Rate 24h", f"{base_24h:.1f}%",
            help="Any random hour: share followed by a 24h drop.")

# ---------------------------------------------------------------------------
# Section 1: Deposit edge across horizons
# ---------------------------------------------------------------------------

st.header("01 // Deposit edge grows with horizon")
st.markdown(
    f"<span style='color:{MUTED}'>Whale sellers are not day-trading. The edge "
    "is thin at 24h but compounds out to months. They appear to price in "
    "structural shifts well ahead of the market.</span>",
    unsafe_allow_html=True,
)

dep_edges = [z(e) for e in B["deposit_edge_by_horizon"]]
fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=HORIZON_LABELS, y=dep_edges,
    marker_color=[edge_colour(e) for e in dep_edges],
    marker_line_color=BORDER, marker_line_width=1,
    text=[f"{e:+.1f}" for e in dep_edges], textposition="outside",
    textfont=dict(family=MONO, color=TEXT),
))
fig1.add_hline(y=0, line_dash="dot", line_color=MUTED)
fig1.update_layout(
    title="DEPOSIT EDGE OVER BASE RATE (unconditional)",
    xaxis_title="HORIZON", yaxis_title="EDGE (pp)",
    yaxis_range=[min(dep_edges) - 2, max(dep_edges) + 3], showlegend=False,
)
st.plotly_chart(style_fig(fig1), width='stretch')

# ---------------------------------------------------------------------------
# Section 2: Yearly stability (alpha decay)
# ---------------------------------------------------------------------------

st.header("02 // Yearly stability: alpha decay")
st.markdown(
    f"<span style='color:{MUTED}'>Each year tested independently. The "
    "withdrawal buy-signal died after 2024. The deposit sell-signal emerged in "
    "2024 and strengthened into 2026 (out-of-sample).</span>",
    unsafe_allow_html=True,
)

tab_dep_yr, tab_wd_yr = st.tabs(["DEPOSITS (sell)", "WITHDRAWALS (buy)"])


def yearly_bar(edges_by_year: dict, title: str, caption: str):
    """One bar per year, coloured by whether the edge clears the base rate."""
    vals = [z(edges_by_year[str(y)]) for y in YEARS]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(y) for y in YEARS], y=vals,
        marker_color=[edge_colour(e, 1.0) for e in vals],
        marker_line_color=BORDER, marker_line_width=1,
        text=[f"{e:+.1f}" for e in vals], textposition="outside",
        textfont=dict(family=MONO, color=TEXT),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=MUTED)
    fig.update_layout(title=title, xaxis_title="YEAR", yaxis_title="EDGE (pp)",
                      yaxis_range=[min(vals) - 3, max(vals) + 3], showlegend=False)
    st.plotly_chart(style_fig(fig, 350), width='stretch')
    st.caption(caption)


with tab_dep_yr:
    yearly_bar(B["yearly"]["deposit_edge"],
               "DEPOSIT EDGE BY YEAR (24h, unconditional)",
               "Deposit edge grew from roughly flat in 2023 to a clear positive edge in 2026 (out-of-sample).")

with tab_wd_yr:
    yearly_bar(B["yearly"]["withdrawal_edge_negfund"],
               "WITHDRAWAL EDGE BY YEAR (24h, negative funding)",
               "Withdrawal edge peaked around +10pp in 2024, then collapsed below zero by 2026.")

# ---------------------------------------------------------------------------
# Section 3: Threshold sensitivity
# ---------------------------------------------------------------------------

st.header("03 // Threshold sensitivity")
st.markdown(
    rf"<span style='color:{MUTED}'>As ETH rose from ~\$1,200 to ~\$4,000+, a \$1M "
    "transaction is fewer ETH and less conviction. Do larger tickets carry a "
    "stronger signal?</span>",
    unsafe_allow_html=True,
)

TS = DATA["threshold_sensitivity"]
tab_t_dep, tab_t_wd = st.tabs(["DEPOSITS // extreme greed", "WITHDRAWALS // neg funding"])


def sensitivity_bar(rows: list, title: str):
    """Edge vs minimum ticket size, with sample sizes annotated."""
    labels = [f"${r['threshold'] // 1_000_000}M+" for r in rows]
    vals = [z(r["edge"]) for r in rows]
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=vals,
        marker_color=[edge_colour(e, 1.0) for e in vals],
        marker_line_color=BORDER, marker_line_width=1,
        text=[f"{e:+.1f}  n={n:,}" for e, n in zip(vals, ns)],
        textposition="outside", textfont=dict(family=MONO, color=TEXT),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=MUTED)
    fig.update_layout(title=title, xaxis_title="MIN TX SIZE", yaxis_title="EDGE (pp)",
                      yaxis_range=[min(vals) - 2, max(vals) + 3], showlegend=False)
    st.plotly_chart(style_fig(fig, 350), width='stretch')


with tab_t_dep:
    sensitivity_bar(TS["deposit_greed"], "DEPOSIT EDGE BY THRESHOLD (extreme greed, 24h)")

with tab_t_wd:
    sensitivity_bar(TS["withdrawal_negfund"], "WITHDRAWAL EDGE BY THRESHOLD (negative funding, 24h)")

# ---------------------------------------------------------------------------
# Section 4: Sentiment-conditioned hit rates
# ---------------------------------------------------------------------------

st.header("04 // Sentiment-conditioned hit rates")

tab_s_dep, tab_s_wd = st.tabs(["DEPOSITS (sell)", "WITHDRAWALS (buy)"])


def sentiment_chart(rows: list, title: str, yrange: list):
    """Whale hit-rate bars over a dashed base-rate reference line."""
    names = [r["name"] for r in rows]
    hits = [z(r["hit"]) for r in rows]
    bases = [z(r["base"]) for r in rows]
    cols = [edge_colour(h - b, 2.0) for h, b in zip(hits, bases)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=hits, name="whale hit rate", marker_color=cols,
        marker_line_color=BORDER, marker_line_width=1,
        text=[f"{h:.1f}" for h in hits], textposition="outside",
        textfont=dict(family=MONO, color=TEXT),
    ))
    fig.add_trace(go.Scatter(
        x=names, y=bases, name="base rate", mode="markers+lines",
        line=dict(color=MUTED, dash="dash"), marker=dict(size=7, color=MUTED),
    ))
    fig.add_hline(y=50, line_dash="dot", line_color=GREY,
                  annotation_text="50% (coin flip)", annotation_font_color=MUTED)
    fig.update_layout(title=title, yaxis_title="HIT RATE (%)", yaxis_range=yrange)
    fig = style_fig(fig, 470)
    # Angle the regime labels so they never run into each other.
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=11))
    fig.update_layout(margin=dict(l=55, r=25, t=70, b=90))
    st.plotly_chart(fig, width='stretch')


with tab_s_dep:
    sentiment_chart(B["sentiment"]["deposit"], "DEPOSIT HIT RATE BY SENTIMENT (24h)", [30, 65])

with tab_s_wd:
    sentiment_chart(B["sentiment"]["withdrawal"], "WITHDRAWAL HIT RATE BY SENTIMENT (24h)", [35, 70])

st.caption(
    "Green = whale edge above base rate. Red = below. Grey = marginal. "
    "Dashed line = base rate for that regime; dotted line = coin flip."
)

# ---------------------------------------------------------------------------
# Section 5: Return distribution
# ---------------------------------------------------------------------------

st.header("05 // Return distribution: deposits in extreme greed")

RD = B["return_dist"]
if RD:
    c1, c2, c3 = st.columns(3)
    c1.metric("Hit Rate (price fell)", f"{z(RD['hit_rate']):.1f}%")
    c1.metric("Avg Hit", f"{z(RD['avg_hit']):.2f}%")
    c2.metric("Miss Rate", f"{z(RD['miss_rate']):.1f}%")
    c2.metric("Avg Miss", f"+{z(RD['avg_miss']):.2f}%")
    c3.metric("Signals", f"{RD['n']:,}")
    c3.metric("Avg Return (all)", f"{z(RD['avg_all']):+.2f}%")

    # Reconstruct the histogram from shared bin edges + per-bin counts.
    edges = RD["edges"]
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    width = edges[1] - edges[0]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(
        x=centres, y=RD["hit_counts"], name="hit (price fell)",
        marker_color=GREEN, marker_line_width=0, width=width, opacity=0.85,
    ))
    fig_hist.add_trace(go.Bar(
        x=centres, y=RD["miss_counts"], name="miss (price rose)",
        marker_color=RED, marker_line_width=0, width=width, opacity=0.85,
    ))
    fig_hist.add_vline(x=0, line_dash="dot", line_color=MUTED)
    fig_hist.update_layout(
        title="24h RETURNS AFTER WHALE DEPOSITS IN EXTREME GREED",
        xaxis_title="24h FORWARD RETURN (%)", yaxis_title="COUNT",
        barmode="overlay",
    )
    st.plotly_chart(style_fig(fig_hist), width='stretch')
else:
    st.info("Not enough deposits above this threshold during extreme greed to plot.")

# ---------------------------------------------------------------------------
# Section 6: Asymmetry
# ---------------------------------------------------------------------------

st.header("06 // The asymmetry: deposits won, withdrawals lost")
st.markdown(
    f"<span style='color:{MUTED}'>Withdrawal edge peaked early then collapsed. "
    "Deposit edge was absent early then grew. One possible reading: whale-watching "
    "tools broadcast buy signals far more than sell signals, leaving the deposit "
    "edge un-arbitraged. Plausible, not proven.</span>",
    unsafe_allow_html=True,
)

dep_yr = [z(B["yearly"]["deposit_edge_uncond"][str(y)]) for y in YEARS]
wd_yr = [z(B["yearly"]["withdrawal_edge_uncond"][str(y)]) for y in YEARS]

fig_asym = go.Figure()
fig_asym.add_trace(go.Bar(
    x=[str(y) for y in YEARS], y=dep_yr, name="deposit edge",
    marker_color=GREEN, marker_line_color=BORDER, marker_line_width=1,
))
fig_asym.add_trace(go.Bar(
    x=[str(y) for y in YEARS], y=wd_yr, name="withdrawal edge",
    marker_color=BLUE, marker_line_color=BORDER, marker_line_width=1,
))
fig_asym.add_hline(y=0, line_dash="dot", line_color=MUTED)
fig_asym.update_layout(
    title="DEPOSIT vs WITHDRAWAL EDGE BY YEAR (24h, unconditional)",
    xaxis_title="YEAR", yaxis_title="EDGE (pp)", barmode="group",
)
st.plotly_chart(style_fig(fig_asym), width='stretch')

# ---------------------------------------------------------------------------
# Section 7: Limitations
# ---------------------------------------------------------------------------

st.header("07 // Limitations")
st.markdown(r"""
1. **Backtested, not live-tested.** Past results do not guarantee future ones.
   If participants start following deposit signals, the edge would likely arbitrage away.
2. **Modest at short horizons.** A +1 to +4pp edge at 24h is statistically
   significant but economically marginal after costs and slippage.
3. **Long-horizon windows overlap.** At 1 month+, thousands of events measure the
   same price move. Hit rates are informative but p-values overstate significance.
4. **No stop-loss modelling.** Long-horizon results assume holding to maturity.
   A trade that ends +5% may have been -20% along the way.
5. **Fixed USD threshold ignores ETH price growth.** \$1M was ~833 ETH in 2023
   but only ~250 ETH in 2026, diluting the pool with smaller actors over time.
6. **The withdrawal signal is dead.** Any strategy built on whale withdrawals
   would have failed in 2025-2026.
""")

st.markdown('<hr class="term-rule">', unsafe_allow_html=True)
st.caption(
    f"Data: {META['n_total']:,} whale transactions "
    f"({META['date_min']} → {META['date_max']}) // "
    f"{META['n_labels']:,} labelled addresses // "
    "Dune Analytics, Binance API, alternative.me"
)
