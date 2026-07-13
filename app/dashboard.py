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
from plotly.subplots import make_subplots
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
def load_data(cache_key: float) -> dict:
    """Load the pre-computed aggregate artefact.

    `cache_key` is the data file's modification time. Passing it (unhashed args
    would be prefixed with '_') means the cache invalidates whenever the file
    changes, so a redeploy with new data never serves a stale cached dict.
    """
    with open(DATA_PATH) as f:
        return json.load(f)


DATA = load_data(DATA_PATH.stat().st_mtime)
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

# Horizon for sections 02, 03, 06 (yearly stability, threshold sensitivity,
# asymmetry). Sections 01, 04, 05 make their own horizon/regime choices and
# are not affected by this control.
default_h = HORIZON_LABELS.index("24h") if "24h" in HORIZON_LABELS else 0
horizon = st.sidebar.selectbox(
    "HORIZON (sections 02, 03, 06)", HORIZON_LABELS, index=default_h,
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


Y = B["yearly"][horizon]

with tab_dep_yr:
    yearly_bar(Y["deposit_edge"],
               f"DEPOSIT EDGE BY YEAR ({horizon}, unconditional)",
               "Deposit edge grew from roughly flat in 2023 to a clear positive edge by 2026 (out-of-sample) "
               "at short horizons. Zero bars mean too few observations at this horizon/threshold to trust "
               "(e.g. long horizons run out of forward data near the end of the dataset).")

with tab_wd_yr:
    yearly_bar(Y["withdrawal_edge_negfund"],
               f"WITHDRAWAL EDGE BY YEAR ({horizon}, negative funding)",
               "Withdrawal edge peaked around 2024, then collapsed. Zero bars mean too few observations "
               "at this horizon/threshold to trust.")

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

TS = DATA["threshold_sensitivity"][horizon]
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
    sensitivity_bar(TS["deposit_greed"], f"DEPOSIT EDGE BY THRESHOLD (extreme greed, {horizon})")

with tab_t_wd:
    sensitivity_bar(TS["withdrawal_negfund"], f"WITHDRAWAL EDGE BY THRESHOLD (negative funding, {horizon})")

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

st.header("05 // Return distribution of deposits, by regime")
st.markdown(
    f"<span style='color:{MUTED}'>Pick a market regime to see the 24h return "
    "distribution after whale deposits. Unconditional deposits sit near a coin "
    "flip; the leftward (bearish) skew only appears in certain regimes. This is "
    "the signal being conditional, not universal.</span>",
    unsafe_allow_html=True,
)

# Condition selector. Default to extreme greed (the strongest case).
# Fall back to the per-block keys if the top-level list is absent (older data).
cond_options = DATA.get("dist_conditions") or list(
    B.get("return_dist_by_condition", {}).keys())
default_idx = cond_options.index("extreme greed") if "extreme greed" in cond_options else 0
condition = st.selectbox("REGIME", cond_options, index=default_idx)

RD = B.get("return_dist_by_condition", {}).get(condition)
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
        title=f"24h RETURNS AFTER WHALE DEPOSITS // {condition.upper()}",
        xaxis_title="24h FORWARD RETURN (%)", yaxis_title="COUNT",
        barmode="overlay",
    )
    st.plotly_chart(style_fig(fig_hist), width='stretch')
    st.caption(
        "A hit = price fell within 24h (the correct call for a sell signal). "
        "Avg Return (all) blends hits and misses: a small directional tilt nets "
        "to a small average because up-moves and down-moves are similar in size."
    )
else:
    st.info("Not enough deposits above this threshold in this regime to plot.")

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

dep_yr = [z(Y["deposit_edge_uncond"][str(y)]) for y in YEARS]
wd_yr = [z(Y["withdrawal_edge_uncond"][str(y)]) for y in YEARS]

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
    title=f"DEPOSIT vs WITHDRAWAL EDGE BY YEAR ({horizon}, unconditional)",
    xaxis_title="YEAR", yaxis_title="EDGE (pp)", barmode="group",
)
st.plotly_chart(style_fig(fig_asym), width='stretch')

# ---------------------------------------------------------------------------
# Section 7: Signal timeline (when it fires, when it pays)
# ---------------------------------------------------------------------------

st.header("07 // When the signal fires, and when it pays")
st.markdown(
    f"<span style='color:{MUTED}'>Monthly view of the deposit (sell) signal at "
    "the base \\$1M+ threshold. Top: ETH price, with bear markets shaded. "
    "Bottom: how many deposits fired each month, and whether that month's 24h "
    "calls beat the month's own base rate. The edge is not spread evenly "
    "through time; it clusters.</span>",
    unsafe_allow_html=True,
)

TL = DATA["timeline"]
BB = DATA["bull_bear"]

# Monthly 24h edge = whale hit rate minus that same month's base rate, so a
# trending month does not get credited to the whales.
tl_edge = [
    (h - b) if (h is not None and b is not None) else None
    for h, b in zip(TL["hit_24h"], TL["base_24h"])
]

# Three stacked panels, each with its own y-axis, sharing one time axis --
# deliberately NOT a dual-axis chart, since overlaying a bar count and an
# edge line on two different scales in one panel invites misreading
# correlation that is not really there.
fig_tl = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    row_heights=[0.34, 0.28, 0.38], vertical_spacing=0.06,
    subplot_titles=("ETH PRICE", "SIGNALS FIRED / MONTH ($1M+ deposits)",
                    "THAT MONTH'S 24h EDGE (whale hit rate minus base rate)"),
)
fig_tl.add_trace(go.Scatter(
    x=TL["months"], y=TL["eth_close"], name="ETH close",
    line=dict(color=BLUE, width=1.6), mode="lines", showlegend=False,
), row=1, col=1)
fig_tl.add_trace(go.Bar(
    x=TL["months"], y=TL["n"], name="deposits fired",
    marker_color=GREY, marker_line_width=0, opacity=0.8, showlegend=False,
), row=2, col=1)
fig_tl.add_trace(go.Scatter(
    x=TL["months"], y=tl_edge, name="monthly 24h edge",
    mode="lines+markers", line=dict(color=GREEN, width=1.6),
    marker=dict(size=5), connectgaps=False, showlegend=False,
), row=3, col=1)
fig_tl.add_hline(y=0, line_dash="dot", line_color=MUTED, row=3, col=1)

# Shade bear-market spans (regime segments from the Section 9 state machine)
# across all three panels so the eye can line up price, volume, and edge.
for seg in BB["segments"]:
    if seg["regime"] == "bear":
        for r in (1, 2, 3):
            fig_tl.add_vrect(x0=seg["start"], x1=seg["end"], fillcolor=RED,
                             opacity=0.10, line_width=0, row=r, col=1)

fig_tl.update_layout(title="DEPOSIT SIGNAL CADENCE // MONTHLY (bear markets shaded red)")
fig_tl = style_fig(fig_tl, 680)
fig_tl.update_annotations(font=dict(family=MONO, color=MUTED, size=11))
fig_tl.update_yaxes(title_text="USD", row=1, col=1)
fig_tl.update_yaxes(title_text="COUNT", row=2, col=1)
fig_tl.update_yaxes(title_text="EDGE (pp)", row=3, col=1)
st.plotly_chart(fig_tl, width='stretch')
st.caption(
    "Read top to bottom: ETH price for context, how many deposits fired that "
    "month, then whether those deposits beat that month's own base rate. "
    "Signal volume roughly tracks market activity; the bottom panel spends "
    "more time above zero in and around the shaded bear stretches, which is "
    "what Section 08 quantifies directly."
)

# ---------------------------------------------------------------------------
# Section 8: Bull vs bear regimes
# ---------------------------------------------------------------------------

st.header("08 // Bull vs bear regimes")
st.markdown(
    f"<span style='color:{MUTED}'>Standard 20% drawdown/rally regime "
    f"definition, not tuned: {BB['hours']['bull']:,} bull hours vs "
    f"{BB['hours']['bear']:,} bear hours across {len(BB['segments'])} "
    "segments. Deposit edge is stronger in bear markets at every size tested; "
    "withdrawals get more wrong exactly when buying conviction should matter "
    "most.</span>",
    unsafe_allow_html=True,
)

bb_horizon = st.radio("HORIZON", BB["horizons"],
                      index=BB["horizons"].index("1w"), horizontal=True)
tab_bb_dep, tab_bb_wd = st.tabs(["DEPOSITS (sell)", "WITHDRAWALS (buy)"])


def bull_bear_bars(category: str, title: str):
    """Grouped bull-vs-bear edge bars per threshold, with n in the hover."""
    rows = [r for r in BB["edges"]
            if r["category"] == category and r["horizon"] == bb_horizon]
    by_key = {(r["threshold"], r["regime"]): r for r in rows}
    labels = [f"${t // 1_000_000}M+" for t in BB["thresholds"]]

    fig = go.Figure()
    for reg, colour in (("bull", GREEN), ("bear", RED)):
        cells = [by_key.get((t, reg)) for t in BB["thresholds"]]
        fig.add_trace(go.Bar(
            x=labels, y=[z(c["edge"]) if c else 0 for c in cells], name=reg,
            marker_color=colour, marker_line_color=BORDER, marker_line_width=1,
            text=[f"{z(c['edge']):+.1f}" if c else "" for c in cells],
            textposition="outside", textfont=dict(family=MONO, color=TEXT),
            customdata=[c["n"] if c else 0 for c in cells],
            hovertemplate="%{x} " + reg + ": %{y:+.1f}pp (n=%{customdata:,})<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=MUTED)
    fig.update_layout(title=title, xaxis_title="MIN TX SIZE",
                      yaxis_title="EDGE (pp)", barmode="group")
    st.plotly_chart(style_fig(fig, 380), width='stretch')


with tab_bb_dep:
    bull_bear_bars("deposit", f"DEPOSIT EDGE BY REGIME ({bb_horizon}, unconditional)")

with tab_bb_wd:
    bull_bear_bars("withdrawal", f"WITHDRAWAL EDGE BY REGIME ({bb_horizon}, unconditional)")

st.caption(
    "Bar colour = market regime (green bull, red bear), not good/bad: for "
    "deposits, the taller bear bars are the finding. Raw p-values, uncorrected "
    "for overlapping observations; the short horizons keep the overstatement "
    "milder than the long-horizon claims that failed (see README Section 9)."
)

# ---------------------------------------------------------------------------
# Section 9: Drawdown before the payoff (MAE)
# ---------------------------------------------------------------------------

st.header("09 // Drawdown before the payoff")
st.markdown(
    f"<span style='color:{MUTED}'>Long-horizon returns only report the "
    "endpoint. Maximum adverse excursion (MAE) is the worst unrealised loss a "
    "trader following the sell signal would have marked-to-market along the "
    "way, computed from the full hourly price path. Shown for trades where the "
    "signal was EVENTUALLY RIGHT; being right does not mean it was "
    "survivable.</span>",
    unsafe_allow_html=True,
)

DD = DATA["drawdown"]
tab_dd_u, tab_dd_g = st.tabs(["UNCONDITIONAL", "EXTREME GREED"])


def mae_chart(condition: str):
    """Median / mean / P90 MAE bars by horizon, eventually-correct trades."""
    rows = {r["horizon"]: r for r in DD["rows"]
            if r["condition"] == condition and r["outcome"] == "hit"}
    hs = [h for h in DD["horizons"] if h in rows]
    fig = go.Figure()
    for field, name, colour in (("median_mae", "median", GREY),
                                ("mean_mae", "mean", BLUE),
                                ("p90_mae", "P90 (worst decile)", RED)):
        fig.add_trace(go.Bar(
            x=hs, y=[z(rows[h][field]) for h in hs], name=name,
            marker_color=colour, marker_line_color=BORDER, marker_line_width=1,
            text=[f"{z(rows[h][field]):.1f}" for h in hs],
            textposition="outside", textfont=dict(family=MONO, size=10, color=TEXT),
        ))
    fig.update_layout(
        title=f"MAE ON EVENTUALLY-CORRECT DEPOSIT TRADES ({condition.upper()})",
        xaxis_title="HOLDING HORIZON", yaxis_title="ADVERSE MOVE (%)",
        barmode="group",
    )
    st.plotly_chart(style_fig(fig, 400), width='stretch')
    six = rows.get("6m")
    if six:
        st.caption(
            f"At 6 months ({condition}), an eventually-correct trade endured a "
            f"{z(six['mean_mae']):.1f}% adverse move on average before paying "
            f"off; the worst decile endured {z(six['p90_mae']):.1f}%. "
            "Eventually-WRONG trades are worse on both counts (unconditional "
            "6-month misses averaged 80.1% MAE). No stop-loss rule is "
            "simulated; MAE is the pain, not a strategy."
        )


with tab_dd_u:
    mae_chart("unconditional")

with tab_dd_g:
    mae_chart("extreme greed")

# ---------------------------------------------------------------------------
# Section 10: Limitations
# ---------------------------------------------------------------------------

st.header("10 // Limitations")
st.markdown(r"""
1. **Backtested, not live-tested.** Past results do not guarantee future ones.
   If participants start following deposit signals, the edge would likely arbitrage away.
2. **Modest at short horizons.** A +1 to +4pp edge at 24h is statistically
   significant but economically marginal after costs and slippage.
3. **Long-horizon windows overlap.** At 1 month+, thousands of events measure the
   same price move. Hit rates are informative but p-values overstate significance.
4. **No stop-loss modelling.** Long-horizon results assume holding to maturity.
   Section 09 measures how bad the ride got (MAE), but no concrete stop-loss
   rule is simulated.
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
