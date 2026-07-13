"""
Pre-compute every number the Streamlit dashboard renders and dump it to a
small JSON file (`app/dashboard_data.json`).

Why: the raw whale dataset is ~187 MB and gitignored, so it cannot ship to
Streamlit Community Cloud. This script crunches it once, locally, into a
few-hundred-KB artefact that commits cleanly and lets the deployed dashboard
render instantly with no heavy processing and no raw data.

Re-run this whenever the underlying data changes:
    python scripts/build_dashboard_data.py
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.features.feature_engineer import assign_transaction_label  # noqa: E402

# Reuse the exact bull/bear state machine from the analysis script, so the
# dashboard's regime classification can never drift from README Section 9's.
sys.path.insert(0, str(ROOT / "scripts"))
from run_bull_bear_analysis import compute_regime  # noqa: E402

# ---------------------------------------------------------------------------
# Parameters (kept identical to the dashboard so numbers match exactly)
# ---------------------------------------------------------------------------

HORIZONS_H = [1, 6, 24, 72, 168, 720, 2160, 4320]
HORIZON_LABELS = ["1h", "6h", "24h", "3d", "1w", "1m", "3m", "6m"]

# Slider values: $1M to $50M in $1M steps (matches the dashboard slider).
THRESHOLDS = list(range(1_000_000, 50_000_001, 1_000_000))

# Fixed thresholds for the threshold-sensitivity section.
SENS_THRESHOLDS = [1_000_000, 2_000_000, 5_000_000, 10_000_000]

# Sentiment regimes. Each maps a dataframe to a boolean mask.
CONDITIONS = {
    "neg funding": lambda df: df["funding_rate"] < 0,
    "extreme fear": lambda df: df["fng_value"] <= 25,
    "fear": lambda df: (df["fng_value"] > 25) & (df["fng_value"] <= 45),
    "neutral": lambda df: (df["fng_value"] > 45) & (df["fng_value"] <= 55),
    "greed": lambda df: (df["fng_value"] > 55) & (df["fng_value"] <= 75),
    "extreme greed": lambda df: df["fng_value"] > 75,
    "pos funding": lambda df: df["funding_rate"] >= 0,
}

MIN_N = 30  # minimum sample size before a hit rate is trustworthy


# ---------------------------------------------------------------------------
# Load and enrich (this is the slow, memory-heavy part done once)
# ---------------------------------------------------------------------------

def load_enriched():
    """Load raw data and attach forward returns + sentiment to whales and prices."""
    print("Loading raw data...")
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
    whale["year"] = whale["timestamp_utc"].dt.year

    prices_sorted = prices.sort_values("timestamp_utc").reset_index(drop=True)
    price_series = prices_sorted.set_index("timestamp_utc")["close"]

    # Entry price + forward returns for every whale (vectorised .map lookups)
    whale = whale.merge(
        price_series.rename("price_t0"),
        left_on="hour_utc", right_index=True, how="left",
    )
    for h, label in zip(HORIZONS_H, HORIZON_LABELS):
        fut = (whale["hour_utc"] + pd.Timedelta(hours=h)).map(price_series)
        whale[f"fwd_{label}"] = (fut - whale["price_t0"]) / whale["price_t0"]

    # Same forward returns for every hourly candle (the base-rate universe)
    for h, label in zip(HORIZONS_H, HORIZON_LABELS):
        fut = (prices_sorted["timestamp_utc"] + pd.Timedelta(hours=h)).map(price_series)
        prices_sorted[f"fwd_{label}"] = (fut - prices_sorted["close"]) / prices_sorted["close"]

    # Attach funding rate (last 8h value before the event)
    funding_sorted = funding.sort_values("timestamp_utc")
    whale = pd.merge_asof(
        whale.sort_values("hour_utc"),
        funding_sorted[["timestamp_utc", "funding_rate"]].rename(
            columns={"timestamp_utc": "hour_utc"}),
        on="hour_utc", direction="backward",
    )
    whale["funding_rate"] = whale["funding_rate"].fillna(0)
    prices_sorted = pd.merge_asof(
        prices_sorted.sort_values("timestamp_utc"),
        funding_sorted[["timestamp_utc", "funding_rate"]],
        on="timestamp_utc", direction="backward",
    )
    prices_sorted["funding_rate"] = prices_sorted["funding_rate"].fillna(0)

    # Attach Fear & Greed (last daily value before the event)
    fng_m = fng.rename(columns={"date": "_date"}).sort_values("_date")
    whale["_date"] = whale["timestamp_utc"].dt.floor("D")
    whale = pd.merge_asof(whale.sort_values("_date"),
                          fng_m[["_date", "fng_value"]], on="_date", direction="backward")
    whale["fng_value"] = whale["fng_value"].fillna(50)
    whale.drop(columns="_date", inplace=True)
    prices_sorted["_date"] = prices_sorted["timestamp_utc"].dt.floor("D")
    prices_sorted = pd.merge_asof(prices_sorted.sort_values("_date"),
                                  fng_m[["_date", "fng_value"]], on="_date", direction="backward")
    prices_sorted["fng_value"] = prices_sorted["fng_value"].fillna(50)
    prices_sorted.drop(columns="_date", inplace=True)
    prices_sorted["year"] = prices_sorted["timestamp_utc"].dt.year

    return whale, prices_sorted


# ---------------------------------------------------------------------------
# Small helpers. All rates are in percent; edge is a difference of percents.
# ---------------------------------------------------------------------------

def hit_pct(returns: pd.Series, direction: str) -> float:
    """Percentage of returns in the predicted direction ('down' = price fell)."""
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    return float((r < 0).mean() * 100 if direction == "down" else (r > 0).mean() * 100)


def r2(x) -> float | None:
    """Round to 2 dp for compact JSON; pass through None/NaN as None."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), 2)


def return_distribution(returns: pd.Series, direction: str) -> dict | None:
    """Histogram + summary stats for a set of forward returns.

    A "hit" is a move in the predicted direction ('down' = price fell, which is
    the correct call after a deposit/sell signal). Returns None if too few
    observations to plot. Bin edges are shared by hits and misses so the two
    colours line up into one continuous distribution.
    """
    r = returns.dropna() * 100  # to percent
    if len(r) <= 10:
        return None
    if direction == "down":
        hits, misses = r[r < 0], r[r >= 0]
    else:
        hits, misses = r[r > 0], r[r <= 0]
    lo, hi = float(np.floor(r.min())), float(np.ceil(r.max()))
    edges = np.linspace(lo, hi, 61)  # 60 shared bins
    hit_counts, _ = np.histogram(hits, bins=edges)
    miss_counts, _ = np.histogram(misses, bins=edges)
    return {
        "n": int(len(r)),
        "hit_rate": r2(len(hits) / len(r) * 100),
        "miss_rate": r2(len(misses) / len(r) * 100),
        "avg_hit": r2(hits.mean()) if len(hits) else None,
        "avg_miss": r2(misses.mean()) if len(misses) else None,
        "avg_all": r2(r.mean()),
        "edges": [round(e, 3) for e in edges.tolist()],
        "hit_counts": hit_counts.tolist(),
        "miss_counts": miss_counts.tolist(),
    }


# Conditions offered in the Section 5 distribution explorer (deposits, sell
# signal). "unconditional" first, then every sentiment regime.
DIST_CONDITIONS = {"unconditional": lambda df: pd.Series(True, index=df.index),
                   **CONDITIONS}


# ---------------------------------------------------------------------------
# Base rates (slider-independent: computed once from the full price universe)
# ---------------------------------------------------------------------------

def build_base_rates(all_prices: pd.DataFrame, years: list[int]) -> dict:
    """Market-wide base rates that every whale hit rate is compared against.

    Yearly and threshold-sensitivity base rates are computed at every horizon
    (sections 2, 3, 6 let the viewer pick a horizon). Sentiment base rates stay
    24h-only (section 4 is not horizon-selectable, to avoid stacking a third
    dimension onto an already-dense 7-regime chart).
    """
    base = {"horizon_down": {}, "yearly_down": {}, "yearly_up": {},
            "yearly_up_negfund": {}, "sentiment_down": {}, "sentiment_up": {},
            "greed_down": {}, "negfund_up": {}}

    # Base rate of a price fall at each horizon (section 1)
    for label in HORIZON_LABELS:
        base["horizon_down"][label] = hit_pct(all_prices[f"fwd_{label}"], "down")

    # Per-year, per-horizon base rates, both directions (sections 2, 6)
    for label in HORIZON_LABELS:
        col = f"fwd_{label}"
        base["yearly_down"][label] = {}
        base["yearly_up"][label] = {}
        base["yearly_up_negfund"][label] = {}
        for year in years:
            yr = all_prices[all_prices["year"] == year][col]
            base["yearly_down"][label][str(year)] = hit_pct(yr, "down")
            base["yearly_up"][label][str(year)] = hit_pct(yr, "up")
            yr_neg = all_prices[(all_prices["year"] == year)
                                & (all_prices["funding_rate"] < 0)][col]
            base["yearly_up_negfund"][label][str(year)] = hit_pct(yr_neg, "up")

    # Per-sentiment base rates at 24h only, both directions (section 4)
    for name, fn in CONDITIONS.items():
        sub = all_prices[fn(all_prices)]["fwd_24h"]
        base["sentiment_down"][name] = hit_pct(sub, "down")
        base["sentiment_up"][name] = hit_pct(sub, "up")

    # Conditioned base rates for threshold sensitivity, at every horizon (section 3)
    for label in HORIZON_LABELS:
        col = f"fwd_{label}"
        base["greed_down"][label] = hit_pct(all_prices[all_prices["fng_value"] > 75][col], "down")
        base["negfund_up"][label] = hit_pct(all_prices[all_prices["funding_rate"] < 0][col], "up")

    base["all_down_24h"] = hit_pct(all_prices["fwd_24h"], "down")
    return base


# ---------------------------------------------------------------------------
# Per-threshold whale aggregates (everything that responds to the slider)
# ---------------------------------------------------------------------------

def build_for_threshold(whale: pd.DataFrame, base: dict, years: list[int]) -> dict:
    """Compute every whale-side number for one minimum-transaction-size cut."""
    deposits = whale[whale["tx_category"] == "exchange_deposit"]
    withdrawals = whale[whale["tx_category"] == "exchange_withdrawal"]

    block: dict = {}
    block["n_filtered"] = int(len(whale))
    block["category_counts"] = {k: int(v) for k, v in
                                whale["tx_category"].value_counts().items()}

    # Key metrics
    dep_hit_24h = hit_pct(deposits["fwd_24h"], "down")
    greed_dep_hit = hit_pct(deposits[deposits["fng_value"] > 75]["fwd_24h"], "down")
    block["deposit_hit_24h"] = r2(dep_hit_24h)
    block["greed_deposit_hit_24h"] = r2(greed_dep_hit)

    # Section 1: deposit edge by horizon (whale hit - base rate)
    block["deposit_edge_by_horizon"] = [
        r2(hit_pct(deposits[f"fwd_{label}"], "down") - base["horizon_down"][label])
        for label in HORIZON_LABELS
    ]

    # Section 2 + 6: yearly edges, at every horizon (horizon-selectable)
    block["yearly"] = {}
    for label in HORIZON_LABELS:
        col = f"fwd_{label}"
        dep_edge, wd_edge_neg, wd_edge_uncond = {}, {}, {}
        for year in years:
            y = str(year)
            yr_dep = deposits[deposits["year"] == year][col]
            yr_wd = withdrawals[withdrawals["year"] == year][col]
            yr_wd_neg = withdrawals[(withdrawals["year"] == year)
                                    & (withdrawals["funding_rate"] < 0)][col]

            dep_edge[y] = (r2(hit_pct(yr_dep, "down") - base["yearly_down"][label][y])
                           if yr_dep.notna().sum() >= MIN_N else 0)
            wd_edge_uncond[y] = (r2(hit_pct(yr_wd, "up") - base["yearly_up"][label][y])
                                 if yr_wd.notna().sum() >= MIN_N else 0)
            wd_edge_neg[y] = (r2(hit_pct(yr_wd_neg, "up") - base["yearly_up_negfund"][label][y])
                              if yr_wd_neg.notna().sum() >= MIN_N else 0)

        # deposit_edge and deposit_edge_uncond are the same series (deposits
        # have no separate "conditional" yearly cut); kept as two keys so the
        # dashboard's section 2 and section 6 lookups stay symmetric with the
        # withdrawal side, which does differ (negative-funding vs unconditional).
        block["yearly"][label] = {
            "deposit_edge": dep_edge, "deposit_edge_uncond": dep_edge,
            "withdrawal_edge_negfund": wd_edge_neg,
            "withdrawal_edge_uncond": wd_edge_uncond,
        }

    # Section 4: sentiment hit rates (store hit + base so the chart draws both)
    def sentiment(source, direction, base_key):
        rows = []
        for name, fn in CONDITIONS.items():
            sub = source[fn(source)]["fwd_24h"]
            if sub.notna().sum() < MIN_N:
                continue
            rows.append({"name": name, "hit": r2(hit_pct(sub, direction)),
                         "base": r2(base[base_key][name])})
        return rows

    block["sentiment"] = {
        "deposit": sentiment(deposits, "down", "sentiment_down"),
        "withdrawal": sentiment(withdrawals, "up", "sentiment_up"),
    }

    # Section 5: 24h return distribution for deposits, under each regime, so the
    # dashboard can offer a condition selector (deposits are a sell signal, so
    # direction is always "down": a hit means price fell).
    block["return_dist_by_condition"] = {
        name: return_distribution(deposits[fn(deposits)]["fwd_24h"], "down")
        for name, fn in DIST_CONDITIONS.items()
    }

    return block


# ---------------------------------------------------------------------------
# Threshold-sensitivity section (independent of the slider)
# ---------------------------------------------------------------------------

def build_threshold_sensitivity(whale: pd.DataFrame, base: dict) -> dict:
    """Edge as a function of minimum ticket size, under two regimes, at every horizon."""
    deposits = whale[whale["tx_category"] == "exchange_deposit"]
    withdrawals = whale[whale["tx_category"] == "exchange_withdrawal"]

    result = {}
    for label in HORIZON_LABELS:
        col = f"fwd_{label}"
        dep_greed, wd_neg = [], []
        for thresh in SENS_THRESHOLDS:
            d = deposits[(deposits["usd_value"] >= thresh)
                         & (deposits["fng_value"] > 75)][col]
            w = withdrawals[(withdrawals["usd_value"] >= thresh)
                            & (withdrawals["funding_rate"] < 0)][col]
            dep_greed.append({
                "threshold": thresh, "n": int(d.notna().sum()),
                "edge": r2(hit_pct(d, "down") - base["greed_down"][label])
                if d.notna().sum() >= MIN_N else 0,
            })
            wd_neg.append({
                "threshold": thresh, "n": int(w.notna().sum()),
                "edge": r2(hit_pct(w, "up") - base["negfund_up"][label])
                if w.notna().sum() >= MIN_N else 0,
            })
        result[label] = {"deposit_greed": dep_greed, "withdrawal_negfund": wd_neg}
    return result


# ---------------------------------------------------------------------------
# Bull/bear regime block (README Section 9; independent of the slider)
# ---------------------------------------------------------------------------

# Short-to-mid horizons only, matching the README's Section 9 rationale:
# long horizons have too few independent observations to trust.
BB_HORIZONS = {"24h": 24, "3d": 72, "1w": 168}


def gap_filled_series(all_prices: pd.DataFrame) -> pd.Series:
    """Hourly close reindexed to a complete hourly grid and forward-filled.

    The raw feed has one missing hour. Row-count-based windows (rolling max,
    shift) only equal calendar hours on a complete grid, so both the regime
    and MAE blocks work on this filled series, matching the source scripts.
    """
    full_index = pd.date_range(
        all_prices["timestamp_utc"].min(), all_prices["timestamp_utc"].max(),
        freq="h", tz="UTC",
    )
    return all_prices.set_index("timestamp_utc")["close"].reindex(full_index).ffill()


def build_bull_bear(whale: pd.DataFrame, all_prices: pd.DataFrame) -> dict:
    """Edge by regime x threshold x horizon, plus the regime segments
    themselves (consecutive bull/bear runs) for shading timeline charts."""
    ps = gap_filled_series(all_prices)
    price_df = pd.DataFrame({"close": ps}).reset_index(names="timestamp_utc")
    price_df["regime"] = compute_regime(price_df)

    # Base-rate forward returns on the filled grid (shift(-h) = h hours ahead)
    for label, h in BB_HORIZONS.items():
        fut = ps.shift(-h)
        price_df[f"fwd_{label}"] = (fut.values - price_df["close"].values) / price_df["close"].values

    regime_by_hour = price_df.set_index("timestamp_utc")["regime"]
    wh = whale.copy()
    wh["regime"] = wh["hour_utc"].map(regime_by_hour)

    edges = []
    for thresh in SENS_THRESHOLDS:
        sized = wh[wh["usd_value"] >= thresh]
        for label in BB_HORIZONS:
            col = f"fwd_{label}"
            for reg in ("bull", "bear"):
                base_pool = price_df.loc[price_df["regime"] == reg, col].dropna()
                for cat, tx_cat, direction in (
                    ("deposit", "exchange_deposit", "down"),
                    ("withdrawal", "exchange_withdrawal", "up"),
                ):
                    sub = sized.loc[
                        (sized["tx_category"] == tx_cat) & (sized["regime"] == reg), col
                    ].dropna()
                    if len(sub) < MIN_N or len(base_pool) < MIN_N:
                        continue
                    edges.append({
                        "threshold": thresh, "horizon": label, "regime": reg,
                        "category": cat, "n": int(len(sub)),
                        "edge": r2(hit_pct(sub, direction) - hit_pct(base_pool, direction)),
                    })

    # Consecutive same-regime runs become vrect shading segments in the app.
    run_id = (price_df["regime"] != price_df["regime"].shift()).cumsum()
    segments = [
        {"regime": g["regime"].iloc[0],
         "start": g["timestamp_utc"].iloc[0].strftime("%Y-%m-%d"),
         "end": g["timestamp_utc"].iloc[-1].strftime("%Y-%m-%d")}
        for _, g in price_df.groupby(run_id)
    ]
    hours = {reg: int((price_df["regime"] == reg).sum()) for reg in ("bull", "bear")}
    return {"edges": edges, "segments": segments, "hours": hours,
            "horizons": list(BB_HORIZONS), "thresholds": SENS_THRESHOLDS}


# ---------------------------------------------------------------------------
# Drawdown / MAE block (README Section 8; independent of the slider)
# ---------------------------------------------------------------------------

MAE_HORIZONS = [(168, "1w"), (336, "2w"), (720, "1m"), (2160, "3m"), (4320, "6m")]


def build_drawdown(whale: pd.DataFrame, all_prices: pd.DataFrame) -> dict:
    """Maximum adverse excursion for the deposit (sell) signal, replicating
    scripts/run_drawdown_analysis.py: the worst unrealised loss (highest price
    reached) inside the holding window, split by eventual outcome."""
    ps = gap_filled_series(all_prices)
    deposits = whale[whale["tx_category"] == "exchange_deposit"].copy()
    # Entry price from the filled grid so the one gap hour is not dropped.
    deposits["p0"] = deposits["hour_utc"].map(ps)

    rows = []
    for h, label in MAE_HORIZONS:
        fut = (deposits["hour_utc"] + pd.Timedelta(hours=h)).map(ps)
        ret = (fut - deposits["p0"]) / deposits["p0"]
        # Reverse -> trailing rolling max -> reverse back = forward-looking
        # max: the highest price reached anywhere in [t0, t0+h].
        fwd_max = ps[::-1].rolling(window=h + 1, min_periods=1).max()[::-1]
        mae = (deposits["hour_utc"].map(fwd_max) - deposits["p0"]) / deposits["p0"]
        d = deposits.assign(ret=ret, mae=mae).dropna(subset=["ret", "mae"])

        for cond_name, mask in (
            ("unconditional", pd.Series(True, index=d.index)),
            ("extreme greed", d["fng_value"] > 75),
        ):
            sub = d[mask]
            # hit = the sell signal was eventually right (price lower at t0+h)
            for outcome, o in (("hit", sub[sub["ret"] < 0]),
                               ("miss", sub[sub["ret"] >= 0])):
                if len(o) < MIN_N:
                    continue
                rows.append({
                    "horizon": label, "condition": cond_name, "outcome": outcome,
                    "n": int(len(o)),
                    "mean_mae": r2(o["mae"].mean() * 100),
                    "median_mae": r2(o["mae"].median() * 100),
                    "p90_mae": r2(o["mae"].quantile(0.90) * 100),
                    "mean_final_return": r2(o["ret"].mean() * 100),
                })
    return {"rows": rows, "horizons": [lbl for _, lbl in MAE_HORIZONS]}


# ---------------------------------------------------------------------------
# Signal timeline block (when the signal fires, and when it pays)
# ---------------------------------------------------------------------------

def build_timeline(whale: pd.DataFrame, all_prices: pd.DataFrame) -> dict:
    """Monthly cadence of the deposit signal at the base $1M+ threshold:
    how many deposits fired each month, that month's 24h hit rate, the
    matching monthly base rate, and a month-end ETH close for context."""
    deposits = whale[whale["tx_category"] == "exchange_deposit"]
    dep_month = deposits["timestamp_utc"].dt.to_period("M").astype(str)
    price_month = all_prices["timestamp_utc"].dt.to_period("M").astype(str)

    months = sorted(price_month.unique())
    out = {"months": months, "n": [], "hit_24h": [], "base_24h": [], "eth_close": []}
    for m in months:
        d = deposits[dep_month == m]
        a = all_prices[price_month == m]
        out["n"].append(int(len(d)))
        # Only report a monthly hit rate when the month has enough signals.
        out["hit_24h"].append(
            r2(hit_pct(d["fwd_24h"], "down"))
            if d["fwd_24h"].notna().sum() >= MIN_N else None
        )
        out["base_24h"].append(r2(hit_pct(a["fwd_24h"], "down")))
        out["eth_close"].append(
            r2(a.sort_values("timestamp_utc")["close"].iloc[-1])
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    whale, all_prices = load_enriched()
    years = sorted(int(y) for y in whale["year"].unique())
    print(f"Loaded {len(whale):,} whale txs across years {years}.")

    print("Computing base rates...")
    base = build_base_rates(all_prices, years)

    payload = {
        "meta": {
            "n_total": int(len(whale)),
            "date_min": whale["timestamp_utc"].min().strftime("%b %Y"),
            "date_max": whale["timestamp_utc"].max().strftime("%b %Y"),
            "n_labels": 52768,
            "base_rate_24h": r2(base["all_down_24h"]),
        },
        "thresholds": THRESHOLDS,
        "years": years,
        "horizon_labels": HORIZON_LABELS,
        "sens_thresholds": SENS_THRESHOLDS,
        "dist_conditions": list(DIST_CONDITIONS.keys()),
        "threshold_sensitivity": build_threshold_sensitivity(whale, base),
        "by_threshold": {},
    }

    print("Computing bull/bear regime block...")
    payload["bull_bear"] = build_bull_bear(whale, all_prices)
    print("Computing drawdown (MAE) block...")
    payload["drawdown"] = build_drawdown(whale, all_prices)
    print("Computing signal timeline block...")
    payload["timeline"] = build_timeline(whale, all_prices)

    print(f"Computing aggregates for {len(THRESHOLDS)} threshold cuts...")
    for i, thresh in enumerate(THRESHOLDS, 1):
        cut = whale[whale["usd_value"] >= thresh]
        payload["by_threshold"][str(thresh)] = build_for_threshold(cut, base, years)
        if i % 10 == 0:
            print(f"  ...{i}/{len(THRESHOLDS)}")

    out_path = ROOT / "app" / "dashboard_data.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))  # compact
    size_kb = out_path.stat().st_size / 1024
    print(f"\nWrote {out_path} ({size_kb:.0f} KB).")


if __name__ == "__main__":
    main()
