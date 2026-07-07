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


# ---------------------------------------------------------------------------
# Base rates (slider-independent: computed once from the full price universe)
# ---------------------------------------------------------------------------

def build_base_rates(all_prices: pd.DataFrame, years: list[int]) -> dict:
    """Market-wide base rates that every whale hit rate is compared against."""
    base = {"horizon_down": {}, "yearly_down": {}, "yearly_up": {},
            "yearly_up_negfund": {}, "sentiment_down": {}, "sentiment_up": {}}

    # Base rate of a price fall at each horizon
    for label in HORIZON_LABELS:
        base["horizon_down"][label] = hit_pct(all_prices[f"fwd_{label}"], "down")

    # Per-year base rates at 24h, both directions
    for year in years:
        yr = all_prices[all_prices["year"] == year]["fwd_24h"]
        base["yearly_down"][str(year)] = hit_pct(yr, "down")
        base["yearly_up"][str(year)] = hit_pct(yr, "up")
        yr_neg = all_prices[(all_prices["year"] == year)
                            & (all_prices["funding_rate"] < 0)]["fwd_24h"]
        base["yearly_up_negfund"][str(year)] = hit_pct(yr_neg, "up")

    # Per-sentiment base rates at 24h, both directions
    for name, fn in CONDITIONS.items():
        sub = all_prices[fn(all_prices)]["fwd_24h"]
        base["sentiment_down"][name] = hit_pct(sub, "down")
        base["sentiment_up"][name] = hit_pct(sub, "up")

    # Special conditioned base rates for the threshold-sensitivity section
    base["greed_down"] = hit_pct(all_prices[all_prices["fng_value"] > 75]["fwd_24h"], "down")
    base["negfund_up"] = hit_pct(all_prices[all_prices["funding_rate"] < 0]["fwd_24h"], "up")
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

    # Section 2 + 6: yearly edges
    dep_edge, wd_edge_neg, dep_edge_uncond, wd_edge_uncond = {}, {}, {}, {}
    for year in years:
        y = str(year)
        yr_dep = deposits[deposits["year"] == year]["fwd_24h"]
        yr_wd = withdrawals[withdrawals["year"] == year]["fwd_24h"]
        yr_wd_neg = withdrawals[(withdrawals["year"] == year)
                                & (withdrawals["funding_rate"] < 0)]["fwd_24h"]

        dep_edge[y] = (r2(hit_pct(yr_dep, "down") - base["yearly_down"][y])
                       if yr_dep.notna().sum() >= MIN_N else 0)
        dep_edge_uncond[y] = dep_edge[y]
        wd_edge_uncond[y] = (r2(hit_pct(yr_wd, "up") - base["yearly_up"][y])
                             if yr_wd.notna().sum() >= MIN_N else 0)
        wd_edge_neg[y] = (r2(hit_pct(yr_wd_neg, "up") - base["yearly_up_negfund"][y])
                          if yr_wd_neg.notna().sum() >= MIN_N else 0)

    block["yearly"] = {"deposit_edge": dep_edge, "withdrawal_edge_negfund": wd_edge_neg,
                       "deposit_edge_uncond": dep_edge_uncond,
                       "withdrawal_edge_uncond": wd_edge_uncond}

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

    # Section 5: return distribution for deposits during extreme greed
    greed_ret = (deposits[deposits["fng_value"] > 75]["fwd_24h"].dropna() * 100)
    if len(greed_ret) > 10:
        wins = greed_ret[greed_ret < 0]     # price fell = hit
        losses = greed_ret[greed_ret >= 0]  # price rose = miss
        lo, hi = float(np.floor(greed_ret.min())), float(np.ceil(greed_ret.max()))
        edges = np.linspace(lo, hi, 61)     # 60 shared bins
        hit_counts, _ = np.histogram(wins, bins=edges)
        miss_counts, _ = np.histogram(losses, bins=edges)
        block["return_dist"] = {
            "n": int(len(greed_ret)),
            "hit_rate": r2(len(wins) / len(greed_ret) * 100),
            "miss_rate": r2(len(losses) / len(greed_ret) * 100),
            "avg_hit": r2(wins.mean()) if len(wins) else None,
            "avg_miss": r2(losses.mean()) if len(losses) else None,
            "avg_all": r2(greed_ret.mean()),
            "edges": [round(e, 3) for e in edges.tolist()],
            "hit_counts": hit_counts.tolist(),
            "miss_counts": miss_counts.tolist(),
        }
    else:
        block["return_dist"] = None

    return block


# ---------------------------------------------------------------------------
# Threshold-sensitivity section (independent of the slider)
# ---------------------------------------------------------------------------

def build_threshold_sensitivity(whale: pd.DataFrame, base: dict) -> dict:
    """Edge as a function of minimum ticket size, under two regimes."""
    deposits = whale[whale["tx_category"] == "exchange_deposit"]
    withdrawals = whale[whale["tx_category"] == "exchange_withdrawal"]

    dep_greed, wd_neg = [], []
    for thresh in SENS_THRESHOLDS:
        d = deposits[(deposits["usd_value"] >= thresh)
                     & (deposits["fng_value"] > 75)]["fwd_24h"]
        w = withdrawals[(withdrawals["usd_value"] >= thresh)
                        & (withdrawals["funding_rate"] < 0)]["fwd_24h"]
        dep_greed.append({
            "threshold": thresh, "n": int(d.notna().sum()),
            "edge": r2(hit_pct(d, "down") - base["greed_down"]) if d.notna().sum() >= MIN_N else 0,
        })
        wd_neg.append({
            "threshold": thresh, "n": int(w.notna().sum()),
            "edge": r2(hit_pct(w, "up") - base["negfund_up"]) if w.notna().sum() >= MIN_N else 0,
        })
    return {"deposit_greed": dep_greed, "withdrawal_negfund": wd_neg}


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
        "threshold_sensitivity": build_threshold_sensitivity(whale, base),
        "by_threshold": {},
    }

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
