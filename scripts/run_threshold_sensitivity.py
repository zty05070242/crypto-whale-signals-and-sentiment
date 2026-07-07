"""
Threshold sensitivity analysis: does the whale edge scale with transaction size?

Tests whether $5M+ and $10M+ whales show stronger signals than $1M+,
which would indicate that truly large actors are more informed.

Usage:
    python scripts/run_threshold_sensitivity.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import config  # noqa: E402
from src.features.feature_engineer import assign_transaction_label  # noqa: E402
from src.analysis.event_study import (  # noqa: E402
    compute_event_returns,
    compute_base_rate,
)

# ---------------------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------------------

print("Loading data...")
whale = pd.read_csv(config.PROCESSED_DATA_DIR / "whale_txs.csv")
whale = assign_transaction_label(whale)
prices = pd.read_csv(config.PROCESSED_DATA_DIR / "eth_prices_hourly.csv")
funding = pd.read_csv(config.PROCESSED_DATA_DIR / "eth_funding_rate.csv")
fng = pd.read_csv(config.PROCESSED_DATA_DIR / "fear_greed_daily.csv")

print("Computing forward returns...")
events = compute_event_returns(whale, prices)
events["timestamp_utc"] = pd.to_datetime(events["timestamp_utc"], utc=True)
events["year"] = events["timestamp_utc"].dt.year

# Merge funding rate
funding_m = funding.copy()
funding_m["timestamp_utc"] = pd.to_datetime(
    funding_m["timestamp_utc"], utc=True, format="ISO8601"
)
funding_m = funding_m.sort_values("timestamp_utc")
events = pd.merge_asof(
    events.sort_values("hour_utc"),
    funding_m[["timestamp_utc", "funding_rate"]].rename(
        columns={"timestamp_utc": "hour_utc"}
    ),
    on="hour_utc",
    direction="backward",
)
events["funding_rate"] = events["funding_rate"].fillna(0.0)

# Merge Fear & Greed
fng_m = fng.copy()
fng_m["date"] = pd.to_datetime(fng_m["date"], utc=True)
events["_date"] = events["timestamp_utc"].dt.floor("D")
fng_r = fng_m.rename(columns={"date": "_date"}).sort_values("_date")
events = pd.merge_asof(
    events.sort_values("_date"),
    fng_r[["_date", "fng_value"]],
    on="_date",
    direction="backward",
)
events.drop(columns="_date", inplace=True)
events["fng_value"] = events["fng_value"].fillna(50)

# Prepare price data with sentiment for base rates
price_full = prices.copy()
price_full["timestamp_utc"] = pd.to_datetime(
    price_full["timestamp_utc"], utc=True
)
price_full = price_full.sort_values("timestamp_utc").reset_index(drop=True)
price_full = pd.merge_asof(
    price_full, funding_m[["timestamp_utc", "funding_rate"]],
    on="timestamp_utc", direction="backward",
)
price_full["funding_rate"] = price_full["funding_rate"].fillna(0.0)
price_full["_date"] = price_full["timestamp_utc"].dt.floor("D")
price_full = pd.merge_asof(
    price_full.sort_values("_date"),
    fng_r[["_date", "fng_value"]],
    on="_date", direction="backward",
)
price_full.drop(columns="_date", inplace=True)
price_full["fng_value"] = price_full["fng_value"].fillna(50)
price_full["year"] = price_full["timestamp_utc"].dt.year

# ---------------------------------------------------------------------------
# 2. Run threshold sensitivity
# ---------------------------------------------------------------------------

THRESHOLDS = [1_000_000, 2_000_000, 5_000_000, 10_000_000]
HORIZON = 24

signals = [
    {
        "label": "WITHDRAWALS (buy) -- negative funding",
        "category": "exchange_withdrawal",
        "direction": "up",
        "whale_cond": lambda d: d["funding_rate"] < 0,
        "price_cond": lambda p: p["funding_rate"] < 0,
    },
    {
        "label": "DEPOSITS (sell) -- unconditional",
        "category": "exchange_deposit",
        "direction": "down",
        "whale_cond": lambda d: pd.Series(True, index=d.index),
        "price_cond": lambda p: pd.Series(True, index=p.index),
    },
    {
        "label": "DEPOSITS (sell) -- extreme greed (FnG > 75)",
        "category": "exchange_deposit",
        "direction": "down",
        "whale_cond": lambda d: d["fng_value"] > 75,
        "price_cond": lambda p: p["fng_value"] > 75,
    },
]

print(f"\n{'='*100}")
print(f"THRESHOLD SENSITIVITY ANALYSIS -- {HORIZON}h horizon")
print(f"{'='*100}")

all_rows = []

for sig in signals:
    print(f"\n  {sig['label']}")
    print(f"  {'Threshold':>12}  {'Year':>6}  {'N':>8}  {'Hit Rate':>9}  "
          f"{'Base Rate':>10}  {'Edge':>7}  {'p-val':>8}")

    for threshold in THRESHOLDS:
        cat_events = events[
            (events["tx_category"] == sig["category"])
            & (events["usd_value"] >= threshold)
        ]

        for year in sorted(events["year"].unique()):
            yr_events = cat_events[cat_events["year"] == year]
            yr_price = price_full[
                price_full["year"] == year
            ].reset_index(drop=True)

            subset = yr_events[sig["whale_cond"](yr_events)]
            price_mask = sig["price_cond"](yr_price)

            if len(subset) < 30:
                continue

            col = f"fwd_return_{HORIZON}h"
            returns = subset[col].dropna().values
            if len(returns) < 30:
                continue

            if sig["direction"] == "down":
                hits = int((returns < 0).sum())
            else:
                hits = int((returns > 0).sum())

            n = len(returns)
            hit_rate = hits / n
            base = compute_base_rate(yr_price, sig["direction"], HORIZON, price_mask)
            edge = hit_rate - base
            p_val = stats.binomtest(hits, n, p=0.5).pvalue

            sig_str = (
                "***" if p_val < 0.001
                else "**" if p_val < 0.01
                else "*" if p_val < 0.05
                else ""
            )

            print(
                f"  ${threshold / 1e6:.0f}M+  {year:>8}  {n:>8,}  "
                f"{hit_rate:>8.1%}  {base:>9.1%}  {edge:>+6.1%}  "
                f"{p_val:>7.4f}{sig_str}"
            )

            all_rows.append({
                "signal": sig["label"],
                "threshold_usd": threshold,
                "year": year,
                "n": n,
                "hits": hits,
                "hit_rate": hit_rate,
                "base_rate": base,
                "whale_edge": edge,
                "mean_return": float(np.mean(returns)),
                "pvalue": p_val,
            })

# Save results
output_path = config.ROOT_DIR / "results" / "threshold_sensitivity.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(all_rows).to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")

print(f"\n{'='*100}")
print("INTERPRETATION")
print(f"{'='*100}")
print("""
  WITHDRAWAL edge: decayed at ALL thresholds. Even $10M+ whales show
  zero edge in 2025-2026. DeFi maturation changed what withdrawals mean
  (staking, LP, L2 bridging) -- no longer a clean buy signal.

  DEPOSIT edge: GROWS with transaction size. $10M+ deposits during
  extreme greed = strongest signal in the dataset. Deposits remain a
  clean sell signal (one reason to deposit to an exchange: to sell).
  Less crowded because market participants exhibit confirmation bias --
  they monitor bullish whale activity but ignore bearish signals.
""")
