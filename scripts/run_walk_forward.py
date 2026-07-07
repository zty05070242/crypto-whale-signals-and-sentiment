"""
Walk-forward event study: do whale signals hold across different years?

Runs the hit rate analysis independently for each calendar year
(2023, 2024, 2025, 2026) to test stability. Also computes base rates
to isolate the whale-specific edge from market drift.

Usage:
    python scripts/run_walk_forward.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.features.feature_engineer import assign_transaction_label  # noqa: E402
from src.analysis.event_study import (  # noqa: E402
    compute_event_returns,
    walk_forward_by_year,
)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

print("Loading whale transactions...")
whale = pd.read_csv(config.PROCESSED_DATA_DIR / "whale_txs.csv")
print(f"  {len(whale):,} transactions")

print("Assigning transaction categories...")
whale = assign_transaction_label(whale)

print("\nLoading ETH hourly prices...")
prices = pd.read_csv(config.PROCESSED_DATA_DIR / "eth_prices_hourly.csv")
print(f"  {len(prices):,} hourly candles")

fng_path = config.PROCESSED_DATA_DIR / "fear_greed_daily.csv"
fng = pd.read_csv(fng_path) if fng_path.exists() else None
if fng is not None:
    print(f"  {len(fng):,} daily Fear & Greed records")

funding_path = config.PROCESSED_DATA_DIR / "eth_funding_rate.csv"
funding = pd.read_csv(funding_path) if funding_path.exists() else None
if funding is not None:
    print(f"  {len(funding):,} funding rate records")

# ---------------------------------------------------------------------------
# 2. Compute forward returns
# ---------------------------------------------------------------------------

print("\nComputing forward returns...")
events = compute_event_returns(whale, prices)
print(f"  {len(events):,} events with valid forward returns")

# ---------------------------------------------------------------------------
# 3. Walk-forward analysis
# ---------------------------------------------------------------------------

print("\nRunning walk-forward analysis by year...")
wf = walk_forward_by_year(events, prices, fng_df=fng, funding_df=funding)

# ---------------------------------------------------------------------------
# 4. Display results
# ---------------------------------------------------------------------------

# Focus on the strongest signals
print(f"\n{'='*90}")
print("WALK-FORWARD EVENT STUDY: Whale Hit Rates by Year")
print(f"{'='*90}")

for cat in ["exchange_withdrawal", "exchange_deposit"]:
    cat_label = "WITHDRAWALS (buy signal)" if cat == "exchange_withdrawal" else "DEPOSITS (sell signal)"
    print(f"\n  {cat_label}")

    for cond in ["all", "funding_negative", "extreme_greed", "extreme_fear"]:
        subset = wf[(wf["category"] == cat) & (wf["condition"] == cond)]
        if subset.empty:
            continue

        cond_label = {
            "all": "Unconditional",
            "funding_negative": "During negative funding",
            "extreme_greed": "During extreme greed (FnG > 75)",
            "extreme_fear": "During extreme fear (FnG <= 25)",
        }[cond]

        print(f"\n    {cond_label}")
        print(f"    {'Year':>6}  {'Horizon':>8}  {'N':>8}  {'Hit Rate':>9}  "
              f"{'Base Rate':>10}  {'Edge':>7}  {'p-value':>8}  {'Verdict':>10}")

        for _, row in subset.sort_values(["year", "horizon_h"]).iterrows():
            sig = ""
            if row["pvalue"] < 0.001:
                sig = "***"
            elif row["pvalue"] < 0.01:
                sig = "**"
            elif row["pvalue"] < 0.05:
                sig = "*"

            edge = row["whale_edge"]
            if edge > 0.02:
                verdict = "SMART"
            elif edge > 0:
                verdict = "slight"
            elif edge > -0.02:
                verdict = "~random"
            else:
                verdict = "WRONG"

            print(f"    {row['year']:>6}  {row['horizon_h']:>6}h  {row['n']:>8,}  "
                  f"{row['hit_rate']:>8.1%}  {row['base_rate']:>9.1%}  "
                  f"{edge:>+6.1%}  {row['pvalue']:>8.4f}{sig:>3}  "
                  f"{verdict:>10}")

# Save results
output_path = config.ROOT_DIR / "results" / "walk_forward_results.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
wf.to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*90}")
print("KEY QUESTION: Does the whale edge persist out-of-sample in 2026?")
print(f"{'='*90}")

oos = wf[(wf["year"] == 2026) & (wf["condition"] == "funding_negative")
         & (wf["category"] == "exchange_withdrawal") & (wf["horizon_h"] == 24)]
if not oos.empty:
    row = oos.iloc[0]
    print(f"\n  Withdrawals + negative funding, 24h, 2026 (out-of-sample):")
    print(f"    N = {row['n']:,}")
    print(f"    Hit rate: {row['hit_rate']:.1%}")
    print(f"    Base rate: {row['base_rate']:.1%}")
    print(f"    Whale edge: {row['whale_edge']:+.1%}")
    print(f"    p-value: {row['pvalue']:.4f}")
    if row["whale_edge"] > 0.02:
        print(f"    --> Edge SURVIVES out-of-sample")
    elif row["whale_edge"] > 0:
        print(f"    --> Modest edge, may not be economically significant")
    else:
        print(f"    --> Edge does NOT survive out-of-sample")
else:
    print("\n  Insufficient 2026 data for this condition (< 30 events)")
