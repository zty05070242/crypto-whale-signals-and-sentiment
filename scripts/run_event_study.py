"""
Run the event study: are whales smart money?

When a whale deposits to an exchange (sell signal), does price drop?
When they withdraw (buy signal), does price rise?
Hit rate > 50% = smart money. Hit rate = 50% = random.

Usage:
    python scripts/run_event_study.py
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
    compute_hit_rates,
    compute_conditioned_hit_rates,
    print_hit_rate_results,
    print_conditioned_hit_rates,
)

# ---------------------------------------------------------------------------
# 1. Load whale transactions
# ---------------------------------------------------------------------------

print("Loading whale transactions...")
whale = pd.read_csv(config.PROCESSED_DATA_DIR / "whale_txs.csv")
print(f"  {len(whale):,} transactions")

print("Assigning transaction categories...")
whale = assign_transaction_label(whale)

print(f"  Category distribution:")
for cat, count in whale["tx_category"].value_counts().items():
    print(f"    {cat}: {count:,} ({count/len(whale):.1%})")

# ---------------------------------------------------------------------------
# 2. Load ETH prices
# ---------------------------------------------------------------------------

print("\nLoading ETH hourly prices...")
prices = pd.read_csv(config.PROCESSED_DATA_DIR / "eth_prices_hourly.csv")
print(f"  {len(prices):,} hourly candles")

# ---------------------------------------------------------------------------
# 3. Compute forward returns
# ---------------------------------------------------------------------------

print("\nComputing forward returns for each whale event...")
events = compute_event_returns(whale, prices)
print(f"  {len(events):,} events with valid forward returns")

# ---------------------------------------------------------------------------
# 4. Hit rate analysis (the core test)
# ---------------------------------------------------------------------------

print("\nComputing hit rates...")
hit_results = compute_hit_rates(events)
print_hit_rate_results(hit_results)

# ---------------------------------------------------------------------------
# 5. Conditioned hit rates (does sentiment help?)
# ---------------------------------------------------------------------------

print("\nLoading sentiment data...")

fng_path = config.PROCESSED_DATA_DIR / "fear_greed_daily.csv"
fng = pd.read_csv(fng_path) if fng_path.exists() else None
if fng is not None:
    print(f"  {len(fng):,} daily Fear & Greed records")

funding_path = config.PROCESSED_DATA_DIR / "eth_funding_rate.csv"
funding = pd.read_csv(funding_path) if funding_path.exists() else None
if funding is not None:
    print(f"  {len(funding):,} funding rate records")

print("\nComputing conditioned hit rates...")
conditioned = compute_conditioned_hit_rates(events, fng_df=fng, funding_df=funding)
print_conditioned_hit_rates(conditioned)

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*80}")
print("INTERPRETATION")
print(f"{'='*80}")
print("""
  SMART  = hit rate significantly above 50% (whales predicted correctly)
  WRONG  = hit rate significantly below 50% (whales predicted incorrectly)
  random = not significantly different from 50% (no predictive ability)

  If whales are smart money, we expect:
    - Exchange deposits: hit rate > 50% (they sell before price drops)
    - Exchange withdrawals: hit rate > 50% (they buy before price rises)

  The conditioned analysis tests whether sentiment regime matters:
    - Do whales predict better during extreme fear or extreme greed?
    - Does funding rate (futures positioning) interact with whale signals?
""")
