"""
Event study: are whales smart money?

Core question: when a whale deposits to an exchange (sell signal), does the
price subsequently drop? When they withdraw (buy signal), does price rise?

We measure the HIT RATE -- what percentage of whale actions correctly predict
the subsequent price direction. If > 50%, whales are smart money. If = 50%,
they are no better than random.

Important: we are NOT claiming whales cause price movements. A $1M transaction
does not move ETH's price. We are testing whether whales ACT AHEAD of price
moves -- whether they have informational advantage.

Methodology:
  1. For each exchange deposit, check if price was LOWER at t+1h, t+6h, t+24h.
     If yes, the whale "correctly" sold before a drop.
  2. For each exchange withdrawal, check if price was HIGHER at t+1h, t+6h, t+24h.
     If yes, the whale "correctly" bought before a rise.
  3. Compute hit rates and test significance with a binomial test.
  4. Condition on sentiment to see if whales are smarter in certain regimes.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

import config


# Prediction horizons to measure (in hours)
HORIZONS = [1, 6, 24]


def compute_event_returns(
    whale_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each whale transaction, compute the forward ETH return at each horizon.

    Parameters
    ----------
    whale_df : pd.DataFrame
        Whale transactions with timestamp_utc and tx_category columns.
    price_df : pd.DataFrame
        Hourly ETH prices with timestamp_utc and close columns.

    Returns
    -------
    pd.DataFrame
        whale_df with fwd_return_Xh columns (percentage return after event).
    """
    whale = whale_df.copy()
    whale["timestamp_utc"] = pd.to_datetime(whale["timestamp_utc"], utc=True)

    price = price_df.copy()
    price["timestamp_utc"] = pd.to_datetime(price["timestamp_utc"], utc=True)
    price = price.sort_values("timestamp_utc").reset_index(drop=True)

    # Floor whale timestamps to the hour for price lookup
    whale["hour_utc"] = whale["timestamp_utc"].dt.floor("h")

    # Build lookup dict: hour -> close price
    price_lookup = price.set_index("timestamp_utc")["close"].to_dict()

    for h in HORIZONS:
        offset = pd.Timedelta(hours=h)
        col = f"fwd_return_{h}h"

        def _calc(row, _offset=offset):
            current_hour = row["hour_utc"]
            future_hour = current_hour + _offset
            current_price = price_lookup.get(current_hour)
            future_price = price_lookup.get(future_hour)

            if current_price is None or future_price is None or current_price == 0:
                return np.nan

            return (future_price - current_price) / current_price

        whale[col] = whale.apply(_calc, axis=1)

    # Drop rows where any return is NaN
    return_cols = [f"fwd_return_{h}h" for h in HORIZONS]
    whale = whale.dropna(subset=return_cols).reset_index(drop=True)

    return whale


def compute_hit_rates(events_df: pd.DataFrame) -> dict:
    """
    Compute hit rates: how often did the whale's action match price direction?

    - Exchange deposit (sell signal): hit = price went DOWN afterward
    - Exchange withdrawal (buy signal): hit = price went UP afterward
    - DeFi interaction: treated as buy signal (deploying capital)

    A hit rate of 50% = random. Above 50% = smart money. Below 50% = dumb money.

    Uses a binomial test to determine if the hit rate is significantly
    different from 50% (random chance).

    Parameters
    ----------
    events_df : pd.DataFrame
        Output of compute_event_returns().

    Returns
    -------
    dict
        results[category][horizon] = {
            'n': int, 'hits': int, 'hit_rate': float,
            'pvalue': float, 'direction': str
        }
    """
    # Define what counts as a "hit" for each category
    # Deposit = sell signal, so a hit is price going DOWN (return < 0)
    # Withdrawal = buy signal, so a hit is price going UP (return > 0)
    category_directions = {
        "exchange_deposit": "down",      # selling before a drop = smart
        "exchange_withdrawal": "up",     # buying before a rise = smart
        "defi_interaction": "up",        # deploying capital = bullish bet
    }

    results = {}

    for cat, expected_dir in category_directions.items():
        cat_data = events_df[events_df["tx_category"] == cat]

        if len(cat_data) < 30:
            print(f"  Skipping {cat}: only {len(cat_data)} events")
            continue

        results[cat] = {}

        for h in HORIZONS:
            col = f"fwd_return_{h}h"
            returns = cat_data[col].values

            if expected_dir == "down":
                # Hit = price dropped (return < 0)
                hits = int((returns < 0).sum())
            else:
                # Hit = price rose (return > 0)
                hits = int((returns > 0).sum())

            n = len(returns)
            hit_rate = hits / n

            # Binomial test: is hit rate significantly different from 50%?
            # binom_test tests if the number of successes in n trials
            # differs from what we'd expect under p=0.5 (random)
            p_value = stats.binomtest(hits, n, p=0.5).pvalue

            results[cat][h] = {
                "n": n,
                "hits": hits,
                "hit_rate": hit_rate,
                "pvalue": p_value,
                "expected_direction": expected_dir,
                "mean_return": np.mean(returns),
            }

    return results


def compute_conditioned_hit_rates(
    events_df: pd.DataFrame,
    fng_df: Optional[pd.DataFrame] = None,
    funding_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Hit rates conditioned on sentiment regime.

    Are whales smarter during extreme fear vs extreme greed?
    Does sentiment improve the whale signal?

    Parameters
    ----------
    events_df : pd.DataFrame
        Output of compute_event_returns().
    fng_df : pd.DataFrame, optional
        Daily Fear & Greed data.
    funding_df : pd.DataFrame, optional
        Binance funding rate data.

    Returns
    -------
    dict
        results[category][condition][horizon] = stats dict
    """
    df = events_df.copy()

    # Merge Fear & Greed Index
    if fng_df is not None:
        fng = fng_df.copy()
        fng["date"] = pd.to_datetime(fng["date"], utc=True)
        df["_date"] = df["timestamp_utc"].dt.floor("D")
        fng = fng.rename(columns={"date": "_date"}).sort_values("_date")

        df = pd.merge_asof(
            df.sort_values("_date"),
            fng[["_date", "fng_value"]],
            on="_date",
            direction="backward",
        )
        df.drop(columns="_date", inplace=True)
        df["fng_value"] = df["fng_value"].fillna(50)

    # Merge funding rate
    if funding_df is not None:
        funding = funding_df.copy()
        funding["timestamp_utc"] = pd.to_datetime(
            funding["timestamp_utc"], utc=True, format="ISO8601"
        )
        funding = funding.sort_values("timestamp_utc")

        df = pd.merge_asof(
            df.sort_values("hour_utc"),
            funding[["timestamp_utc", "funding_rate"]].rename(
                columns={"timestamp_utc": "hour_utc"}
            ),
            on="hour_utc",
            direction="backward",
        )
        df["funding_rate"] = df["funding_rate"].fillna(0.0)

    # Define sentiment conditions
    conditions = {}

    if "fng_value" in df.columns:
        conditions["extreme_fear"] = df["fng_value"] <= 25
        conditions["fear"] = (df["fng_value"] > 25) & (df["fng_value"] <= 45)
        conditions["neutral"] = (df["fng_value"] > 45) & (df["fng_value"] <= 55)
        conditions["greed"] = (df["fng_value"] > 55) & (df["fng_value"] <= 75)
        conditions["extreme_greed"] = df["fng_value"] > 75

    if "funding_rate" in df.columns:
        conditions["funding_negative"] = df["funding_rate"] < 0
        conditions["funding_positive"] = df["funding_rate"] >= 0

    category_directions = {
        "exchange_deposit": "down",
        "exchange_withdrawal": "up",
    }

    results = {}

    for cat, expected_dir in category_directions.items():
        results[cat] = {}
        cat_data = df[df["tx_category"] == cat]

        for cond_name, cond_mask in conditions.items():
            subset = cat_data[cond_mask[cat_data.index]]

            if len(subset) < 30:
                continue

            results[cat][cond_name] = {}

            for h in HORIZONS:
                col = f"fwd_return_{h}h"
                returns = subset[col].values

                if expected_dir == "down":
                    hits = int((returns < 0).sum())
                else:
                    hits = int((returns > 0).sum())

                n = len(returns)
                hit_rate = hits / n
                p_value = stats.binomtest(hits, n, p=0.5).pvalue

                results[cat][cond_name][h] = {
                    "n": n,
                    "hits": hits,
                    "hit_rate": hit_rate,
                    "pvalue": p_value,
                    "mean_return": np.mean(returns),
                }

    return results


def print_hit_rate_results(results: dict) -> None:
    """Print hit rate results."""
    print(f"\n{'='*80}")
    print("WHALE SMART MONEY TEST: Hit Rates")
    print("(Hit = whale's action correctly predicted price direction)")
    print("(50% = random, >50% = smart money, <50% = dumb money)")
    print(f"{'='*80}")

    for cat, horizons in results.items():
        if not horizons:
            continue

        direction = horizons[HORIZONS[0]]["expected_direction"]
        action = "sold before drop" if direction == "down" else "bought before rise"

        print(f"\n  {cat.upper().replace('_', ' ')}  (hit = {action})")
        print(f"  {'Horizon':>8}  {'N':>8}  {'Hits':>8}  {'Hit Rate':>9}  "
              f"{'p-value':>8}  {'Sig':>5}  {'Verdict':>12}")

        for h, s in sorted(horizons.items()):
            sig = ""
            if s["pvalue"] < 0.001:
                sig = "***"
            elif s["pvalue"] < 0.01:
                sig = "**"
            elif s["pvalue"] < 0.05:
                sig = "*"

            if s["hit_rate"] > 0.50 and s["pvalue"] < 0.05:
                verdict = "SMART"
            elif s["hit_rate"] < 0.50 and s["pvalue"] < 0.05:
                verdict = "WRONG"
            else:
                verdict = "random"

            print(f"  {h:>6}h  {s['n']:>8,}  {s['hits']:>8,}  "
                  f"{s['hit_rate']:>8.1%}  "
                  f"{s['pvalue']:>8.4f}  "
                  f"{sig:>5}  "
                  f"{verdict:>12}")


def print_conditioned_hit_rates(results: dict) -> None:
    """Print sentiment-conditioned hit rate results."""
    print(f"\n{'='*80}")
    print("CONDITIONED HIT RATES: Are Whales Smarter in Certain Sentiment Regimes?")
    print(f"{'='*80}")

    for cat, conditions in results.items():
        if not conditions:
            continue

        print(f"\n  {cat.upper().replace('_', ' ')}")

        for cond_name, horizons in sorted(conditions.items()):
            print(f"\n    {cond_name}")
            print(f"    {'Horizon':>8}  {'N':>8}  {'Hit Rate':>9}  "
                  f"{'p-value':>8}  {'Sig':>5}  {'Verdict':>12}")

            for h, s in sorted(horizons.items()):
                sig = ""
                if s["pvalue"] < 0.001:
                    sig = "***"
                elif s["pvalue"] < 0.01:
                    sig = "**"
                elif s["pvalue"] < 0.05:
                    sig = "*"

                if s["hit_rate"] > 0.50 and s["pvalue"] < 0.05:
                    verdict = "SMART"
                elif s["hit_rate"] < 0.50 and s["pvalue"] < 0.05:
                    verdict = "WRONG"
                else:
                    verdict = "random"

                print(f"    {h:>6}h  {s['n']:>8,}  "
                      f"{s['hit_rate']:>8.1%}  "
                      f"{s['pvalue']:>8.4f}  "
                      f"{sig:>5}  "
                      f"{verdict:>12}")
