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

    # Vectorised: merge current-hour price, then future-hour prices
    price_series = price.set_index("timestamp_utc")["close"]

    # Current price at event hour
    whale = whale.merge(
        price_series.rename("price_t0"),
        left_on="hour_utc", right_index=True, how="left",
    )

    for h in HORIZONS:
        col = f"fwd_return_{h}h"
        future_hour = whale["hour_utc"] + pd.Timedelta(hours=h)
        future_price = future_hour.map(price_series)
        whale[col] = (future_price - whale["price_t0"]) / whale["price_t0"]

    whale.drop(columns="price_t0", inplace=True)

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


def compute_base_rate(
    price_df: pd.DataFrame,
    direction: str,
    horizon: int,
    condition_mask: Optional[pd.Series] = None,
) -> float:
    """
    Compute the base rate: what fraction of ALL hours saw price go in the
    expected direction? This is the benchmark a whale must beat to show edge.

    Parameters
    ----------
    price_df : pd.DataFrame
        Hourly prices with timestamp_utc and close columns.
    direction : str
        'up' or 'down'.
    horizon : int
        Hours forward to measure.
    condition_mask : pd.Series, optional
        Boolean mask (positionally aligned with price_df) to restrict
        to a sentiment regime.

    Returns
    -------
    float
        Fraction of hours where price moved in the expected direction.
    """
    price = price_df.copy()
    price["timestamp_utc"] = pd.to_datetime(price["timestamp_utc"], utc=True)
    price = price.sort_values("timestamp_utc").reset_index(drop=True)

    price_series = price.set_index("timestamp_utc")["close"]
    future = price["timestamp_utc"] + pd.Timedelta(hours=horizon)
    future_price = future.map(price_series)
    fwd_return = (future_price - price["close"]) / price["close"]

    # Apply condition mask by position (numpy array) to avoid index mismatch
    if condition_mask is not None:
        mask_arr = condition_mask.values if hasattr(condition_mask, "values") else condition_mask
        fwd_return = fwd_return[mask_arr]

    valid = fwd_return.dropna()

    if len(valid) == 0:
        return 0.5

    if direction == "up":
        return float((valid > 0).sum() / len(valid))
    else:
        return float((valid < 0).sum() / len(valid))


def walk_forward_by_year(
    events_df: pd.DataFrame,
    price_df: pd.DataFrame,
    fng_df: Optional[pd.DataFrame] = None,
    funding_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Run the event study independently for each calendar year to show
    whether the whale signal is stable over time.

    For each year, computes:
    - Whale hit rate for withdrawals and deposits
    - Base rate under the same conditions
    - Whale edge (hit rate minus base rate)

    Parameters
    ----------
    events_df : pd.DataFrame
        Output of compute_event_returns() with sentiment already merged.
    price_df : pd.DataFrame
        Hourly ETH prices.
    fng_df : pd.DataFrame, optional
        Daily Fear & Greed data.
    funding_df : pd.DataFrame, optional
        Binance funding rate data.

    Returns
    -------
    pd.DataFrame
        One row per year x category x condition x horizon.
    """
    df = events_df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # Merge sentiment if not already present
    if fng_df is not None and "fng_value" not in df.columns:
        fng = fng_df.copy()
        fng["date"] = pd.to_datetime(fng["date"], utc=True)
        df["_date"] = df["timestamp_utc"].dt.floor("D")
        fng = fng.rename(columns={"date": "_date"}).sort_values("_date")
        df = pd.merge_asof(
            df.sort_values("_date"), fng[["_date", "fng_value"]],
            on="_date", direction="backward",
        )
        df.drop(columns="_date", inplace=True)
        df["fng_value"] = df["fng_value"].fillna(50)

    if funding_df is not None and "funding_rate" not in df.columns:
        funding = funding_df.copy()
        funding["timestamp_utc"] = pd.to_datetime(
            funding["timestamp_utc"], utc=True, format="ISO8601",
        )
        funding = funding.sort_values("timestamp_utc")
        df = pd.merge_asof(
            df.sort_values("hour_utc"),
            funding[["timestamp_utc", "funding_rate"]].rename(
                columns={"timestamp_utc": "hour_utc"}
            ),
            on="hour_utc", direction="backward",
        )
        df["funding_rate"] = df["funding_rate"].fillna(0.0)

    # Prepare price data with sentiment for base rate computation
    price = price_df.copy()
    price["timestamp_utc"] = pd.to_datetime(price["timestamp_utc"], utc=True)
    price = price.sort_values("timestamp_utc").reset_index(drop=True)

    if fng_df is not None:
        fng2 = fng_df.copy()
        fng2["date"] = pd.to_datetime(fng2["date"], utc=True)
        price["_date"] = price["timestamp_utc"].dt.floor("D")
        fng2 = fng2.rename(columns={"date": "_date"}).sort_values("_date")
        price = pd.merge_asof(
            price.sort_values("_date"), fng2[["_date", "fng_value"]],
            on="_date", direction="backward",
        )
        price.drop(columns="_date", inplace=True)
        price["fng_value"] = price["fng_value"].fillna(50)

    if funding_df is not None:
        funding2 = funding_df.copy()
        funding2["timestamp_utc"] = pd.to_datetime(
            funding2["timestamp_utc"], utc=True, format="ISO8601",
        )
        funding2 = funding2.sort_values("timestamp_utc")
        price = pd.merge_asof(
            price.sort_values("timestamp_utc"),
            funding2[["timestamp_utc", "funding_rate"]],
            on="timestamp_utc", direction="backward",
        )
        price["funding_rate"] = price["funding_rate"].fillna(0.0)

    price["year"] = price["timestamp_utc"].dt.year
    df["year"] = df["timestamp_utc"].dt.year

    category_directions = {
        "exchange_withdrawal": "up",
        "exchange_deposit": "down",
    }

    # Conditions to test (label, whale_mask_fn, price_mask_fn)
    cond_defs = [
        ("all", lambda d: pd.Series(True, index=d.index), lambda p: pd.Series(True, index=p.index)),
        ("funding_negative",
         lambda d: d["funding_rate"] < 0,
         lambda p: p["funding_rate"] < 0),
        ("extreme_greed",
         lambda d: d["fng_value"] > 75,
         lambda p: p["fng_value"] > 75),
        ("extreme_fear",
         lambda d: d["fng_value"] <= 25,
         lambda p: p["fng_value"] <= 25),
    ]

    rows = []
    years = sorted(df["year"].unique())

    for year in years:
        yr_events = df[df["year"] == year]
        yr_price = price[price["year"] == year]

        price_series_yr = yr_price.set_index("timestamp_utc")["close"]

        for cat, expected_dir in category_directions.items():
            cat_data = yr_events[yr_events["tx_category"] == cat]

            for cond_label, whale_mask_fn, price_mask_fn in cond_defs:
                subset = cat_data[whale_mask_fn(cat_data)]
                price_subset_mask = price_mask_fn(yr_price)

                if len(subset) < 30:
                    continue

                for h in HORIZONS:
                    col = f"fwd_return_{h}h"
                    if col not in subset.columns:
                        continue

                    returns = subset[col].dropna().values
                    if len(returns) < 30:
                        continue

                    if expected_dir == "down":
                        hits = int((returns < 0).sum())
                    else:
                        hits = int((returns > 0).sum())

                    n = len(returns)
                    hit_rate = hits / n

                    # Base rate for this year and condition
                    base = compute_base_rate(
                        yr_price, expected_dir, h, price_subset_mask,
                    )

                    p_value = stats.binomtest(hits, n, p=0.5).pvalue

                    rows.append({
                        "year": year,
                        "category": cat,
                        "condition": cond_label,
                        "horizon_h": h,
                        "n": n,
                        "hits": hits,
                        "hit_rate": hit_rate,
                        "base_rate": base,
                        "whale_edge": hit_rate - base,
                        "mean_return": float(np.mean(returns)),
                        "pvalue": p_value,
                    })

    return pd.DataFrame(rows)
