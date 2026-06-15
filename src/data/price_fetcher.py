"""
ETH/USD hourly price fetcher using Binance's public klines API.

Binance returns up to 1000 hourly candles per request (~41 days). For our
2.5-year window we paginate using startTime/endTime parameters.

No API key required. Rate limit is generous (1200 requests/min).
"""

import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BINANCE_URL = "https://api.binance.com/api/v3/klines"

# Binance returns a max of 1000 candles per request.
# At 1h interval that is ~41 days.
MAX_CANDLES = 1000

# Seconds between API calls — Binance is generous but no need to hammer it
REQUEST_DELAY = 1

DEFAULT_OUTPUT = config.PROCESSED_DATA_DIR / "eth_prices_hourly.csv"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_hourly_prices(
    start_date: str = "2023-01-01",
    end_date: str = "2025-07-01",
) -> pd.DataFrame:
    """
    Fetch hourly ETH/USDT candles from Binance.

    Parameters
    ----------
    start_date : str
        ISO date string for the start of the range (inclusive).
    end_date : str
        ISO date string for the end of the range (exclusive).

    Returns
    -------
    pd.DataFrame
        Columns: timestamp_utc, open, high, low, close, volume.
        One row per hour, sorted chronologically.
    """
    # Convert to millisecond UNIX timestamps (Binance uses ms)
    start_ms = _date_to_ms(start_date)
    end_ms = _date_to_ms(end_date)

    print(f"Fetching ETH/USDT hourly candles: {start_date} to {end_date}")

    all_candles = []
    current_ms = start_ms
    request_count = 0

    while current_ms < end_ms:
        params = {
            "symbol": "ETHUSDT",
            "interval": "1h",
            "startTime": current_ms,
            "endTime": end_ms,
            "limit": MAX_CANDLES,
        }

        response = _request_with_retry(BINANCE_URL, params)
        candles = response.json()

        if not candles:
            break

        all_candles.extend(candles)
        request_count += 1

        # Move start to 1ms after the last candle's open time to avoid
        # duplicates on the next request
        current_ms = candles[-1][0] + 1

        if request_count % 5 == 0:
            print(f"  {len(all_candles):,} candles fetched so far...")

        time.sleep(REQUEST_DELAY)

    print(f"  Done: {len(all_candles):,} candles in {request_count} requests")

    # Binance kline format: [open_time, open, high, low, close, volume,
    #   close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
    # We only need open_time, OHLC, and volume.
    df = pd.DataFrame(all_candles, columns=[
        "open_time_ms", "open", "high", "low", "close", "volume",
        "close_time_ms", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "_ignore",
    ])

    # Convert types — Binance returns all values as strings
    df["timestamp_utc"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Keep only the columns we need
    df = df[["timestamp_utc", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    print(f"  Date range: {df['timestamp_utc'].iloc[0]} to {df['timestamp_utc'].iloc[-1]}")
    print(f"  Price range: ${df['close'].min():,.2f} to ${df['close'].max():,.2f}")

    return df


def save_hourly_prices(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> None:
    """Save hourly prices to CSV."""
    path = output_path or DEFAULT_OUTPUT
    path = type(path) is str and __import__("pathlib").Path(path) or path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} hourly prices to {path}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _date_to_ms(date_str: str) -> int:
    """Convert an ISO date string to a UNIX timestamp in milliseconds."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _request_with_retry(
    url: str,
    params: dict,
    max_retries: int = 5,
) -> requests.Response:
    """Make a GET request with retry logic for rate limits (429) and errors."""
    for attempt in range(max_retries):
        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            return response

        if response.status_code == 429:
            wait = 10 * (2 ** attempt)
            print(f"  Rate limited. Waiting {wait}s (attempt {attempt+1})...")
            time.sleep(wait)
            continue

        response.raise_for_status()

    raise RuntimeError(
        f"Binance request failed after {max_retries} retries. URL: {url}"
    )
