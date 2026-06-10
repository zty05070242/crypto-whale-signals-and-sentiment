"""
GDELT news article fetcher for crypto sentiment analysis.

Queries the GDELT DOC 2.0 API to collect English-language news headlines
mentioning Ethereum, Bitcoin, or crypto. Each day is split into 6-hour
windows (00-06, 06-12, 12-18, 18-24) to stay under the 250-article cap
per request and ensure coverage across the full day.

GDELT constraints:
  - Rate limit: 1 request per 5 seconds (we use 8s to be safe).
  - Max 250 articles per request.
  - Historical queries with startdatetime/enddatetime work back to 2017.

The fetcher saves progress incrementally to a CSV after each day, so an
interruption does not lose already-fetched data.
"""

import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Search terms: covers the main crypto assets relevant to our whale data.
# Parentheses are required by GDELT for OR queries.
DEFAULT_QUERY = "(ethereum OR bitcoin OR crypto)"

# Delay between requests. GDELT's stated limit is 5s, but it throttles
# more aggressively after bursts. 8s balances speed vs reliability.
REQUEST_DELAY_SECONDS = 8

# Articles per window. Set below 250 to keep responses small and reduce
# rate-limiting. 50 per 6-hour window = up to 200 per day, well
# distributed across the clock.
MAX_RECORDS_PER_WINDOW = 50

# 6-hour windows within each day (hour boundaries).
# Each tuple is (start_hour, end_hour).
DAY_WINDOWS = [(0, 6), (6, 12), (12, 18), (18, 24)]

# Default output path for raw article data.
DEFAULT_OUTPUT_PATH: Path = config.ROOT_DIR / "data" / "raw" / "gdelt_articles.csv"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_date_range(
    start: date,
    end: date,
    query: str = DEFAULT_QUERY,
    output_path: Optional[Path] = None,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Fetch GDELT articles for each day from start to end (inclusive).

    Each day is split into 4 x 6-hour windows to get broad coverage
    without hitting the 250-article cap. Saves progress incrementally
    so the process can be interrupted and resumed.

    Parameters
    ----------
    start : date
        First date to fetch (inclusive).
    end : date
        Last date to fetch (inclusive).
    query : str
        GDELT search query. Default searches for ethereum/bitcoin/crypto.
    output_path : Path, optional
        Where to save the CSV. Defaults to data/raw/gdelt_articles.csv.
    resume : bool
        If True and output_path already exists, skip dates that have
        already been fetched. Default True.

    Returns
    -------
    pd.DataFrame
        All fetched articles with columns: title, url, seendate, domain,
        language, sourcecountry, fetch_date.
    """
    save_path = output_path or DEFAULT_OUTPUT_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data if resuming
    existing_df = pd.DataFrame()
    fetched_dates: set[str] = set()
    if resume and save_path.exists():
        existing_df = pd.read_csv(save_path, dtype=str)
        # fetch_date column tracks which calendar date each row came from
        fetched_dates = set(existing_df["fetch_date"].unique())
        print(f"Resuming: {len(fetched_dates)} dates already fetched.")

    all_dfs = [existing_df] if len(existing_df) > 0 else []

    # Generate list of dates to fetch
    current = start
    dates_to_fetch = []
    while current <= end:
        date_str = current.isoformat()
        if date_str not in fetched_dates:
            dates_to_fetch.append(current)
        current += timedelta(days=1)

    total = len(dates_to_fetch)
    if total == 0:
        print("All dates already fetched.")
        return existing_df

    print(f"Fetching {total} days from GDELT ({start} to {end})...")
    print(f"  {len(DAY_WINDOWS)} windows/day, {MAX_RECORDS_PER_WINDOW} articles/window")

    for i, day in enumerate(dates_to_fetch):
        day_articles = _fetch_day_windowed(day, query=query)

        if day_articles is not None and len(day_articles) > 0:
            day_articles["fetch_date"] = day.isoformat()
            all_dfs.append(day_articles)

        # Save progress after each day
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(save_path, index=False)

        count = len(day_articles) if day_articles is not None else 0
        print(f"  [{i + 1}/{total}] {day.isoformat()}: {count} articles")

    result = pd.concat(all_dfs, ignore_index=True)
    result.to_csv(save_path, index=False)
    print(f"\nDone. {len(result):,} total articles saved to {save_path}")
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_day_windowed(
    day: date,
    query: str = DEFAULT_QUERY,
) -> Optional[pd.DataFrame]:
    """
    Fetch articles for a single day using 6-hour windows.

    Makes 4 requests per day (00-06, 06-12, 12-18, 18-24) and combines
    the results. This ensures coverage across the full day rather than
    only getting the first 250 articles (which skew toward one time zone).

    Parameters
    ----------
    day : date
        The date to query.
    query : str
        GDELT search terms.

    Returns
    -------
    pd.DataFrame or None
        Combined articles from all windows. None if all windows failed.
    """
    window_dfs = []

    for start_hour, end_hour in DAY_WINDOWS:
        df = _fetch_single_window(day, start_hour, end_hour, query=query)
        if df is not None and len(df) > 0:
            window_dfs.append(df)

        # Rate limit between each window request
        time.sleep(REQUEST_DELAY_SECONDS)

    if not window_dfs:
        return pd.DataFrame()

    # pd.concat stacks the DataFrames from each window vertically
    return pd.concat(window_dfs, ignore_index=True)


def _fetch_single_window(
    day: date,
    start_hour: int,
    end_hour: int,
    query: str = DEFAULT_QUERY,
    max_retries: int = 5,
) -> Optional[pd.DataFrame]:
    """
    Fetch articles from GDELT for a single time window within a day.

    Parameters
    ----------
    day : date
        The date to query.
    start_hour : int
        Start hour (0-23).
    end_hour : int
        End hour (1-24). 24 means midnight of the next day.
    query : str
        GDELT search terms.
    max_retries : int
        Number of retries on rate-limit (429) errors.

    Returns
    -------
    pd.DataFrame or None
        Columns: title, url, seendate, domain, language, sourcecountry.
        Returns None if the request fails after retries.
    """
    # GDELT datetime format: YYYYMMDDHHmmSS
    start_dt = day.strftime("%Y%m%d") + f"{start_hour:02d}0000"
    if end_hour == 24:
        # Midnight of next day — use 235959 of current day
        end_dt = day.strftime("%Y%m%d") + "235959"
    else:
        end_dt = day.strftime("%Y%m%d") + f"{end_hour:02d}0000"

    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": MAX_RECORDS_PER_WINDOW,
        "format": "json",
        "startdatetime": start_dt,
        "enddatetime": end_dt,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(GDELT_API_URL, params=params, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                articles = data.get("articles", [])
                if not articles:
                    return pd.DataFrame()
                # Convert list of dicts to DataFrame, keeping only useful columns
                df = pd.DataFrame(articles)
                keep_cols = ["title", "url", "seendate", "domain",
                             "language", "sourcecountry"]
                available = [c for c in keep_cols if c in df.columns]
                return df[available]

            if resp.status_code == 429:
                wait = REQUEST_DELAY_SECONDS * (attempt + 2)
                print(f"    Rate limited ({start_hour:02d}-{end_hour:02d}), "
                      f"waiting {wait}s (attempt {attempt + 1})...")
                time.sleep(wait)
                continue

            print(f"    HTTP {resp.status_code} for {day} "
                  f"{start_hour:02d}-{end_hour:02d}: {resp.text[:200]}")
            return None

        except requests.RequestException as e:
            print(f"    Request error for {day} {start_hour:02d}-{end_hour:02d}: {e}")
            if attempt < max_retries - 1:
                time.sleep(REQUEST_DELAY_SECONDS)
            continue

    print(f"    Failed after {max_retries} retries for {day} "
          f"{start_hour:02d}-{end_hour:02d}")
    return None
