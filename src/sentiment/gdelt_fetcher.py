"""
GDELT news article fetcher for crypto sentiment analysis.

Queries the GDELT DOC 2.0 API day by day to collect English-language news
headlines mentioning Ethereum, Bitcoin, or crypto. The titles are later
scored with VADER for sentiment.

GDELT constraints:
  - Rate limit: 1 request per 5 seconds (we use 6s to be safe).
  - Max 250 articles per request.
  - Rolling 3-month search window, but historical queries with
    startdatetime/enddatetime work for any date back to 2017.

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

# GDELT rate limit is 1 request per 5 seconds. We use 6 to be safe.
REQUEST_DELAY_SECONDS = 6

# Max articles GDELT returns per request.
MAX_RECORDS_PER_REQUEST = 250

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
    Fetch GDELT articles day by day from start to end (inclusive).

    Saves progress incrementally to output_path after each day, so the
    process can be interrupted and resumed without losing data.

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

    for i, day in enumerate(dates_to_fetch):
        articles = fetch_single_day(day, query=query)

        if articles is not None and len(articles) > 0:
            # Tag each row with the calendar date we queried for
            articles["fetch_date"] = day.isoformat()
            all_dfs.append(articles)

        # Save progress after each day
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(save_path, index=False)

        progress = f"[{i + 1}/{total}]"
        count = len(articles) if articles is not None else 0
        print(f"  {progress} {day.isoformat()}: {count} articles")

        # Rate limit: wait between requests
        if i < total - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    result = pd.concat(all_dfs, ignore_index=True)
    result.to_csv(save_path, index=False)
    print(f"\nDone. {len(result):,} total articles saved to {save_path}")
    return result


def fetch_single_day(
    day: date,
    query: str = DEFAULT_QUERY,
    max_retries: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Fetch articles from GDELT for a single calendar day.

    Parameters
    ----------
    day : date
        The date to query.
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
    start_dt = day.strftime("%Y%m%d") + "000000"
    end_dt = day.strftime("%Y%m%d") + "235959"

    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": MAX_RECORDS_PER_REQUEST,
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
                # Select and rename columns we need
                keep_cols = ["title", "url", "seendate", "domain",
                             "language", "sourcecountry"]
                # Only keep columns that exist (some may be missing)
                available = [c for c in keep_cols if c in df.columns]
                return df[available]

            if resp.status_code == 429:
                # Rate limited — wait longer and retry
                wait = REQUEST_DELAY_SECONDS * (attempt + 2)
                print(f"    Rate limited, waiting {wait}s (attempt {attempt + 1})...")
                time.sleep(wait)
                continue

            # Other error — log and return None
            print(f"    HTTP {resp.status_code} for {day}: {resp.text[:200]}")
            return None

        except requests.RequestException as e:
            print(f"    Request error for {day}: {e}")
            if attempt < max_retries - 1:
                time.sleep(REQUEST_DELAY_SECONDS)
            continue

    print(f"    Failed after {max_retries} retries for {day}")
    return None