"""
Whale transaction fetcher.

Responsibilities:
  1. Run the Dune query via dune_client.run_query()
  2. Validate that the returned schema matches what we expect
  3. Parse string columns into correct Python types
  4. Save the result to disk

All whale-specific knowledge (column names, threshold defaults, output path)
lives here. dune_client.py knows nothing about whales.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

import config
from src.data.dune_client import run_query


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------

# These are the exact column names the Dune query must return.
# If the query is edited and a column is renamed, _validate_schema() will
# catch the mismatch immediately with a clear error rather than a cryptic
# KeyError somewhere downstream.
EXPECTED_COLUMNS: set[str] = {
    "timestamp_utc",
    "block_number",
    "tx_hash",
    "from_address",
    "to_address",
    "eth_value",
    "usd_value",
    "eth_usd_price",
    "gas_price_gwei",
    "gas_used",
    "tx_fee_eth",
    "is_contract_call",
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def fetch_whale_transactions(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    usd_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Run the Dune whale query and return a validated, typed DataFrame.

    Parameters default to values in config if not provided, which lets you
    override per-call without changing global config — useful for tests or
    one-off analyses with a narrower date range.

    Parameters
    ----------
    start_date : str, optional
        ISO date string, e.g. "2023-01-01". Passed as a Dune query parameter.
    end_date : str, optional
        ISO date string, e.g. "2025-01-01". End is exclusive in the SQL.
    usd_threshold : float, optional
        Minimum USD value at transaction time. Default: config.WHALE_USD_THRESHOLD.

    Returns
    -------
    pd.DataFrame
        Columns as defined in EXPECTED_COLUMNS, with correct Python types.
        Sorted ascending by timestamp_utc.
    """
    if not config.DUNE_QUERY_ID:
        raise ValueError(
            "DUNE_QUERY_ID is not set. Create the query in the Dune UI, "
            "then add DUNE_QUERY_ID=<id> to your .env file."
        )

    params = {
        "start_date": start_date or config.FETCH_START_DATE,
        "end_date": end_date or config.FETCH_END_DATE,
        "whale_usd_threshold": usd_threshold or config.WHALE_USD_THRESHOLD,
        "eth_prefilter": config.WHALE_ETH_PREFILTER,
    }

    raw_df = run_query(config.DUNE_QUERY_ID, parameters=params)
    validated_df = _validate_schema(raw_df)
    typed_df = _parse_types(validated_df)

    # Ensure chronological order — Dune returns sorted by block_time but
    # make this explicit so downstream code can rely on it
    typed_df = typed_df.sort_values("timestamp_utc").reset_index(drop=True)

    return typed_df


def save_whale_transactions(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save whale transactions to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Output of fetch_whale_transactions().
    output_path : Path, optional
        Defaults to config.PROCESSED_DATA_DIR / "whale_txs.csv".

    Returns
    -------
    Path
        The path the file was written to.
    """
    if output_path is None:
        output_path = config.PROCESSED_DATA_DIR / "whale_txs.csv"

    # mkdir with parents=True creates intermediate directories if they do not exist.
    # exist_ok=True means no error if the directory already exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # index=False prevents pandas from writing the integer row index as a column
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} whale transactions to {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check that the Dune result contains all expected columns.

    Raises ValueError if any are missing — extra columns from Dune are
    silently dropped so they do not propagate into the processed CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from fetch_results_csv().

    Returns
    -------
    pd.DataFrame
        Same data, restricted to EXPECTED_COLUMNS.
    """
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dune result is missing expected columns: {missing}.\n"
            "Check that the Dune query output matches the schema in "
            "docs/dune_queries/whale_transactions.sql."
        )

    # Select only declared columns in a consistent order for reproducibility
    return df[sorted(EXPECTED_COLUMNS)]


def _parse_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast each column from string (Dune CSV default) to the correct Python type.

    Dune's CSV endpoint returns all values as strings. This function makes
    explicit type coercions so downstream code can rely on correct types
    (e.g. arithmetic on eth_value, datetime operations on timestamp_utc).

    Parameters
    ----------
    df : pd.DataFrame
        Output of _validate_schema() — all columns present, all values strings.

    Returns
    -------
    pd.DataFrame
        Same data with correct dtypes.
    """
    df = df.copy()  # avoid mutating the input; makes function behaviour predictable

    # pd.to_datetime parses ISO 8601 strings into timezone-aware Timestamp objects.
    # utc=True ensures the result is always UTC, not timezone-naive — important when
    # later joining with Reddit/news data which may come from different time zones.
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # Floating-point columns — errors="coerce" replaces unparseable values with NaN
    # rather than raising an exception, so one bad row does not abort the whole load.
    float_cols = ("eth_value", "usd_value", "eth_usd_price", "gas_price_gwei", "tx_fee_eth")
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Integer columns — Int64 (capital I) is pandas' nullable integer type.
    # Standard int64 cannot represent NaN; Int64 can. This matters if Dune ever
    # returns a row with a missing block_number or gas_used.
    int_cols = ("block_number", "gas_used")
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Boolean column — Dune CSV outputs Python booleans as the strings "True"/"False".
    # The dict map handles both the string and native bool cases defensively.
    df["is_contract_call"] = df["is_contract_call"].map(
        {"True": True, "False": False, "true": True, "false": False,
         True: True, False: False}
    )

    return df