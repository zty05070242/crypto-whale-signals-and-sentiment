"""
Generic Dune Analytics API client.

Dune's API works in 3 steps:
    1. POST to execute a saved query -> returns an execution_id
    2. GET status until state == COMPLETED (poll in a loop)
    3. GET results/csv to download the data

This module handles all three steps. It has no whale-specific logic - 
it is reusable for any Dune query in the project.
"""

import time
from io import StringIO
from typing import Optional

import pandas as pd
import requests

import config
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _auth_header() -> dict[str, str]:
    """Return the HTTP header Dune requires on every authenticated request."""
    raise RuntimeError(
        "Dune API credits are exhausted. Do not call the Dune API. "
        "All data is available locally — see docs/project_arc.md."
    )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# -------------------------------------Execute Query--------------------------------------
def execute_query(query_id: int, parameters: Optional[dict] = None) -> str:
    """
    Tell Dune to start executing a saved query.

    PARAMETERS
        query_id:   The integer ID in the Dune query URL.
        parameters: Maps query parameter name to value.

    RETURNS
        string:     The execution_id string, used to poll status and fetch results.
    """
    url = f"{config.DUNE_BASE_URL}/query/{query_id}/execute"

    body: dict = {}
    if parameters: 
        body["query_parameters"] = parameters

    response = requests.post(url, json=body, headers=_auth_header(), timeout=30)
    response.raise_for_status()
    return response.json()["execution_id"]

# -------------------------------------Dune State Check--------------------------------------

def poll_until_complete(execution_id:str, poll_interval_seconds:int=5, timeout_seconds:int=600) -> None:
    """
    Repeatedly check Dune's status endpoint until the query finishes.

    PARAMETERS:
        execution_id:           Returned by execute_query().
        poll_interval_seconds:  Seconds to wait between status checks.
        timeout_seconds:        Max total wait time before giving up.

    Raises RuntimeError     If Dune reports the query failed.
    Raises TimeoutError     If the query does not complete within timeout_seconds.
    """
    url = f"{config.DUNE_BASE_URL}/execution/{execution_id}/status"
    elapsed = 0

    while elapsed < timeout_seconds:
        response = requests.get(url, headers=_auth_header(), timeout=30)
        response.raise_for_status()

        state:str = response.json()["state"]

        if state == "QUERY_STATE_COMPLETED":
            return
        if state == "QUERY_STATE_FAILED":
            raise RuntimeError(f"Dune execution '{execution_id}' failed. Check Dune UI for error details.")
        
        # Any other state (PENDING, EXECUTING) - wait and try again
        print(f"Dune status: {state} ({elapsed}s elapsed)...")
        time.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds

    raise TimeoutError(f"Dune execution '{execution_id}' did not complete within {timeout_seconds}s. The query may be too large for the free tier - consider narrowing the date range.")

# -------------------------------------Download the Data--------------------------------------

def fetch_results_csv(execution_id:str) -> pd.DataFrame:
    """
    Download the results of a completed Dune execution as a DataFrame.

    Uses the /results/csv endpoint, which stream a plain CSV text response.
    This is simpler and more memory-efficient than the JSON results endpoint.

    Returns pd.DataFrame: All columns as strings - type parsing is done by the caller.
    """
    url = f"{config.DUNE_BASE_URL}/execution/{execution_id}/results/csv"
    response = requests.get(url, headers=_auth_header(), timeout=120)
    response.raise_for_status()

    # StringIO wraps the raw CSV text string so pd.read_csv can treat it as a file-like object
    return pd.read_csv(StringIO(response.text))


# =========================================== Ultimate Function to Use ===========================================

def run_query(query_id:int, parameters:Optional[dict]=None) -> pd.DataFrame:
    """
    Very convenient function: execute -> poll -> fetch in one call.

    PARAMETERS:
        query_id:       Dune query ID
        parameters:     Query parameters to pass.

    RETURNS:
        pd.DataFrame:   Raw results with all columns as strings.
    """
    print(f"Submitting Dune query {query_id} with parameters: {parameters}")
    execution_id = execute_query(query_id, parameters)

    print(f"Execution started (id={execution_id}). Polling for completion...")
    poll_until_complete(execution_id)

    print("Query complete. Downloading results...")
    df = fetch_results_csv(execution_id)
    print(f"Downloaded {len(df):,} rows, {len(df.columns)} columns.")

    return df
