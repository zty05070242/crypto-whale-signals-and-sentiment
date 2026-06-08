"""
Tests for src/data/dune_client.py.

All tests mock out HTTP calls — no network connection or real API key required.

SKIPPED: Dune API credits are exhausted and _auth_header() now raises
RuntimeError to prevent accidental API calls. These tests are retained
for reference but skipped in the test suite.

unittest.mock.patch replaces a name in the module under test with a fake object
for the duration of the test. The path "src.data.dune_client.requests.post"
means: in the dune_client module, replace the 'requests.post' attribute with
our MagicMock. This intercepts the call before it reaches the network.
"""

import pytest
from unittest.mock import MagicMock, patch
from io import StringIO

import pandas as pd

# Skip all tests in this module — Dune API is permanently disabled
pytestmark = pytest.mark.skip(reason="Dune API credits exhausted, _auth_header disabled")

# We must set config values before importing dune_client, because dune_client
# imports config at module load time and reads DUNE_API_KEY from it.
import config
config.DUNE_API_KEY = "test-api-key"
config.DUNE_BASE_URL = "https://api.dune.com/api/v1"

from src.data.dune_client import (
    execute_query,
    poll_until_complete,
    fetch_results_csv,
    run_query,
)


# ---------------------------------------------------------------------------
# execute_query
# ---------------------------------------------------------------------------

class TestExecuteQuery:
    def _make_mock_response(self, execution_id: str) -> MagicMock:
        """Build a fake requests.Response object with the expected JSON payload."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"execution_id": execution_id}
        # raise_for_status() on a MagicMock does nothing by default — correct
        # behaviour for a simulated 200 OK response
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    def test_returns_execution_id(self):
        with patch("src.data.dune_client.requests.post") as mock_post:
            mock_post.return_value = self._make_mock_response("exec-abc-123")
            result = execute_query(query_id=999)

        assert result == "exec-abc-123"

    def test_posts_to_correct_url(self):
        with patch("src.data.dune_client.requests.post") as mock_post:
            mock_post.return_value = self._make_mock_response("x")
            execute_query(query_id=42)

        # mock_post.call_args.args[0] is the first positional arg to requests.post
        called_url = mock_post.call_args.args[0]
        assert called_url == "https://api.dune.com/api/v1/query/42/execute"

    def test_sends_parameters_when_provided(self):
        with patch("src.data.dune_client.requests.post") as mock_post:
            mock_post.return_value = self._make_mock_response("x")
            execute_query(query_id=1, parameters={"start_date": "2023-01-01"})

        # Dune expects a flat dict, not a list of typed objects
        sent_body = mock_post.call_args.kwargs["json"]
        assert sent_body["query_parameters"] == {"start_date": "2023-01-01"}

    def test_raises_value_error_when_api_key_missing(self):
        original_key = config.DUNE_API_KEY
        config.DUNE_API_KEY = ""
        try:
            with pytest.raises(ValueError, match="DUNE_API_KEY"):
                execute_query(query_id=1)
        finally:
            config.DUNE_API_KEY = original_key  # restore so other tests are not affected


# ---------------------------------------------------------------------------
# poll_until_complete
# ---------------------------------------------------------------------------

class TestPollUntilComplete:
    def _make_status_response(self, state: str) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"state": state}
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    def test_returns_immediately_when_already_complete(self):
        with patch("src.data.dune_client.requests.get") as mock_get:
            mock_get.return_value = self._make_status_response("QUERY_STATE_COMPLETED")
            # Should not raise and should call requests.get exactly once
            poll_until_complete("exec-123", poll_interval_seconds=0)

        assert mock_get.call_count == 1

    def test_polls_multiple_times_before_completion(self):
        # Simulate: first call returns EXECUTING, second returns COMPLETED
        with patch("src.data.dune_client.requests.get") as mock_get, \
             patch("src.data.dune_client.time.sleep"):  # patch sleep so test runs instantly
            mock_get.side_effect = [
                self._make_status_response("QUERY_STATE_EXECUTING"),
                self._make_status_response("QUERY_STATE_COMPLETED"),
            ]
            poll_until_complete("exec-123", poll_interval_seconds=1)

        assert mock_get.call_count == 2

    def test_raises_runtime_error_on_failure(self):
        with patch("src.data.dune_client.requests.get") as mock_get:
            mock_get.return_value = self._make_status_response("QUERY_STATE_FAILED")
            with pytest.raises(RuntimeError, match="failed"):
                poll_until_complete("exec-123")

    def test_raises_timeout_error_when_stuck(self):
        with patch("src.data.dune_client.requests.get") as mock_get, \
             patch("src.data.dune_client.time.sleep"):
            # Always returns EXECUTING — should hit timeout
            mock_get.return_value = self._make_status_response("QUERY_STATE_EXECUTING")
            with pytest.raises(TimeoutError):
                poll_until_complete("exec-123", poll_interval_seconds=1, timeout_seconds=2)


# ---------------------------------------------------------------------------
# fetch_results_csv
# ---------------------------------------------------------------------------

class TestFetchResultsCsv:
    def test_returns_dataframe_from_csv_response(self):
        csv_text = "col_a,col_b\n1,hello\n2,world\n"

        mock_resp = MagicMock()
        mock_resp.text = csv_text
        mock_resp.raise_for_status.return_value = None

        with patch("src.data.dune_client.requests.get", return_value=mock_resp):
            df = fetch_results_csv("exec-123")

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["col_a", "col_b"]
        assert len(df) == 2

    def test_fetches_from_correct_url(self):
        mock_resp = MagicMock()
        mock_resp.text = "a\n1\n"
        mock_resp.raise_for_status.return_value = None

        with patch("src.data.dune_client.requests.get", return_value=mock_resp) as mock_get:
            fetch_results_csv("exec-456")

        called_url = mock_get.call_args.args[0]
        assert called_url == "https://api.dune.com/api/v1/execution/exec-456/results/csv"