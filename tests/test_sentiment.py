"""
Tests for the sentiment pipeline: VADER scorer and hourly aggregator.

GDELT fetcher is not tested here — it makes real HTTP requests and is
tested manually. These tests use synthetic DataFrames only.
"""

import pandas as pd
import pytest

from src.sentiment.vader_scorer import score_text, score_dataframe
from src.sentiment.aggregator import (
    aggregate_hourly,
    POSITIVE_THRESHOLD,
    NEGATIVE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_articles_df(
    titles: list[str],
    timestamps: list[str],
) -> pd.DataFrame:
    """Build a minimal article DataFrame for testing."""
    return pd.DataFrame({
        "title": titles,
        "seendate": timestamps,
        "url": [f"https://example.com/{i}" for i in range(len(titles))],
        "domain": ["example.com"] * len(titles),
    })


# ---------------------------------------------------------------------------
# VADER scorer
# ---------------------------------------------------------------------------

class TestScoreText:
    def test_positive_text(self):
        """A clearly positive headline should have a positive compound score."""
        result = score_text("Bitcoin surges to all-time high, investors celebrate")
        assert result["compound"] > 0.0

    def test_negative_text(self):
        """A clearly negative headline should have a negative compound score."""
        result = score_text("Crypto market crashes, billions wiped out in panic sell-off")
        assert result["compound"] < 0.0

    def test_neutral_text(self):
        """A factual headline should have a near-zero compound score."""
        result = score_text("Ethereum block number reaches 18 million")
        assert abs(result["compound"]) < 0.5

    def test_returns_all_keys(self):
        """VADER should return neg, neu, pos, compound."""
        result = score_text("test headline")
        assert set(result.keys()) == {"neg", "neu", "pos", "compound"}

    def test_compound_range(self):
        """Compound score must be between -1.0 and 1.0."""
        result = score_text("absolutely terrible catastrophic disaster")
        assert -1.0 <= result["compound"] <= 1.0

    def test_empty_string(self):
        """Empty string should return zero scores."""
        result = score_text("")
        assert result["compound"] == 0.0


class TestScoreDataframe:
    def test_adds_four_columns(self):
        """Should add vader_neg, vader_neu, vader_pos, vader_compound."""
        df = make_articles_df(
            ["Bitcoin surges", "Crypto crashes"],
            ["20230101T120000Z", "20230101T130000Z"],
        )
        result = score_dataframe(df)
        expected_cols = {"vader_neg", "vader_neu", "vader_pos", "vader_compound"}
        assert expected_cols.issubset(set(result.columns))

    def test_row_count_preserved(self):
        """Output should have the same number of rows as input."""
        df = make_articles_df(
            ["headline one", "headline two", "headline three"],
            ["20230101T120000Z", "20230101T130000Z", "20230101T140000Z"],
        )
        result = score_dataframe(df)
        assert len(result) == 3

    def test_does_not_mutate_input(self):
        """Should return a copy, not modify the input."""
        df = make_articles_df(["test"], ["20230101T120000Z"])
        original_cols = set(df.columns)
        score_dataframe(df)
        assert set(df.columns) == original_cols

    def test_handles_nan_titles(self):
        """NaN titles should not crash — scored as empty string."""
        df = pd.DataFrame({
            "title": ["Bitcoin surges", None, "Crypto news"],
            "seendate": ["20230101T120000Z"] * 3,
        })
        result = score_dataframe(df)
        assert len(result) == 3
        # NaN title -> compound 0.0 (empty string)
        assert result["vader_compound"].iloc[1] == 0.0

    def test_missing_column_raises(self):
        """Should raise ValueError if text column is missing."""
        df = pd.DataFrame({"not_title": ["test"]})
        with pytest.raises(ValueError, match="not found"):
            score_dataframe(df)

    def test_positive_headline_has_positive_compound(self):
        """Sanity check: positive headline -> positive vader_compound."""
        df = make_articles_df(
            ["Incredible growth, massive gains, investors thrilled"],
            ["20230101T120000Z"],
        )
        result = score_dataframe(df)
        assert result["vader_compound"].iloc[0] > 0.0


# ---------------------------------------------------------------------------
# Hourly aggregator
# ---------------------------------------------------------------------------

class TestAggregateHourly:
    def test_groups_by_hour(self):
        """Articles in the same hour should be grouped together."""
        df = make_articles_df(
            ["Good news", "Bad news", "More good news"],
            ["20230101T143000Z", "20230101T145500Z", "20230101T150000Z"],
        )
        scored = score_dataframe(df)
        hourly = aggregate_hourly(scored)
        # 14:30 and 14:55 -> 14:00 hour; 15:00 -> 15:00 hour
        assert len(hourly) == 2

    def test_article_count(self):
        """article_count should reflect how many articles fell in each hour."""
        df = make_articles_df(
            ["a", "b", "c"],
            ["20230101T140000Z", "20230101T143000Z", "20230101T150000Z"],
        )
        scored = score_dataframe(df)
        hourly = aggregate_hourly(scored)
        # Two articles in 14:00 hour, one in 15:00
        hour_14 = hourly[hourly["hour_utc"].dt.hour == 14]
        hour_15 = hourly[hourly["hour_utc"].dt.hour == 15]
        assert hour_14["article_count"].iloc[0] == 2
        assert hour_15["article_count"].iloc[0] == 1

    def test_sentiment_mean_range(self):
        """Mean sentiment should be between -1 and 1."""
        df = make_articles_df(
            ["Great", "Terrible", "Neutral thing"],
            ["20230101T140000Z", "20230101T140100Z", "20230101T140200Z"],
        )
        scored = score_dataframe(df)
        hourly = aggregate_hourly(scored)
        assert (hourly["sentiment_mean"] >= -1.0).all()
        assert (hourly["sentiment_mean"] <= 1.0).all()

    def test_positive_ratio(self):
        """Positive ratio should be fraction of articles above threshold."""
        df = make_articles_df(
            # Two clearly positive, one clearly negative
            ["Amazing incredible wonderful", "Fantastic great superb", "Terrible awful disaster"],
            ["20230101T140000Z", "20230101T140100Z", "20230101T140200Z"],
        )
        scored = score_dataframe(df)
        hourly = aggregate_hourly(scored)
        # 2 out of 3 are positive -> ratio ~0.667
        assert abs(hourly["positive_ratio"].iloc[0] - 2 / 3) < 0.01

    def test_chronological_order(self):
        """Output should be sorted by hour_utc."""
        df = make_articles_df(
            ["a", "b", "c"],
            ["20230103T140000Z", "20230101T100000Z", "20230102T080000Z"],
        )
        scored = score_dataframe(df)
        hourly = aggregate_hourly(scored)
        hours = hourly["hour_utc"].tolist()
        assert hours == sorted(hours)

    def test_missing_timestamp_raises(self):
        """Should raise ValueError if timestamp column missing."""
        df = pd.DataFrame({"title": ["test"], "vader_compound": [0.5]})
        with pytest.raises(ValueError, match="not found"):
            aggregate_hourly(df)

    def test_missing_score_column_raises(self):
        """Should raise ValueError if score column missing."""
        df = pd.DataFrame({"seendate": ["20230101T140000Z"], "title": ["test"]})
        with pytest.raises(ValueError, match="not found"):
            aggregate_hourly(df)

    def test_empty_dataframe(self):
        """Should handle empty DataFrame without errors."""
        df = pd.DataFrame({
            "title": pd.Series(dtype=str),
            "seendate": pd.Series(dtype=str),
            "vader_compound": pd.Series(dtype=float),
        })
        hourly = aggregate_hourly(df)
        assert len(hourly) == 0

    def test_single_article_std_is_nan(self):
        """Standard deviation of a single article should be NaN."""
        df = make_articles_df(["test headline"], ["20230101T140000Z"])
        scored = score_dataframe(df)
        hourly = aggregate_hourly(scored)
        # std of a single value is NaN in pandas (ddof=1 by default)
        assert pd.isna(hourly["sentiment_std"].iloc[0])