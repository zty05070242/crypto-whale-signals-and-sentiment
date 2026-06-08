"""
Hourly sentiment aggregator.

Takes VADER-scored articles and aggregates them into hourly bins aligned
with UTC timestamps. This produces the sentiment features that Phase 4
joins with whale transaction data.

Aggregation per hour:
  - sentiment_mean:     mean compound score across all articles in the hour
  - sentiment_std:      standard deviation (captures disagreement/uncertainty)
  - article_count:      number of articles (proxy for media attention)
  - positive_ratio:     fraction of articles with compound > 0.05
  - negative_ratio:     fraction of articles with compound < -0.05

Hours with zero articles get NaN, which can be forward-filled with decay
in Phase 4 (older sentiment = weaker proxy). We do NOT fill here — that
decision belongs to the feature engineering step in Phase 4.
"""

from typing import Optional

import pandas as pd

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# VADER compound score thresholds for classifying articles as positive/negative.
# These are standard VADER thresholds from the original paper:
#   compound >=  0.05 -> positive
#   compound <= -0.05 -> negative
#   otherwise         -> neutral
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

DEFAULT_OUTPUT_PATH = config.ROOT_DIR / "data" / "processed" / "hourly_sentiment.csv"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate_hourly(
    df: pd.DataFrame,
    timestamp_column: str = "seendate",
    score_column: str = "vader_compound",
) -> pd.DataFrame:
    """
    Aggregate VADER-scored articles into hourly sentiment features.

    Parameters
    ----------
    df : pd.DataFrame
        Output of vader_scorer.score_dataframe(). Must have a timestamp
        column and vader_compound column.
    timestamp_column : str
        Column containing article timestamps. Default 'seendate'.
    score_column : str
        Column containing VADER compound scores. Default 'vader_compound'.

    Returns
    -------
    pd.DataFrame
        One row per UTC hour with columns: hour_utc, sentiment_mean,
        sentiment_std, article_count, positive_ratio, negative_ratio.
    """
    if timestamp_column not in df.columns:
        raise ValueError(f"Column '{timestamp_column}' not found.")
    if score_column not in df.columns:
        raise ValueError(f"Column '{score_column}' not found.")

    df = df.copy()

    # Parse timestamps — GDELT format is like '20230115T210000Z'
    df["_ts"] = pd.to_datetime(df[timestamp_column], utc=True)

    # Floor to the hour — pd.Timestamp.floor('h') rounds down to the
    # nearest hour boundary. e.g. 14:37 -> 14:00
    df["hour_utc"] = df["_ts"].dt.floor("h")

    # --- Compute per-hour aggregates ---
    # groupby('hour_utc') splits the DataFrame into groups sharing the same hour.
    # .agg() applies multiple aggregation functions to each group.
    hourly = df.groupby("hour_utc")[score_column].agg(
        sentiment_mean="mean",
        sentiment_std="std",
        article_count="count",
    ).reset_index()

    # --- Compute positive/negative ratios ---
    # These require row-level boolean checks, then group-level averaging.
    df["_is_positive"] = df[score_column] > POSITIVE_THRESHOLD
    df["_is_negative"] = df[score_column] < NEGATIVE_THRESHOLD

    ratios = df.groupby("hour_utc").agg(
        positive_ratio=("_is_positive", "mean"),  # mean of booleans = proportion
        negative_ratio=("_is_negative", "mean"),
    ).reset_index()

    # pd.merge joins the two aggregations on the shared hour_utc column
    hourly = hourly.merge(ratios, on="hour_utc", how="left")

    # Sort chronologically
    hourly = hourly.sort_values("hour_utc").reset_index(drop=True)

    return hourly


def save_hourly_sentiment(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> None:
    """
    Save hourly sentiment DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Output of aggregate_hourly().
    output_path : str, optional
        Defaults to data/processed/hourly_sentiment.csv.
    """
    path = output_path or DEFAULT_OUTPUT_PATH
    path = type(path) is str and __import__("pathlib").Path(path) or path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} hourly records to {path}")
