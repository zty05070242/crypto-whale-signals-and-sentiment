"""
Phase 3 sentiment pipeline runner.

Loads the Kaggle Bitcoin news dataset, scores each headline with VADER,
aggregates into hourly bins, and saves the result to data/processed/.

The Kaggle dataset already includes a pre-computed sentiment score
(DistilRoBERTa fine-tuned on financial news). We keep both:
  - vader_compound:     our own VADER score (transparent, reproducible)
  - roberta_sentiment:  the Kaggle-provided score (transformer-based)
This lets Phase 4 compare which sentiment signal is more predictive.

Usage:
    python scripts/run_sentiment_pipeline.py
"""

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so we can import our modules
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.sentiment.vader_scorer import score_dataframe  # noqa: E402
from src.sentiment.aggregator import aggregate_hourly, save_hourly_sentiment  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_PATH = config.RAW_DATA_DIR / "bitcoin_sentiments_21_24.csv"
SCORED_OUTPUT = config.PROCESSED_DATA_DIR / "bitcoin_news_scored.csv"

# ---------------------------------------------------------------------------
# 1. Load raw Kaggle data
# ---------------------------------------------------------------------------

print(f"Loading {INPUT_PATH} ...")
raw = pd.read_csv(INPUT_PATH)
print(f"  {len(raw):,} articles loaded")
print(f"  Columns: {list(raw.columns)}")
print(f"  Date range: {raw['Date'].min()} to {raw['Date'].max()}")

# ---------------------------------------------------------------------------
# 2. Filter to the whale transaction overlap window (Jan 2023 - Sep 2024)
# ---------------------------------------------------------------------------

# pd.to_datetime parses the Date column into proper datetime objects
raw["Date"] = pd.to_datetime(raw["Date"])

# Filter to the period covered by our whale transaction data
overlap = raw[
    (raw["Date"] >= "2023-01-01") & (raw["Date"] <= "2024-09-30")
].copy()

print(f"\n  Filtered to whale overlap window (2023-01 to 2024-09): "
      f"{len(overlap):,} articles")

# ---------------------------------------------------------------------------
# 3. Rename columns for clarity
# ---------------------------------------------------------------------------

# Rename 'Accurate Sentiments' to 'roberta_sentiment' — this is the
# pre-computed score from a DistilRoBERTa model fine-tuned on financial news.
overlap = overlap.rename(columns={
    "Short Description": "title",
    "Accurate Sentiments": "roberta_sentiment",
})

# ---------------------------------------------------------------------------
# 4. Score with VADER
# ---------------------------------------------------------------------------

print("\nScoring with VADER ...")
scored = score_dataframe(overlap, text_column="title")
print(f"  Done. VADER compound range: "
      f"[{scored['vader_compound'].min():.3f}, "
      f"{scored['vader_compound'].max():.3f}]")

# Quick comparison between VADER and the RoBERTa score
# Correlation tells us how much the two scorers agree
corr = scored["vader_compound"].corr(scored["roberta_sentiment"])
print(f"  Correlation between VADER and RoBERTa: {corr:.3f}")

# ---------------------------------------------------------------------------
# 5. Save scored articles
# ---------------------------------------------------------------------------

scored.to_csv(SCORED_OUTPUT, index=False)
print(f"\nSaved scored articles to {SCORED_OUTPUT}")

# ---------------------------------------------------------------------------
# 6. Aggregate to hourly sentiment
# ---------------------------------------------------------------------------

print("\nAggregating to hourly bins ...")
hourly = aggregate_hourly(scored, timestamp_column="Date", score_column="vader_compound")
print(f"  {len(hourly):,} hourly bins")
print(f"  Hours with articles: {(hourly['article_count'] > 0).sum():,}")
print(f"  Mean articles per hour: {hourly['article_count'].mean():.1f}")

# ---------------------------------------------------------------------------
# 7. Save hourly sentiment
# ---------------------------------------------------------------------------

save_hourly_sentiment(hourly)

print("\nPhase 3 sentiment pipeline complete.")
