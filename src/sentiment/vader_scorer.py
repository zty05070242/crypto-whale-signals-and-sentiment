"""
VADER sentiment scorer for crypto news headlines.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon-based
sentiment tool designed for social media and short text. It works by:
  1. Looking up each word in a pre-built dictionary of ~7,500 words,
     each rated on a scale from -4 (very negative) to +4 (very positive).
  2. Applying rules for punctuation, capitalisation, negation, and
     degree modifiers ("very", "slightly").
  3. Outputting four scores per text:
       - neg:      proportion of negative words (0.0 to 1.0)
       - neu:      proportion of neutral words (0.0 to 1.0)
       - pos:      proportion of positive words (0.0 to 1.0)
       - compound: normalised aggregate score (-1.0 to +1.0)
     The compound score is what we use as the single sentiment measure.

Why VADER over FinBERT or a transformer model?
  - Transparent: you can trace exactly why a headline got its score.
  - Fast: scores 100k headlines in seconds (no GPU needed).
  - Reproducible: deterministic, no model weights or random seeds.
  - Defensible in interviews: "I used VADER because..." is easier to
    articulate than "I used a fine-tuned BERT variant because...".
  - Known limitation: VADER was trained on social media, not financial
    text. "Crash" is negative in VADER but might be used neutrally in
    crypto contexts ("flash crash recovery"). This is documented.
"""

from typing import Optional

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------------------------------------------------------------------------
# Module-level analyser (instantiated once, reused across calls)
# ---------------------------------------------------------------------------

# SentimentIntensityAnalyzer loads the lexicon on init. Creating it once
# avoids re-reading the dictionary file on every call.
_analyser: Optional[SentimentIntensityAnalyzer] = None


def _get_analyser() -> SentimentIntensityAnalyzer:
    """Lazy-initialise and return the VADER analyser singleton."""
    global _analyser
    if _analyser is None:
        _analyser = SentimentIntensityAnalyzer()
    return _analyser


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_text(text: str) -> dict[str, float]:
    """
    Score a single text string with VADER.

    Parameters
    ----------
    text : str
        A headline, title, or short text to score.

    Returns
    -------
    dict[str, float]
        Keys: 'neg', 'neu', 'pos', 'compound'.
        compound ranges from -1.0 (most negative) to +1.0 (most positive).
    """
    analyser = _get_analyser()
    # polarity_scores returns a dict with neg, neu, pos, compound keys
    return analyser.polarity_scores(text)


def score_dataframe(
    df: pd.DataFrame,
    text_column: str = "title",
) -> pd.DataFrame:
    """
    Score all rows in a DataFrame and add sentiment columns.

    Adds four columns: vader_neg, vader_neu, vader_pos, vader_compound.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a text column (default: 'title').
    text_column : str
        Name of the column containing text to score.

    Returns
    -------
    pd.DataFrame
        Copy of df with four additional sentiment columns.
    """
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()

    analyser = _get_analyser()

    # Apply VADER to each row's text. fillna handles missing/NaN titles.
    # .apply() runs a function on each element of a Series — slower than
    # vectorised operations but VADER has no vectorised interface.
    texts = df[text_column].fillna("")
    scores = texts.apply(analyser.polarity_scores)

    # pd.json_normalize converts a Series of dicts into a DataFrame
    # with one column per dict key (neg, neu, pos, compound)
    score_df = pd.json_normalize(scores)

    # Prefix columns to make their origin clear
    score_df.columns = [f"vader_{col}" for col in score_df.columns]

    # Reset index to align with df (apply preserves the original index,
    # but concat needs them to match)
    score_df.index = df.index

    # pd.concat with axis=1 joins DataFrames column-wise (side by side)
    df = pd.concat([df, score_df], axis=1)

    return df
