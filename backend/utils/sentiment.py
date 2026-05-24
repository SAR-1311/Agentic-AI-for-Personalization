"""Emotional intensity e(m) — Eq. 3 of the proposal.

`e(m) = |Sentiment Score(m)|` normalised to [0, 1].

Default: VADER (fast, no GPU, English).
Upgrade path: cardiffnlp/twitter-roberta-base-sentiment-latest via transformers.
"""
from __future__ import annotations

from functools import lru_cache
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@lru_cache
def _vader() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def emotional_intensity(text: str) -> float:
    """Return |compound sentiment| in [0, 1]."""
    if not text or not text.strip():
        return 0.0
    score = _vader().polarity_scores(text)["compound"]   # in [-1, 1]
    return abs(score)


def sentiment_label(text: str) -> str:
    """Return 'positive' | 'neutral' | 'negative'."""
    if not text or not text.strip():
        return "neutral"
    c = _vader().polarity_scores(text)["compound"]
    if c >= 0.05:
        return "positive"
    if c <= -0.05:
        return "negative"
    return "neutral"
