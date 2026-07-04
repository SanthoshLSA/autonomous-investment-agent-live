"""
Sentiment analysis module for the Autonomous Investment Research Agent.

Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) for rule-based
sentiment scoring of news headlines and article descriptions.  Provides both
single-headline and batch-level analysis with exponential time-decay weighting.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.data.models import NewsArticle
from src.logger import get_logger

logger = get_logger(__name__)

# Module-level singleton – VADER is stateless & thread-safe.
_analyzer: SentimentIntensityAnalyzer | None = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    """Lazy-init the VADER analyser (avoids startup cost if never used).

    Returns:
        A shared ``SentimentIntensityAnalyzer`` instance.
    """
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
        logger.debug("vader_analyzer_initialized")
    return _analyzer


# ═══════════════════════════════════════════════════════════════════════════════
# Single-Headline Analysis
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_headline(headline: str) -> dict:
    """Score a single headline using VADER.

    Args:
        headline: The text to analyse (headline or short snippet).

    Returns:
        Dictionary with keys:
            - ``compound``:  Overall sentiment score in [-1, 1].
            - ``positive``: Proportion of positive tokens.
            - ``negative``: Proportion of negative tokens.
            - ``neutral``:  Proportion of neutral tokens.
    """
    if not headline or not headline.strip():
        logger.warning("analyze_headline_empty_text")
        return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}

    analyzer = _get_analyzer()
    scores = analyzer.polarity_scores(headline)

    return {
        "compound": scores["compound"],
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Classification
# ═══════════════════════════════════════════════════════════════════════════════


def classify_sentiment(score: float) -> str:
    """Map a compound sentiment score to a human-readable label.

    Thresholds:
        - ``very_bullish``:  score > 0.5
        - ``bullish``:       0.15 < score ≤ 0.5
        - ``neutral``:      -0.15 ≤ score ≤ 0.15
        - ``bearish``:      -0.5 ≤ score < -0.15
        - ``very_bearish``: score < -0.5

    Args:
        score: Compound sentiment score (float, typically -1 to 1).

    Returns:
        One of ``'very_bullish'``, ``'bullish'``, ``'neutral'``,
        ``'bearish'``, ``'very_bearish'``.
    """
    if score > 0.5:
        return "very_bullish"
    elif score > 0.15:
        return "bullish"
    elif score < -0.5:
        return "very_bearish"
    elif score < -0.15:
        return "bearish"
    else:
        return "neutral"


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Analysis with Exponential Decay
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_news_batch(
    articles: list[NewsArticle],
    decay_factor: float = 0.3,
) -> dict:
    """Analyse a batch of news articles with time-decay weighting.

    Recent articles receive higher weights via an exponential decay function:
    ``weight = exp(-decay_factor × days_old)`` where ``days_old`` is the
    number of days since publication.

    Args:
        articles: List of ``NewsArticle`` instances.
        decay_factor: Controls how fast old articles lose influence.
            Higher → faster decay (default 0.3).

    Returns:
        Dictionary with keys:
            - ``weighted_sentiment``: Exponentially-weighted average compound.
            - ``raw_average``: Simple average of compound scores.
            - ``article_count``: Number of articles analysed.
            - ``most_positive``: Dict with ``title`` and ``score`` of the
              most positive article.
            - ``most_negative``: Dict with ``title`` and ``score`` of the
              most negative article.
            - ``sentiment_trend``: ``'improving'`` / ``'declining'`` /
              ``'stable'`` based on first-half vs. second-half comparison.
            - ``classification``: Overall sentiment label (via
              ``classify_sentiment``).
    """
    neutral_default: dict = {
        "weighted_sentiment": 0.0,
        "raw_average": 0.0,
        "article_count": 0,
        "most_positive": {"title": "", "score": 0.0},
        "most_negative": {"title": "", "score": 0.0},
        "sentiment_trend": "stable",
        "classification": "neutral",
    }

    if not articles:
        logger.warning("analyze_news_batch_empty")
        return neutral_default

    # ── Score every article ───────────────────────────────────────────────
    now = datetime.now(UTC)
    scored: list[dict] = []

    for article in articles:
        # Combine title + description for richer signal
        text = article.title or ""
        if article.description:
            text = f"{text}. {article.description}"

        sentiment = analyze_headline(text)
        compound = sentiment["compound"]

        # Compute age in days
        if article.published_at is not None:
            pub = article.published_at
            # Ensure timezone-aware comparison
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=UTC)
            days_old = max((now - pub).total_seconds() / 86400, 0.0)
        else:
            # Unknown date ⇒ assume 3 days old (moderate penalty)
            days_old = 3.0

        weight = math.exp(-decay_factor * days_old)

        scored.append(
            {
                "title": article.title or "(no title)",
                "compound": compound,
                "weight": weight,
                "days_old": days_old,
            }
        )

    # ── Aggregate ─────────────────────────────────────────────────────────
    compounds = [s["compound"] for s in scored]
    weights = [s["weight"] for s in scored]

    total_weight = sum(weights)
    if total_weight == 0:
        weighted_avg = 0.0
    else:
        weighted_avg = sum(c * w for c, w in zip(compounds, weights, strict=False)) / total_weight

    raw_avg = sum(compounds) / len(compounds)

    # Most positive / negative
    best = max(scored, key=lambda s: s["compound"])
    worst = min(scored, key=lambda s: s["compound"])

    # ── Sentiment trend: compare first half vs. second half ───────────────
    mid = len(compounds) // 2
    if mid >= 1 and len(compounds) >= 2:
        first_half_avg = sum(compounds[:mid]) / mid
        second_half_avg = sum(compounds[mid:]) / (len(compounds) - mid)
        diff = second_half_avg - first_half_avg
        if diff > 0.05:
            trend = "improving"
        elif diff < -0.05:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "stable"

    result = {
        "weighted_sentiment": round(weighted_avg, 4),
        "raw_average": round(raw_avg, 4),
        "article_count": len(scored),
        "most_positive": {"title": best["title"], "score": best["compound"]},
        "most_negative": {"title": worst["title"], "score": worst["compound"]},
        "sentiment_trend": trend,
        "classification": classify_sentiment(weighted_avg),
    }

    logger.info(
        "news_batch_analyzed",
        article_count=result["article_count"],
        weighted_sentiment=result["weighted_sentiment"],
        classification=result["classification"],
    )
    return result


__all__ = [
    "analyze_headline",
    "analyze_news_batch",
    "classify_sentiment",
]
