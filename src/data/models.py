"""
Pydantic data models for the data collection layer.

Defines canonical representations for market prices, asset metadata,
news articles, fetch operation results, and bundled market data.
All external data is normalized into these models before downstream use.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "AssetPrice",
    "AssetInfo",
    "NewsArticle",
    "FetchResult",
    "MarketDataBundle",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Price Data
# ═══════════════════════════════════════════════════════════════════════════════


class AssetPrice(BaseModel):
    """Single OHLCV bar for an asset.

    Represents one day (or period) of price data with optional adjusted close.
    """

    date: datetime = Field(..., description="Bar timestamp (start of period)")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="Highest price during the period")
    low: float = Field(..., description="Lowest price during the period")
    close: float = Field(..., description="Closing price")
    volume: int = Field(..., description="Volume traded during the period")
    adjusted_close: float | None = Field(default=None, description="Split/dividend-adjusted close")

    model_config = {"frozen": True}


# ═══════════════════════════════════════════════════════════════════════════════
# Asset Metadata
# ═══════════════════════════════════════════════════════════════════════════════


class AssetInfo(BaseModel):
    """Fundamental / descriptive information about an asset.

    Contains static or slow-changing data like sector, market cap, and
    valuation ratios. Fields are optional because not all data sources
    provide every field (e.g., crypto has no PE ratio).
    """

    ticker: str = Field(..., description="Canonical ticker symbol")
    name: str = Field(..., description="Human-readable asset name")
    sector: str | None = Field(default=None, description="GICS sector")
    industry: str | None = Field(default=None, description="GICS industry")
    market_cap: float | None = Field(default=None, description="Market capitalisation (USD)")
    pe_ratio: float | None = Field(default=None, description="Trailing P/E ratio")
    eps: float | None = Field(default=None, description="Earnings per share (TTM)")
    dividend_yield: float | None = Field(
        default=None, description="Trailing annual dividend yield (decimal)"
    )
    beta: float | None = Field(default=None, description="Beta vs. market benchmark")
    fifty_two_week_high: float | None = Field(default=None, description="52-week high price")
    fifty_two_week_low: float | None = Field(default=None, description="52-week low price")


# ═══════════════════════════════════════════════════════════════════════════════
# News
# ═══════════════════════════════════════════════════════════════════════════════


class NewsArticle(BaseModel):
    """A single news article or headline.

    Sentiment score is populated later by the sentiment analysis agent.
    """

    title: str = Field(..., description="Article headline")
    source: str = Field(..., description="Publisher / source name")
    published_at: datetime = Field(..., description="Publication timestamp")
    url: str = Field(..., description="Canonical URL to the article")
    description: str | None = Field(default=None, description="Article snippet / description")
    sentiment_score: float | None = Field(
        default=None,
        description="Sentiment score in [-1.0, 1.0], populated by sentiment agent",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Fetch Operation Result
# ═══════════════════════════════════════════════════════════════════════════════


class FetchResult(BaseModel):
    """Outcome of a single data-fetch operation.

    Wraps the fetched payload together with metadata about the fetch:
    which source was used, whether it was cached, timing, and any errors.
    """

    status: Literal["success", "cached", "failed"] = Field(..., description="Outcome of the fetch")
    data: Any = Field(default=None, description="Fetched payload (type varies)")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this result was created",
    )
    source_used: str = Field(..., description="Data source that produced this result")
    error_message: str | None = Field(
        default=None, description="Error details when status == 'failed'"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Bundled Market Data
# ═══════════════════════════════════════════════════════════════════════════════


class MarketDataBundle(BaseModel):
    """Complete data package for a single asset.

    Aggregates price history, fundamental info, and news together with
    per-category fetch results so downstream consumers can inspect
    provenance and error state.
    """

    ticker: str = Field(..., description="Canonical ticker symbol")
    prices: list[AssetPrice] = Field(default_factory=list, description="Historical price bars")
    info: AssetInfo | None = Field(
        default=None, description="Fundamental / descriptive information"
    )
    news: list[NewsArticle] = Field(default_factory=list, description="Recent news articles")
    fetch_results: dict[str, FetchResult] = Field(
        default_factory=dict,
        description="Per-category fetch outcomes (e.g. 'prices', 'info', 'news')",
    )
