"""
NewsAPI data source adapter.

Wraps the ``newsapi-python`` client to search for recent articles relevant
to a given asset ticker. Includes a ticker-to-search-term mapping so that
queries like ``"AAPL"`` are translated to more effective search strings
such as ``"Apple stock"``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.data.models import NewsArticle
from src.logger import get_logger

__all__ = ["NewsAPISource"]

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Ticker → human-readable search term mapping
# ═══════════════════════════════════════════════════════════════════════════════

_TICKER_SEARCH_MAP: dict[str, str] = {
    # US stocks
    "AAPL": "Apple stock",
    "MSFT": "Microsoft stock",
    "GOOGL": "Google Alphabet stock",
    "AMZN": "Amazon stock",
    "TSLA": "Tesla stock",
    # Indian stocks
    "RELIANCE.NS": "Reliance Industries stock",
    "TCS.NS": "TCS Tata Consultancy stock",
    "INFY.NS": "Infosys stock",
    "HDFCBANK.NS": "HDFC Bank stock",
    # Crypto
    "BTC-USD": "Bitcoin crypto",
    "ETH-USD": "Ethereum crypto",
    # Indices
    "^GSPC": "S&P 500 index",
    "^NSEI": "NIFTY 50 India index",
}


class NewsAPISource:
    """Adapter for the NewsAPI (https://newsapi.org).

    Requires a valid API key. If no key is provided the source degrades
    gracefully — all methods return empty results with a logged warning.

    Args:
        api_key: NewsAPI key. ``None`` or empty string disables the source.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or ""
        self._client = None

        if not self._api_key:
            logger.warning(
                "newsapi_no_key",
                msg="NewsAPI key not provided — source disabled",
            )
            return

        try:
            from newsapi import NewsApiClient  # type: ignore[import-untyped]

            self._client = NewsApiClient(api_key=self._api_key)
            logger.info("newsapi_source_initialised")
        except ImportError:
            logger.error(
                "newsapi_import_error",
                msg="newsapi-python package not installed",
            )
        except Exception as exc:
            logger.error(
                "newsapi_init_error",
                error=str(exc),
            )

    # ── Public API ────────────────────────────────────────────────────────

    def get_news(
        self,
        query: str,
        days_back: int = 7,
        max_articles: int = 10,
    ) -> list[NewsArticle]:
        """Search for recent news articles matching a query or ticker.

        The *query* is first looked up in the ticker-to-search-term map.
        If no mapping exists the raw query string is used directly.

        Args:
            query: Ticker symbol or free-text search string.
            days_back: How many days of history to search.
            max_articles: Maximum number of articles to return.

        Returns:
            List of ``NewsArticle`` models (may be empty on error or
            missing API key).
        """
        if self._client is None:
            logger.warning(
                "newsapi_skipped",
                query=query,
                reason="client not available",
            )
            return []

        search_term = _TICKER_SEARCH_MAP.get(query, query)
        from_date = (datetime.now(tz=UTC) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = datetime.now(tz=UTC).strftime("%Y-%m-%d")

        logger.info(
            "newsapi_fetching",
            query=query,
            search_term=search_term,
            from_date=from_date,
            to_date=to_date,
        )

        try:
            response = self._client.get_everything(
                q=search_term,
                from_param=from_date,
                to=to_date,
                language="en",
                sort_by="publishedAt",
                page_size=max_articles,
            )
        except Exception as exc:
            error_msg = str(exc).lower()
            if "429" in error_msg or "rate" in error_msg or "quota" in error_msg:
                logger.warning(
                    "newsapi_rate_limited",
                    query=query,
                    error=str(exc),
                )
            else:
                logger.error(
                    "newsapi_fetch_error",
                    query=query,
                    error=str(exc),
                )
            return []

        articles_raw = response.get("articles") or []
        articles: list[NewsArticle] = []

        for raw in articles_raw:
            try:
                published_str = raw.get("publishedAt", "")
                if published_str:
                    # NewsAPI returns ISO 8601 timestamps
                    published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                else:
                    published_at = datetime.now(tz=UTC)

                article = NewsArticle(
                    title=raw.get("title") or "Untitled",
                    source=(raw.get("source") or {}).get("name", "Unknown"),
                    published_at=published_at,
                    url=raw.get("url") or "",
                    description=raw.get("description"),
                    sentiment_score=None,
                )
                articles.append(article)
            except (ValueError, TypeError, KeyError) as exc:
                logger.warning(
                    "newsapi_article_skipped",
                    error=str(exc),
                    raw_title=raw.get("title", "?"),
                )

        logger.info(
            "newsapi_fetched",
            query=query,
            articles=len(articles),
        )
        return articles
