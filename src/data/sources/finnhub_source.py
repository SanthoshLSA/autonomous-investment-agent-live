"""
Finnhub data source adapter.

Provides a backup / secondary data source for quotes and company news
via the Finnhub REST API. Includes rate limiting (60 calls/minute for
the free tier) and graceful degradation when no API key is configured.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.data.models import NewsArticle
from src.logger import get_logger

__all__ = ["FinnhubSource"]

logger = get_logger(__name__)

# Free-tier limit: 60 API calls per minute → 1 call per second
_MIN_CALL_INTERVAL = 1.0


class FinnhubSource:
    """Adapter for the Finnhub REST API (https://finnhub.io).

    Used as a fallback when the primary data source (yfinance) is
    unavailable. Requires a Finnhub API key; without one, all methods
    return empty / default results with a logged warning.

    Args:
        api_key: Finnhub API key. ``None`` or empty string disables the source.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or ""
        self._client = None
        self._last_call_time: float = 0.0

        if not self._api_key:
            logger.warning(
                "finnhub_no_key",
                msg="Finnhub API key not provided — source disabled",
            )
            return

        try:
            import finnhub  # type: ignore[import-untyped]

            self._client = finnhub.Client(api_key=self._api_key)
            logger.info("finnhub_source_initialised")
        except ImportError:
            logger.error(
                "finnhub_import_error",
                msg="finnhub-python package not installed",
            )
        except Exception as exc:
            logger.error(
                "finnhub_init_error",
                error=str(exc),
            )

    # ── Rate limiter ──────────────────────────────────────────────────────

    def _rate_limit(self) -> None:
        """Enforce a minimum inter-call delay to respect the free-tier limit."""
        elapsed = time.time() - self._last_call_time
        if elapsed < _MIN_CALL_INTERVAL:
            time.sleep(_MIN_CALL_INTERVAL - elapsed)
        self._last_call_time = time.time()

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_ticker(ticker: str) -> str:
        """Convert Yahoo Finance ticker format to Finnhub format.

        Finnhub uses ``"."`` as suffix delimiter (e.g. ``RELIANCE.NS``)
        and does not support crypto pairs or index symbols like ``^GSPC``.

        Args:
            ticker: Yahoo Finance style ticker.

        Returns:
            Finnhub-compatible symbol, or the original string if no
            conversion is needed.
        """
        # Indices are not supported on Finnhub free tier
        if ticker.startswith("^"):
            return ""
        # Crypto: BTC-USD → BINANCE:BTCUSDT (approximate)
        crypto_map = {
            "BTC-USD": "BINANCE:BTCUSDT",
            "ETH-USD": "BINANCE:ETHUSDT",
        }
        if ticker in crypto_map:
            return crypto_map[ticker]
        return ticker

    # ── Quote ─────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def get_quote(self, ticker: str) -> dict[str, float]:
        """Fetch a real-time quote for *ticker*.

        Args:
            ticker: Asset symbol (Yahoo Finance format accepted).

        Returns:
            Dict with keys ``current_price``, ``change``, ``percent_change``.
            Returns zeroed dict if the source is unavailable.

        Raises:
            Exception: On unrecoverable Finnhub errors after retries.
        """
        if self._client is None:
            logger.warning("finnhub_quote_skipped", ticker=ticker, reason="no_client")
            return {"current_price": 0.0, "change": 0.0, "percent_change": 0.0}

        symbol = self._normalise_ticker(ticker)
        if not symbol:
            logger.warning(
                "finnhub_unsupported_ticker",
                ticker=ticker,
                reason="no_finnhub_equivalent",
            )
            return {"current_price": 0.0, "change": 0.0, "percent_change": 0.0}

        self._rate_limit()
        logger.info("finnhub_fetching_quote", ticker=ticker, symbol=symbol)

        data = self._client.quote(symbol)

        result = {
            "current_price": float(data.get("c", 0.0)),
            "change": float(data.get("d", 0.0)),
            "percent_change": float(data.get("dp", 0.0)),
        }

        logger.info(
            "finnhub_quote_fetched",
            ticker=ticker,
            current_price=result["current_price"],
        )
        return result

    # ── Company news ──────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def get_company_news(
        self,
        ticker: str,
        days_back: int = 7,
    ) -> list[NewsArticle]:
        """Fetch recent company news articles from Finnhub.

        Args:
            ticker: Asset symbol (Yahoo Finance format accepted).
            days_back: Number of days of history to search.

        Returns:
            List of ``NewsArticle`` models (may be empty).

        Raises:
            Exception: On unrecoverable Finnhub errors after retries.
        """
        if self._client is None:
            logger.warning("finnhub_news_skipped", ticker=ticker, reason="no_client")
            return []

        symbol = self._normalise_ticker(ticker)
        if not symbol:
            logger.warning(
                "finnhub_news_unsupported",
                ticker=ticker,
                reason="no_finnhub_equivalent",
            )
            return []

        now = datetime.now(tz=UTC)
        from_date = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = now.strftime("%Y-%m-%d")

        self._rate_limit()
        logger.info(
            "finnhub_fetching_news",
            ticker=ticker,
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
        )

        raw_list = self._client.company_news(symbol, _from=from_date, to=to_date)

        articles: list[NewsArticle] = []
        for raw in raw_list or []:
            try:
                # Finnhub returns epoch seconds for datetime
                ts = raw.get("datetime", 0)
                published_at = datetime.fromtimestamp(ts, tz=UTC)

                article = NewsArticle(
                    title=raw.get("headline") or "Untitled",
                    source=raw.get("source") or "Finnhub",
                    published_at=published_at,
                    url=raw.get("url") or "",
                    description=raw.get("summary"),
                    sentiment_score=None,
                )
                articles.append(article)
            except (ValueError, TypeError, KeyError) as exc:
                logger.warning(
                    "finnhub_article_skipped",
                    error=str(exc),
                    raw_headline=raw.get("headline", "?"),
                )

        logger.info(
            "finnhub_news_fetched",
            ticker=ticker,
            articles=len(articles),
        )
        return articles
