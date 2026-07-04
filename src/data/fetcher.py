"""
Main data aggregator / fetcher for the data collection layer.

Orchestrates multiple data sources (yfinance, NewsAPI, Finnhub) with an
intermediate SQLite cache. Implements a fallback chain: if the primary source
fails, the next source is tried automatically. Individual ticker failures
never crash the pipeline — partial results are returned with detailed
error metadata.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from src.config import APIKeys, AppConfig
from src.data.cache import DataCache
from src.data.models import (
    AssetInfo,
    AssetPrice,
    FetchResult,
    MarketDataBundle,
    NewsArticle,
)
from src.data.sources.finnhub_source import FinnhubSource
from src.data.sources.newsapi_source import NewsAPISource
from src.data.sources.yfinance_source import YFinanceSource
from src.logger import get_logger

__all__ = ["DataFetcher"]

logger = get_logger(__name__)


class DataFetcher:
    """Central data-collection orchestrator.

    Initialises all data sources and the SQLite cache, then exposes
    high-level methods to fetch prices, info, news, and bundled data
    for one or many tickers.

    Args:
        config: Application configuration.
        api_keys: API key bag (NewsAPI, Finnhub, etc.).
    """

    def __init__(self, config: AppConfig, api_keys: APIKeys) -> None:
        self._config = config
        self._api_keys = api_keys

        # ── Initialise cache ──────────────────────────────────────────────
        db_path = str(
            Path(config.cache.database_path)
            if Path(config.cache.database_path).is_absolute()
            else Path(__file__).parent.parent.parent / config.cache.database_path
        )
        self._cache_enabled = config.cache.enabled
        self._cache = DataCache(
            db_path=db_path,
            default_ttl_hours=config.cache.ttl_hours,
        )

        # ── Initialise sources ────────────────────────────────────────────
        self._yfinance = YFinanceSource()
        self._newsapi = NewsAPISource(api_key=api_keys.newsapi_key)
        self._finnhub = FinnhubSource(api_key=api_keys.finnhub_key)

        logger.info(
            "data_fetcher_initialised",
            cache_enabled=self._cache_enabled,
            has_newsapi=bool(api_keys.newsapi_key),
            has_finnhub=bool(api_keys.finnhub_key),
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Single-asset fetching
    # ═══════════════════════════════════════════════════════════════════════

    def fetch_asset_data(self, ticker: str) -> MarketDataBundle:
        """Fetch prices, info, and news for a single asset.

        Checks cache first; on a miss, tries yfinance → Finnhub fallback
        for prices/info, and NewsAPI → Finnhub for news.

        Args:
            ticker: Canonical ticker symbol.

        Returns:
            A ``MarketDataBundle`` with whatever data could be obtained.
            Individual failures are recorded in ``fetch_results``.
        """
        t0 = time.time()
        logger.info("fetch_asset_data_start", ticker=ticker)

        prices, prices_result = self._fetch_prices(ticker)
        info, info_result = self._fetch_info(ticker)
        news, news_result = self._fetch_news_for_ticker(ticker)

        elapsed = round(time.time() - t0, 2)
        logger.info(
            "fetch_asset_data_complete",
            ticker=ticker,
            prices_status=prices_result.status,
            info_status=info_result.status,
            news_status=news_result.status,
            elapsed_s=elapsed,
        )

        return MarketDataBundle(
            ticker=ticker,
            prices=prices,
            info=info,
            news=news,
            fetch_results={
                "prices": prices_result,
                "info": info_result,
                "news": news_result,
            },
        )

    # ═══════════════════════════════════════════════════════════════════════
    # News for multiple tickers
    # ═══════════════════════════════════════════════════════════════════════

    def fetch_news(self, tickers: list[str]) -> dict[str, list[NewsArticle]]:
        """Fetch news articles for a list of tickers.

        Args:
            tickers: List of ticker symbols.

        Returns:
            Mapping of ticker → list of ``NewsArticle`` objects.
        """
        results: dict[str, list[NewsArticle]] = {}
        for ticker in tickers:
            news, _ = self._fetch_news_for_ticker(ticker)
            results[ticker] = news
        return results

    # ═══════════════════════════════════════════════════════════════════════
    # Batch fetching
    # ═══════════════════════════════════════════════════════════════════════

    def fetch_all(self, tickers: list[str]) -> dict[str, MarketDataBundle]:
        """Fetch full data bundles for every ticker in the list.

        Never crashes on individual failures. Returns partial results with
        error info and logs a summary at the end.

        Args:
            tickers: List of ticker symbols.

        Returns:
            Mapping of ticker → ``MarketDataBundle``.
        """
        t0 = time.time()
        bundles: dict[str, MarketDataBundle] = {}
        succeeded = 0
        failed = 0
        cached = 0

        for ticker in tickers:
            try:
                bundle = self.fetch_asset_data(ticker)
                bundles[ticker] = bundle

                # Count statuses
                statuses = [r.status for r in bundle.fetch_results.values()]
                if all(s == "cached" for s in statuses):
                    cached += 1
                elif any(s == "failed" for s in statuses):
                    failed += 1
                else:
                    succeeded += 1

            except Exception as exc:
                logger.error(
                    "fetch_all_ticker_error",
                    ticker=ticker,
                    error=str(exc),
                )
                bundles[ticker] = MarketDataBundle(
                    ticker=ticker,
                    fetch_results={
                        "prices": FetchResult(
                            status="failed",
                            data=None,
                            source_used="none",
                            error_message=str(exc),
                        ),
                        "info": FetchResult(
                            status="failed",
                            data=None,
                            source_used="none",
                            error_message=str(exc),
                        ),
                        "news": FetchResult(
                            status="failed",
                            data=None,
                            source_used="none",
                            error_message=str(exc),
                        ),
                    },
                )
                failed += 1

        elapsed = round(time.time() - t0, 2)
        logger.info(
            "fetch_all_complete",
            total=len(tickers),
            succeeded=succeeded,
            cached=cached,
            failed=failed,
            elapsed_s=elapsed,
        )
        return bundles

    # ═══════════════════════════════════════════════════════════════════════
    # Private: price fetching with cache + fallback
    # ═══════════════════════════════════════════════════════════════════════

    def _fetch_prices(self, ticker: str) -> tuple[list[AssetPrice], FetchResult]:
        """Fetch prices: cache → yfinance → finnhub (quote only).

        Args:
            ticker: Canonical ticker symbol.

        Returns:
            Tuple of (price list, FetchResult metadata).
        """
        # ── Check cache ───────────────────────────────────────────────────
        if self._cache_enabled:
            cached_json = self._cache.get(ticker, "prices")
            if cached_json is not None:
                try:
                    raw_list = json.loads(cached_json)
                    prices = [AssetPrice(**item) for item in raw_list]
                    return prices, FetchResult(
                        status="cached",
                        data=None,
                        source_used="cache",
                    )
                except (json.JSONDecodeError, Exception) as exc:
                    logger.warning(
                        "cache_deserialize_error",
                        ticker=ticker,
                        data_type="prices",
                        error=str(exc),
                    )

        # ── Try yfinance (primary) ────────────────────────────────────────
        try:
            prices = self._yfinance.get_historical_prices(ticker)
            if prices:
                self._cache_prices(ticker, prices)
                return prices, FetchResult(
                    status="success",
                    data=None,
                    source_used="yfinance",
                )
        except Exception as exc:
            logger.warning(
                "yfinance_prices_failed",
                ticker=ticker,
                error=str(exc),
            )

        # ── Fallback: Finnhub quote (not historical, but better than nothing)
        try:
            quote = self._finnhub.get_quote(ticker)
            if quote.get("current_price", 0) > 0:
                now = datetime.now(tz=UTC)
                fallback_price = AssetPrice(
                    date=now,
                    open=quote["current_price"],
                    high=quote["current_price"],
                    low=quote["current_price"],
                    close=quote["current_price"],
                    volume=0,
                )
                return [fallback_price], FetchResult(
                    status="success",
                    data=None,
                    source_used="finnhub",
                )
        except Exception as exc:
            logger.warning(
                "finnhub_quote_failed",
                ticker=ticker,
                error=str(exc),
            )

        # ── All sources failed ────────────────────────────────────────────
        return [], FetchResult(
            status="failed",
            data=None,
            source_used="none",
            error_message=f"All price sources failed for {ticker}",
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Private: info fetching with cache
    # ═══════════════════════════════════════════════════════════════════════

    def _fetch_info(self, ticker: str) -> tuple[AssetInfo | None, FetchResult]:
        """Fetch asset info: cache → yfinance.

        Args:
            ticker: Canonical ticker symbol.

        Returns:
            Tuple of (AssetInfo or None, FetchResult metadata).
        """
        # ── Check cache ───────────────────────────────────────────────────
        if self._cache_enabled:
            cached_json = self._cache.get(ticker, "info")
            if cached_json is not None:
                try:
                    info = AssetInfo(**json.loads(cached_json))
                    return info, FetchResult(
                        status="cached",
                        data=None,
                        source_used="cache",
                    )
                except (json.JSONDecodeError, Exception) as exc:
                    logger.warning(
                        "cache_deserialize_error",
                        ticker=ticker,
                        data_type="info",
                        error=str(exc),
                    )

        # ── Try yfinance ──────────────────────────────────────────────────
        try:
            info = self._yfinance.get_asset_info(ticker)
            if self._cache_enabled:
                self._cache.set(ticker, "info", info.model_dump_json())
            return info, FetchResult(
                status="success",
                data=None,
                source_used="yfinance",
            )
        except Exception as exc:
            logger.warning(
                "yfinance_info_failed",
                ticker=ticker,
                error=str(exc),
            )

        return None, FetchResult(
            status="failed",
            data=None,
            source_used="none",
            error_message=f"All info sources failed for {ticker}",
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Private: news fetching with cache + fallback
    # ═══════════════════════════════════════════════════════════════════════

    def _fetch_news_for_ticker(self, ticker: str) -> tuple[list[NewsArticle], FetchResult]:
        """Fetch news: cache → NewsAPI → Finnhub company_news.

        Args:
            ticker: Canonical ticker symbol.

        Returns:
            Tuple of (news article list, FetchResult metadata).
        """
        days_back = self._config.sentiment.news_lookback_days
        max_articles = self._config.sentiment.max_articles_per_asset

        # ── Check cache ───────────────────────────────────────────────────
        if self._cache_enabled:
            cached_json = self._cache.get(ticker, "news")
            if cached_json is not None:
                try:
                    raw_list = json.loads(cached_json)
                    articles = [NewsArticle(**item) for item in raw_list]
                    return articles, FetchResult(
                        status="cached",
                        data=None,
                        source_used="cache",
                    )
                except (json.JSONDecodeError, Exception) as exc:
                    logger.warning(
                        "cache_deserialize_error",
                        ticker=ticker,
                        data_type="news",
                        error=str(exc),
                    )

        # ── Try NewsAPI (primary) ─────────────────────────────────────────
        try:
            articles = self._newsapi.get_news(
                query=ticker,
                days_back=days_back,
                max_articles=max_articles,
            )
            if articles:
                self._cache_news(ticker, articles)
                return articles, FetchResult(
                    status="success",
                    data=None,
                    source_used="newsapi",
                )
        except Exception as exc:
            logger.warning(
                "newsapi_news_failed",
                ticker=ticker,
                error=str(exc),
            )

        # ── Fallback: Finnhub company news ────────────────────────────────
        try:
            articles = self._finnhub.get_company_news(
                ticker=ticker,
                days_back=days_back,
            )
            if articles:
                self._cache_news(ticker, articles)
                return articles, FetchResult(
                    status="success",
                    data=None,
                    source_used="finnhub",
                )
        except Exception as exc:
            logger.warning(
                "finnhub_news_failed",
                ticker=ticker,
                error=str(exc),
            )

        # ── No news found (not necessarily a failure) ─────────────────────
        return [], FetchResult(
            status="success",
            data=None,
            source_used="none",
            error_message=f"No news found for {ticker} from any source",
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Private: cache helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _cache_prices(self, ticker: str, prices: list[AssetPrice]) -> None:
        """Serialise and cache a list of prices.

        Args:
            ticker: Canonical ticker symbol.
            prices: Price bars to cache.
        """
        if not self._cache_enabled:
            return
        try:
            data = json.dumps([p.model_dump(mode="json") for p in prices])
            self._cache.set(ticker, "prices", data)
        except Exception as exc:
            logger.warning(
                "cache_write_error",
                ticker=ticker,
                data_type="prices",
                error=str(exc),
            )

    def _cache_news(self, ticker: str, articles: list[NewsArticle]) -> None:
        """Serialise and cache a list of news articles.

        Args:
            ticker: Canonical ticker symbol.
            articles: Articles to cache.
        """
        if not self._cache_enabled:
            return
        try:
            data = json.dumps([a.model_dump(mode="json") for a in articles])
            self._cache.set(ticker, "news", data)
        except Exception as exc:
            logger.warning(
                "cache_write_error",
                ticker=ticker,
                data_type="news",
                error=str(exc),
            )
