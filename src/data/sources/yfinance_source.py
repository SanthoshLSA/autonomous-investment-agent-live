"""
Yahoo Finance data source adapter.

Wraps the ``yfinance`` library to fetch historical prices and asset metadata,
converting raw DataFrames / dicts into the project's canonical Pydantic models.
Includes rate limiting and retry logic for resilience.
"""

from __future__ import annotations

import time

import yfinance as yf
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.data.models import AssetInfo, AssetPrice
from src.logger import get_logger

__all__ = ["YFinanceSource"]

logger = get_logger(__name__)

# Minimum seconds between consecutive yfinance calls
_RATE_LIMIT_SECONDS = 0.5


class YFinanceSource:
    """Adapter for the Yahoo Finance API via the ``yfinance`` library.

    Provides methods to fetch historical OHLCV data, asset metadata, and
    the latest quote price. All methods include retry logic with
    exponential backoff and a small inter-call delay for rate limiting.
    """

    def __init__(self) -> None:
        self._last_call_time: float = 0.0
        logger.info("yfinance_source_initialised")

    # ── Rate limiter ──────────────────────────────────────────────────────

    def _rate_limit(self) -> None:
        """Enforce a minimum delay between consecutive API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < _RATE_LIMIT_SECONDS:
            sleep_for = _RATE_LIMIT_SECONDS - elapsed
            time.sleep(sleep_for)
        self._last_call_time = time.time()

    # ── Historical prices ─────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def get_historical_prices(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> list[AssetPrice]:
        """Fetch historical OHLCV bars for a given ticker.

        Args:
            ticker: Yahoo Finance ticker symbol (e.g. ``"AAPL"``).
            period: Lookback period string (``"1y"``, ``"6mo"``, etc.).
            interval: Bar interval (``"1d"``, ``"1h"``, etc.).

        Returns:
            List of ``AssetPrice`` objects ordered oldest-first.

        Raises:
            ValueError: If the ticker returns no data (delisted / invalid).
            Exception: On unrecoverable yfinance errors after retries.
        """
        self._rate_limit()
        logger.info(
            "yfinance_fetching_prices",
            ticker=ticker,
            period=period,
            interval=interval,
        )

        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df is None or df.empty:
            raise ValueError(
                f"No price data returned for {ticker} " f"(period={period}, interval={interval})"
            )

        prices: list[AssetPrice] = []
        for idx, row in df.iterrows():
            try:
                price = AssetPrice(
                    date=idx.to_pydatetime(),  # type: ignore[union-attr]
                    open=float(row.get("Open", 0.0)),
                    high=float(row.get("High", 0.0)),
                    low=float(row.get("Low", 0.0)),
                    close=float(row.get("Close", 0.0)),
                    volume=int(row.get("Volume", 0)),
                    adjusted_close=(float(row["Adj Close"]) if "Adj Close" in row.index else None),
                )
                prices.append(price)
            except (ValueError, TypeError, KeyError) as exc:
                logger.warning(
                    "yfinance_price_row_skipped",
                    ticker=ticker,
                    date=str(idx),
                    error=str(exc),
                )

        logger.info(
            "yfinance_prices_fetched",
            ticker=ticker,
            bars=len(prices),
        )
        return prices

    # ── Asset information ─────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def get_asset_info(self, ticker: str) -> AssetInfo:
        """Fetch fundamental / descriptive metadata for an asset.

        Args:
            ticker: Yahoo Finance ticker symbol.

        Returns:
            Populated ``AssetInfo`` model. Missing fields default to ``None``.

        Raises:
            ValueError: If no info dict is returned.
            Exception: On unrecoverable yfinance errors after retries.
        """
        self._rate_limit()
        logger.info("yfinance_fetching_info", ticker=ticker)

        stock = yf.Ticker(ticker)
        info: dict = stock.info or {}

        if not info:
            raise ValueError(f"No info returned for {ticker}")

        def _safe_float(key: str) -> float | None:
            """Extract a float from the info dict, returning None on failure."""
            val = info.get(key)
            if val is None:
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        asset_info = AssetInfo(
            ticker=ticker,
            name=info.get("shortName") or info.get("longName") or ticker,
            sector=info.get("sector"),
            industry=info.get("industry"),
            market_cap=_safe_float("marketCap"),
            pe_ratio=_safe_float("trailingPE"),
            eps=_safe_float("trailingEps"),
            dividend_yield=_safe_float("dividendYield"),
            beta=_safe_float("beta"),
            fifty_two_week_high=_safe_float("fiftyTwoWeekHigh"),
            fifty_two_week_low=_safe_float("fiftyTwoWeekLow"),
        )

        logger.info(
            "yfinance_info_fetched",
            ticker=ticker,
            name=asset_info.name,
        )
        return asset_info

    # ── Current price ─────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def get_current_price(self, ticker: str) -> float:
        """Fetch the latest quoted price for a ticker.

        Uses the ``fast_info`` attribute for speed, falling back to the
        full ``info`` dict if needed.

        Args:
            ticker: Yahoo Finance ticker symbol.

        Returns:
            Latest price as a float.

        Raises:
            ValueError: If no price can be determined.
            Exception: On unrecoverable yfinance errors after retries.
        """
        self._rate_limit()
        logger.info("yfinance_fetching_current_price", ticker=ticker)

        stock = yf.Ticker(ticker)

        # Try fast_info first (no extra HTTP call if info was cached)
        try:
            price = float(stock.fast_info["lastPrice"])
            if price > 0:
                logger.info(
                    "yfinance_current_price_fetched",
                    ticker=ticker,
                    price=price,
                )
                return price
        except (KeyError, TypeError, AttributeError):
            pass

        # Fallback: full info dict
        info = stock.info or {}
        for key in ("currentPrice", "regularMarketPrice", "previousClose"):
            val = info.get(key)
            if val is not None:
                try:
                    price = float(val)
                    if price > 0:
                        logger.info(
                            "yfinance_current_price_fetched",
                            ticker=ticker,
                            price=price,
                            via=key,
                        )
                        return price
                except (ValueError, TypeError):
                    continue

        raise ValueError(f"Could not determine current price for {ticker}")
