"""
Technical analysis module for the Autonomous Investment Research Agent.

Provides functions for computing standard technical indicators (SMA, EMA, RSI,
MACD, Bollinger Bands) and aggregating them into actionable trading signals.
All indicator calculations delegate to **pandas-ta-classic** for correctness.
"""

from __future__ import annotations

import pandas as pd
import pandas_ta_classic as ta

from src.config import AnalysisConfig
from src.logger import get_logger

logger = get_logger(__name__)

# Minimum data points required for each indicator family.
_MIN_POINTS_SMA200 = 200
_MIN_POINTS_DEFAULT = 30


# ═══════════════════════════════════════════════════════════════════════════════
# Individual Indicator Functions
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_sma(prices_series: pd.Series, period: int) -> pd.Series:
    """Compute the Simple Moving Average using *pandas-ta*.

    Args:
        prices_series: Series of closing prices (float), indexed by date.
        period: Look-back window length (e.g. 20, 50, 200).

    Returns:
        pd.Series of SMA values aligned with the input index.
        Leading entries (fewer than *period* data points) are ``NaN``.

    Raises:
        ValueError: If *period* is less than 1.
    """
    if period < 1:
        raise ValueError(f"SMA period must be >= 1, got {period}")

    if prices_series.empty or len(prices_series) < period:
        logger.warning(
            "insufficient_data_for_sma",
            data_points=len(prices_series),
            required=period,
        )
        return pd.Series(dtype=float, index=prices_series.index)

    result = ta.sma(prices_series, length=period)
    if result is None:
        logger.warning("pandas_ta_sma_returned_none", period=period)
        return pd.Series(dtype=float, index=prices_series.index)
    return result


def calculate_ema(prices_series: pd.Series, period: int) -> pd.Series:
    """Compute the Exponential Moving Average using *pandas-ta*.

    Args:
        prices_series: Series of closing prices (float).
        period: Look-back window length.

    Returns:
        pd.Series of EMA values.  Leading ``NaN`` values are expected.

    Raises:
        ValueError: If *period* is less than 1.
    """
    if period < 1:
        raise ValueError(f"EMA period must be >= 1, got {period}")

    if prices_series.empty or len(prices_series) < period:
        logger.warning(
            "insufficient_data_for_ema",
            data_points=len(prices_series),
            required=period,
        )
        return pd.Series(dtype=float, index=prices_series.index)

    result = ta.ema(prices_series, length=period)
    if result is None:
        logger.warning("pandas_ta_ema_returned_none", period=period)
        return pd.Series(dtype=float, index=prices_series.index)
    return result


def calculate_rsi(prices_series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (0–100) using *pandas-ta*.

    Args:
        prices_series: Series of closing prices (float).
        period: RSI look-back window (default 14).

    Returns:
        pd.Series with RSI values in the range [0, 100].
        Returns an empty series if data is insufficient.
    """
    if prices_series.empty or len(prices_series) < period + 1:
        logger.warning(
            "insufficient_data_for_rsi",
            data_points=len(prices_series),
            required=period + 1,
        )
        return pd.Series(dtype=float, index=prices_series.index)

    result = ta.rsi(prices_series, length=period)
    if result is None:
        logger.warning("pandas_ta_rsi_returned_none", period=period)
        return pd.Series(dtype=float, index=prices_series.index)
    return result


def calculate_macd(
    prices_series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    """Compute MACD, signal line and histogram using *pandas-ta*.

    Args:
        prices_series: Series of closing prices.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal line EMA period (default 9).

    Returns:
        Dictionary with keys ``'macd_line'``, ``'signal_line'``, and
        ``'histogram'``, each mapping to a ``pd.Series``.
    """
    empty: dict[str, pd.Series] = {
        "macd_line": pd.Series(dtype=float, index=prices_series.index),
        "signal_line": pd.Series(dtype=float, index=prices_series.index),
        "histogram": pd.Series(dtype=float, index=prices_series.index),
    }

    min_required = slow + signal
    if prices_series.empty or len(prices_series) < min_required:
        logger.warning(
            "insufficient_data_for_macd",
            data_points=len(prices_series),
            required=min_required,
        )
        return empty

    result = ta.macd(prices_series, fast=fast, slow=slow, signal=signal)
    if result is None or result.empty:
        logger.warning("pandas_ta_macd_returned_none")
        return empty

    # pandas-ta returns a DataFrame with columns like MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
    cols = result.columns.tolist()
    return {
        "macd_line": result[cols[0]],
        "signal_line": result[cols[1]],
        "histogram": result[cols[2]],
    }


def calculate_bollinger_bands(
    prices_series: pd.Series,
    period: int = 20,
    std_dev: int = 2,
) -> dict[str, pd.Series]:
    """Compute Bollinger Bands (upper, middle, lower) using *pandas-ta*.

    Args:
        prices_series: Series of closing prices.
        period: Look-back window for the moving-average centre line (default 20).
        std_dev: Number of standard deviations for upper/lower bands (default 2).

    Returns:
        Dictionary with keys ``'upper'``, ``'middle'``, ``'lower'``.
    """
    empty: dict[str, pd.Series] = {
        "upper": pd.Series(dtype=float, index=prices_series.index),
        "middle": pd.Series(dtype=float, index=prices_series.index),
        "lower": pd.Series(dtype=float, index=prices_series.index),
    }

    if prices_series.empty or len(prices_series) < period:
        logger.warning(
            "insufficient_data_for_bollinger",
            data_points=len(prices_series),
            required=period,
        )
        return empty

    result = ta.bbands(prices_series, length=period, std=float(std_dev))
    if result is None or result.empty:
        logger.warning("pandas_ta_bbands_returned_none")
        return empty

    # pandas-ta columns: BBL_{period}_{std}, BBM_{period}_{std}, BBU_{period}_{std}, BBB, BBP
    cols = result.columns.tolist()
    return {
        "lower": result[cols[0]],
        "middle": result[cols[1]],
        "upper": result[cols[2]],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Composite Signal Generation
# ═══════════════════════════════════════════════════════════════════════════════


def _latest_valid(series: pd.Series) -> float | None:
    """Return the last non-NaN value in *series*, or ``None``."""
    if series.empty:
        return None
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])


def generate_signals(prices_series: pd.Series, config: AnalysisConfig) -> dict:
    """Run all technical indicators and produce a unified signal assessment.

    Args:
        prices_series: Series of closing prices (ideally ≥ 252 data points).
        config: ``AnalysisConfig`` instance controlling indicator parameters.

    Returns:
        Dictionary containing:
            - ``sma_signal``:  ``'buy'`` / ``'sell'`` / ``'neutral'``
            - ``rsi_signal``:  ``'oversold'`` / ``'overbought'`` / ``'neutral'``
            - ``rsi_value``:   Current RSI as ``float`` (or ``None``)
            - ``macd_signal``: ``'bullish'`` / ``'bearish'`` / ``'neutral'``
            - ``bollinger_signal``: ``'oversold'`` / ``'overbought'`` / ``'neutral'``
            - ``overall_signal``:   ``'strong_buy'`` / ``'buy'`` / ``'hold'`` / ``'sell'`` / ``'strong_sell'``
            - ``confidence``:  ``float`` in [0, 1] — fraction of indicators agreeing
    """
    logger.info("generating_technical_signals", data_points=len(prices_series))

    # ── Defaults (safe fallback when data is scarce) ──────────────────────
    signals: dict = {
        "sma_signal": "neutral",
        "rsi_signal": "neutral",
        "rsi_value": None,
        "macd_signal": "neutral",
        "bollinger_signal": "neutral",
        "overall_signal": "hold",
        "confidence": 0.0,
    }

    if prices_series.empty:
        logger.warning("empty_price_series_for_signals")
        return signals

    current_price = float(prices_series.iloc[-1])

    # ── SMA 200 ───────────────────────────────────────────────────────────
    sma200 = calculate_sma(prices_series, 200)
    sma200_val = _latest_valid(sma200)
    if sma200_val is not None:
        if current_price > sma200_val:
            signals["sma_signal"] = "buy"
        elif current_price < sma200_val:
            signals["sma_signal"] = "sell"

    # ── RSI ────────────────────────────────────────────────────────────────
    rsi = calculate_rsi(prices_series, period=config.rsi_period)
    rsi_val = _latest_valid(rsi)
    signals["rsi_value"] = rsi_val
    if rsi_val is not None:
        if rsi_val < 30:
            signals["rsi_signal"] = "oversold"
        elif rsi_val > 70:
            signals["rsi_signal"] = "overbought"

    # ── MACD ───────────────────────────────────────────────────────────────
    macd_data = calculate_macd(
        prices_series,
        fast=config.macd_fast,
        slow=config.macd_slow,
        signal=config.macd_signal,
    )
    macd_val = _latest_valid(macd_data["macd_line"])
    signal_val = _latest_valid(macd_data["signal_line"])
    hist_val = _latest_valid(macd_data["histogram"])

    if macd_val is not None and signal_val is not None and hist_val is not None:
        if macd_val > signal_val and hist_val > 0:
            signals["macd_signal"] = "bullish"
        else:
            signals["macd_signal"] = "bearish"

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb = calculate_bollinger_bands(
        prices_series,
        period=config.bollinger_period,
        std_dev=config.bollinger_std,
    )
    bb_upper = _latest_valid(bb["upper"])
    bb_lower = _latest_valid(bb["lower"])

    if bb_upper is not None and bb_lower is not None:
        if current_price < bb_lower:
            signals["bollinger_signal"] = "oversold"
        elif current_price > bb_upper:
            signals["bollinger_signal"] = "overbought"

    # ── Aggregate: weighted vote ──────────────────────────────────────────
    score = 0.0  # positive = bullish, negative = bearish
    n_active = 0  # indicators that produced a non-neutral result

    # SMA signal (weight 1)
    if signals["sma_signal"] == "buy":
        score += 1.0
        n_active += 1
    elif signals["sma_signal"] == "sell":
        score -= 1.0
        n_active += 1

    # RSI signal (weight 1)
    if signals["rsi_signal"] == "oversold":
        score += 1.0
        n_active += 1
    elif signals["rsi_signal"] == "overbought":
        score -= 1.0
        n_active += 1

    # MACD signal (weight 1)
    if signals["macd_signal"] == "bullish":
        score += 1.0
        n_active += 1
    elif signals["macd_signal"] == "bearish":
        score -= 1.0
        n_active += 1

    # Bollinger signal (weight 1)
    if signals["bollinger_signal"] == "oversold":
        score += 1.0
        n_active += 1
    elif signals["bollinger_signal"] == "overbought":
        score -= 1.0
        n_active += 1

    total_indicators = 4  # SMA, RSI, MACD, BB

    # Overall signal mapping based on net score
    if score >= 3:
        signals["overall_signal"] = "strong_buy"
    elif score >= 1:
        signals["overall_signal"] = "buy"
    elif score <= -3:
        signals["overall_signal"] = "strong_sell"
    elif score <= -1:
        signals["overall_signal"] = "sell"
    else:
        signals["overall_signal"] = "hold"

    # Confidence = fraction of indicators pointing in the same direction
    if n_active > 0:
        # How many indicators agree with the overall direction?
        direction = 1 if score >= 0 else -1
        agreeing = 0
        for s_name, buy_val, sell_val in [
            ("sma_signal", "buy", "sell"),
            ("rsi_signal", "oversold", "overbought"),
            ("macd_signal", "bullish", "bearish"),
            ("bollinger_signal", "oversold", "overbought"),
        ]:
            val = signals[s_name]
            if direction > 0 and val == buy_val or direction < 0 and val == sell_val:
                agreeing += 1
        signals["confidence"] = round(agreeing / total_indicators, 2)
    else:
        signals["confidence"] = 0.0

    logger.info(
        "technical_signals_generated",
        overall=signals["overall_signal"],
        confidence=signals["confidence"],
        score=score,
    )
    return signals


__all__ = [
    "calculate_sma",
    "calculate_ema",
    "calculate_rsi",
    "calculate_macd",
    "calculate_bollinger_bands",
    "generate_signals",
]
