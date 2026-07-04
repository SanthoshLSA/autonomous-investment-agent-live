"""
Risk analysis module for the Autonomous Investment Research Agent.

Provides functions for calculating risk metrics including volatility, Value at
Risk (VaR), Conditional VaR (CVaR/Expected Shortfall), Monte Carlo simulation,
maximum drawdown, Sharpe ratio, Sortino ratio, beta, and stress testing.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.logger import get_logger

logger = get_logger(__name__)

# Trading days in a year for annualization
_TRADING_DAYS = 252


# ═══════════════════════════════════════════════════════════════════════════════
# Helper – Guard Clause
# ═══════════════════════════════════════════════════════════════════════════════


def _validate_returns(returns: pd.Series, func_name: str) -> bool:
    """Check that *returns* is usable; log a warning and return False if not.

    Args:
        returns: The returns series to validate.
        func_name: Name of the calling function (for logging).

    Returns:
        True if the series has at least 2 valid (non-NaN) entries.
    """
    if returns is None or returns.empty:
        logger.warning(f"{func_name}_empty_series")
        return False
    valid = returns.dropna()
    if len(valid) < 2:
        logger.warning(f"{func_name}_insufficient_data", valid_points=len(valid))
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Returns & Volatility
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_daily_returns(prices: pd.Series) -> pd.Series:
    """Compute simple daily returns: (P_t - P_{t-1}) / P_{t-1}.

    Args:
        prices: Series of closing prices indexed by date.

    Returns:
        pd.Series of daily percentage returns.  The first entry is ``NaN``.
    """
    if prices is None or prices.empty:
        logger.warning("calculate_daily_returns_empty_prices")
        return pd.Series(dtype=float)

    returns = prices.pct_change()
    return returns


def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """Compute the standard deviation of returns, optionally annualized.

    Args:
        returns: Series of daily returns.
        annualize: If True, multiply by √252 for annualized volatility.

    Returns:
        Annualized (or daily) volatility as a positive float.
        Returns ``0.0`` if data is insufficient.
    """
    if not _validate_returns(returns, "calculate_volatility"):
        return 0.0

    clean = returns.dropna()
    vol = float(clean.std())

    if annualize:
        vol *= math.sqrt(_TRADING_DAYS)

    logger.debug("volatility_calculated", volatility=round(vol, 6), annualized=annualize)
    return vol


# ═══════════════════════════════════════════════════════════════════════════════
# Value at Risk (VaR)
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk (percentile-based).

    Args:
        returns: Series of daily returns.
        confidence: Confidence level (e.g. 0.95 for 95 %).

    Returns:
        VaR as a **negative** float representing the loss threshold at the
        given confidence.  Returns ``0.0`` on insufficient data.
    """
    if not _validate_returns(returns, "calculate_var_historical"):
        return 0.0

    clean = returns.dropna()
    var = float(np.percentile(clean, (1 - confidence) * 100))

    logger.debug("historical_var", var=round(var, 6), confidence=confidence)
    return var


def calculate_var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric (Gaussian) Value at Risk.

    Assumes returns are normally distributed.

    Args:
        returns: Series of daily returns.
        confidence: Confidence level.

    Returns:
        VaR as a negative float.  Returns ``0.0`` on insufficient data.
    """
    if not _validate_returns(returns, "calculate_var_parametric"):
        return 0.0

    clean = returns.dropna()
    mu = float(clean.mean())
    sigma = float(clean.std())

    if sigma == 0:
        logger.warning("parametric_var_zero_std")
        return 0.0

    z_score = sp_stats.norm.ppf(1 - confidence)
    var = mu + z_score * sigma

    logger.debug("parametric_var", var=round(var, 6), confidence=confidence)
    return var


# ═══════════════════════════════════════════════════════════════════════════════
# Conditional VaR (Expected Shortfall)
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR — average of losses beyond the historical VaR.

    Args:
        returns: Series of daily returns.
        confidence: Confidence level.

    Returns:
        CVaR as a **negative** float (≤ VaR).  Returns ``0.0`` if data is
        insufficient.
    """
    if not _validate_returns(returns, "calculate_cvar"):
        return 0.0

    var = calculate_var_historical(returns, confidence)
    clean = returns.dropna()
    tail = clean[clean <= var]

    if tail.empty:
        logger.warning("cvar_no_tail_losses")
        return var  # degenerate case – return VaR itself

    cvar = float(tail.mean())
    logger.debug("cvar_calculated", cvar=round(cvar, 6), confidence=confidence)
    return cvar


# ═══════════════════════════════════════════════════════════════════════════════
# Monte Carlo Simulation
# ═══════════════════════════════════════════════════════════════════════════════


def monte_carlo_simulation(
    returns: pd.Series,
    simulations: int = 10_000,
    days: int = _TRADING_DAYS,
) -> dict:
    """Run a Monte Carlo simulation of future returns.

    Generates *simulations* random paths of *days* trading days using the
    historical mean and standard deviation of *returns* under a geometric
    Brownian-motion assumption.

    Args:
        returns: Historical daily returns.
        simulations: Number of simulation paths (default 10 000).
        days: Forecast horizon in trading days (default 252 ≈ 1 year).

    Returns:
        Dictionary with keys:
            - ``var_95``:  5th-percentile terminal return (float).
            - ``var_99``:  1st-percentile terminal return (float).
            - ``expected_return``: Mean terminal return (float).
            - ``worst_case``: Minimum terminal return (float).
            - ``best_case``: Maximum terminal return (float).
            - ``median_return``: Median terminal return (float).
            - ``simulation_summary``: Dict of percentile → return value.
    """
    default: dict = {
        "var_95": 0.0,
        "var_99": 0.0,
        "expected_return": 0.0,
        "worst_case": 0.0,
        "best_case": 0.0,
        "median_return": 0.0,
        "simulation_summary": {},
    }

    if not _validate_returns(returns, "monte_carlo_simulation"):
        return default

    clean = returns.dropna()
    mu = float(clean.mean())
    sigma = float(clean.std())

    if sigma == 0:
        logger.warning("monte_carlo_zero_std")
        return default

    logger.info(
        "monte_carlo_starting",
        simulations=simulations,
        days=days,
        mu=round(mu, 6),
        sigma=round(sigma, 6),
    )

    # Geometric Brownian Motion: S_T / S_0 = exp(Σ daily_log_returns)
    rng = np.random.default_rng(seed=42)
    random_returns = rng.normal(mu, sigma, size=(simulations, days))
    # Cumulative return for each path
    terminal_returns = np.exp(np.sum(np.log(1 + random_returns), axis=1)) - 1

    percentiles = {
        "1st": float(np.percentile(terminal_returns, 1)),
        "5th": float(np.percentile(terminal_returns, 5)),
        "10th": float(np.percentile(terminal_returns, 10)),
        "25th": float(np.percentile(terminal_returns, 25)),
        "50th": float(np.percentile(terminal_returns, 50)),
        "75th": float(np.percentile(terminal_returns, 75)),
        "90th": float(np.percentile(terminal_returns, 90)),
        "95th": float(np.percentile(terminal_returns, 95)),
        "99th": float(np.percentile(terminal_returns, 99)),
    }

    result = {
        "var_95": percentiles["5th"],
        "var_99": percentiles["1st"],
        "expected_return": float(np.mean(terminal_returns)),
        "worst_case": float(np.min(terminal_returns)),
        "best_case": float(np.max(terminal_returns)),
        "median_return": percentiles["50th"],
        "simulation_summary": percentiles,
    }

    logger.info(
        "monte_carlo_complete",
        expected=round(result["expected_return"], 4),
        var_95=round(result["var_95"], 4),
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Drawdown
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_max_drawdown(prices: pd.Series) -> dict:
    """Compute the maximum drawdown from peak to trough.

    Args:
        prices: Series of closing prices indexed by date.

    Returns:
        Dictionary with keys:
            - ``max_drawdown_pct``: Negative float (e.g. -0.25 for 25 % drawdown).
            - ``peak_date``: Datetime of the peak before the deepest trough.
            - ``trough_date``: Datetime of the deepest trough.
            - ``recovery_date``: Datetime when price recovered to the peak,
              or ``None`` if not yet recovered.
            - ``current_drawdown_pct``: Current drawdown from the last peak.
    """
    default: dict = {
        "max_drawdown_pct": 0.0,
        "peak_date": None,
        "trough_date": None,
        "recovery_date": None,
        "current_drawdown_pct": 0.0,
    }

    if prices is None or prices.empty or len(prices) < 2:
        logger.warning(
            "max_drawdown_insufficient_data", length=0 if prices is None else len(prices)
        )
        return default

    prices = prices.dropna()
    if len(prices) < 2:
        return default

    # Running maximum
    cumulative_max = prices.cummax()
    drawdown = (prices - cumulative_max) / cumulative_max

    # Max drawdown
    max_dd = float(drawdown.min())
    trough_idx = drawdown.idxmin()
    # Peak is the last cummax value before the trough
    peak_idx = prices.loc[:trough_idx].idxmax()

    # Recovery: first date after trough where price >= peak price
    peak_price = prices.loc[peak_idx]
    post_trough = prices.loc[trough_idx:]
    recovered = post_trough[post_trough >= peak_price]
    recovery_date: datetime | None = None
    if len(recovered) > 1:
        # Skip the trough itself if it happens to equal peak (unlikely)
        recovery_candidates = recovered.iloc[1:]
        if not recovery_candidates.empty:
            recovery_date = recovery_candidates.index[0]
            # Convert to datetime if needed
            if hasattr(recovery_date, "to_pydatetime"):
                recovery_date = recovery_date.to_pydatetime()

    # Current drawdown
    current_dd = float(drawdown.iloc[-1])

    peak_dt = peak_idx
    trough_dt = trough_idx
    if hasattr(peak_dt, "to_pydatetime"):
        peak_dt = peak_dt.to_pydatetime()
    if hasattr(trough_dt, "to_pydatetime"):
        trough_dt = trough_dt.to_pydatetime()

    result = {
        "max_drawdown_pct": max_dd,
        "peak_date": peak_dt,
        "trough_date": trough_dt,
        "recovery_date": recovery_date,
        "current_drawdown_pct": current_dd,
    }

    logger.debug(
        "max_drawdown_calculated",
        max_drawdown=round(max_dd, 4),
        current_drawdown=round(current_dd, 4),
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Risk-Adjusted Return Ratios
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """Annualized Sharpe ratio.

    Formula: (mean_daily_return × 252 − risk_free_rate) / (daily_std × √252)

    Args:
        returns: Daily returns series.
        risk_free_rate: Annualized risk-free rate (default 5 %).

    Returns:
        Sharpe ratio as a float.  Returns ``0.0`` on insufficient data or
        zero volatility.
    """
    if not _validate_returns(returns, "calculate_sharpe_ratio"):
        return 0.0

    clean = returns.dropna()
    ann_return = float(clean.mean()) * _TRADING_DAYS
    ann_vol = float(clean.std()) * math.sqrt(_TRADING_DAYS)

    if ann_vol == 0:
        logger.warning("sharpe_zero_volatility")
        return 0.0

    sharpe = (ann_return - risk_free_rate) / ann_vol
    logger.debug("sharpe_ratio", sharpe=round(sharpe, 4))
    return sharpe


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """Annualized Sortino ratio (uses downside deviation only).

    Args:
        returns: Daily returns series.
        risk_free_rate: Annualized risk-free rate (default 5 %).

    Returns:
        Sortino ratio as a float.  Returns ``0.0`` on insufficient data or
        zero downside deviation.
    """
    if not _validate_returns(returns, "calculate_sortino_ratio"):
        return 0.0

    clean = returns.dropna()
    ann_return = float(clean.mean()) * _TRADING_DAYS

    # Downside deviation: std of negative returns only
    downside = clean[clean < 0]
    if downside.empty:
        logger.warning("sortino_no_downside_returns")
        return 0.0  # Cannot compute – no negative returns

    downside_dev = float(downside.std()) * math.sqrt(_TRADING_DAYS)

    if downside_dev == 0:
        logger.warning("sortino_zero_downside_deviation")
        return 0.0

    sortino = (ann_return - risk_free_rate) / downside_dev
    logger.debug("sortino_ratio", sortino=round(sortino, 4))
    return sortino


# ═══════════════════════════════════════════════════════════════════════════════
# Beta
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """Compute beta: cov(asset, market) / var(market).

    Args:
        asset_returns: Daily returns for the asset.
        market_returns: Daily returns for the market benchmark.

    Returns:
        Beta as a float.  Returns ``1.0`` on insufficient data (assume
        market-neutral).
    """
    if not _validate_returns(asset_returns, "calculate_beta_asset"):
        return 1.0
    if not _validate_returns(market_returns, "calculate_beta_market"):
        return 1.0

    # Align on common index
    combined = pd.DataFrame({"asset": asset_returns, "market": market_returns}).dropna()

    if len(combined) < 2:
        logger.warning("beta_insufficient_overlapping_data", overlap=len(combined))
        return 1.0

    cov_matrix = combined.cov()
    market_var = cov_matrix.loc["market", "market"]

    if market_var == 0:
        logger.warning("beta_zero_market_variance")
        return 1.0

    beta = float(cov_matrix.loc["asset", "market"] / market_var)
    logger.debug("beta_calculated", beta=round(beta, 4))
    return beta


# ═══════════════════════════════════════════════════════════════════════════════
# Stress Testing
# ═══════════════════════════════════════════════════════════════════════════════


def stress_test(
    returns: pd.Series,
    scenarios: dict[str, dict] | None = None,
) -> dict:
    """Run stress-test scenarios and estimate portfolio impact.

    Default scenarios simulate sudden market drops and volatility spikes.

    Args:
        returns: Historical daily returns of the asset.
        scenarios: Optional custom scenarios.  Each key maps to a dict with
            ``'type'`` (``'crash'`` or ``'volatility'``) and ``'magnitude'``
            (float).

    Returns:
        Dictionary mapping scenario name → estimated impact dict containing
        ``'portfolio_impact_pct'`` and ``'description'``.
    """
    if scenarios is None:
        scenarios = {
            "market_crash_10pct": {"type": "crash", "magnitude": -0.10},
            "market_crash_20pct": {"type": "crash", "magnitude": -0.20},
            "volatility_2x": {"type": "volatility", "multiplier": 2.0},
            "volatility_3x": {"type": "volatility", "multiplier": 3.0},
        }

    if not _validate_returns(returns, "stress_test"):
        return {
            name: {"portfolio_impact_pct": 0.0, "description": "Insufficient data"}
            for name in scenarios
        }

    clean = returns.dropna()
    current_vol = float(clean.std()) * math.sqrt(_TRADING_DAYS)
    float(clean.mean()) * _TRADING_DAYS

    results: dict = {}
    for name, params in scenarios.items():
        scenario_type = params.get("type", "crash")

        if scenario_type == "crash":
            magnitude = params.get("magnitude", -0.10)
            # Direct crash impact
            impact = magnitude
            desc = f"Immediate market decline of {abs(magnitude)*100:.0f}%"
        elif scenario_type == "volatility":
            multiplier = params.get("multiplier", 2.0)
            # Higher volatility ⇒ wider potential loss band
            stressed_vol = current_vol * multiplier
            # Approximate worst-day loss at 2σ under stressed volatility
            impact = -(stressed_vol / math.sqrt(_TRADING_DAYS)) * 2
            desc = (
                f"Volatility increases {multiplier:.0f}x "
                f"(from {current_vol*100:.1f}% to {stressed_vol*100:.1f}%)"
            )
        else:
            impact = 0.0
            desc = "Unknown scenario type"

        results[name] = {
            "portfolio_impact_pct": round(float(impact), 4),
            "description": desc,
        }

    logger.info("stress_test_complete", scenarios=list(results.keys()))
    return results


__all__ = [
    "calculate_daily_returns",
    "calculate_volatility",
    "calculate_var_historical",
    "calculate_var_parametric",
    "calculate_cvar",
    "monte_carlo_simulation",
    "calculate_max_drawdown",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_beta",
    "stress_test",
]
