"""
Composite risk scoring module for the Autonomous Investment Research Agent.

Combines technical signals, risk metrics, and sentiment data into a single
risk score (0–100) with a buy/sell signal, confidence level, component
breakdown, and human-readable reasoning.
"""

from __future__ import annotations

from datetime import UTC

import pandas as pd

from src.analysis.risk import (
    calculate_beta,
    calculate_daily_returns,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_var_historical,
    calculate_volatility,
    monte_carlo_simulation,
    stress_test,
)
from src.analysis.sentiment import analyze_news_batch, classify_sentiment
from src.analysis.technical import generate_signals
from src.config import AnalysisConfig, RiskConfig, ScoringConfig, SentimentConfig
from src.data.models import NewsArticle
from src.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════════════


def normalize_to_100(value: float, min_val: float, max_val: float) -> float:
    """Linearly normalize *value* into the [0, 100] range, clamped.

    Args:
        value: The raw value to normalize.
        min_val: The value that maps to 0.
        max_val: The value that maps to 100.

    Returns:
        Normalized score in [0, 100].
        If ``min_val == max_val`` the function returns 50.0 (midpoint).
    """
    if min_val == max_val:
        return 50.0

    # Ensure min_val < max_val for correct direction
    if min_val > max_val:
        min_val, max_val = max_val, min_val

    normalized = ((value - min_val) / (max_val - min_val)) * 100
    return float(max(0.0, min(100.0, normalized)))


# ═══════════════════════════════════════════════════════════════════════════════
# Signal Mapping
# ═══════════════════════════════════════════════════════════════════════════════

_SIGNAL_MAP: dict[str, tuple[float, float]] = {
    "strong_buy": (0, 20),
    "buy": (20, 40),
    "hold": (40, 60),
    "sell": (60, 80),
    "strong_sell": (80, 100),
}


def _score_to_signal(risk_score: float) -> str:
    """Convert a 0-100 risk score to a signal label.

    Args:
        risk_score: Composite risk score.

    Returns:
        One of ``'strong_buy'``, ``'buy'``, ``'hold'``, ``'sell'``, ``'strong_sell'``.
    """
    if risk_score < 20:
        return "strong_buy"
    elif risk_score < 40:
        return "buy"
    elif risk_score < 60:
        return "hold"
    elif risk_score < 80:
        return "sell"
    else:
        return "strong_sell"


# ═══════════════════════════════════════════════════════════════════════════════
# Composite Score
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_composite_score(
    technical_signals: dict,
    risk_metrics: dict,
    sentiment_data: dict,
    scoring_config: ScoringConfig,
) -> dict:
    """Blend technical, risk and sentiment dimensions into a single risk score.

    Formula:
        ``risk_score = w1 × vol_norm + w2 × neg_sent_norm + w3 × beta_norm + w4 × drawdown_norm``

    Args:
        technical_signals: Output of ``generate_signals()``.
        risk_metrics: Dictionary containing ``volatility``, ``beta``,
            ``max_drawdown_pct`` keys (from the risk module).
        sentiment_data: Output of ``analyze_news_batch()`` (must include
            ``weighted_sentiment``).
        scoring_config: ``ScoringConfig`` with component weights.

    Returns:
        Dictionary with keys:
            - ``risk_score``:  int 0–100 (0 = safest, 100 = riskiest).
            - ``signal``:      Trading signal string.
            - ``confidence``:  float in [0, 1].
            - ``breakdown``:   Dict of component-name → {normalized, weight, raw}.
            - ``reasoning``:   Human-readable explanation.
    """
    logger.info("calculating_composite_score")

    # ── Extract raw values with safe defaults ─────────────────────────────
    volatility = risk_metrics.get("volatility", 0.0)
    beta = risk_metrics.get("beta", 1.0)
    max_dd = abs(risk_metrics.get("max_drawdown_pct", 0.0))  # work with positive number
    weighted_sent = sentiment_data.get("weighted_sentiment", 0.0)

    # ── Normalize each component to 0-100 ─────────────────────────────────
    # Volatility: 0% → 0, 100% → 100 (higher vol = higher risk)
    vol_norm = normalize_to_100(volatility, 0.0, 1.0)

    # Negative sentiment: sentiment -1 → risk 100, +1 → risk 0
    # Invert the sentiment scale so negative sentiment = higher risk
    neg_sent_norm = normalize_to_100(-weighted_sent, -1.0, 1.0)

    # Beta: 0 → risk 0, 2 → risk 100 (higher beta = higher market risk)
    beta_norm = normalize_to_100(beta, 0.0, 2.0)

    # Max drawdown: 0% → 0, 50% → 100
    dd_norm = normalize_to_100(max_dd, 0.0, 0.5)

    # ── Weighted sum ──────────────────────────────────────────────────────
    w1 = scoring_config.volatility_weight
    w2 = scoring_config.sentiment_weight
    w3 = scoring_config.market_beta_weight
    w4 = scoring_config.drawdown_weight

    risk_score_raw = w1 * vol_norm + w2 * neg_sent_norm + w3 * beta_norm + w4 * dd_norm
    risk_score = int(round(max(0, min(100, risk_score_raw))))

    signal = _score_to_signal(risk_score)

    # ── Confidence from technical signals ─────────────────────────────────
    tech_confidence = technical_signals.get("confidence", 0.5)
    # Blend with data-availability heuristic
    data_quality = 1.0
    if volatility == 0:
        data_quality -= 0.25
    if beta == 1.0 and max_dd == 0:
        data_quality -= 0.25
    confidence = round(min(1.0, max(0.0, tech_confidence * data_quality)), 2)

    # ── Breakdown ─────────────────────────────────────────────────────────
    breakdown = {
        "volatility": {
            "raw": round(volatility, 4),
            "normalized": round(vol_norm, 2),
            "weight": w1,
        },
        "sentiment": {
            "raw": round(weighted_sent, 4),
            "normalized": round(neg_sent_norm, 2),
            "weight": w2,
        },
        "beta": {
            "raw": round(beta, 4),
            "normalized": round(beta_norm, 2),
            "weight": w3,
        },
        "max_drawdown": {
            "raw": round(max_dd, 4),
            "normalized": round(dd_norm, 2),
            "weight": w4,
        },
    }

    # ── Human-readable reasoning ──────────────────────────────────────────
    reasons: list[str] = []
    if vol_norm > 60:
        reasons.append(f"High volatility ({volatility*100:.1f}%) increases risk")
    elif vol_norm < 30:
        reasons.append(f"Low volatility ({volatility*100:.1f}%) is favorable")

    sent_label = sentiment_data.get("classification", classify_sentiment(weighted_sent))
    reasons.append(f"Sentiment is {sent_label} (score: {weighted_sent:.2f})")

    if beta > 1.3:
        reasons.append(f"High beta ({beta:.2f}) — amplifies market moves")
    elif beta < 0.7:
        reasons.append(f"Low beta ({beta:.2f}) — defensive profile")

    if max_dd > 0.2:
        reasons.append(f"Significant max drawdown ({max_dd*100:.1f}%)")

    tech_overall = technical_signals.get("overall_signal", "hold")
    reasons.append(f"Technical indicators suggest: {tech_overall}")

    reasoning = ". ".join(reasons) + "."

    result = {
        "risk_score": risk_score,
        "signal": signal,
        "confidence": confidence,
        "breakdown": breakdown,
        "reasoning": reasoning,
    }

    logger.info(
        "composite_score_calculated",
        risk_score=risk_score,
        signal=signal,
        confidence=confidence,
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# All-in-One Asset Report
# ═══════════════════════════════════════════════════════════════════════════════


def generate_asset_report(
    ticker: str,
    prices: pd.Series,
    news: list[NewsArticle],
    market_prices: pd.Series,
    analysis_config: AnalysisConfig | None = None,
    risk_config: RiskConfig | None = None,
    sentiment_config: SentimentConfig | None = None,
    scoring_config: ScoringConfig | None = None,
) -> dict:
    """Run the full analysis pipeline for a single asset.

    Orchestrates technical analysis, risk calculations, sentiment analysis,
    and composite scoring into a single comprehensive report dictionary.

    Args:
        ticker: Asset ticker symbol (e.g. ``'AAPL'``).
        prices: Historical closing prices for the asset.
        news: List of ``NewsArticle`` instances related to the asset.
        market_prices: Historical closing prices for the market benchmark
            (e.g. S&P 500) – used for beta calculation.
        analysis_config: Technical analysis parameters (defaults applied if None).
        risk_config: Risk analysis parameters (defaults applied if None).
        sentiment_config: Sentiment parameters (defaults applied if None).
        scoring_config: Scoring weights (defaults applied if None).

    Returns:
        Comprehensive dictionary with top-level keys:
            ``ticker``, ``technical``, ``risk``, ``sentiment``, ``composite``,
            ``timestamp``.
    """
    logger.info("generating_asset_report", ticker=ticker)

    # ── Defaults ──────────────────────────────────────────────────────────
    a_cfg = analysis_config or AnalysisConfig()
    r_cfg = risk_config or RiskConfig()
    s_cfg = sentiment_config or SentimentConfig()
    sc_cfg = scoring_config or ScoringConfig()

    # ── Technical Analysis ────────────────────────────────────────────────
    technical_signals = generate_signals(prices, a_cfg)

    # ── Risk Analysis ─────────────────────────────────────────────────────
    returns = calculate_daily_returns(prices)
    market_returns = calculate_daily_returns(market_prices)

    volatility = calculate_volatility(returns)
    var_hist = calculate_var_historical(returns, confidence=r_cfg.var_confidence)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=r_cfg.risk_free_rate)
    sortino = calculate_sortino_ratio(returns, risk_free_rate=r_cfg.risk_free_rate)
    beta = calculate_beta(returns, market_returns)
    drawdown = calculate_max_drawdown(prices)
    mc = monte_carlo_simulation(
        returns,
        simulations=r_cfg.monte_carlo_simulations,
    )
    stress = stress_test(returns)

    risk_metrics = {
        "volatility": volatility,
        "var_historical_95": var_hist,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "beta": beta,
        "max_drawdown_pct": drawdown["max_drawdown_pct"],
        "max_drawdown": drawdown,
        "monte_carlo": mc,
        "stress_test": stress,
    }

    # ── Sentiment Analysis ────────────────────────────────────────────────
    sentiment_data = analyze_news_batch(news, decay_factor=s_cfg.decay_factor)

    # ── Composite Score ───────────────────────────────────────────────────
    composite = calculate_composite_score(
        technical_signals=technical_signals,
        risk_metrics=risk_metrics,
        sentiment_data=sentiment_data,
        scoring_config=sc_cfg,
    )

    from datetime import datetime

    report = {
        "ticker": ticker,
        "current_price": float(prices.iloc[-1]) if not prices.empty else None,
        "technical": technical_signals,
        "risk": risk_metrics,
        "sentiment": sentiment_data,
        "composite": composite,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    logger.info(
        "asset_report_generated",
        ticker=ticker,
        risk_score=composite["risk_score"],
        signal=composite["signal"],
    )
    return report


__all__ = [
    "normalize_to_100",
    "calculate_composite_score",
    "generate_asset_report",
]
