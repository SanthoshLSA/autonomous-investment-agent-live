"""
Analyst agent node calculating all technical indicators, risk parameters, and sentiments.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.agents.state import InvestmentAgentState
from src.analysis.risk import (
    calculate_beta,
    calculate_cvar,
    calculate_daily_returns,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_var_historical,
    calculate_volatility,
)
from src.analysis.scoring import calculate_composite_score
from src.analysis.sentiment import analyze_news_batch
from src.analysis.technical import generate_signals
from src.config import get_config
from src.data.models import NewsArticle
from src.logger import get_logger

logger = get_logger(__name__)


def analyst_node(state: InvestmentAgentState) -> dict[str, Any]:
    """Processes downloaded raw market data and news.

    Calculates RSI, MACD, historical Value at Risk, news sentiment scores,
    and returns a combined composite risk score.

    Args:
        state: Shared LangGraph state.

    Returns:
        State updates containing technical, risk, sentiment, and composite dicts.
    """
    market_data = state.get("market_data")
    if not market_data:
        err = "Analyst failed: No market data found in state."
        logger.error(err)
        return {"error_log": [err], "audit_log": ["Analyst aborted: Missing market data."]}

    config = get_config()
    tech_results = {}
    risk_results = {}
    sent_results = {}
    composite_results = {}
    audit_messages = []

    # Get market reference index returns (default S&P 500) if available to compute beta
    benchmark_ticker = config.backtest.benchmark
    benchmark_returns = None
    if benchmark_ticker in market_data:
        benchmark_prices = market_data[benchmark_ticker].get("prices", [])
        if benchmark_prices:
            bench_df = pd.DataFrame(benchmark_prices)
            bench_df["date"] = pd.to_datetime(bench_df["date"], utc=True)
            bench_df.set_index("date", inplace=True)
            bench_df.sort_index(inplace=True)
            benchmark_returns = calculate_daily_returns(bench_df["close"])

    for ticker, bundle in market_data.items():
        # Do not calculate scores for index benchmarks themselves in basic analysis
        if ticker.startswith("^"):
            continue

        prices = bundle.get("prices", [])
        if not prices:
            logger.warning("No price data for ticker, skipping analysis", ticker=ticker)
            continue

        try:
            # Load into pandas DataFrame
            df = pd.DataFrame(prices)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            close_series = df["close"]
            returns = calculate_daily_returns(close_series)

            # 1. Technical Indicators
            signals = generate_signals(close_series, config.analysis)
            tech_results[ticker] = signals

            # 2. Risk Metrics
            vol = calculate_volatility(returns)
            var_hist = calculate_var_historical(returns, config.risk.var_confidence)
            cvar = calculate_cvar(returns, config.risk.cvar_confidence)
            sharpe = calculate_sharpe_ratio(returns, config.risk.risk_free_rate)
            drawdown_metrics = calculate_max_drawdown(close_series)

            beta = 1.0
            if benchmark_returns is not None:
                beta = calculate_beta(returns, benchmark_returns)

            risk_results[ticker] = {
                "volatility": vol,
                "var_95": var_hist,
                "cvar_95": cvar,
                "sharpe_ratio": sharpe,
                "max_drawdown": drawdown_metrics.get("max_drawdown_pct", 0.0),
                "beta": beta,
            }

            # 3. Sentiment Metrics
            news_list = [NewsArticle(**art) for art in bundle.get("news", [])]
            sentiment = analyze_news_batch(news_list, config.sentiment.decay_factor)
            sent_results[ticker] = sentiment

            # 4. Composite Risk & Signal
            composite = calculate_composite_score(
                signals, risk_results[ticker], sentiment, config.scoring
            )
            composite_results[ticker] = composite

            audit_messages.append(
                f"{ticker}: Signal={composite['signal']}, Risk Score={composite['risk_score']}"
            )

        except Exception as e:
            logger.exception("Error analyzing asset", ticker=ticker)
            state["error_log"].append(f"Analyst ticker {ticker} error: {str(e)}")

    logger.info("Analyst processing complete")
    return {
        "technical_analysis": tech_results,
        "risk_analysis": risk_results,
        "sentiment_analysis": sent_results,
        "composite_scores": composite_results,
        "audit_log": [f"Analyst calculations complete. {'; '.join(audit_messages)}"],
    }
