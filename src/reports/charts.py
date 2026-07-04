"""
Plotly interactive chart generation helper functions.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.logger import get_logger

logger = get_logger(__name__)


def build_price_candlestick(ticker: str, prices: list[dict[str, Any]]) -> go.Figure:
    """Builds a Plotly Candlestick chart for the ticker asset.

    Args:
        ticker: Asset ticker string.
        prices: List of serialized AssetPrice objects.

    Returns:
        Plotly Figure instance.
    """
    logger.info("Building candlestick chart", ticker=ticker)
    df = pd.DataFrame(prices)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="OHLC",
            )
        ]
    )

    fig.update_layout(
        title=f"{ticker} Price Candlestick",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        yaxis_title="Price ($ / ₹)",
        xaxis_title="Date",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def build_risk_heatmap(composite_scores: dict[str, Any]) -> go.Figure:
    """Generates a vertical bar chart heatmap representing normalized risk scores.

    Args:
        composite_scores: Watchlist composite scores dict.

    Returns:
        Plotly Figure.
    """
    logger.info("Building risk heat representation...")
    tickers = list(composite_scores.keys())
    scores = [composite_scores[t]["risk_score"] for t in tickers]
    signals = [composite_scores[t]["signal"].upper() for t in tickers]

    # Create visual pandas structure
    df = pd.DataFrame({"Ticker": tickers, "Risk Score": scores, "Signal": signals})
    df.sort_values(by="Risk Score", ascending=False, inplace=True)

    fig = px.bar(
        df,
        x="Ticker",
        y="Risk Score",
        color="Risk Score",
        color_continuous_scale="RdYlGn_r",  # Red (high risk) to Green (low risk) reversed
        text="Signal",
        range_color=[0, 100],
    )

    fig.update_layout(
        title="Asset Risk Profile Comparison",
        template="plotly_dark",
        yaxis_title="Normalized Risk Score (0-100)",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def build_allocation_pie(weights: dict[str, float]) -> go.Figure:
    """Creates a pie chart breakdown of target asset allocations.

    Args:
        weights: Optimized asset weight mappings.

    Returns:
        Plotly Figure.
    """
    logger.info("Building allocation pie...")
    labels = list(weights.keys())
    values = [w for w in weights.values()]

    # Filter out empty allocations to make the pie chart clean
    active_labels = [lbl for lbl, v in zip(labels, values, strict=False) if v > 0.0]
    active_values = [v for v in values if v > 0.0]

    # Calculate Cash allocation
    total_active_w = sum(active_values)
    if total_active_w < 0.999:
        active_labels.append("Cash")
        active_values.append(1.0 - total_active_w)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=active_labels,
                values=active_values,
                hole=0.4,
                textinfo="label+percent",
            )
        ]
    )

    fig.update_layout(
        title="Target Portfolio Allocations",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def build_equity_curve(curve_values: list[float], dates: list[str]) -> go.Figure:
    """Generates the strategy equity curve line chart.

    Args:
        curve_values: Value history starting at initial capital.
        dates: List of matching date strings.

    Returns:
        Plotly Figure.
    """
    logger.info("Building equity curve...")
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=curve_values,
            mode="lines",
            name="Portfolio Value",
            line=dict(color="#00D2FF", width=2),
        )
    )

    fig.update_layout(
        title="Historical Portfolio Value (Backtest)",
        template="plotly_dark",
        yaxis_title="Value ($ / ₹)",
        xaxis_title="Date",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig
