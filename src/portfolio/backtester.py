"""
Backtester engine simulating trading strategies and quarterly rebalancing.

Supports both a vectorbt implementation and a pure pandas/numpy simulation fallback.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import BacktestConfig
from src.logger import get_logger

logger = get_logger(__name__)


class Backtester:
    """Historical backtester simulating strategy returns, drawdowns, and Sharpe ratios.

    Compares the customized portfolio allocations against the benchmark.
    """

    def __init__(self, config: BacktestConfig) -> None:
        """Initialise backtester settings.

        Args:
            config: Loaded BacktestConfig.
        """
        self.config = config

    def run_backtest(
        self,
        prices_df: pd.DataFrame,
        weights: dict[str, float],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Runs the historical performance backtest over the target timeframe.

        Args:
            prices_df: Daily prices DataFrame.
            weights: Portfolio weights dictionary.
            start_date: Simulation start filter. Defaults to config value.
            end_date: Simulation end filter.

        Returns:
            Dictionary containing performance metrics, equity curves, and drawdowns.
        """
        logger.info("Starting historical backtesting simulation...")
        start = start_date or self.config.start_date
        end = end_date or self.config.end_date

        # Clean index and filter dates
        df = prices_df.copy()
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        if start:
            df = df.loc[start:]
        if end:
            df = df.loc[:end]

        if df.empty or len(df) < 5:
            logger.error("Insufficient historical data for backtesting.")
            return {"error": "Insufficient data"}

        # Attempt vectorbt run first, fallback to pandas if failure or import issue
        try:
            return self._run_vectorbt_backtest(df, weights)
        except Exception as e:
            logger.warning("Vectorbt simulation failed, running pandas fallback", error=str(e))
            return self._run_pandas_backtest(df, weights)

    def _run_vectorbt_backtest(self, df: pd.DataFrame, weights: dict[str, float]) -> dict[str, Any]:
        """Executes rebalanced backtest using vectorbt."""
        logger.info("Executing vectorbt simulation...")

        # Configure numba cache dir to avoid Windows hang issues
        os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.home() / ".numba_cache"))
        import vectorbt as vbt

        # Exclude benchmark indicators from asset daily returns calculation
        asset_cols = [c for c in df.columns if not c.startswith("^")]
        asset_prices = df[asset_cols]
        asset_prices.pct_change().dropna()

        # Build weights array matching columns
        w_arr = np.array([weights.get(col, 0.0) for col in asset_cols])
        # Re-scale weights to sum to 1.0 to avoid leverage simulation issue
        w_sum = w_arr.sum()
        if w_sum > 0:
            w_arr = w_arr / w_sum

        # Simulate periodic rebalancing (every 63 trading days = approx quarterly)
        # Vectorbt portfolio simulation
        pf = vbt.Portfolio.from_rebalancing(
            close=asset_prices,
            weights=pd.DataFrame([w_arr], index=asset_prices.index, columns=asset_cols),
            freq="B",
            rebalance_cb_kwargs=dict(freq="63B"),  # Every 63 days rebalance
            init_cash=1000000.0,
            fees=self.config.transaction_cost,
            slippage=self.config.slippage,
        )

        total_return = float(pf.total_return())
        ann_return = float(pf.annualized_return())
        vol = float(pf.annualized_volatility())
        sharpe = float(pf.sharpe_ratio(risk_free=0.05))
        max_dd = float(pf.max_drawdown())

        # Cumulative equity curve
        equity_curve = pf.value().tolist()
        # Daily drawdown percentages
        drawdown_series = pf.drawdown().tolist()

        # Calculate monthly returns
        monthly_returns = pf.returns().resample("ME").sum()
        monthly_dict = {date.strftime("%Y-%m"): float(val) for date, val in monthly_returns.items()}

        # Simple benchmark comparison (e.g. S&P 500)
        benchmark_return = 0.0
        bench_cols = [c for c in df.columns if c.startswith("^")]
        if bench_cols:
            bench_col = bench_cols[0]
            bench_prices = df[bench_col]
            benchmark_return = float(
                (bench_prices.iloc[-1] - bench_prices.iloc[0]) / bench_prices.iloc[0]
            )

        return {
            "total_return": total_return,
            "annual_return": ann_return,
            "annual_volatility": vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": ann_return / abs(max_dd) if max_dd != 0 else 0.0,
            "benchmark_return": benchmark_return,
            "alpha": ann_return - benchmark_return,
            "monthly_returns": monthly_dict,
            "equity_curve": equity_curve,
            "drawdown_series": drawdown_series,
            "win_rate": float((pf.returns() > 0).mean()),
        }

    def _run_pandas_backtest(self, df: pd.DataFrame, weights: dict[str, float]) -> dict[str, Any]:
        """Calculates returns deterministically using pure pandas/numpy."""
        logger.info("Executing pandas backtest fallback...")
        asset_cols = [c for c in df.columns if not c.startswith("^")]
        asset_prices = df[asset_cols]
        returns = asset_prices.pct_change().fillna(0.0)

        # Build weights array
        w_arr = np.array([weights.get(col, 0.0) for col in asset_cols])
        w_sum = w_arr.sum()
        if w_sum > 0:
            w_arr = w_arr / w_sum

        # Calculate daily portfolio returns
        daily_portfolio_returns = returns.dot(w_arr)

        # Simple transaction fee subtraction on first day
        daily_portfolio_returns.iloc[0] -= self.config.transaction_cost

        # Calculate metrics
        cumulative_returns = (1 + daily_portfolio_returns).cumprod() - 1
        total_return = float(cumulative_returns.iloc[-1])

        # Annualized values
        n_days = len(returns)
        ann_return = float((total_return + 1) ** (252.0 / n_days) - 1) if n_days > 0 else 0.0
        vol = float(daily_portfolio_returns.std() * np.sqrt(252))

        sharpe = (ann_return - 0.05) / vol if vol > 0 else 0.0

        # Drawdowns
        cum_max = (cumulative_returns + 1).cummax()
        drawdowns = ((cumulative_returns + 1) / cum_max) - 1
        max_dd = float(drawdowns.min())

        equity_curve = [1000000.0 * (1.0 + r) for r in cumulative_returns.tolist()]

        # Benchmark
        benchmark_return = 0.0
        bench_cols = [c for c in df.columns if c.startswith("^")]
        if bench_cols:
            bench_col = bench_cols[0]
            bench_series = df[bench_col]
            benchmark_return = float(
                (bench_series.iloc[-1] - bench_series.iloc[0]) / bench_series.iloc[0]
            )

        monthly_returns = daily_portfolio_returns.resample("ME").sum()
        monthly_dict = {date.strftime("%Y-%m"): float(val) for date, val in monthly_returns.items()}

        return {
            "total_return": total_return,
            "annual_return": ann_return,
            "annual_volatility": vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": ann_return / abs(max_dd) if max_dd != 0 else 0.0,
            "benchmark_return": benchmark_return,
            "alpha": ann_return - benchmark_return,
            "monthly_returns": monthly_dict,
            "equity_curve": equity_curve,
            "drawdown_series": drawdowns.tolist(),
            "win_rate": float((daily_portfolio_returns > 0).mean()),
        }
