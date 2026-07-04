"""
Portfolio Optimization wrapper using PyPortfolioOpt.

Implements Efficient Frontier (max Sharpe & min volatility) with asset weight limits.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation

from src.config import PortfolioConfig
from src.logger import get_logger

logger = get_logger(__name__)


class PortfolioOptimizer:
    """Wrapper class around PyPortfolioOpt to calculate target portfolio allocations.

    Configured with weight limits and optimization goals.
    """

    def __init__(self, config: PortfolioConfig) -> None:
        """Initialise optimizer with configuration preferences.

        Args:
            config: Loaded PortfolioConfig settings.
        """
        self.config = config
        logger.info(
            "PortfolioOptimizer initialized",
            risk_tolerance=config.risk_tolerance,
            min_position=config.min_position_size,
            max_position=config.max_single_position,
        )

    def optimize(self, prices_df: pd.DataFrame, method: str | None = None) -> dict[str, Any]:
        """Calculates optimal asset allocations based on historical daily prices.

        Args:
            prices_df: DataFrame index=Date, columns=tickers, values=close prices.
            method: Optimization method ('min_volatility', 'max_sharpe'). If None,
                    defaults to the config setting.

        Returns:
            Dictionary containing optimized asset weights and expected performance.
        """
        logger.info("Optimizing portfolio allocations...")
        opt_method = method or self.config.optimization_method
        tickers = list(prices_df.columns)

        try:
            # 1. Expected Returns & Sample Covariance
            # Since PyPortfolioOpt expects daily prices:
            mu = expected_returns.mean_historical_return(prices_df)
            S = risk_models.sample_cov(prices_df)  # noqa: N806

            # 2. Set weight limits (min position limit, max single exposure limit)
            ef = EfficientFrontier(
                mu,
                S,
                weight_bounds=(self.config.min_position_size, self.config.max_single_position),
            )

            # 3. Apply objective function
            if opt_method == "max_sharpe":
                logger.info("Running Maximum Sharpe ratio optimization")
                ef.max_sharpe(risk_free_rate=0.05)
            else:
                logger.info("Running Minimum Volatility optimization")
                ef.min_volatility()

            # 4. Clean weights and performance metrics
            weights = ef.clean_weights()
            ret, vol, sharpe = ef.portfolio_performance(verbose=False, risk_free_rate=0.05)

            logger.info("Optimization succeeded", sharpe=sharpe, expected_return=ret)
            return {
                "weights": dict(weights),
                "expected_return": float(ret),
                "annual_volatility": float(vol),
                "sharpe_ratio": float(sharpe),
                "method": opt_method,
            }

        except Exception as e:
            logger.exception("Portfolio optimization failed, using equal weight fallback")

            # Equal weight fallback
            n_assets = len(tickers)
            fallback_weight = round(1.0 / n_assets, 4) if n_assets > 0 else 0.0
            weights = {ticker: fallback_weight for ticker in tickers}

            return {
                "weights": weights,
                "expected_return": 0.0,
                "annual_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "method": "equal_weight_fallback",
                "error": str(e),
            }

    def get_discrete_allocation(
        self, weights: dict[str, float], latest_prices: pd.Series, total_value: float
    ) -> dict[str, Any]:
        """Converts percentage allocations into discrete share buy counts.

        Args:
            weights: Optimized ticker-to-weight dictionary.
            latest_prices: Series mapping tickers to latest close prices.
            total_value: Portfolio capital amount.

        Returns:
            Dictionary containing bought share quantities and cash leftover.
        """
        logger.info("Calculating discrete allocation counts", portfolio_value=total_value)
        try:
            da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_value)
            allocation, leftover = da.lp_portfolio()
            return {
                "allocation": dict(allocation),
                "leftover": float(leftover),
            }
        except Exception as e:
            logger.exception("Discrete allocation calculation failed, using basic rounding")
            allocation = {}
            leftover = total_value

            for ticker, weight in weights.items():
                price = latest_prices.get(ticker)
                if price and price > 0:
                    shares = int((total_value * weight) // price)
                    allocation[ticker] = shares
                    leftover -= shares * price

            return {
                "allocation": allocation,
                "leftover": float(leftover),
                "warning": f"Basic rounding fallback used: {str(e)}",
            }
