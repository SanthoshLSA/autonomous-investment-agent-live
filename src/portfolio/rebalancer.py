"""
Portfolio drift detection and rebalancing engine.
"""

from __future__ import annotations

from typing import Any

from src.config import PortfolioConfig
from src.logger import get_logger

logger = get_logger(__name__)


class Rebalancer:
    """Detects deviations from target allocations and issues buy/sell execution actions."""

    def __init__(self, config: PortfolioConfig) -> None:
        """Initialise rebalancer with drift threshold bounds.

        Args:
            config: PortfolioConfig settings.
        """
        self.config = config

    def check_drift(
        self, current_weights: dict[str, float], target_weights: dict[str, float]
    ) -> dict[str, Any]:
        """Compares current holdings allocations to targets to determine if drift threshold exceeded.

        Args:
            current_weights: Actual percentage breakdown of current portfolio.
            target_weights: Target optimal allocations.

        Returns:
            Dictionary specifying drift details per asset and rebalance flag.
        """
        logger.info("Checking portfolio allocation drift...")
        drifted_assets = []
        max_drift = 0.0
        total_drift = 0.0

        # Combine all unique tickers across both sets
        all_tickers = set(current_weights.keys()).union(target_weights.keys())

        for ticker in all_tickers:
            current = current_weights.get(ticker, 0.0)
            target = target_weights.get(ticker, 0.0)
            drift = current - target
            abs_drift = abs(drift)

            total_drift += abs_drift
            if abs_drift > max_drift:
                max_drift = abs_drift

            drifted_assets.append(
                {
                    "ticker": ticker,
                    "current_weight": current,
                    "target_weight": target,
                    "drift": drift,
                    "abs_drift": abs_drift,
                }
            )

        # Rebalance is triggered if maximum single asset drift exceeds the configuration threshold
        needs_rebalance = max_drift >= self.config.rebalance_threshold

        logger.info(
            "Drift calculation complete",
            needs_rebalance=needs_rebalance,
            max_drift=max_drift,
            total_drift=total_drift,
        )

        return {
            "needs_rebalance": needs_rebalance,
            "drifted_assets": drifted_assets,
            "max_drift": max_drift,
            "total_drift": total_drift,
        }

    def generate_rebalance_orders(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        portfolio_value: float,
    ) -> list[dict[str, Any]]:
        """Constructs target execution buy/sell transactions to realign portfolio.

        Args:
            current_weights: Actual percentage breakdown.
            target_weights: Target optimal percentage allocations.
            portfolio_value: Capital scaling amount.

        Returns:
            List of order dictionaries containing ticker, action, percentage shift, and dollar value.
        """
        logger.info("Generating rebalance trade orders...")
        drift_report = self.check_drift(current_weights, target_weights)
        orders = []

        for asset in drift_report["drifted_assets"]:
            ticker = asset["ticker"]
            drift = asset[
                "drift"
            ]  # Positive drift means we are overweighted, Negative means underweighted

            # Skip tiny adjustments under 0.5% weight drift
            if asset["abs_drift"] < 0.005:
                continue

            dollar_amount = drift * portfolio_value
            action = "sell" if dollar_amount > 0 else "buy"

            orders.append(
                {
                    "ticker": ticker,
                    "action": action,
                    "weight_change": -drift,  # Target is current - drift
                    "dollar_amount": abs(dollar_amount),
                    "reason": f"Drift of {drift:+.2%} exceeds boundary. Rebalancing to target {asset['target_weight']:.2%}.",
                }
            )

        # Sort orders so we sell first to release cash before buying (largest dollar size first)
        orders.sort(key=lambda x: (0 if x["action"] == "sell" else 1, -x["dollar_amount"]))
        return orders
