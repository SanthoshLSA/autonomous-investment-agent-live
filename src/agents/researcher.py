"""
Researcher agent node responsible for downloading price, fundamentals, and news data.
"""

from __future__ import annotations

from typing import Any

from src.agents.state import InvestmentAgentState
from src.config import get_api_keys, get_config
from src.data.fetcher import DataFetcher
from src.logger import get_logger

logger = get_logger(__name__)


def researcher_node(state: InvestmentAgentState) -> dict[str, Any]:
    """Retrieves pricing histories, company info, and relevant news headlines.

    This node operates deterministically using our configured fetchers.

    Args:
        state: Shared LangGraph state.

    Returns:
        State updates containing downloaded market data.
    """
    tickers = state.get("tickers", [])
    logger.info("Researcher agent starting fetch operation", tickers=tickers)

    if not tickers:
        error_msg = "No tickers provided in the state watchlist."
        logger.error(error_msg)
        return {
            "error_log": [error_msg],
            "audit_log": ["Researcher failed: No tickers provided."],
        }

    try:
        config = get_config()
        api_keys = get_api_keys()
        fetcher = DataFetcher(config, api_keys)

        logger.info("Fetching price bundles & metadata...")
        bundles = fetcher.fetch_all(tickers)

        # Dump to model representations for json serialization in state
        serialized_bundles = {}
        for ticker, bundle in bundles.items():
            serialized_bundles[ticker] = bundle.model_dump()

        logger.info("Researcher fetch complete", count=len(serialized_bundles))
        return {
            "market_data": serialized_bundles,
            "audit_log": [f"Researcher: Retrieved market data and news for {', '.join(tickers)}."],
        }
    except Exception as e:
        logger.exception("Researcher node critical failure")
        return {
            "error_log": [f"Researcher error: {str(e)}"],
            "audit_log": ["Researcher failed due to an exception."],
        }
