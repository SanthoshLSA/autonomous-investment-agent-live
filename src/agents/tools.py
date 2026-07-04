"""
LangChain tools wrapping data retrieval and analysis functionality for the AI agents.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from src.analysis.sentiment import analyze_news_batch
from src.analysis.technical import generate_signals
from src.config import get_config
from src.logger import get_logger

logger = get_logger(__name__)


@tool
def fetch_market_data_summary(tickers: list[str]) -> str:
    """Fetch raw market prices and news for the given assets and return a summary of data availability.

    Args:
        tickers: List of asset symbols.

    Returns:
        JSON string describing the success, cached, or failed fetch results.
    """
    logger.info("fetch_market_data_summary called", tickers=tickers)
    try:
        from src.config import get_api_keys
        from src.data.fetcher import DataFetcher

        config = get_config()
        api_keys = get_api_keys()
        fetcher = DataFetcher(config, api_keys)

        results = fetcher.fetch_all(tickers)
        summary = {}
        for ticker, bundle in results.items():
            summary[ticker] = {
                "prices_count": len(bundle.prices),
                "has_info": bundle.info is not None,
                "news_count": len(bundle.news),
                "status": {k: v.status for k, v in bundle.fetch_results.items()},
            }
        return json.dumps(summary)
    except Exception as e:
        logger.exception("Error fetching market data summary", tickers=tickers)
        return json.dumps({"error": str(e)})


@tool
def run_technical_signals(ticker: str, prices_json: str) -> str:
    """Calculate technical analysis signals for a specific ticker from JSON price data.

    Args:
        ticker: Asset ticker symbol.
        prices_json: JSON serialized list of AssetPrice structures.

    Returns:
        JSON string containing moving averages, RSI, MACD, and overall recommendation.
    """
    logger.info("run_technical_signals called", ticker=ticker)
    try:
        import pandas as pd

        raw_prices = json.loads(prices_json)
        if not raw_prices:
            return json.dumps({"error": "No price data provided"})

        df = pd.DataFrame(raw_prices)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        config = get_config()
        signals = generate_signals(df["close"], config.analysis)
        return json.dumps(signals)
    except Exception as e:
        logger.exception("Error running technical signals", ticker=ticker)
        return json.dumps({"error": str(e)})


@tool
def analyze_news_sentiment(ticker: str, news_json: str) -> str:
    """Analyze recent headlines sentiment for a given ticker from JSON news articles.

    Args:
        ticker: Asset ticker symbol.
        news_json: JSON serialized list of NewsArticle structures.

    Returns:
        JSON string with weighted sentiment score and classification.
    """
    logger.info("analyze_news_sentiment called", ticker=ticker)
    try:
        from src.data.models import NewsArticle

        raw_news = json.loads(news_json)
        articles = [NewsArticle(**art) for art in raw_news]

        config = get_config()
        sentiment = analyze_news_batch(articles, config.sentiment.decay_factor)
        return json.dumps(sentiment)
    except Exception as e:
        logger.exception("Error analyzing news sentiment", ticker=ticker)
        return json.dumps({"error": str(e)})
