"""
Data sources package — re-exports all source adapters.
"""

from src.data.sources.finnhub_source import FinnhubSource
from src.data.sources.newsapi_source import NewsAPISource
from src.data.sources.yfinance_source import YFinanceSource

__all__ = [
    "YFinanceSource",
    "NewsAPISource",
    "FinnhubSource",
]
