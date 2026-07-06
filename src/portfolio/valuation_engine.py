"""
Simulation Valuation Engine.

Computes holdings valuations, daily snapshots, and profit/loss parameters using SQLite.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import yfinance as yf

from src.logger import get_logger
from src.portfolio.simulator_db import (
    get_most_recent_snapshot,
    get_snapshot_by_date,
    get_user_cash,
    get_user_holdings,
    upsert_daily_snapshot,
)

logger = get_logger(__name__)


def get_latest_price(ticker: str) -> float:
    """Fetches the latest closing/current market price for a ticker using yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker)
        # Try fast_info first
        info = ticker_obj.fast_info
        if hasattr(info, "last_price") and info.last_price is not None:
            return float(info.last_price)

        # Fallback to daily close history
        history = ticker_obj.history(period="1d")
        if not history.empty:
            return float(history["Close"].iloc[-1])
    except Exception:
        logger.exception("Failed to fetch price for ticker", ticker=ticker)
    return 0.0


def run_daily_valuation_for_user(username: str) -> bool:
    """Calculates holdings value and updates the daily snapshot for a single user."""
    username_clean = username.strip().lower()
    today = datetime.now().date()

    try:
        cash = get_user_cash(username_clean)
        holdings = get_user_holdings(username_clean)

        holdings_value = 0.0
        for pos in holdings:
            ticker = pos["ticker"]
            shares = pos["shares"]
            price = get_latest_price(ticker)
            if price > 0:
                holdings_value += shares * price
            else:
                # If price fails, fallback to cost basis to avoid showing complete zero values
                holdings_value += shares * pos["avg_cost"]

        total_value = cash + holdings_value

        # Check for yesterday's snapshot to compute daily PnL
        yesterday = today - timedelta(days=1)
        prev_snap = get_snapshot_by_date(username_clean, str(yesterday))

        # If yesterday doesn't exist, search for the most recent snapshot
        if not prev_snap:
            prev_snap = get_most_recent_snapshot(username_clean)

        prev_total = prev_snap["total_value"] if prev_snap else 1000000.0  # Initial starting amount
        daily_pnl = total_value - prev_total

        # Upsert today's snapshot
        upsert_daily_snapshot(
            username_clean,
            str(today),
            total_value,
            cash,
            holdings_value,
            daily_pnl,
        )
        logger.info(
            "Daily snapshot updated",
            username=username_clean,
            total_value=total_value,
            daily_pnl=daily_pnl,
        )
        return True
    except Exception:
        logger.exception("Error running daily valuation snapshot", username=username_clean)
        return False


def run_all_valuations() -> None:
    """Runs daily valuation snapshots for all registered users in the database."""
    logger.info("Starting daily valuation loop for all simulator accounts...")
    try:
        # Auto-initialize database tables in case the CLI/daemon runs before Streamlit
        from src.portfolio.simulator_db import init_db

        init_db()

        conn = get_db_connection()
        cursor = conn.execute("SELECT username FROM users")
        users = [row[0] for row in cursor.fetchall()]
        conn.close()

        success_count = 0
        for user in users:
            if run_daily_valuation_for_user(user):
                success_count += 1

        logger.info(
            "Daily valuations snapshot completed",
            total=len(users),
            succeeded=success_count,
        )
    except Exception:
        logger.exception("Failed to execute valuation loop")
