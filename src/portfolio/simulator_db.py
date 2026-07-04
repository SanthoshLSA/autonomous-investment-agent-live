"""
SQLite Database Manager for the Paper Trading Simulator and User Authentication.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)


def get_db_connection() -> sqlite3.Connection:
    """Establishes connection to the SQLite database file, enabling foreign key checks."""
    Path("data").mkdir(exist_ok=True)
    conn = sqlite3.connect("data/portfolio.db")
    # Enable foreign keys support in SQLite
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Initializes the database and creates required tables if they don't exist."""
    logger.info("Initializing SQLite Portfolio Database...")
    try:
        conn = get_db_connection()
        with conn:
            # Users table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    cash_balance REAL DEFAULT 1000000.0
                )
            """)

            # Holdings table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS holdings (
                    username TEXT,
                    ticker TEXT,
                    shares REAL DEFAULT 0.0,
                    avg_cost REAL DEFAULT 0.0,
                    PRIMARY KEY (username, ticker),
                    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
                )
            """)

            # Transactions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    ticker TEXT,
                    action TEXT,
                    shares REAL,
                    price REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
                )
            """)

            # Daily snapshots table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_snapshots (
                    username TEXT,
                    date TEXT,
                    total_value REAL,
                    cash REAL,
                    holdings_value REAL,
                    daily_pnl REAL,
                    PRIMARY KEY (username, date),
                    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
                )
            """)

            # Migration: Convert misnamed 'CASH' holdings back to user cash balance
            cursor = conn.execute(
                "SELECT username, shares, avg_cost FROM holdings WHERE ticker = 'CASH'"
            )
            cash_holdings = cursor.fetchall()
            for username_item, shares_item, avg_cost_item in cash_holdings:
                refund = float(shares_item) * float(avg_cost_item)
                conn.execute(
                    "UPDATE users SET cash_balance = cash_balance + ? WHERE username = ?",
                    (refund, username_item),
                )
                conn.execute(
                    "DELETE FROM holdings WHERE username = ? AND ticker = 'CASH'",
                    (username_item,),
                )
                logger.info(
                    "Migrated CASH holding back to cash balance",
                    username=username_item,
                    refund=refund,
                )

        conn.close()
        logger.info("SQLite Database tables initialized successfully.")
    except Exception as e:
        logger.exception("Database initialization failed")
        raise e


def hash_password(password: str) -> str:
    """Computes SHA256 digest of plaintext password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# ─── Auth Handlers ────────────────────────────────────────────────────────────


def create_user(username: str, password: str) -> bool:
    """Registers a new user inside the database. Initial cash is 10 Lakh INR."""
    username_clean = username.strip().lower()
    if not username_clean or not password:
        return False

    pw_hash = hash_password(password)
    try:
        conn = get_db_connection()
        with conn:
            # Check if user already exists
            cursor = conn.execute(
                "SELECT username FROM users WHERE username = ?", (username_clean,)
            )
            if cursor.fetchone():
                conn.close()
                return False

            # Insert new user record
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username_clean, pw_hash),
            )
        conn.close()
        logger.info("User registered successfully", username=username_clean)
        return True
    except Exception:
        logger.exception("Error creating user", username=username_clean)
        return False


def verify_user(username: str, password: str) -> bool:
    """Checks if username and password hash match database records."""
    username_clean = username.strip().lower()
    if not username_clean or not password:
        return False

    pw_hash = hash_password(password)
    try:
        conn = get_db_connection()
        cursor = conn.execute(
            "SELECT username FROM users WHERE username = ? AND password_hash = ?",
            (username_clean, pw_hash),
        )
        result = cursor.fetchone()
        conn.close()
        return result is not None
    except Exception:
        logger.exception("Error verifying user", username=username_clean)
        return False


# ─── Cash & Position Handlers ──────────────────────────────────────────────────


def get_user_cash(username: str) -> float:
    """Retrieves current cash balance for specified username."""
    username_clean = username.strip().lower()
    try:
        conn = get_db_connection()
        cursor = conn.execute(
            "SELECT cash_balance FROM users WHERE username = ?", (username_clean,)
        )
        row = cursor.fetchone()
        conn.close()
        return float(row[0]) if row else 0.0
    except Exception:
        logger.exception("Error getting user cash", username=username_clean)
        return 0.0


def add_funds(username: str, amount: float) -> bool:
    """Adds play money funds to user's cash balance."""
    username_clean = username.strip().lower()
    if amount <= 0:
        return False

    try:
        conn = get_db_connection()
        with conn:
            conn.execute(
                "UPDATE users SET cash_balance = cash_balance + ? WHERE username = ?",
                (amount, username_clean),
            )
            # Log deposit as a dummy transaction
            conn.execute(
                "INSERT INTO transactions (username, ticker, action, shares, price) VALUES (?, ?, ?, ?, ?)",
                (username_clean, "INR", "DEPOSIT", 1.0, amount),
            )
        conn.close()
        logger.info("Funds deposited successfully", username=username_clean, amount=amount)
        return True
    except Exception:
        logger.exception("Error adding funds", username=username_clean)
        return False


# ─── Trade Handler ────────────────────────────────────────────────────────────


def execute_trade(username: str, ticker: str, action: str, shares: float, price: float) -> str:
    """Executes paper trade transaction, adjusts cash and holdings records.

    Returns:
        String message describing success or reason for failure.
    """
    username_clean = username.strip().lower()
    ticker_clean = ticker.strip().upper()
    action_clean = action.strip().upper()

    if shares <= 0 or price <= 0:
        return "Quantity and Price must be positive values."

    if action_clean not in ["BUY", "SELL"]:
        return f"Invalid action: {action_clean}"

    total_cost = shares * price

    try:
        conn = get_db_connection()
        with conn:
            # 1. Fetch current cash balance
            cursor = conn.execute(
                "SELECT cash_balance FROM users WHERE username = ?", (username_clean,)
            )
            row = cursor.fetchone()
            if not row:
                conn.close()
                return "User record not found."
            cash = float(row[0])

            # 2. Fetch current holdings status
            cursor = conn.execute(
                "SELECT shares, avg_cost FROM holdings WHERE username = ? AND ticker = ?",
                (username_clean, ticker_clean),
            )
            holding_row = cursor.fetchone()
            held_shares = float(holding_row[0]) if holding_row else 0.0
            held_avg_cost = float(holding_row[1]) if holding_row else 0.0

            if action_clean == "BUY":
                if cash < total_cost:
                    conn.close()
                    return f"Insufficient cash. Required: {total_cost:,.2f} INR. Available: {cash:,.2f} INR."

                # Update cash
                conn.execute(
                    "UPDATE users SET cash_balance = cash_balance - ? WHERE username = ?",
                    (total_cost, username_clean),
                )

                # Update holdings
                if held_shares > 0:
                    new_shares = held_shares + shares
                    new_avg = (held_shares * held_avg_cost + total_cost) / new_shares
                    conn.execute(
                        "UPDATE holdings SET shares = ?, avg_cost = ? WHERE username = ? AND ticker = ?",
                        (new_shares, new_avg, username_clean, ticker_clean),
                    )
                else:
                    conn.execute(
                        "INSERT INTO holdings (username, ticker, shares, avg_cost) VALUES (?, ?, ?, ?)",
                        (username_clean, ticker_clean, shares, price),
                    )

            elif action_clean == "SELL":
                if held_shares < shares:
                    conn.close()
                    return f"Insufficient shares. Trying to sell: {shares}. Owned: {held_shares}."

                # Update cash
                conn.execute(
                    "UPDATE users SET cash_balance = cash_balance + ? WHERE username = ?",
                    (total_cost, username_clean),
                )

                # Update holdings
                new_shares = held_shares - shares
                if new_shares > 0:
                    conn.execute(
                        "UPDATE holdings SET shares = ? WHERE username = ? AND ticker = ?",
                        (new_shares, username_clean, ticker_clean),
                    )
                else:
                    conn.execute(
                        "DELETE FROM holdings WHERE username = ? AND ticker = ?",
                        (username_clean, ticker_clean),
                    )

            # 3. Log Transaction
            conn.execute(
                "INSERT INTO transactions (username, ticker, action, shares, price) VALUES (?, ?, ?, ?, ?)",
                (username_clean, ticker_clean, action_clean, shares, price),
            )

        conn.close()
        logger.info(
            "Trade executed successfully",
            username=username_clean,
            ticker=ticker_clean,
            action=action_clean,
            shares=shares,
        )
        return "SUCCESS"
    except Exception as e:
        logger.exception("Error executing trade", username=username_clean)
        return f"Database Error: {str(e)}"


# ─── Query Retrieval Handlers ──────────────────────────────────────────────────


def get_user_holdings(username: str) -> list[dict[str, Any]]:
    """Fetches list of holdings for specified username."""
    username_clean = username.strip().lower()
    try:
        conn = get_db_connection()
        cursor = conn.execute(
            "SELECT ticker, shares, avg_cost FROM holdings WHERE username = ?",
            (username_clean,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {"ticker": row[0], "shares": float(row[1]), "avg_cost": float(row[2])} for row in rows
        ]
    except Exception:
        logger.exception("Error getting user holdings", username=username_clean)
        return []


def get_user_transactions(username: str) -> list[dict[str, Any]]:
    """Retrieves trade transaction log for specified username."""
    username_clean = username.strip().lower()
    try:
        conn = get_db_connection()
        cursor = conn.execute(
            "SELECT ticker, action, shares, price, timestamp FROM transactions WHERE username = ? ORDER BY id DESC",
            (username_clean,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "ticker": row[0],
                "action": row[1],
                "shares": float(row[2]),
                "price": float(row[3]),
                "timestamp": row[4],
            }
            for row in rows
        ]
    except Exception:
        logger.exception("Error getting user transactions", username=username_clean)
        return []


def get_daily_snapshots(username: str) -> list[dict[str, Any]]:
    """Fetches historical daily snapshots for username."""
    username_clean = username.strip().lower()
    try:
        conn = get_db_connection()
        cursor = conn.execute(
            "SELECT date, total_value, cash, holdings_value, daily_pnl FROM daily_snapshots WHERE username = ? ORDER BY date ASC",
            (username_clean,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "date": row[0],
                "total_value": float(row[1]),
                "cash": float(row[2]),
                "holdings_value": float(row[3]),
                "daily_pnl": float(row[4]),
            }
            for row in rows
        ]
    except Exception:
        logger.exception("Error getting daily snapshots", username=username_clean)
        return []


def reset_user_portfolio(username: str) -> bool:
    """Resets user portfolio back to 10 Lakh default balance, deleting all asset holdings and transaction logs."""
    username_clean = username.strip().lower()
    try:
        conn = get_db_connection()
        with conn:
            conn.execute(
                "UPDATE users SET cash_balance = 1000000.0 WHERE username = ?",
                (username_clean,),
            )
            conn.execute("DELETE FROM holdings WHERE username = ?", (username_clean,))
            conn.execute("DELETE FROM transactions WHERE username = ?", (username_clean,))
            conn.execute("DELETE FROM daily_snapshots WHERE username = ?", (username_clean,))
        conn.close()
        logger.info("User portfolio reset successfully", username=username_clean)
        return True
    except Exception:
        logger.exception("Error resetting user portfolio", username=username_clean)
        return False
