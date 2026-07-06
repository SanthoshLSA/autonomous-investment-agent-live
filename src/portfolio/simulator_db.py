"""
Database Manager supporting both SQLite (local) and MongoDB (cloud persistence) for the Paper Trading Simulator.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from pymongo import MongoClient

from src.logger import get_logger

logger = get_logger(__name__)

# Detect MongoDB URI from environment or Streamlit Secrets
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            MONGO_URI = st.secrets.get("MONGO_URI")
    except Exception:
        pass


def is_mongo_active() -> bool:
    """Returns True if MongoDB should be used instead of SQLite."""
    return MONGO_URI is not None


def get_mongo_db() -> Any:
    """Returns the MongoDB database object if MONGO_URI is configured."""
    if MONGO_URI:
        try:
            client = MongoClient(MONGO_URI)
            # Default database name is 'portfolio_db'
            return client.get_database("portfolio_db")
        except Exception:
            logger.exception("Failed to connect to MongoDB")
    return None


def get_db_connection() -> sqlite3.Connection:
    """Establishes connection to the SQLite database file, enabling foreign key checks."""
    Path("data").mkdir(exist_ok=True)
    conn = sqlite3.connect("data/portfolio.db")
    # Enable foreign keys support in SQLite
    conn.pragma_update = lambda *args: None  # Dummy
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Initializes the database and creates required tables/indexes if they don't exist."""
    if is_mongo_active():
        logger.info("Initializing MongoDB Collections & Indexes...")
        try:
            db = get_mongo_db()
            if db is not None:
                # Ensure unique index on username
                db.users.create_index("username", unique=True)
                # Ensure composite unique index on holdings
                db.holdings.create_index([("username", 1), ("ticker", 1)], unique=True)
                # Ensure index on transaction logs
                db.transactions.create_index([("username", 1), ("timestamp", -1)])
                # Ensure composite unique index on daily snapshots
                db.daily_snapshots.create_index([("username", 1), ("date", 1)], unique=True)
                logger.info("MongoDB collections and indexes initialized successfully.")
            else:
                logger.error("Failed to initialize MongoDB: connection object is None")
        except Exception as e:
            logger.exception("MongoDB initialization failed")
            raise e
    else:
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
    if is_mongo_active():
        try:
            db = get_mongo_db()
            if db.users.find_one({"username": username_clean}):
                return False
            db.users.insert_one({
                "username": username_clean,
                "password_hash": pw_hash,
                "cash_balance": 1000000.0
            })
            logger.info("User registered successfully via Mongo", username=username_clean)
            return True
        except Exception:
            logger.exception("Error creating user in Mongo", username=username_clean)
            return False
    else:
        try:
            conn = get_db_connection()
            with conn:
                cursor = conn.execute(
                    "SELECT username FROM users WHERE username = ?", (username_clean,)
                )
                if cursor.fetchone():
                    conn.close()
                    return False

                conn.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username_clean, pw_hash),
                )
            conn.close()
            logger.info("User registered successfully via SQLite", username=username_clean)
            return True
        except Exception:
            logger.exception("Error creating user in SQLite", username=username_clean)
            return False


def verify_user(username: str, password: str) -> bool:
    """Checks if username and password hash match database records."""
    username_clean = username.strip().lower()
    if not username_clean or not password:
        return False

    pw_hash = hash_password(password)
    if is_mongo_active():
        try:
            db = get_mongo_db()
            user = db.users.find_one({"username": username_clean, "password_hash": pw_hash})
            return user is not None
        except Exception:
            logger.exception("Error verifying user in Mongo", username=username_clean)
            return False
    else:
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
            logger.exception("Error verifying user in SQLite", username=username_clean)
            return False


# ─── Cash & Position Handlers ──────────────────────────────────────────────────


def get_user_cash(username: str) -> float:
    """Retrieves current cash balance for specified username."""
    username_clean = username.strip().lower()
    if is_mongo_active():
        try:
            db = get_mongo_db()
            user = db.users.find_one({"username": username_clean})
            return float(user["cash_balance"]) if user else 0.0
        except Exception:
            logger.exception("Error getting user cash in Mongo", username=username_clean)
            return 0.0
    else:
        try:
            conn = get_db_connection()
            cursor = conn.execute(
                "SELECT cash_balance FROM users WHERE username = ?", (username_clean,)
            )
            row = cursor.fetchone()
            conn.close()
            return float(row[0]) if row else 0.0
        except Exception:
            logger.exception("Error getting user cash in SQLite", username=username_clean)
            return 0.0


def add_funds(username: str, amount: float) -> bool:
    """Adds play money funds to user's cash balance."""
    username_clean = username.strip().lower()
    if amount <= 0:
        return False

    if is_mongo_active():
        try:
            db = get_mongo_db()
            res = db.users.update_one(
                {"username": username_clean},
                {"$inc": {"cash_balance": amount}}
            )
            if res.matched_count == 0:
                return False
            # Log deposit
            db.transactions.insert_one({
                "username": username_clean,
                "ticker": "INR",
                "action": "DEPOSIT",
                "shares": 1.0,
                "price": amount,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            logger.info("Funds deposited successfully via Mongo", username=username_clean, amount=amount)
            return True
        except Exception:
            logger.exception("Error adding funds in Mongo", username=username_clean)
            return False
    else:
        try:
            conn = get_db_connection()
            with conn:
                conn.execute(
                    "UPDATE users SET cash_balance = cash_balance + ? WHERE username = ?",
                    (amount, username_clean),
                )
                conn.execute(
                    "INSERT INTO transactions (username, ticker, action, shares, price) VALUES (?, ?, ?, ?, ?)",
                    (username_clean, "INR", "DEPOSIT", 1.0, amount),
                )
            conn.close()
            logger.info("Funds deposited successfully via SQLite", username=username_clean, amount=amount)
            return True
        except Exception:
            logger.exception("Error adding funds in SQLite", username=username_clean)
            return False


# ─── Trade Handler ────────────────────────────────────────────────────────────


def execute_trade(username: str, ticker: str, action: str, shares: float, price: float) -> str:
    """Executes paper trade transaction, adjusts cash and holdings records."""
    username_clean = username.strip().lower()
    ticker_clean = ticker.strip().upper()
    action_clean = action.strip().upper()

    if shares <= 0 or price <= 0:
        return "Quantity and Price must be positive values."

    if action_clean not in ["BUY", "SELL"]:
        return f"Invalid action: {action_clean}"

    total_cost = shares * price

    if is_mongo_active():
        try:
            db = get_mongo_db()
            # 1. Fetch current cash balance
            user = db.users.find_one({"username": username_clean})
            if not user:
                return "User record not found."
            cash = float(user["cash_balance"])

            # 2. Fetch current holdings status
            holding = db.holdings.find_one({"username": username_clean, "ticker": ticker_clean})
            held_shares = float(holding["shares"]) if holding else 0.0
            held_avg_cost = float(holding["avg_cost"]) if holding else 0.0

            if action_clean == "BUY":
                if cash < total_cost:
                    return f"Insufficient cash. Required: {total_cost:,.2f} INR. Available: {cash:,.2f} INR."

                # Update cash
                db.users.update_one({"username": username_clean}, {"$inc": {"cash_balance": -total_cost}})

                # Update holdings
                if held_shares > 0:
                    new_shares = held_shares + shares
                    new_avg = (held_shares * held_avg_cost + total_cost) / new_shares
                    db.holdings.update_one(
                        {"username": username_clean, "ticker": ticker_clean},
                        {"$set": {"shares": new_shares, "avg_cost": new_avg}}
                    )
                else:
                    db.holdings.insert_one({
                        "username": username_clean,
                        "ticker": ticker_clean,
                        "shares": shares,
                        "avg_cost": price
                    })

            elif action_clean == "SELL":
                if held_shares < shares:
                    return f"Insufficient shares. Trying to sell: {shares}. Owned: {held_shares}."

                # Update cash
                db.users.update_one({"username": username_clean}, {"$inc": {"cash_balance": total_cost}})

                # Update holdings
                new_shares = held_shares - shares
                if new_shares > 0:
                    db.holdings.update_one(
                        {"username": username_clean, "ticker": ticker_clean},
                        {"$set": {"shares": new_shares}}
                    )
                else:
                    db.holdings.delete_one({"username": username_clean, "ticker": ticker_clean})

            # 3. Log Transaction
            db.transactions.insert_one({
                "username": username_clean,
                "ticker": ticker_clean,
                "action": action_clean,
                "shares": shares,
                "price": price,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            logger.info(
                "Trade executed successfully via Mongo",
                username=username_clean,
                ticker=ticker_clean,
                action=action_clean,
                shares=shares,
            )
            return "SUCCESS"
        except Exception as e:
            logger.exception("Error executing trade in Mongo", username=username_clean)
            return f"Database Error: {str(e)}"
    else:
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
                "Trade executed successfully via SQLite",
                username=username_clean,
                ticker=ticker_clean,
                action=action_clean,
                shares=shares,
            )
            return "SUCCESS"
        except Exception as e:
            logger.exception("Error executing trade in SQLite", username=username_clean)
            return f"Database Error: {str(e)}"


# ─── Query Retrieval Handlers ──────────────────────────────────────────────────


def get_user_holdings(username: str) -> list[dict[str, Any]]:
    """Fetches list of holdings for specified username."""
    username_clean = username.strip().lower()
    if is_mongo_active():
        try:
            db = get_mongo_db()
            cursor = db.holdings.find({"username": username_clean})
            return [
                {"ticker": h["ticker"], "shares": float(h["shares"]), "avg_cost": float(h["avg_cost"])}
                for h in cursor
            ]
        except Exception:
            logger.exception("Error getting user holdings in Mongo", username=username_clean)
            return []
    else:
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
            logger.exception("Error getting user holdings in SQLite", username=username_clean)
            return []


def get_user_transactions(username: str) -> list[dict[str, Any]]:
    """Retrieves trade transaction log for specified username."""
    username_clean = username.strip().lower()
    if is_mongo_active():
        try:
            db = get_mongo_db()
            cursor = db.transactions.find({"username": username_clean}).sort("timestamp", -1)
            return [
                {
                    "ticker": t["ticker"],
                    "action": t["action"],
                    "shares": float(t["shares"]),
                    "price": float(t["price"]),
                    "timestamp": t["timestamp"],
                }
                for t in cursor
            ]
        except Exception:
            logger.exception("Error getting user transactions in Mongo", username=username_clean)
            return []
    else:
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
            logger.exception("Error getting user transactions in SQLite", username=username_clean)
            return []


def get_daily_snapshots(username: str) -> list[dict[str, Any]]:
    """Fetches historical daily snapshots for username."""
    username_clean = username.strip().lower()
    if is_mongo_active():
        try:
            db = get_mongo_db()
            cursor = db.daily_snapshots.find({"username": username_clean}).sort("date", 1)
            return [
                {
                    "date": s["date"],
                    "total_value": float(s["total_value"]),
                    "cash": float(s["cash"]),
                    "holdings_value": float(s["holdings_value"]),
                    "daily_pnl": float(s["daily_pnl"]),
                }
                for s in cursor
            ]
        except Exception:
            logger.exception("Error getting daily snapshots in Mongo", username=username_clean)
            return []
    else:
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
            logger.exception("Error getting daily snapshots in SQLite", username=username_clean)
            return []


def reset_user_portfolio(username: str) -> bool:
    """Resets user portfolio back to 10 Lakh default balance, deleting all holdings and transactions."""
    username_clean = username.strip().lower()
    if is_mongo_active():
        try:
            db = get_mongo_db()
            db.users.update_one({"username": username_clean}, {"$set": {"cash_balance": 1000000.0}})
            db.holdings.delete_many({"username": username_clean})
            db.transactions.delete_many({"username": username_clean})
            db.daily_snapshots.delete_many({"username": username_clean})
            logger.info("User portfolio reset successfully via Mongo", username=username_clean)
            return True
        except Exception:
            logger.exception("Error resetting user portfolio in Mongo", username=username_clean)
            return False
    else:
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
            logger.info("User portfolio reset successfully via SQLite", username=username_clean)
            return True
        except Exception:
            logger.exception("Error resetting user portfolio in SQLite", username=username_clean)
            return False


# ─── Snapshot Refactored Helpers ──────────────────────────────────────────────


def get_snapshot_by_date(username: str, date_str: str) -> dict[str, Any] | None:
    """Fetches daily snapshot for username by date."""
    username_clean = username.strip().lower()
    if is_mongo_active():
        try:
            db = get_mongo_db()
            snap = db.daily_snapshots.find_one({"username": username_clean, "date": date_str})
            if snap:
                return {
                    "total_value": float(snap["total_value"]),
                    "cash": float(snap["cash"]),
                    "holdings_value": float(snap["holdings_value"]),
                    "daily_pnl": float(snap["daily_pnl"]),
                }
            return None
        except Exception:
            logger.exception("Error getting snapshot by date in Mongo", username=username_clean)
            return None
    else:
        try:
            conn = get_db_connection()
            cursor = conn.execute(
                "SELECT total_value, cash, holdings_value, daily_pnl FROM daily_snapshots WHERE username = ? AND date = ?",
                (username_clean, date_str),
            )
            row = cursor.fetchone()
            conn.close()
            if row:
                return {
                    "total_value": float(row[0]),
                    "cash": float(row[1]),
                    "holdings_value": float(row[2]),
                    "daily_pnl": float(row[3]),
                }
            return None
        except Exception:
            logger.exception("Error getting snapshot by date in SQLite", username=username_clean)
            return None


def get_most_recent_snapshot(username: str) -> dict[str, Any] | None:
    """Fetches the most recent daily snapshot for username."""
    username_clean = username.strip().lower()
    if is_mongo_active():
        try:
            db = get_mongo_db()
            snap = db.daily_snapshots.find_one({"username": username_clean}, sort=[("date", -1)])
            if snap:
                return {
                    "total_value": float(snap["total_value"]),
                    "cash": float(snap["cash"]),
                    "holdings_value": float(snap["holdings_value"]),
                    "daily_pnl": float(snap["daily_pnl"]),
                }
            return None
        except Exception:
            logger.exception("Error getting most recent snapshot in Mongo", username=username_clean)
            return None
    else:
        try:
            conn = get_db_connection()
            cursor = conn.execute(
                "SELECT total_value, cash, holdings_value, daily_pnl FROM daily_snapshots WHERE username = ? ORDER BY date DESC LIMIT 1",
                (username_clean,),
            )
            row = cursor.fetchone()
            conn.close()
            if row:
                return {
                    "total_value": float(row[0]),
                    "cash": float(row[1]),
                    "holdings_value": float(row[2]),
                    "daily_pnl": float(row[3]),
                }
            return None
        except Exception:
            logger.exception("Error getting most recent snapshot in SQLite", username=username_clean)
            return None


def upsert_daily_snapshot(
    username: str,
    date_str: str,
    total_value: float,
    cash: float,
    holdings_value: float,
    daily_pnl: float,
) -> bool:
    """Inserts or updates today's daily snapshot for username."""
    username_clean = username.strip().lower()
    if is_mongo_active():
        try:
            db = get_mongo_db()
            db.daily_snapshots.update_one(
                {"username": username_clean, "date": date_str},
                {
                    "$set": {
                        "total_value": total_value,
                        "cash": cash,
                        "holdings_value": holdings_value,
                        "daily_pnl": daily_pnl,
                    }
                },
                upsert=True,
            )
            return True
        except Exception:
            logger.exception("Error upserting daily snapshot in Mongo", username=username_clean)
            return False
    else:
        try:
            conn = get_db_connection()
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO daily_snapshots (username, date, total_value, cash, holdings_value, daily_pnl)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (username_clean, date_str, total_value, cash, holdings_value, daily_pnl),
                )
            conn.close()
            return True
        except Exception:
            logger.exception("Error upserting daily snapshot in SQLite", username=username_clean)
            return False
