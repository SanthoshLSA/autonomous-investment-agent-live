"""
SQLite-based data cache for market data.

Provides a thread-safe, TTL-aware caching layer that sits between
the data sources and the rest of the pipeline. Cached data is stored
as serialised JSON keyed by (ticker, data_type). Each entry carries
a ``fetched_at`` epoch timestamp used for freshness checks.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from src.logger import get_logger

__all__ = ["DataCache"]

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SQLite Data Cache
# ═══════════════════════════════════════════════════════════════════════════════

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cache (
    ticker     TEXT NOT NULL,
    data_type  TEXT NOT NULL,
    data_json  TEXT NOT NULL,
    fetched_at REAL NOT NULL,
    PRIMARY KEY (ticker, data_type)
)
"""

_UPSERT_SQL = """
INSERT INTO cache (ticker, data_type, data_json, fetched_at)
VALUES (?, ?, ?, ?)
ON CONFLICT(ticker, data_type)
DO UPDATE SET data_json = excluded.data_json,
              fetched_at = excluded.fetched_at
"""

_SELECT_SQL = """
SELECT data_json, fetched_at FROM cache
WHERE ticker = ? AND data_type = ?
"""

_DELETE_EXPIRED_SQL = """
DELETE FROM cache WHERE fetched_at < ?
"""

_DELETE_ALL_SQL = "DELETE FROM cache"


class DataCache:
    """Thread-safe SQLite cache with configurable TTL.

    Each thread gets its own ``sqlite3.Connection`` via ``threading.local()``
    to avoid SQLite's thread-affinity restrictions.

    Args:
        db_path: Filesystem path for the SQLite database file.
                 Parent directories are created automatically.
        default_ttl_hours: Default time-to-live for cached entries (hours).
    """

    def __init__(self, db_path: str, default_ttl_hours: float = 4.0) -> None:
        self._db_path = db_path
        self._default_ttl_hours = default_ttl_hours
        self._local = threading.local()

        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialise table on the calling thread's connection
        self._get_connection()

        logger.info(
            "cache_initialised",
            db_path=db_path,
            default_ttl_hours=default_ttl_hours,
        )

    # ── Connection management ─────────────────────────────────────────────

    def _get_connection(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection, creating if needed.

        Returns:
            sqlite3.Connection bound to the current thread.
        """
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()
            self._local.conn = conn
        return conn

    # ── Public API ────────────────────────────────────────────────────────

    def get(
        self,
        ticker: str,
        data_type: str,
        max_age_hours: float | None = None,
    ) -> str | None:
        """Retrieve a cached JSON string if it exists and is fresh.

        Args:
            ticker: Asset ticker symbol.
            data_type: Category key (e.g. ``"prices"``, ``"info"``, ``"news"``).
            max_age_hours: Override default TTL. ``None`` uses instance default.

        Returns:
            The cached JSON string, or ``None`` on a miss or stale entry.
        """
        ttl = max_age_hours if max_age_hours is not None else self._default_ttl_hours
        conn = self._get_connection()
        cursor = conn.execute(_SELECT_SQL, (ticker, data_type))
        row = cursor.fetchone()

        if row is None:
            logger.debug("cache_miss", ticker=ticker, data_type=data_type, reason="not_found")
            return None

        data_json, fetched_at = row
        age_hours = (time.time() - fetched_at) / 3600.0

        if age_hours > ttl:
            logger.debug(
                "cache_miss",
                ticker=ticker,
                data_type=data_type,
                reason="expired",
                age_hours=round(age_hours, 2),
                ttl_hours=ttl,
            )
            return None

        logger.debug(
            "cache_hit",
            ticker=ticker,
            data_type=data_type,
            age_hours=round(age_hours, 2),
        )
        return data_json

    def set(self, ticker: str, data_type: str, data_json: str) -> None:
        """Store (upsert) a JSON string in the cache.

        Args:
            ticker: Asset ticker symbol.
            data_type: Category key (e.g. ``"prices"``, ``"info"``, ``"news"``).
            data_json: Serialised JSON payload to cache.
        """
        conn = self._get_connection()
        conn.execute(_UPSERT_SQL, (ticker, data_type, data_json, time.time()))
        conn.commit()

        logger.debug("cache_set", ticker=ticker, data_type=data_type)

    def clear_expired(self, max_age_hours: float | None = None) -> int:
        """Remove all entries older than the TTL.

        Args:
            max_age_hours: Override default TTL. ``None`` uses instance default.

        Returns:
            Number of rows deleted.
        """
        ttl = max_age_hours if max_age_hours is not None else self._default_ttl_hours
        cutoff = time.time() - (ttl * 3600.0)
        conn = self._get_connection()
        cursor = conn.execute(_DELETE_EXPIRED_SQL, (cutoff,))
        conn.commit()
        deleted = cursor.rowcount

        logger.info("cache_cleared_expired", deleted=deleted, ttl_hours=ttl)
        return deleted

    def clear_all(self) -> None:
        """Delete every entry in the cache."""
        conn = self._get_connection()
        conn.execute(_DELETE_ALL_SQL)
        conn.commit()
        logger.info("cache_cleared_all")

    def close(self) -> None:
        """Close the current thread's connection if open."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
            logger.debug("cache_connection_closed")
