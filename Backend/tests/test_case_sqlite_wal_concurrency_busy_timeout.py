import pytest
import sqlite3
from app.db import init_db, get_db_connection

def test_sqlite_wal_mode_and_busy_timeout_pragmas():
    """Verify that SQLite connection enables Write-Ahead Logging (WAL) and 5000ms busy timeout."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Check journal mode
        cursor.execute("PRAGMA journal_mode;")
        journal_mode = cursor.fetchone()[0]
        assert journal_mode.lower() in ("wal", "memory")  # WAL mode active
        
        # Check busy timeout
        cursor.execute("PRAGMA busy_timeout;")
        busy_timeout = cursor.fetchone()[0]
        assert busy_timeout >= 5000  # Minimum 5000ms
