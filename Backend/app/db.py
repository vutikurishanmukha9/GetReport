import sqlite3
import json
import logging
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = "tasks.db"

def init_db():
    """
    Initialize the SQLite database with the required schema.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Jobs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        task_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        filename TEXT,
        message TEXT,
        progress INTEGER DEFAULT 0,
        result_json TEXT, -- JSON blob of analysis results
        error TEXT,
        report_path TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")

@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False) # SQLite objects created in a thread can only be used in that thread usually, but check_same_thread=False allows sharing if careful.
    # Note: For high concurrency, use aiosqlite or connection pool, but for this scale sync sqlite is fine.
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
