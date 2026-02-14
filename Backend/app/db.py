import sqlite3
import json
import logging
import re
import os
from datetime import datetime
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)

DB_PATH = "tasks.db"

# ─── PostgreSQL Wrapper ──────────────────────────────────────────────────────
class PostgresCursor:
    """
    Wraps psycopg2 cursor to support SQLite-style '?' placeholders
    by converting them to '%s' on the fly.
    """
    def __init__(self, cursor):
        self.cursor = cursor

    def execute(self, sql: str, params: Tuple = ()) -> Any:
        # Convert ? to %s for Postgres
        # We use a simple replace because ? is standard SQL placeholder
        # but we must be careful about ? inside strings. 
        # For this specific app, we know query structure is simple.
        pg_sql = sql.replace("?", "%s")
        return self.cursor.execute(pg_sql, params)

    def fetchone(self) -> Optional[Any]:
        return self.cursor.fetchone()

    def fetchall(self) -> List[Any]:
        return self.cursor.fetchall()
    
    def close(self):
        self.cursor.close()
        
    def __getattr__(self, name):
        return getattr(self.cursor, name)

class PostgresConnection:
    """
    Wraps psycopg2 connection to mimic sqlite3 connection behavior
    expected by the application (row_factory access, execute shortcut).
    """
    def __init__(self, conn):
        self.conn = conn
        self.row_factory = None # Psycopg2 uses DictCursor for this
        
    def cursor(self):
        return PostgresCursor(self.conn.cursor())

    def execute(self, sql: str, params: Tuple = ()) -> PostgresCursor:
        cursor = self.cursor()
        cursor.execute(sql, params)
        return cursor

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()
        
    def __getattr__(self, name):
        return getattr(self.conn, name)

# ─── Database Initialization ─────────────────────────────────────────────────
def init_db():
    """
    Initialize the database (SQLite or Postgres).
    """
    if settings.DATABASE_URL:
        _init_postgres()
    else:
        _init_sqlite()

def _init_sqlite():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    _create_schema(cursor)
    conn.commit()
    conn.close()
    logger.info(f"SQLite Database initialized at {DB_PATH}")

def _init_postgres():
    try:
        import psycopg2
        conn = psycopg2.connect(settings.DATABASE_URL)
        cursor = conn.cursor()
        _create_schema_postgres(cursor)
        conn.commit()
        conn.close()
        logger.info("PostgreSQL Database initialized.")
    except Exception as e:
        logger.error(f"PostgreSQL Init Failed: {e}")
        # Fallback? No, fail hard if config is wrong.
        raise e

def _create_schema(cursor):
    """SQLite Schema"""
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        task_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        filename TEXT,
        message TEXT,
        progress INTEGER DEFAULT 0,
        result_path TEXT,
        error TEXT,
        report_path TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    # Migration: add result_path if table already exists without it
    try:
        cursor.execute("ALTER TABLE jobs ADD COLUMN result_path TEXT")
    except Exception:
        pass  # Column already exists
    
    # Migration: Drop result_json if it exists (Optional cleanup)
    # SQLite < 3.35 doesn't support DROP COLUMN, so we skip it to be safe
    # or just leave it as 'dead' data.

def _create_schema_postgres(cursor):
    """Postgres Schema"""
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        task_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        filename TEXT,
        message TEXT,
        progress INTEGER DEFAULT 0,
        result_path TEXT,
        error TEXT,
        report_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    # Migration: add result_path if table already exists without it
    try:
        cursor.execute("ALTER TABLE jobs ADD COLUMN result_path TEXT")
    except Exception:
        pass  # Column already exists

    # ─── Vector Store (pgvector) ─────────────────────────────────────────────
    try:
        # 1. Enable Extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # 2. Create Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                task_id TEXT NOT NULL,
                content TEXT,
                chunk_index INTEGER,
                metadata JSONB,
                embedding vector(1536),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 3. Create Index
        try:
             cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
                ON document_chunks USING hnsw (embedding vector_cosine_ops)
            """)
        except Exception as e:
            logger.warning(f"Vector Index Creation Failed (Non-critical): {e}")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_task_id ON document_chunks(task_id)")

    except Exception as e:
        logger.warning(f"Vector Store Init Failed (Is 'vector' extension installed?): {e}")

# ─── Connection Factory ──────────────────────────────────────────────────────
@contextmanager
def get_db_connection():
    """
    Yields a database connection (SQLite or Postgres) based on config.
    """
    if settings.DATABASE_URL:
        # PostgreSQL
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            conn = psycopg2.connect(settings.DATABASE_URL, cursor_factory=RealDictCursor)
            pg_conn = PostgresConnection(conn)
            try:
                yield pg_conn
            finally:
                pg_conn.close()
        except ImportError:
            logger.error("psycopg2 not installed. Install via 'pip install psycopg2-binary'")
            raise
    else:
        # SQLite
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
