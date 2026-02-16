import sqlite3
import json
import logging
import re
import os
import asyncio
from datetime import datetime
from contextlib import contextmanager, asynccontextmanager
from typing import Any, List, Optional, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)

DB_PATH = "tasks.db"

# Global Connection Pools
pg_pool = None
async_pg_pool = None

# ─── SYNC PostgreSQL Wrapper (Legacy/Celery) ─────────────────────────────────
class PostgresCursor:
    """Wraps psycopg2 cursor for Sync contexts"""
    def __init__(self, cursor):
        self.cursor = cursor

    def execute(self, sql: str, params: Tuple = ()) -> Any:
        def replace_placeholder(match):
            if match.group(1): return match.group(1)
            return "%s"
        pattern = r"(\'[^\']*\'|\"[^\"]*\")|\?"
        pg_sql = re.sub(pattern, replace_placeholder, sql)
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
    """Wraps psycopg2 connection for Sync contexts"""
    def __init__(self, conn, pool=None):
        self.conn = conn
        self.pool = pool
        self.row_factory = None
        
    def cursor(self):
        return PostgresCursor(self.conn.cursor())

    def execute(self, sql: str, params: Tuple = ()) -> PostgresCursor:
        cursor = self.cursor()
        cursor.execute(sql, params)
        return cursor

    def commit(self):
        self.conn.commit()

    def close(self):
        if self.pool:
            self.pool.putconn(self.conn)
        else:
            self.conn.close()
        
    def __getattr__(self, name):
        return getattr(self.conn, name)

# ─── ASYNC PostgreSQL Wrapper (FastAPI) ──────────────────────────────────────
class AsyncPostgresCursor:
    """Wraps asyncpg connection to support '?' placeholders"""
    def __init__(self, conn):
        self.conn = conn
        self._last_result = None

    async def execute(self, sql: str, params: Tuple = ()) -> Any:
        # asyncpg uses $1, $2, $3. We must convert ? -> $n
        # Regex to ignore strings matches
        params = list(params) # Tuple to list
        
        counter = 0
        def replace_placeholder(match):
            nonlocal counter
            if match.group(1): return match.group(1)
            counter += 1
            return f"${counter}"

        pattern = r"(\'[^\']*\'|\"[^\"]*\")|\?"
        pg_sql = re.sub(pattern, replace_placeholder, sql)
        
        self._last_result = await self.conn.fetch(pg_sql, *params)
        return self

    async def fetchone(self) -> Optional[Any]:
        if self._last_result and len(self._last_result) > 0:
            return self._last_result[0]
        return None

    async def fetchall(self) -> List[Any]:
        return self._last_result or []
    
    async def close(self):
        pass # Nothing to close for asyncpg cursor-less execution

class AsyncPostgresConnection:
    """Wraps asyncpg pool connection"""
    def __init__(self, conn, pool=None):
        self.conn = conn
        self.pool = pool
        self.row_factory = None # Not used directly, asyncpg returns Record objects
        
    def cursor(self):
        # asyncpg doesn't strictly have cursors in the same way, 
        # but we return a wrapper that can 'execute'
        return AsyncPostgresCursor(self.conn)

    async def execute(self, sql: str, params: Tuple = ()) -> AsyncPostgresCursor:
        cursor = self.cursor()
        await cursor.execute(sql, params)
        return cursor

    async def commit(self):
        pass # Auto-commit is default in asyncpg usually unless in transaction

    async def close(self):
        # Released via context manager usually, but if manual:
        if self.pool:
            await self.pool.release(self.conn)
        else:
            await self.conn.close()

# ─── ASYNC SQLite Wrapper (FastAPI) ──────────────────────────────────────────
class AsyncSqliteConnection:
    """Wraps aiosqlite connection"""
    def __init__(self, conn):
        self.conn = conn
        self.row_factory = None
        
    async def execute(self, sql: str, params: Tuple = ()):
        return await self.conn.execute(sql, params)
        
    async def commit(self):
        await self.conn.commit()
        
    async def close(self):
        await self.conn.close()

# ─── Initialization ──────────────────────────────────────────────────────────
def init_db():
    """Sync Initialization (Schema Creation) - Runs on Startup"""
    if settings.DATABASE_URL:
        _init_postgres_sync()
    else:
        _init_sqlite_sync()

async def init_async_db():
    """Async Initialization (Pool Creation)"""
    if settings.DATABASE_URL:
        global async_pg_pool
        import asyncpg
        if not async_pg_pool:
            async_pg_pool = await asyncpg.create_pool(
                dsn=settings.DATABASE_URL, 
                min_size=1, 
                max_size=20
            )
            logger.info("Async PostgreSQL Pool initialized.")

def close_db():
    """Sync Cleanup"""
    global pg_pool
    if pg_pool:
        pg_pool.closeall()
        logger.info("Sync PostgreSQL Pool closed.")

async def close_async_db():
    """Async Cleanup"""
    global async_pg_pool
    if async_pg_pool:
        await async_pg_pool.close()
        logger.info("Async PostgreSQL Pool closed.")

# ─── Sync Implementation Details ─────────────────────────────────────────────
def _init_sqlite_sync():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    _create_schema(cursor)
    conn.commit()
    conn.close()

def _init_postgres_sync():
    global pg_pool
    try:
        import psycopg2
        from psycopg2 import pool
        from psycopg2.extras import RealDictCursor
        
        if not pg_pool:
            pg_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1, maxconn=20,
                dsn=settings.DATABASE_URL,
                cursor_factory=RealDictCursor
            )
            
        conn = pg_pool.getconn()
        try:
            cursor = conn.cursor()
            _create_core_tables(cursor)
            conn.commit()
            try:
                _enable_vector_extension(cursor)
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.warning(f"Vector extension init failed: {e}")
        finally:
            pg_pool.putconn(conn)
    except Exception as e:
        logger.error(f"Postgres Init Failed: {e}")
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
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        version INTEGER DEFAULT 0
    )
    """)
    # Migrations
    for col in ["result_path TEXT", "version INTEGER DEFAULT 0"]:
        try: cursor.execute(f"ALTER TABLE jobs ADD COLUMN {col}")
        except: pass

def _create_core_tables(cursor):
    """Postgres Core Schema"""
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
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        version INTEGER DEFAULT 0
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)")
    
    for col in ["result_path TEXT", "version INTEGER DEFAULT 0"]:
        try: cursor.execute(f"ALTER TABLE jobs ADD COLUMN {col}")
        except: pass

def _enable_vector_extension(cursor):
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
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
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops)")
    except: pass
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_task_id ON document_chunks(task_id)")

# ─── Context Factories ───────────────────────────────────────────────────────
@contextmanager
def get_db_connection():
    """Sync Connection (Celery/Legacy)"""
    if settings.DATABASE_URL:
        # Postgres Sync
        global pg_pool
        if not pg_pool: _init_postgres_sync()
        conn = pg_pool.getconn()
        pg_conn = PostgresConnection(conn, pg_pool)
        try: yield pg_conn
        finally: pg_conn.close()
    else:
        # SQLite Sync
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try: yield conn
        finally: conn.close()

@asynccontextmanager
async def get_async_db_connection():
    """Async Connection (FastAPI)"""
    if settings.DATABASE_URL:
        # Postgres Async
        global async_pg_pool
        if not async_pg_pool: await init_async_db()
        async with async_pg_pool.acquire() as conn:
            yield AsyncPostgresConnection(conn, async_pg_pool)
    else:
        # SQLite Async
        import aiosqlite
        async with aiosqlite.connect(DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            yield AsyncSqliteConnection(conn)
