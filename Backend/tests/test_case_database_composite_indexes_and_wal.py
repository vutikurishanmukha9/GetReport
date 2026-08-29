import sqlite3
import pytest
from app.db import get_db_connection, init_db

def test_database_composite_indexes_and_wal():
    """
    Verify Phase 2 database performance hardening:
    1. Composite indexes are created and active.
    2. Query planner utilizes indexes for polling and historical lookups.
    3. WAL mode and tuned PRAGMAs are properly set.
    """
    init_db()
    
    with get_db_connection() as conn:
        # 1. Verify indexes exist
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index';")
        indexes = {row["name"] for row in cursor.fetchall()}
        
        assert "idx_jobs_status_created" in indexes
        assert "idx_jobs_filename_status" in indexes
        assert "idx_jobs_hash_batch" in indexes
        assert "idx_jobs_report_status" in indexes
        
        # 2. Verify EXPLAIN QUERY PLAN uses index on polling query
        explain_cursor = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC", 
            ("COMPLETED",)
        )
        plan_rows = explain_cursor.fetchall()
        plan_detail = " ".join(str(r["detail"]) for r in plan_rows)
        assert "idx_jobs_status_created" in plan_detail or "USING INDEX" in plan_detail
        
        # 3. Verify EXPLAIN QUERY PLAN uses index on filename + status lookup
        explain_cursor_2 = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM jobs WHERE filename = ? AND status = ? ORDER BY created_at DESC",
            ("dataset.csv", "COMPLETED")
        )
        plan_rows_2 = explain_cursor_2.fetchall()
        plan_detail_2 = " ".join(str(r["detail"]) for r in plan_rows_2)
        assert "idx_jobs_filename_status" in plan_detail_2 or "USING INDEX" in plan_detail_2
        
        # 4. Verify WAL Mode
        mode_cursor = conn.execute("PRAGMA journal_mode;")
        mode = mode_cursor.fetchone()[0].upper()
        assert mode == "WAL"
