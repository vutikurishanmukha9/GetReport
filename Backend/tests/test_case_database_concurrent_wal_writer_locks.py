import concurrent.futures
import uuid
import pytest
from app.db import get_db_connection, init_db

def _insert_task(task_id: str, filename: str) -> bool:
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO jobs (task_id, status, filename, progress) VALUES (?, ?, ?, ?)",
            (task_id, "PENDING", filename, 0)
        )
        conn.commit()
    return True

def _update_task(task_id: str, progress: int) -> bool:
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE jobs SET progress = ?, status = ? WHERE task_id = ?",
            (progress, "PROCESSING", task_id)
        )
        conn.commit()
    return True

def test_database_concurrent_wal_writer_locks():
    """
    Verify concurrent multi-threaded writes execute smoothly under SQLite WAL mode
    without raising database locking / busy errors.
    """
    init_db()
    
    task_ids = [f"task_conc_{uuid.uuid4().hex[:8]}" for _ in range(30)]
    
    # 1. Concurrent Inserts across 8 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_insert_task, tid, f"file_{tid}.csv") for tid in task_ids]
        for f in concurrent.futures.as_completed(futures):
            assert f.result() is True
            
    # 2. Concurrent Updates across 8 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        update_futures = [executor.submit(_update_task, tid, 50) for tid in task_ids]
        for f in concurrent.futures.as_completed(update_futures):
            assert f.result() is True
            
    # 3. Verify all records updated
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT count(*) FROM jobs WHERE progress = 50")
        count = cursor.fetchone()[0]
        assert count >= len(task_ids)
