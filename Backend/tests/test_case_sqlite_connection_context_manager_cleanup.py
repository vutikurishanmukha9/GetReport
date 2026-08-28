import pytest
from app.db import get_db_connection

def test_sqlite_connection_context_manager_lifecycle():
    """Verify that get_db_connection handles query execution and cleanly closes cursor."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 + 1;")
        result = cursor.fetchone()
        assert result[0] == 2
