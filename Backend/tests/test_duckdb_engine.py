"""
tests/test_duckdb_engine.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive unit tests for the DuckDBAnalyticalSession engine.
Verifies analytical query capabilities, zero-copy registrations, and security guards.
"""
import pytest
import tempfile
import os
import polars as pl
from app.services.duckdb_engine import DuckDBAnalyticalSession, DuckDBSecurityError


@pytest.fixture
def sample_polars_df():
    return pl.DataFrame({
        "employee_id": [101, 102, 103, 104, 105],
        "department": ["Engineering", "Engineering", "Sales", "Sales", "HR"],
        "salary": [120000.0, 135000.0, 95000.0, 110000.0, 85000.0],
        "rating": [4.8, 4.2, 3.9, 4.5, 4.0]
    })


def test_duckdb_session_initialization():
    session = DuckDBAnalyticalSession(memory_limit="512MB", threads=2)
    tables = session.list_tables()
    assert isinstance(tables, list)
    session.close()


def test_duckdb_polars_registration_and_query(sample_polars_df):
    session = DuckDBAnalyticalSession()
    session.register_polars("employees", sample_polars_df)
    
    assert "employees" in session.list_tables()
    
    # Simple query
    results = session.execute_read_query("SELECT COUNT(*) as cnt FROM employees")
    assert len(results) == 1
    assert results[0]["cnt"] == 5
    
    # Aggregation with group by
    dept_results = session.execute_read_query("""
        SELECT department, AVG(salary) as avg_sal, COUNT(*) as headcount
        FROM employees
        GROUP BY department
        ORDER BY avg_sal DESC
    """)
    assert len(dept_results) == 3
    assert dept_results[0]["department"] == "Engineering"
    assert dept_results[0]["headcount"] == 2
    assert dept_results[0]["avg_sal"] == 127500.0

    session.close()


def test_duckdb_parquet_registration(sample_polars_df):
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sample_polars_df.write_parquet(tmp_path)
        
        session = DuckDBAnalyticalSession()
        session.register_parquet("parquet_staff", tmp_path)
        
        results = session.execute_read_query(
            "SELECT employee_id, salary FROM parquet_staff WHERE salary > 100000 ORDER BY salary ASC"
        )
        assert len(results) == 3
        assert results[0]["employee_id"] == 104
        assert results[-1]["employee_id"] == 102
        session.close()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_duckdb_column_profile(sample_polars_df):
    session = DuckDBAnalyticalSession()
    session.register_polars("staff", sample_polars_df)
    
    profile = session.get_column_profile("staff", "salary")
    assert profile["total_records"] == 5
    assert profile["non_null_count"] == 5
    assert profile["null_count"] == 0
    assert profile["min_value"] == 85000.0
    assert profile["max_value"] == 135000.0
    assert profile["mean_value"] == 109000.0
    
    session.close()


def test_duckdb_security_blocks_mutation(sample_polars_df):
    session = DuckDBAnalyticalSession()
    session.register_polars("staff", sample_polars_df)
    
    # Block DROP
    with pytest.raises(DuckDBSecurityError):
        session.execute_read_query("DROP TABLE staff")

    # Block DELETE
    with pytest.raises(DuckDBSecurityError):
        session.execute_read_query("DELETE FROM staff WHERE salary > 100000")

    # Block UPDATE
    with pytest.raises(DuckDBSecurityError):
        session.execute_read_query("UPDATE staff SET salary = 999999")

    # Block INSERT
    with pytest.raises(DuckDBSecurityError):
        session.execute_read_query("INSERT INTO staff VALUES (999, 'Bad', 0, 0)")

    # Block ALTER
    with pytest.raises(DuckDBSecurityError):
        session.execute_read_query("ALTER TABLE staff ADD COLUMN leaked INT")

    # Block statement chaining injection
    with pytest.raises(DuckDBSecurityError):
        session.execute_read_query("SELECT * FROM staff; DROP TABLE staff;")

    session.close()
