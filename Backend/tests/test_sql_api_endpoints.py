"""
tests/test_sql_api_endpoints.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration tests for:
- POST /api/jobs/{task_id}/query-sql
- GET /api/jobs/{task_id}/export-gx
"""
import pytest
import io
import polars as pl
from fastapi.testclient import TestClient
from app.main import app
from app.services.task_manager import title_task_manager, TaskStatus
from app.services.storage import get_storage_provider

client = TestClient(app)


@pytest.fixture
def setup_completed_job():
    storage = get_storage_provider()
    
    # Create sample dataframe
    df = pl.DataFrame({
        "product_id": [1, 2, 3, 4, 5],
        "category": ["Hardware", "Hardware", "Software", "Software", "Cloud"],
        "price": [150.0, 300.0, 45.0, 99.0, 1200.0]
    })
    
    # Create job in database
    task_id = title_task_manager.create_job("products.csv")
    
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    buffer.seek(0)
    
    cleaned_file_ref = storage.save_upload(buffer, f"cleaned_{task_id}.parquet")
    
    job_result = {
        "filename": "products.csv",
        "cleaned_file_ref": cleaned_file_ref,
        "info": {
            "rows": 5,
            "columns": 3,
            "grade": "A",
            "score": 96.0
        },
        "issue_ledger": {
            "issues": [
                {
                    "id": "iss_1",
                    "column": "price",
                    "issue_type": "missing_values",
                    "status": "approved"
                }
            ]
        }
    }
    
    title_task_manager.update_status(task_id, TaskStatus.COMPLETED, job_result)
    
    yield task_id, cleaned_file_ref
    
    # Teardown
    try:
        storage.delete(cleaned_file_ref)
    except Exception:
        pass


def test_query_sql_endpoint_success(setup_completed_job):
    task_id, _ = setup_completed_job
    
    payload = {
        "sql": "SELECT category, AVG(price) as avg_price, COUNT(*) as cnt FROM dataset GROUP BY category ORDER BY avg_price DESC"
    }
    
    response = client.post(f"/api/jobs/{task_id}/query-sql", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["task_id"] == task_id
    assert "records" in data
    assert len(data["records"]) == 3
    assert data["records"][0]["category"] == "Cloud"
    assert data["records"][0]["avg_price"] == 1200.0


def test_query_sql_security_blocked(setup_completed_job):
    task_id, _ = setup_completed_job
    
    payload = {
        "sql": "DROP TABLE dataset"
    }
    
    response = client.post(f"/api/jobs/{task_id}/query-sql", json=payload)
    assert response.status_code == 403


def test_export_gx_endpoint_success(setup_completed_job):
    task_id, _ = setup_completed_job
    
    response = client.get(f"/api/jobs/{task_id}/export-gx")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert "expectation_suite.json" in response.headers["content-disposition"]
    
    suite = response.json()
    assert suite["data_asset_type"] == "Dataset"
    assert "expectations" in suite
    assert len(suite["expectations"]) > 0
    assert suite["meta"]["quality_grade"] == "A"


def test_golden_query_crud_endpoints(setup_completed_job):
    task_id, _ = setup_completed_job

    # 1. Create golden query
    post_payload = {
        "question": "What is the average price of cloud products?",
        "sql_query": "SELECT AVG(price) as avg_price FROM dataset WHERE category = 'Cloud'",
        "description": "Cloud pricing benchmark",
    }
    res_post = client.post(f"/api/jobs/{task_id}/golden-queries", json=post_payload)
    assert res_post.status_code == 200
    created = res_post.json()
    assert created["task_id"] == task_id
    assert created["question"] == post_payload["question"]
    assert created["sql_query"] == post_payload["sql_query"]
    query_id = created["id"]

    # 2. List golden queries
    res_list = client.get(f"/api/jobs/{task_id}/golden-queries")
    assert res_list.status_code == 200
    queries = res_list.json()
    assert len(queries) == 1
    assert queries[0]["id"] == query_id

    # 3. Delete golden query
    res_del = client.delete(f"/api/jobs/{task_id}/golden-queries/{query_id}")
    assert res_del.status_code == 200
    assert res_del.json()["deleted"] is True

    # 4. Verify deletion
    res_list_after = client.get(f"/api/jobs/{task_id}/golden-queries")
    assert res_list_after.status_code == 200
    assert len(res_list_after.json()) == 0


def test_get_sql_schema_endpoint(setup_completed_job):
    task_id, _ = setup_completed_job

    response = client.get(f"/api/jobs/{task_id}/sql-schema")
    assert response.status_code == 200
    data = response.json()

    assert data["task_id"] == task_id
    assert data["success"] is True
    assert data["engine"] == "DuckDB-InProcess"
    assert data["total_tables"] == 1
    table_meta = data["tables"][0]
    assert table_meta["table_name"] == "dataset"
    assert table_meta["row_count"] == 5
    assert table_meta["column_count"] == 3
    col_names = [col["name"] for col in table_meta["columns"]]
    assert "product_id" in col_names
    assert "category" in col_names
    assert "price" in col_names


def test_query_sql_execution_ms_and_error_reporting(setup_completed_job):
    task_id, _ = setup_completed_job

    # Valid query returns execution_ms
    response = client.post(
        f"/api/jobs/{task_id}/query-sql",
        json={"sql": "SELECT * FROM dataset"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "execution_ms" in data
    assert isinstance(data["execution_ms"], (int, float))

    # Unknown column gives helpful Binder Error instead of vague error
    bad_res = client.post(
        f"/api/jobs/{task_id}/query-sql",
        json={"sql": "SELECT nonexistent_column FROM dataset"}
    )
    assert bad_res.status_code == 400
    detail = bad_res.json()["detail"]
    assert "DuckDB Query Error" in detail


def test_query_sql_capped_flag_and_ast_safety(setup_completed_job):
    task_id, _ = setup_completed_job

    # 1. Peek limit+1 detects true truncation cleanly (5 rows > limit 2)
    capped_res = client.post(
        f"/api/jobs/{task_id}/query-sql",
        json={"sql": "SELECT product_id, price FROM dataset ORDER BY product_id", "limit": 2},
    )
    assert capped_res.status_code == 200
    capped_data = capped_res.json()
    assert capped_data["total_returned"] == 2
    assert capped_data["capped"] is True
    assert capped_data["column_types"] == ["Int64", "Float64"]

    # 2. String literals containing semicolons or keyword-like words must NOT false-positive
    safe_literal_res = client.post(
        f"/api/jobs/{task_id}/query-sql",
        json={"sql": "SELECT category FROM dataset WHERE category <> 'delete;update'"},
    )
    assert safe_literal_res.status_code == 200
    assert safe_literal_res.json()["capped"] is False

    # 3. Stacked statement injection & table functions stay strictly blocked (403)
    stacked = client.post(
        f"/api/jobs/{task_id}/query-sql",
        json={"sql": "SELECT 1 FROM dataset; DROP TABLE dataset"},
    )
    assert stacked.status_code == 403

    read_fn = client.post(
        f"/api/jobs/{task_id}/query-sql",
        json={"sql": "SELECT * FROM read_parquet('secret.parquet')"},
    )
    assert read_fn.status_code == 403


def test_sql_studio_extended_endpoints(setup_completed_job):
    task_id, _ = setup_completed_job

    # 1. Column Profile
    prof_res = client.get(f"/api/jobs/{task_id}/sql-column-profile", params={"column": "price"})
    assert prof_res.status_code == 200
    prof = prof_res.json()
    assert prof["min_value"] == 45.0
    assert prof["max_value"] == 1200.0
    assert prof["null_count"] == 0

    bad_prof = client.get(f"/api/jobs/{task_id}/sql-column-profile", params={"column": "missing_col"})
    assert bad_prof.status_code == 400

    # 2. Explain plan JSON
    exp_res = client.post(
        f"/api/jobs/{task_id}/sql-explain",
        json={"sql": "SELECT category, COUNT(*) FROM dataset GROUP BY category"},
    )
    assert exp_res.status_code == 200
    assert len(exp_res.json()["plan"]) >= 1

    # 3. SQL Format endpoint
    fmt_res = client.post(
        f"/api/jobs/{task_id}/sql-format",
        json={"sql": "select category, count(*) from dataset group by category"},
    )
    assert fmt_res.status_code == 200
    assert "FROM dataset" in fmt_res.json()["sql"]

    # 4. Streaming CSV Export & Parameterized Query & Golden Patch
    param_res = client.post(
        f"/api/jobs/{task_id}/query-sql",
        json={"sql": "SELECT product_id FROM dataset WHERE category = {{ cat }}", "params": {"cat": "Cloud"}},
    )
    assert param_res.status_code == 200
    assert param_res.json()["total_returned"] == 1

    csv_res = client.post(
        f"/api/jobs/{task_id}/query-sql/export",
        json={"sql": "SELECT category, price FROM dataset ORDER BY price", "format": "csv"},
    )
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.headers["content-type"]
    assert "Hardware" in csv_res.text

    created = client.post(
        f"/api/jobs/{task_id}/golden-queries",
        json={"question": "Initial Q", "sql_query": "SELECT COUNT(*) FROM dataset"},
    ).json()
    patched = client.patch(
        f"/api/jobs/{task_id}/golden-queries/{created['id']}",
        json={"question": "Renamed Golden Q"},
    )
    assert patched.status_code == 200
    assert patched.json()["question"] == "Renamed Golden Q"


