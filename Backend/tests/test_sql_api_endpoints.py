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

