"""
tests/test_phase3_api_endpoints.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration tests for Phase 3 endpoints:
- POST /api/jobs/{task_id}/sandbox-exec
- POST /api/jobs/{task_id}/concepts/derive
- GET  /api/jobs/{task_id}/concepts
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
def setup_phase3_job():
    storage = get_storage_provider()

    df = pl.DataFrame({
        "revenue": [1000.0, 2500.0, 4000.0, 5500.0],
        "cogs": [400.0, 1100.0, 1900.0, 2800.0],
        "department": ["East", "West", "East", "Central"],
    })

    task_id = title_task_manager.create_job("financials.csv")

    buffer = io.BytesIO()
    df.write_parquet(buffer)
    buffer.seek(0)
    cleaned_file_ref = storage.save_upload(buffer, f"cleaned_{task_id}.parquet")

    job_result = {
        "filename": "financials.csv",
        "cleaned_file_ref": cleaned_file_ref,
        "info": {
            "rows": 4,
            "columns": 3,
            "grade": "A",
            "score": 98.0,
        },
        "transformation_dag": {
            "dataset_name": "financials.csv",
            "original_state": {"rows": 4, "cols": 3},
            "nodes": {},
        },
    }

    title_task_manager.update_status(task_id, TaskStatus.COMPLETED, job_result)
    return task_id


def test_sandbox_exec_valid_code(setup_phase3_job):
    task_id = setup_phase3_job
    code = """
profit = df['revenue'] - df['cogs']
print(f"TOTAL_PROFIT={profit.sum():.2f}")
"""
    resp = client.post(
        f"/api/jobs/{task_id}/sandbox-exec",
        json={"code": code},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "TOTAL_PROFIT=6800.00" in data["output"]
    assert data["error"] is None
    assert data["execution_time_ms"] >= 0.0


def test_sandbox_exec_plot_generation(setup_phase3_job):
    task_id = setup_phase3_job
    code = """
plt.figure(figsize=(5, 3))
plt.bar(df['department'].to_list(), df['revenue'].to_list(), color='teal')
plt.title("Revenue by Department")
print("Bar chart generated")
"""
    resp = client.post(
        f"/api/jobs/{task_id}/sandbox-exec",
        json={"code": code},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "Bar chart generated" in data["output"]
    assert data["chart_base64"] is not None
    assert len(data["chart_base64"]) > 50


def test_sandbox_exec_blocks_malicious_code(setup_phase3_job):
    task_id = setup_phase3_job
    code = """
import os
os.system("echo compromised")
"""
    resp = client.post(
        f"/api/jobs/{task_id}/sandbox-exec",
        json={"code": code},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert "Forbidden import or AST pattern" in (data["error"] or "")


def test_derive_virtual_concept_endpoint(setup_phase3_job):
    task_id = setup_phase3_job
    payload = {
        "concept_name": "gross_profit",
        "formula_or_intent": "revenue - cogs",
        "description": "Total revenue minus cost of goods sold",
    }
    resp = client.post(
        f"/api/jobs/{task_id}/concepts/derive",
        json=payload,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["concept_name"] == "gross_profit"
    assert data["total_columns"] == 4
    assert data["total_rows"] == 4
    assert data["node"]["operation"] == "concept_derivation"
    assert data["node"]["target_column"] == "gross_profit"


def test_list_derived_concepts_endpoint(setup_phase3_job):
    task_id = setup_phase3_job

    # Derive two concepts sequentially
    client.post(
        f"/api/jobs/{task_id}/concepts/derive",
        json={"concept_name": "profit", "formula_or_intent": "revenue - cogs"},
    )
    client.post(
        f"/api/jobs/{task_id}/concepts/derive",
        json={"concept_name": "margin", "formula_or_intent": "(revenue - cogs) / revenue"},
    )

    resp = client.get(f"/api/jobs/{task_id}/concepts")
    assert resp.status_code == 200
    concepts = resp.json()
    assert len(concepts) >= 2
    concept_names = [c["concept_name"] for c in concepts]
    assert "profit" in concept_names
    assert "margin" in concept_names
