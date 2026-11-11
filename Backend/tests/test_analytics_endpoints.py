"""
Integration Tests for Analytics & Superset BI Companion API Endpoints (/api/analytics).
"""

from datetime import date
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import polars as pl
from slowapi.middleware import SlowAPIMiddleware

from app.api.routes.analytics import router as analytics_router
from app.core.limiter import limiter
from app.db import init_db
from app.services.superset_engine import superset_engine

# Initialize database schema for isolated endpoint testing
try:
    init_db()
except Exception:
    pass

# Dedicated test app avoiding external DB lifecycle and heavy disk traversals
_test_app = FastAPI()
_test_app.state.limiter = limiter
_test_app.add_middleware(SlowAPIMiddleware)
_test_app.include_router(analytics_router, prefix="/api")


@pytest.fixture
def client():
    return TestClient(_test_app)


@pytest.fixture
def mock_dataset(monkeypatch):
    """Mocks dataset resolution inside superset_engine."""
    df = pl.DataFrame({
        "order_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "category": ["Tech", "Fashion", "Tech", "Food", "Fashion", "Tech", "Food", "Tech", "Fashion", "Food"],
        "region": ["North", "South", "East", "West", "North", "South", "East", "West", "North", "South"],
        "sales": [100.5, 200.0, 150.25, 50.0, 300.0, 120.0, 80.0, 220.5, 180.0, 95.0],
        "order_date": [date(2026, 1, i + 1) for i in range(10)],
    })
    monkeypatch.setattr(superset_engine, "resolve_dataset_df", lambda task_id: df)
    return df


def test_get_analytics_status(client):
    response = client.get("/api/analytics/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "superset_url" in data
    assert "enabled" in data


def test_list_dashboards(client):
    response = client.get("/api/analytics/dashboards")
    assert response.status_code == 200
    data = response.json()
    assert "dashboards" in data
    assert len(data["dashboards"]) >= 1


def test_list_datasets(client):
    response = client.get("/api/analytics/datasets")
    assert response.status_code == 200
    data = response.json()
    assert "datasets" in data


def test_get_superset_dataset_schema(client, mock_dataset):
    response = client.get("/api/analytics/explore/mock_task_001/schema")
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "mock_task_001"
    assert data["total_rows"] == 10
    assert "sales" in data["numeric_columns"]
    assert "category" in data["categorical_columns"]
    assert "time_grains" in data


def test_execute_superset_explore_query(client, mock_dataset):
    payload = {
        "task_id": "mock_task_001",
        "chart_type": "echarts_bar",
        "dimension": "category",
        "metric": "sales",
        "aggregate": "SUM",
        "row_limit": 5,
        "sort_order": "desc",
        "color_palette": "modernDark",
    }
    response = client.post("/api/analytics/explore/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["dimension"] == "category"
    assert "echarts_option" in data
    assert data["kpis"]["total_rows"] == 10


def test_execute_superset_explore_query_with_filters(client, mock_dataset):
    payload = {
        "task_id": "mock_task_001",
        "chart_type": "echarts_line",
        "dimension": "category",
        "metric": "sales",
        "aggregate": "AVG",
        "filters": [
            {"column": "region", "operator": "eq", "value": "North"}
        ],
        "row_limit": 10,
    }
    response = client.post("/api/analytics/explore/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["filters"]) == 1
    assert data["kpis"]["total_rows"] == 3  # Only 3 North rows in mock dataset


def test_execute_superset_explore_query_invalid_dimension(client, mock_dataset):
    payload = {
        "task_id": "mock_task_001",
        "chart_type": "echarts_bar",
        "dimension": "non_existent_column",
    }
    response = client.post("/api/analytics/explore/query", json=payload)
    assert response.status_code == 400


def test_aggregate_dataset_metrics(client, mock_dataset):
    payload = {
        "dimension": "category",
        "metric": "sales",
        "aggregation": "avg",
        "limit": 5,
        "sort_order": "desc",
    }
    response = client.post("/api/analytics/aggregate/mock_task_001", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["dimension"] == "category"
    assert len(data["records"]) <= 5


def test_generate_guest_token_offline_fallback(client):
    payload = {
        "dashboard_id": "1",
        "task_id": "mock_task_001",
    }
    response = client.post("/api/analytics/guest-token", json=payload)
    assert response.status_code == 200
    data = response.json()
    # When Superset container is not running, it gracefully returns status without failing
    assert data["status"] in ("offline", "online", "error", "auth_error")
