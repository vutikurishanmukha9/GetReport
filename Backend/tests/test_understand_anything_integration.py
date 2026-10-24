"""
Unit & Integration Test Suite for Understand-Anything Backend Capabilities in GetReport.
Tests:
- Architectural Layer Detection (detect_dataset_layers)
- Guided Audit Tour Generation (generate_dataset_audit_tour)
- Semantic Fingerprinting & Drift Detection (compute_dataset_fingerprint, compare_dataset_fingerprints)
- Understand-Anything KnowledgeGraph Schema Export
- FastAPI Endpoints (/layers, /tour, /fingerprint, /ua-export, /drift)
"""
import pytest
import os
from pathlib import Path
from fastapi.testclient import TestClient

from app.core.graph_store import DatasetGraphStore
from app.services.understand_anything.layer_detector import detect_dataset_layers
from app.services.understand_anything.tour_generator import generate_dataset_audit_tour
from app.services.understand_anything.fingerprint import compute_dataset_fingerprint, compare_dataset_fingerprints
from app.services.understand_anything.exporter import export_to_understand_anything_schema
from app.services.task_manager import title_task_manager
from app.main import app


@pytest.fixture
def populated_dataset_graph():
    """Builds a rich sample graph with ingestion, features, quality issues, PII, and SQL lineage."""
    store = DatasetGraphStore("test_ua_task")
    
    # 1. Dataset Node
    store.add_entity("dataset:sales_2026.csv", "Dataset", {"name": "sales_2026.csv", "total_rows": 100000, "total_cols": 4})
    
    # 2. Feature Columns
    store.add_entity("col:user_id", "FeatureColumn", {"name": "user_id", "data_type": "Int64", "null_count": 0, "null_percentage": 0.0, "unique_count": 50000})
    store.add_entity("col:ssn", "FeatureColumn", {"name": "ssn", "data_type": "Utf8", "null_count": 50, "null_percentage": 0.05, "unique_count": 49950})
    store.add_entity("col:transaction_amount", "FeatureColumn", {"name": "transaction_amount", "data_type": "Float64", "null_count": 1200, "null_percentage": 1.2, "unique_count": 85000, "benford_status": "Passed"})
    store.add_entity("col:tax_amount", "FeatureColumn", {"name": "tax_amount", "data_type": "Float64", "null_count": 1200, "null_percentage": 1.2, "unique_count": 85000})
    
    # 3. Quality & Security Issues
    store.add_entity(
        "issue:tx_nulls",
        "DataQualityIssue",
        {"column_name": "transaction_amount", "issue_type": "MissingValues", "severity": "critical", "description": "Transaction amount has 1.2% missing entries."}
    )
    store.add_entity(
        "pii:ssn",
        "SecurityRisk",
        {"column_name": "ssn", "pii_type": "US_SSN", "severity": "critical", "description": "Unencrypted Social Security Number column detected."}
    )
    
    # 4. Relations & Lineage
    store.add_relation("dataset:sales_2026.csv", "col:user_id", "CONTAINS_COLUMN", {"weight": 1.0})
    store.add_relation("dataset:sales_2026.csv", "col:ssn", "CONTAINS_COLUMN", {"weight": 1.0})
    store.add_relation("dataset:sales_2026.csv", "col:transaction_amount", "CONTAINS_COLUMN", {"weight": 1.0})
    store.add_relation("dataset:sales_2026.csv", "col:tax_amount", "CONTAINS_COLUMN", {"weight": 1.0})
    store.add_relation("col:transaction_amount", "issue:tx_nulls", "HAS_QUALITY_ISSUE", {"weight": 2.0})
    store.add_relation("col:ssn", "pii:ssn", "HAS_PII_RISK", {"weight": 3.0})
    store.add_relation("col:transaction_amount", "col:tax_amount", "CORRELATED_WITH", {"weight": 0.98, "r_value": 0.98})
    
    store.attach_sql("CREATE VIEW reporting.revenue_summary AS SELECT user_id, transaction_amount FROM raw.sales_2026;")
    
    store.cluster()
    return store


def test_layer_detection(populated_dataset_graph):
    """Verifies partitioning into canonical Understand-Anything architectural data layers."""
    layers = detect_dataset_layers(populated_dataset_graph.graph)
    assert len(layers) >= 4

    layer_ids = [l["id"] for l in layers]
    assert "layer:ingestion" in layer_ids
    assert "layer:features" in layer_ids
    assert "layer:quality" in layer_ids
    assert "layer:security" in layer_ids

    # Verify node counts
    for layer in layers:
        assert len(layer["nodeIds"]) == layer["nodeCount"]
        assert layer["name"] != ""


def test_guided_audit_tour(populated_dataset_graph):
    """Verifies pedagogical generation of ordered investigation steps."""
    tour = generate_dataset_audit_tour(
        populated_dataset_graph.graph,
        analysis_results={"summary": {"total_rows": 100000}}
    )
    assert len(tour) >= 4
    
    # Check ordering is sequential
    orders = [step["order"] for step in tour]
    assert orders == list(range(1, len(tour) + 1))

    # Step 1 should be ingestion
    assert "Ingestion" in tour[0]["title"]
    # Should include quality issues step
    titles = [s["title"] for s in tour]
    assert any("Quality" in t for t in titles)
    assert any("Security" in t or "Governance" in t for t in titles)


def test_fingerprinting_and_drift_detection(populated_dataset_graph):
    """Verifies deterministic SHA-256 fingerprinting and drift comparison."""
    fp1 = compute_dataset_fingerprint(populated_dataset_graph.graph)
    assert fp1["content_hash"] != ""
    assert fp1["node_count"] > 0

    # Same graph yields identical fingerprint
    fp2 = compute_dataset_fingerprint(populated_dataset_graph.graph)
    assert fp1["content_hash"] == fp2["content_hash"]

    # Mutate graph (simulate schema drift: drop a column, add a new one)
    mutated_store = DatasetGraphStore.from_json("mutated_task", populated_dataset_graph.to_json())
    mutated_store.graph.remove_node("col:tax_amount")
    mutated_store.add_entity("col:discount_amount", "FeatureColumn", {"name": "discount_amount", "data_type": "Float64"})

    mutated_fp = compute_dataset_fingerprint(mutated_store.graph)
    assert mutated_fp["content_hash"] != fp1["content_hash"]

    comparison = compare_dataset_fingerprints(fp1, mutated_fp)
    assert comparison["identical"] is False
    assert comparison["drift_score"] > 0
    assert "col:tax_amount" in comparison["removed_nodes"]
    assert "col:discount_amount" in comparison["added_nodes"]


def test_understand_anything_schema_export(populated_dataset_graph):
    """Verifies export to Understand-Anything KnowledgeGraph schema specification."""
    ua_data = export_to_understand_anything_schema(
        populated_dataset_graph.graph,
        project_name="Test Sales Audit"
    )

    assert ua_data["version"] == "1.0.0"
    assert ua_data["kind"] == "codebase"
    assert "project" in ua_data
    assert "nodes" in ua_data
    assert "edges" in ua_data
    assert "layers" in ua_data
    assert "tour" in ua_data

    # Check canonical node types (must be one of the 27 allowed types)
    valid_ua_types = {
        "file", "function", "class", "module", "concept",
        "config", "document", "service", "table", "endpoint",
        "pipeline", "schema", "resource", "domain", "flow", "step",
        "article", "entity", "topic", "claim", "source",
        "page", "screen", "component", "componentSet", "instance", "token"
    }
    for n in ua_data["nodes"]:
        assert n["type"] in valid_ua_types
        assert n["complexity"] in ("simple", "moderate", "complex")


def test_understand_anything_api_endpoints(populated_dataset_graph):
    """Verifies FastAPI REST endpoints for layers, tour, fingerprint, ua-export, and drift."""
    task_id = title_task_manager.create_job("sales_2026.csv", owner_id=None)
    title_task_manager.complete_job(task_id, result={
        "filename": "sales_2026.csv",
        "analysis": {"summary": {"total_rows": 100000}}
    })

    base_dir = Path(__file__).resolve().parent.parent
    cache_dir = os.path.join(base_dir, "temp_cache")
    os.makedirs(cache_dir, exist_ok=True)
    graph_path = os.path.join(cache_dir, f"{task_id}_graph.json")
    populated_dataset_graph.save_to_file(graph_path)

    client = TestClient(app)

    # 1. GET /api/jobs/{task_id}/graph/layers
    resp_layers = client.get(f"/api/jobs/{task_id}/graph/layers")
    assert resp_layers.status_code == 200
    layers = resp_layers.json()
    assert len(layers) >= 4

    # 2. GET /api/jobs/{task_id}/graph/tour
    resp_tour = client.get(f"/api/jobs/{task_id}/graph/tour")
    assert resp_tour.status_code == 200
    tour = resp_tour.json()
    assert len(tour) >= 4

    # 3. GET /api/jobs/{task_id}/graph/fingerprint
    resp_fp = client.get(f"/api/jobs/{task_id}/graph/fingerprint")
    assert resp_fp.status_code == 200
    fp = resp_fp.json()
    assert "content_hash" in fp

    # 4. GET /api/jobs/{task_id}/graph/ua-export
    resp_ua = client.get(f"/api/jobs/{task_id}/graph/ua-export")
    assert resp_ua.status_code == 200
    ua = resp_ua.json()
    assert ua["version"] == "1.0.0"

    # 5. POST /api/jobs/{task_id}/graph/drift (compare with self)
    resp_drift = client.post(f"/api/jobs/{task_id}/graph/drift", json={"baseline_task_id": task_id})
    assert resp_drift.status_code == 200
    drift = resp_drift.json()
    assert drift["comparison"]["identical"] is True
