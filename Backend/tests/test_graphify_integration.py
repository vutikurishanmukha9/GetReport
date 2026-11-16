"""
Unit and Integration Tests for Graphify Integration in GetReport.
Tests Leiden clustering, hub labeling, god nodes, surprising connections,
fuzzy entity search, causal path traversal, and blast radius calculations.
"""
import pytest
import networkx as nx
from app.core.graph_store import DatasetGraphStore
from app.services.graphify.cluster_bridge import cluster_and_analyze_dataset_graph
from app.services.graphify.query_bridge import (
    search_graph_entities,
    find_causal_path,
    compute_blast_radius
)
from app.services.graphify.export_bridge import (
    generate_interactive_html,
    generate_typst_vector_svg,
    export_graph_to_format
)


@pytest.fixture
def sample_dataset_graph():
    """Constructs a sample dataset graph with columns, issues, and correlations."""
    store = DatasetGraphStore("task_test_123")
    store.add_entity("dataset:sales.csv", "Dataset", {"name": "sales.csv", "total_rows": 1000})
    store.add_entity("col:CustomerID", "FeatureColumn", {"name": "CustomerID", "null_percentage": 0.0})
    store.add_entity("col:TransactionAmount", "FeatureColumn", {"name": "TransactionAmount", "null_percentage": 5.2, "benford_status": "Passed"})
    store.add_entity("col:FraudScore", "FeatureColumn", {"name": "FraudScore", "null_percentage": 1.0})
    store.add_entity("issue:benford_spike", "DataQualityIssue", {"issue_type": "BenfordAnomaly", "severity": "high"})
    store.add_entity("pii:SSN", "SecurityRisk", {"pii_type": "SSN", "severity": "critical"})

    store.add_relation("dataset:sales.csv", "col:CustomerID", "CONTAINS_COLUMN", {"weight": 1.0})
    store.add_relation("dataset:sales.csv", "col:TransactionAmount", "CONTAINS_COLUMN", {"weight": 1.0})
    store.add_relation("dataset:sales.csv", "col:FraudScore", "CONTAINS_COLUMN", {"weight": 1.0})
    store.add_relation("col:TransactionAmount", "issue:benford_spike", "HAS_QUALITY_ISSUE", {"weight": 2.0})
    store.add_relation("issue:benford_spike", "dataset:sales.csv", "AFFECTS_CONFIDENCE", {"weight": 1.5})
    store.add_relation("col:TransactionAmount", "col:FraudScore", "CORRELATED_WITH", {"weight": 0.85, "r_value": 0.85})
    store.add_relation("col:CustomerID", "pii:SSN", "HAS_PII_RISK", {"weight": 3.0})

    return store


def test_graphify_clustering_and_analysis(sample_dataset_graph):
    """Verifies Leiden community detection, hub labeling, and god node detection."""
    store = sample_dataset_graph
    analysis = store.cluster()

    assert "total_communities" in analysis
    assert analysis["total_communities"] > 0
    assert "god_nodes" in analysis
    assert len(analysis["god_nodes"]) > 0

    # Ensure nodes have been annotated with community IDs and hub labels
    for nid, data in store.graph.nodes(data=True):
        assert "community" in data
        assert "community_label" in data

    # Verify God Node ranking has valid entries
    top_hub = analysis["god_nodes"][0]
    assert "id" in top_hub
    assert "degree" in top_hub
    assert top_hub["degree"] >= 1


def test_causal_path_traversal(sample_dataset_graph):
    """Verifies shortest and causal path finding between entities."""
    store = sample_dataset_graph
    # Trace path: col:TransactionAmount -> issue:benford_spike -> dataset:sales.csv
    res = store.find_path("col:TransactionAmount", "dataset:sales.csv")

    assert res["found"] is True
    assert res["hop_count"] >= 1
    assert len(res["path_nodes"]) >= 2
    assert "explanation_markdown" in res
    assert "Causal Path" in res["explanation_markdown"]


def test_blast_radius_computation(sample_dataset_graph):
    """Verifies upstream and downstream impact blast radius."""
    store = sample_dataset_graph
    blast = store.get_blast_radius("col:TransactionAmount")

    assert blast["found"] is True
    assert blast["total_impacted_count"] > 0
    assert "risk_level" in blast
    assert blast["risk_score"] > 0

    # Downstream should include issue:benford_spike and dataset:sales.csv
    impacted_ids = [n["id"] for n in blast["downstream_impact"]]
    assert "issue:benford_spike" in impacted_ids


def test_trigram_fuzzy_entity_search(sample_dataset_graph):
    """Verifies in-memory trigram fuzzy lookup."""
    store = sample_dataset_graph
    # Search with partial typo / token
    matches = store.search_entities("transaction", limit=5)

    assert len(matches) > 0
    match_ids = [m["id"] for m in matches]
    assert "col:TransactionAmount" in match_ids


def test_graph_exporters(sample_dataset_graph, tmp_path):
    """Verifies HTML, SVG, and database exporter generation."""
    store = sample_dataset_graph
    store.cluster()

    html_file = str(tmp_path / "test_graph.html")
    svg_file = str(tmp_path / "test_graph.svg")
    cypher_file = str(tmp_path / "test_graph.cyp")

    # Generate HTML
    out_html = store.generate_html(html_file)
    assert out_html == html_file
    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()
        assert "html" in html_content.lower()

    # Generate SVG
    out_svg = store.generate_svg(svg_file)
    assert out_svg == svg_file
    with open(svg_file, "r", encoding="utf-8") as f:
        svg_content = f.read()
        assert "<svg" in svg_content

    # Export Cypher
    out_cypher = export_graph_to_format(store.graph, "cypher", cypher_file)
    assert out_cypher == cypher_file


def test_graph_serialization_and_cache(sample_dataset_graph, tmp_path):
    """Verifies DatasetGraphStore serialization, deserialization, and caching."""
    import os
    store = sample_dataset_graph
    store.cluster()

    file_path = str(tmp_path / "serialized_graph.json")
    store.save_to_file(file_path)
    assert os.path.exists(file_path)

    loaded_store = DatasetGraphStore.load_from_file("task_test_123", file_path)
    assert loaded_store.task_id == "task_test_123"
    assert loaded_store.graph.number_of_nodes() == store.graph.number_of_nodes()
    assert loaded_store.graph.number_of_edges() == store.graph.number_of_edges()

    analysis = loaded_store.get_analysis()
    assert "total_communities" in analysis
    assert len(analysis["god_nodes"]) > 0


def test_sql_lineage_extraction_and_bridge(sample_dataset_graph):
    """Verifies sqlglot lineage extraction and attachment to DatasetGraphStore."""
    from app.services.graphify.sql_lineage_bridge import extract_sql_lineage, attach_sql_lineage_to_dataset_graph

    sql = """
    CREATE OR REPLACE VIEW reporting.customer_summary AS
    SELECT u.customer_id, u.email, o.total_amount
    FROM raw.users u
    JOIN raw.orders o ON u.customer_id = o.customer_id;
    """

    lineage = extract_sql_lineage(sql, dialect="snowflake")
    assert "users" in lineage["read_tables"]
    assert "orders" in lineage["read_tables"]
    assert "customer_summary" in lineage["write_tables"]
    assert len(lineage["edges"]) >= 2

    # Attach directly to store
    store = sample_dataset_graph
    res = store.attach_sql(sql, dialect="snowflake")
    assert res["write_tables"] == ["customer_summary"]
    assert store.graph.has_node("table:users")
    assert store.graph.has_node("view:customer_summary")
    assert len(res["read_tables"]) + len(res["write_tables"]) >= 3
    assert len(res["nodes"]) >= 3
