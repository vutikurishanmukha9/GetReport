import pytest
import os
import json
import asyncio
from app.core.rag_utils import TableSemanticChunker, CrossEncoderReranker, SimpleVectorStore
from app.core.graph_store import DatasetGraphStore
from app.services.dataset_graph_builder import build_dataset_graph
from app.services.query_router import QueryRouter
from app.services.rag_service import EnhancedRAGService

# Sample Mock Job Result
MOCK_JOB_RESULT = {
    "filename": "financial_transactions.csv",
    "analysis": {
        "metadata": {
            "total_rows": 15000,
            "total_columns": 5,
            "missing_value_pct": 2.4,
        },
        "domain": "Financial Analytics",
        "summary": {
            "transaction_id": {
                "null_count": 0,
                "null_percentage": 0.0,
                "unique_count": 15000,
                "data_type": "Int64"
            },
            "amount": {
                "null_count": 360,
                "null_percentage": 2.4,
                "unique_count": 12400,
                "mean": 342.50,
                "min": 10.0,
                "max": 15000.0,
                "std": 128.40,
                "skewness": 2.85,
                "benford_status": "Conforming (Chi-Square p=0.45)"
            },
            "discount": {
                "null_count": 0,
                "null_percentage": 0.0,
                "unique_count": 25,
                "mean": 0.08,
                "min": 0.0,
                "max": 0.35,
                "std": 0.05
            },
            "customer_age": {
                "null_count": 50,
                "null_percentage": 0.33,
                "unique_count": 70,
                "mean": 38.2,
                "min": 18,
                "max": 95
            }
        },
        "columns": {
            "transaction_id": {"data_type": "Int64", "semantic_type": "Identifier"},
            "amount": {"data_type": "Float64", "semantic_type": "MonetaryAmount"},
            "discount": {"data_type": "Float64", "semantic_type": "PercentageRate"},
            "customer_age": {"data_type": "Int64", "semantic_type": "DemographicAge"}
        },
        "correlation": {
            "strong_correlations": [
                {"column_a": "amount", "column_b": "discount", "r_value": -0.74}
            ]
        }
    },
    "confidence": {
        "dataset_confidence": 78.5,
        "dataset_grade": "B",
        "critical_issues": [
            {
                "id": "crit_1",
                "column": "amount",
                "issue_type": "missing_values",
                "severity": "high",
                "description": "360 missing values in primary monetary column",
                "suggested_action": "Median Imputation"
            }
        ]
    },
    "cleaning_report": {
        "total_changes": 360,
        "columns_renamed": {"TransactionID": "transaction_id", "AMT": "amount"}
    }
}

MOCK_LEDGER_ISSUES = [
    {
        "id": "iss_1",
        "column": "amount",
        "issue_type": "missing_values",
        "severity": "high",
        "description": "360 null values detected in amount column",
        "suggested_action": "Impute with median value ($342.50)",
        "status": "pending"
    },
    {
        "id": "iss_2",
        "column": "customer_age",
        "issue_type": "outliers",
        "severity": "medium",
        "description": "3 records with age > 90 years",
        "suggested_action": "Winsorize at 99th percentile",
        "status": "approved"
    }
]

# --- 1. Test TableSemanticChunker ---
def test_table_semantic_chunker_markdown_anchoring():
    chunker = TableSemanticChunker(chunk_size=500, chunk_overlap=50)
    raw_markdown = """# Financial Summary
Here is the performance table:

| Metric | Q1 | Q2 |
|---|---|---|
| Revenue | $100k | $120k |
| Margin | 20% | 22% |

End of table.
"""
    chunks = chunker.split_text(raw_markdown)
    assert len(chunks) > 0
    combined = " ".join(chunks)
    # Ensure header-cell key-value binding was performed
    assert "Metric: Revenue" in combined
    assert "Q1: $100k" in combined
    assert "Metric: Margin" in combined

def test_table_semantic_chunker_structured_dataset_profile():
    chunker = TableSemanticChunker()
    chunks = chunker.chunk_dataset_profile(MOCK_JOB_RESULT, MOCK_LEDGER_ISSUES)
    assert len(chunks) >= 4  # Overview, amount, discount, customer_age, correlations, ledger
    
    types = [c["metadata"]["type"] for c in chunks]
    assert "dataset_overview" in types
    assert "column_profile" in types
    assert "feature_correlations" in types
    assert "issue_ledger" in types
    
    # Check that column profile contains explicit key-value anchors
    amount_chunk = next(c for c in chunks if c["metadata"].get("column_name") == "amount")
    assert "Feature Name: amount" in amount_chunk["content"] or "Column Name: amount" in amount_chunk["content"]
    assert "Mean: 342.5" in amount_chunk["content"]
    assert "missing_values" in amount_chunk["content"]

# --- 2. Test CrossEncoderReranker ---
def test_cross_encoder_reranker_scoring():
    reranker = CrossEncoderReranker()
    query = "What data cleaning is recommended for amount outliers and missing values?"
    candidates = [
        {"content": "Customer age has 50 null values and 3 records above 90 years.", "id": 1},
        {"content": "The amount column has 360 missing values and is flagged for median imputation.", "id": 2},
        {"content": "Dataset contains 15000 transactions across financial analytics.", "id": 3},
    ]
    reranked = reranker.rerank(query, candidates, top_n=3)
    assert len(reranked) == 3
    # Candidate #2 should rank highest due to matching 'amount', 'missing', 'values', 'cleaning'
    assert reranked[0]["id"] == 2
    assert "rerank_score" in reranked[0]
    assert reranked[0]["rerank_score"] > reranked[2]["rerank_score"]

# --- 3. Test DatasetGraphStore ---
def test_dataset_graph_store_local_and_global():
    store = DatasetGraphStore("test_task_123")
    
    # Add nodes
    store.add_entity("dataset:test", "Dataset", {"name": "test.csv", "grade": "B"})
    store.add_entity("col:amount", "FeatureColumn", {"name": "amount", "mean": 342.5})
    store.add_entity("col:discount", "FeatureColumn", {"name": "discount", "mean": 0.08})
    store.add_entity("issue:iss_1", "DataQualityIssue", {"issue_type": "missing_values", "severity": "high"})
    
    # Add relations
    store.add_relation("dataset:test", "col:amount", "CONTAINS_COLUMN", {"description": "Contains amount"})
    store.add_relation("col:amount", "col:discount", "CORRELATED_WITH", {"r_value": -0.74, "description": "Inverse correlation r=-0.74"})
    store.add_relation("col:amount", "issue:iss_1", "HAS_QUALITY_ISSUE", {"description": "360 nulls in amount"})
    
    # Local Subgraph (1-hop from col:amount)
    subgraph = store.get_local_subgraph(["col:amount"], max_hops=1)
    node_ids = [n["id"] for n in subgraph["nodes"]]
    assert "col:amount" in node_ids
    assert "col:discount" in node_ids
    assert "issue:iss_1" in node_ids
    
    # Format Subgraph Markdown
    md = store.format_subgraph_markdown(subgraph)
    assert "col:amount" in md
    assert "CORRELATED_WITH" in md
    
    # Global Relations (Thematic)
    global_rels = store.get_global_relations(["correlation", "quality"])
    assert len(global_rels) >= 2
    
    # JSON Serialization roundtrip
    json_str = store.to_json()
    reloaded = DatasetGraphStore.from_json("test_task_123", json_str)
    assert reloaded.graph.number_of_nodes() == store.graph.number_of_nodes()
    assert reloaded.graph.number_of_edges() == store.graph.number_of_edges()

# --- 4. Test build_dataset_graph ---
def test_build_dataset_graph_builder():
    task_id = "task_builder_test"
    graph_store = build_dataset_graph(task_id, MOCK_JOB_RESULT, MOCK_LEDGER_ISSUES)
    
    assert graph_store.graph.number_of_nodes() > 5
    assert graph_store.graph.has_node("dataset:financial_transactions.csv")
    assert graph_store.graph.has_node("col:amount")
    assert graph_store.graph.has_node("col:discount")
    
    # Verify correlation edge
    edge_data = graph_store.graph.get_edge_data("col:amount", "col:discount")
    assert edge_data is not None
    key = list(edge_data.keys())[0]
    assert edge_data[key]["relation_type"] == "CORRELATED_WITH"
    assert edge_data[key]["r_value"] == -0.74

# --- 5. Test QueryRouter ---
def test_query_router_classification():
    async def _run():
        router = QueryRouter()
        cols = ["transaction_id", "amount", "discount", "customer_age"]
        
        # 1. Exact Math Query -> DuckDB OLAP
        q1 = "What is the average transaction amount for customers?"
        res1 = await router.route_query(q1, cols)
        assert res1["duckdb_sql_needed"] is True
        assert "amount" in res1["ll_keywords"]
        assert res1["query_type"] == "analytical_sql"
        
        # 2. Relational Query -> Graph Local
        q2 = "Why does amount have a negative correlation with discount?"
        res2 = await router.route_query(q2, cols)
        assert "amount" in res2["ll_keywords"]
        assert "discount" in res2["ll_keywords"]
        assert "correlation" in res2["hl_keywords"]
        assert res2["query_type"] == "graph_local"
        
        # 3. Global Quality / Overview Query -> Graph Global
        q3 = "What are the key data quality issues and overall health of this dataset?"
        res3 = await router.route_query(q3, cols)
        assert res3["query_type"] in ("graph_global", "hybrid")
        assert any(k in res3["hl_keywords"] for k in ["quality", "health", "issues", "data health"])
    
    asyncio.run(_run())

# --- 6. Test EnhancedRAGService Integration ---
def test_rag_service_hybrid_and_graph():
    async def _run():
        rag = EnhancedRAGService()
        task_id = "test_rag_pipeline_task"
        
        # Ingest structured text
        chunker = TableSemanticChunker()
        semantic_chunks = chunker.chunk_dataset_profile(MOCK_JOB_RESULT, MOCK_LEDGER_ISSUES)
        combined_text = "\n\n".join([c["content"] for c in semantic_chunks])
        
        # Ingest
        ingest_res = await rag.ingest_report(task_id, combined_text)
        assert ingest_res["success"] is True
        assert ingest_res["num_chunks"] > 0
        
        # Query with report
        chat_res = await rag.chat_with_report(
            task_id=task_id,
            question="What is the correlation between amount and discount?",
            job_result=MOCK_JOB_RESULT,
            include_sources=True
        )
        assert chat_res["success"] is True
        assert "answer" in chat_res
        assert len(chat_res["sources"]) > 0
        assert any("Knowledge Graph" in s or "amount" in s.lower() for s in chat_res["sources"])
    
    asyncio.run(_run())
