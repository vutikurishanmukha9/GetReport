import pytest
from app.services.rag_service import RAGMetrics

def test_rag_metrics_query_counting():
    """Verify RAGMetrics records total queries and tracks failure rates accurately."""
    metrics = RAGMetrics()
    assert metrics.total_queries == 0
    assert metrics.failed_queries == 0
    
    metrics.record_query(success=True)
    metrics.record_query(success=True)
    metrics.record_query(success=False)
    
    stats = metrics.get_stats()
    assert stats["total_queries"] == 3
    assert stats["failed_queries"] == 1
