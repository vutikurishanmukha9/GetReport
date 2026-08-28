import pytest
from app.services.rag_service import _extract_targeted_structured_context

def test_rag_targeted_context_extraction():
    """Verify structured metadata extraction for specific query mentions."""
    job_result = {
        "analysis": {
            "summary": {
                "profit": {"mean": 12500.0, "min": -500.0, "max": 45000.0}
            }
        }
    }
    
    context = _extract_targeted_structured_context("Tell me about the profit column", job_result)
    assert "profit" in context.lower()
    assert "12500" in context
