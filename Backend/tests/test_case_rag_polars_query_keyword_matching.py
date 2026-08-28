import pytest
from app.services.rag_service import _execute_polars_data_query

def test_rag_polars_query_keyword_matching():
    """Verify that questions requesting math metrics (mean, min, max, average) trigger exact Polars lookup."""
    job_result = {
        "analysis": {
            "summary": {
                "salary": {"mean": 85000.0, "min": 40000.0, "max": 160000.0, "std": 25000.0}
            },
            "columns": {
                "salary": {"type": "numeric"}
            }
        }
    }
    
    question = "What is the average salary?"
    res = _execute_polars_data_query(question, job_result)
    assert "--- VERIFIED EXACT POLARS CALCULATIONS ---" in res
    assert "Column 'salary'" in res
    assert "Mean=85000.0" in res
