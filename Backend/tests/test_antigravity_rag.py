import pytest
from app.services.rag_service import ANTIGRAVITY_AVAILABLE, _build_antigravity_tools, _build_structured_job_context

def test_antigravity_sdk_availability():
    """Verify that google-antigravity SDK is installed and recognized."""
    assert ANTIGRAVITY_AVAILABLE is True

def test_antigravity_tools_construction():
    """Verify that dataset tools are dynamically registered for Antigravity Agent."""
    mock_job_result = {
        "filename": "sales_q3.csv",
        "analysis": {
            "metadata": {"total_rows": 1500, "total_columns": 8, "missing_value_pct": 2.5},
            "summary": {
                "revenue": {"mean": 5400.5, "min": 100.0, "max": 25000.0}
            },
            "strong_correlations": [
                {"column_a": "ad_spend", "column_b": "revenue", "r_value": 0.89}
            ],
            "column_quality_flags": {}
        },
        "cleaning_report": {"total_changes": 12, "duplicate_rows_removed": 2, "empty_rows_dropped": 0},
        "insights": {"insights_text": "Strong growth observed in Q3 sales."}
    }
    
    # 1. Test Structured Job Context
    context_str = _build_structured_job_context(mock_job_result)
    assert "sales_q3.csv" in context_str
    assert "Rows: 1500" in context_str
    assert "revenue" in context_str

    # 2. Test Tools Construction
    tools = _build_antigravity_tools(mock_job_result)
    assert len(tools) == 4
    
    tool_names = [t.__name__ for t in tools]
    assert "get_dataset_overview" in tool_names
    assert "query_column_statistics" in tool_names
    assert "get_correlation_insights" in tool_names
    assert "get_data_quality_report" in tool_names
    
    # Test tool execution
    stats_func = next(t for t in tools if t.__name__ == "query_column_statistics")
    output = stats_func("revenue")
    assert "5400.5" in output or "revenue" in output
