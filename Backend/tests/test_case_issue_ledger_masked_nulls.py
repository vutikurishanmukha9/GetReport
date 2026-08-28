import pytest
import polars as pl
from app.services.issue_ledger import _detect_masked_null_issues

def test_issue_ledger_masked_nulls_detection():
    """Verify detection of dirty string placeholders like N/A, -999, NULL, ? as missing values."""
    df = pl.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4", "C5"],
        "income": ["$50,000", "N/A", "missing", "65000", "?"],
        "score": [100, -999, 95, 9999, 88]
    })
    
    issues = _detect_masked_null_issues(df)
    assert len(issues) >= 2
    
    cols_flagged = {i.column for i in issues}
    assert "income" in cols_flagged
    assert "score" in cols_flagged
    
    income_issue = next(i for i in issues if i.column == "income")
    assert income_issue.affected_rows == 3
    assert income_issue.issue_type == "missing_values"
