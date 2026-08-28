import pytest
import polars as pl
from app.services.issue_ledger import _detect_business_rule_violations

def test_business_rule_human_age_bounds_validation():
    """Verify detection and clipping recommendation for negative or impossible human ages (>120)."""
    df = pl.DataFrame({
        "user_age": [25, 34, -5, 142, 88, -1]
    })
    
    issues = _detect_business_rule_violations(df)
    assert len(issues) >= 1
    age_issue = next(i for i in issues if i.column == "user_age")
    assert age_issue.affected_rows == 3 # -5, 142, -1
    assert "120" in age_issue.description
