import pytest
import polars as pl
from datetime import datetime, timedelta
from app.services.issue_ledger import _detect_business_rule_violations

def test_business_rule_future_timestamp_validation():
    """Verify detection of timestamp records that exceed the current datetime."""
    now = datetime.now()
    past_date = now - timedelta(days=30)
    future_date = now + timedelta(days=365)
    
    df = pl.DataFrame({
        "created_at": [past_date, now, future_date]
    })
    
    issues = _detect_business_rule_violations(df)
    assert len(issues) >= 1
    future_issue = next(i for i in issues if i.column == "created_at")
    assert future_issue.affected_rows == 1
    assert "future" in future_issue.description.lower()
