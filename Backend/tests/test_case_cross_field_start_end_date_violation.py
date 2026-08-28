import pytest
import polars as pl
from datetime import date
from app.services.issue_ledger import _detect_business_rule_violations

def test_cross_field_start_end_date_chronology_violations():
    """Verify detection of chronology violations where ship_date occurs earlier than order_date."""
    df = pl.DataFrame({
        "order_date": [date(2024, 1, 10), date(2024, 2, 1), date(2024, 3, 15)],
        "ship_date": [date(2024, 1, 15), date(2024, 1, 20), date(2024, 3, 20)] # Row 2 is violation
    })
    
    issues = _detect_business_rule_violations(df)
    assert len(issues) >= 1
    chronology_issue = next((i for i in issues if "chronology" in i.description.lower() or "earlier" in i.description.lower()), None)
    assert chronology_issue is not None
    assert chronology_issue.affected_rows == 1
