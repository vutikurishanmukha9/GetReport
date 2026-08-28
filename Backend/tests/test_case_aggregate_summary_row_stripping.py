import pytest
import polars as pl
from app.services.issue_ledger import _detect_summary_rows_and_mid_headers

def test_aggregate_summary_row_detection():
    """Verify detection of Excel/report aggregate summary rows ('Total', 'Grand Total', 'Average') in data."""
    df = pl.DataFrame({
        "item_name": ["Widget A", "Widget B", "Widget C", "Total", "Average"],
        "sales": ["100", "250", "300", "650", "216.6"]
    })
    
    issues = _detect_summary_rows_and_mid_headers(df)
    assert len(issues) >= 1
    summary_issue = next(i for i in issues if "aggregate summary" in i.description.lower() or "total" in i.description.lower())
    assert summary_issue.affected_rows == 2 # "Total", "Average"
