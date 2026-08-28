import pytest
import polars as pl
from app.services.issue_ledger import _detect_summary_rows_and_mid_headers

def test_mid_file_header_detection():
    """Verify detection of repeated CSV header lines accidentally embedded inside the dataset body."""
    df = pl.DataFrame({
        "customer_id": ["C100", "C101", "customer_id", "C102", "C103"],
        "amount": ["50", "120", "amount", "85", "90"]
    })
    
    issues = _detect_summary_rows_and_mid_headers(df)
    assert len(issues) >= 1
    header_issue = next(i for i in issues if "repeated header" in i.description.lower())
    assert header_issue.affected_rows == 1
    assert header_issue.column == "customer_id"
