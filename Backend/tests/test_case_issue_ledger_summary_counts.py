import pytest
from app.services.issue_ledger import Issue, IssueLedger

def test_issue_ledger_summary_counts_by_status():
    """Verify get_summary computes accurate counts across pending, approved, rejected, and modified statuses."""
    ledger = IssueLedger()
    
    # 1 approved, 1 rejected, 1 modified, 1 pending
    ledger.add_issue(Issue("i1", "missing_values", "high", "a", 1, 1.0, "", "", ""))
    ledger.add_issue(Issue("i2", "outliers", "medium", "b", 2, 2.0, "", "", ""))
    ledger.add_issue(Issue("i3", "duplicates", "low", "c", 3, 3.0, "", "", ""))
    ledger.add_issue(Issue("i4", "format_issue", "high", "d", 4, 4.0, "", "", ""))
    
    ledger.approve("i1")
    ledger.reject("i2")
    ledger.modify("i3", "new_code")
    
    summary = ledger.get_summary()
    assert summary["approved"] == 1
    assert summary["rejected"] == 1
    assert summary["modified"] == 1
    assert summary["pending"] == 1
    assert summary["total"] == 4
