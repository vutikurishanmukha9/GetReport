import pytest
from app.services.issue_ledger import Issue, IssueLedger

def test_issue_ledger_filter_pending_vs_approved():
    """Verify isolation of pending issues versus approved action items."""
    ledger = IssueLedger()
    ledger.add_issue(Issue("i1", "missing_values", "high", "col_a", 10, 10.0, "", "", ""))
    ledger.add_issue(Issue("i2", "outliers", "medium", "col_b", 5, 5.0, "", "", ""))
    
    ledger.approve("i1")
    
    pending = [i for i in ledger.issues if i.status == "pending"]
    approved = ledger.get_approved_issues()
    
    assert len(pending) == 1
    assert pending[0].id == "i2"
    assert len(approved) == 1
    assert approved[0].id == "i1"
