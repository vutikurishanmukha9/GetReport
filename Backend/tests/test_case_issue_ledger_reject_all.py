import pytest
from app.services.issue_ledger import Issue, IssueLedger

def test_issue_ledger_reject_all_pending():
    """Verify reject_all rejects all pending issues in bulk."""
    ledger = IssueLedger()
    for i in range(4):
        ledger.add_issue(Issue(
            id=f"iss_{i}",
            issue_type="outliers",
            severity="low",
            column=f"col_{i}",
            affected_rows=2,
            affected_pct=2.0,
            description="Outliers",
            suggested_fix="Clip",
            fix_code=""
        ))
    
    count = ledger.reject_all()
    assert count == 4
    assert all(iss.status == "rejected" for iss in ledger.issues)
    assert len(ledger.get_approved_issues()) == 0
