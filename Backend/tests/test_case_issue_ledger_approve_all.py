import pytest
from app.services.issue_ledger import Issue, IssueLedger

def test_issue_ledger_approve_all_pending():
    """Verify approve_all approves all pending issues in bulk and returns exact count."""
    ledger = IssueLedger()
    for i in range(3):
        ledger.add_issue(Issue(
            id=f"iss_{i}",
            issue_type="missing_values",
            severity="medium",
            column=f"col_{i}",
            affected_rows=10,
            affected_pct=10.0,
            description=f"Nulls in col_{i}",
            suggested_fix="Impute",
            fix_code=""
        ))
    
    count = ledger.approve_all()
    assert count == 3
    assert all(iss.status == "approved" for iss in ledger.issues)
    assert len(ledger.get_approved_issues()) == 3
