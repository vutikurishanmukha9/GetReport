import pytest
from app.services.issue_ledger import Issue, IssueLedger

def test_issue_ledger_reject_workflow_with_note():
    """Verify rejecting an issue updates its status to 'rejected' and preserves the audit note."""
    ledger = IssueLedger()
    issue = Issue(
        id="iss_99",
        issue_type="missing_values",
        severity="medium",
        column="notes",
        affected_rows=10,
        affected_pct=10.0,
        description="Missing values in notes",
        suggested_fix="Drop column",
        fix_code="df.drop('notes')"
    )
    ledger.add_issue(issue)
    assert ledger.reject("iss_99", note="Keep column as-is per client policy") is True
    
    rejected = ledger.issues[0]
    assert rejected.status == "rejected"
    assert rejected.user_note == "Keep column as-is per client policy"
