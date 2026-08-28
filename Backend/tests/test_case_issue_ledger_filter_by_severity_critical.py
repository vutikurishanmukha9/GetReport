import pytest
from app.services.issue_ledger import Issue, IssueLedger

def test_issue_ledger_filter_by_severity():
    """Verify filtering issues by critical and high severity levels."""
    ledger = IssueLedger()
    ledger.add_issue(Issue("1", "missing_values", "critical", "col1", 100, 100.0, "", "", ""))
    ledger.add_issue(Issue("2", "outliers", "low", "col2", 5, 2.0, "", "", ""))
    ledger.add_issue(Issue("3", "duplicates", "high", "col3", 20, 15.0, "", "", ""))
    
    critical_issues = [i for i in ledger.issues if i.severity == "critical"]
    high_or_crit = [i for i in ledger.issues if i.severity in ("critical", "high")]
    
    assert len(critical_issues) == 1
    assert len(high_or_crit) == 2
