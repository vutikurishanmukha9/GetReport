import pytest
from app.services.issue_ledger import Issue, IssueLedger

def test_issue_ledger_to_dict_serialization():
    """Verify IssueLedger serializes completely to dictionary format for JSON transmission."""
    ledger = IssueLedger()
    ledger.add_issue(Issue("i1", "missing_values", "high", "age", 5, 5.0, "Missing age", "Impute", "code"))
    
    d = ledger.to_dict()
    assert "issues" in d
    assert "summary" in d
    assert "locked" in d
    assert "created_at" in d
    assert len(d["issues"]) == 1
    assert d["issues"][0]["column"] == "age"
