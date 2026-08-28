import pytest
from app.services.issue_ledger import Issue, IssueLedger

def test_issue_ledger_locking_enforcement():
    """Verify that a locked IssueLedger strictly rejects new issues and modifications."""
    ledger = IssueLedger()
    issue1 = Issue(
        id="iss_01",
        issue_type="missing_values",
        severity="high",
        column="age",
        affected_rows=15,
        affected_pct=15.0,
        description="15 missing values in age",
        suggested_fix="Impute with median",
        fix_code="df.with_columns(pl.col('age').fill_null(pl.col('age').median()))"
    )
    ledger.add_issue(issue1)
    assert len(ledger.issues) == 1
    assert ledger.approve("iss_01") is True
    assert ledger.issues[0].status == "approved"
    
    # Lock the ledger
    ledger.locked = True
    
    # Adding issue must raise ValueError
    issue2 = Issue(
        id="iss_02",
        issue_type="duplicates",
        severity="medium",
        column=None,
        affected_rows=3,
        affected_pct=3.0,
        description="3 duplicate rows",
        suggested_fix="Drop duplicates",
        fix_code="df.unique()"
    )
    with pytest.raises(ValueError, match="Cannot add issues to a locked ledger"):
        ledger.add_issue(issue2)
        
    # Modifying/approving must raise ValueError
    with pytest.raises(ValueError, match="Cannot modify a locked ledger"):
        ledger.approve("iss_01")
