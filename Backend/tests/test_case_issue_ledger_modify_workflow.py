import pytest
from app.services.issue_ledger import Issue, IssueLedger

def test_issue_ledger_modify_fix_code_workflow():
    """Verify modifying an issue updates its status to 'modified' and stores the new fix code."""
    ledger = IssueLedger()
    issue = Issue(
        id="iss_101",
        issue_type="missing_values",
        severity="high",
        column="salary",
        affected_rows=5,
        affected_pct=5.0,
        description="Missing values in salary",
        suggested_fix="Impute with mean",
        fix_code="df.with_columns(pl.col('salary').fill_null(pl.col('salary').mean()))"
    )
    ledger.add_issue(issue)
    
    custom_code = "df.with_columns(pl.col('salary').fill_null(75000.0))"
    assert ledger.modify("iss_101", new_fix_code=custom_code, note="Hardcode baseline salary") is True
    
    modified = ledger.issues[0]
    assert modified.status == "modified"
    assert modified.fix_code == custom_code
    assert modified.user_note == "Hardcode baseline salary"
