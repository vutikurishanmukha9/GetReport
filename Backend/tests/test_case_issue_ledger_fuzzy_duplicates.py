import pytest
import polars as pl
from app.services.issue_ledger import _detect_fuzzy_duplicate_issues, _detect_duplicate_issues

def test_issue_ledger_fuzzy_duplicates_detection():
    """Verify detection of fuzzy near-duplicate text entities and exact duplicate rows."""
    df_dups = pl.DataFrame({
        "name": ["Acme Corp", "Beta LLC", "Acme Corp", "Gamma Inc", "Beta LLC"],
        "revenue": [1000, 2000, 1000, 3000, 2000]
    })
    
    dup_issues = _detect_duplicate_issues(df_dups)
    assert len(dup_issues) == 1
    assert dup_issues[0].issue_type == "duplicates"
    assert dup_issues[0].affected_rows == 2

    df_fuzzy = pl.DataFrame({
        "company_name": ["Microsoft Corp", "Microsft Corp", "Google LLC", "Googlle LLC", "Apple Inc"] * 3
    })
    fuzzy_issues = _detect_fuzzy_duplicate_issues(df_fuzzy)
    assert len(fuzzy_issues) >= 1
    assert fuzzy_issues[0].column == "company_name"
