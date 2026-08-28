import pytest
import polars as pl
from app.services.issue_ledger import disambiguate_categorical_entities, detect_issues, apply_remediation

def test_issue_ledger_entity_disambiguation_and_auto_remediation():
    """
    Verify Jaro-Winkler entity clustering detects near-duplicate variations and standardizes them to canonical modal strings.
    """
    # Dataset with intentional typographical variations
    df = pl.DataFrame({
        "city": [
            "San Francisco", "San Francisco", "San Francisco", "San Fransisco", # Typo
            "Los Angeles", "Los Angeles", "Los Angelos"                         # Typo
        ]
    })
    
    # 1. Test direct disambiguation
    res = disambiguate_categorical_entities(df, "city", threshold=0.85)
    reps = res["replacements"]
    assert "San Fransisco" in reps
    assert reps["San Fransisco"] == "San Francisco"
    assert "Los Angelos" in reps
    assert reps["Los Angelos"] == "Los Angeles"
    
    # 2. Test ledger detection and remediation
    ledger = detect_issues(df)
    fuzzy_issue = next((iss for iss in ledger.issues if iss.column == "city" and iss.issue_type == "duplicates"), None)
    assert fuzzy_issue is not None
    
    # Approve and remediate
    ledger.approve(fuzzy_issue.id)
    remediated_df = apply_remediation(df, ledger)
    
    unique_cities = remediated_df["city"].unique().to_list()
    assert "San Fransisco" not in unique_cities
    assert "Los Angelos" not in unique_cities
    assert set(unique_cities) == {"San Francisco", "Los Angeles"}
