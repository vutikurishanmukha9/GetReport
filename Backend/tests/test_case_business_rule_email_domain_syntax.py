import pytest
import polars as pl
from app.services.issue_ledger import _detect_business_rule_violations

def test_business_rule_email_domain_syntax_validation():
    """Verify detection of malformed email addresses missing @ or domain extensions."""
    df = pl.DataFrame({
        "customer_email": [
            "valid.user@example.com",
            "not-an-email",
            "missing_domain@",
            "@missing_username.org",
            "support@company.io"
        ]
    })
    
    issues = _detect_business_rule_violations(df)
    assert len(issues) >= 1
    email_issue = next(i for i in issues if i.column == "customer_email")
    assert email_issue.affected_rows == 3
    assert email_issue.issue_type == "format_issue"
