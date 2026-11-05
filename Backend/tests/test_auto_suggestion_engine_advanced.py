import polars as pl
import pytest
from app.services.auto_suggestion_engine import (
    analyze,
    build_remediation_plan,
    apply_suggestion,
    apply_suggestions,
    profile_columns,
    EngineConfig,
)


def test_full_pipeline_analyze():
    """Verify full end-to-end analyze() pipeline produces quality scores, plan, and suggestions."""
    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"],
        "price": ["$10.50", "$20.00", "$30.25", "$40.00", "$50.00", "$60.00", "$70.00", "$80.00", "$90.00", "$100.00"],
        "status": ["Active", "active", "ACTIVE", "Active", "Active", "Inactive", "Active", "Active", "Active", "Active"],
        "notes": ["fine", "N/A", "ok", "-", "none", "good", "all set", "N/A", "ok", "done"],
        "constant_col": ["FIXED"] * 10,
    })

    report = analyze(df)
    assert "summary" in report
    assert "quality" in report
    assert "suggestions" in report
    assert "plan" in report
    assert "profile" in report

    # Quality scores before and after
    quality = report["quality"]
    assert "overall_before" in quality
    assert "overall_after" in quality
    assert quality["overall_after"] >= quality["overall_before"]

    # Remediations suggested
    actions = [s["action"] for s in report["suggestions"]]
    assert "clean_masked_nulls" in actions
    assert "coerce_numeric" in actions
    assert "drop_column" in actions


def test_apply_suggestions_execution():
    """Verify apply_suggestions applies the remediation plan correctly to clean the DataFrame."""
    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "amount": ["$100.00", "$200.00", "$300.00", "$400.00", "$500.00", "$600.00", "$700.00", "$800.00", "$900.00", "$1,000.00"],
        "dummy": ["X"] * 10,
    })

    report = analyze(df)
    plan = [s for s in report["suggestions"] if s["id"] in report["plan"]]
    cleaned_df, logs = apply_suggestions(df, plan)

    assert "dummy" not in cleaned_df.columns
    assert cleaned_df["amount"].dtype in (pl.Int64, pl.Float64)
    assert cleaned_df["amount"][0] == 100.0
    assert cleaned_df["amount"][9] == 1000.0


def test_pii_detection():
    """Verify PII (emails, SSNs) is identified and flagged with privacy recommendations."""
    df = pl.DataFrame({
        "customer_email": [f"user_{i}@company.com" for i in range(20)],
    })
    report = analyze(df)
    pii_suggestions = [s for s in report["suggestions"] if s["type"] == "pii_detected"]
    assert len(pii_suggestions) == 1
    assert pii_suggestions[0]["action"] == "hash_pii"
    assert pii_suggestions[0]["target_pillar"] == "Compliance"


def test_duplicate_row_detection_and_removal():
    """Verify duplicate row detection and removal in the engine."""
    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 1, 2],
        "val": ["A", "B", "C", "D", "A", "B"]
    })
    report = analyze(df)
    dup_s = [s for s in report["suggestions"] if s["action"] == "drop_duplicate_rows"]
    assert len(dup_s) == 1
    assert dup_s[0]["count"] == 2

    cleaned_df = apply_suggestion(df, dup_s[0])
    assert cleaned_df.height == 4


def test_profile_columns():
    """Verify profile_columns semantic profiling."""
    df = pl.DataFrame({
        "user_id": [1, 2, 3, 4],
        "email": ["a@b.com", "c@d.com", "e@f.com", "g@h.com"],
        "is_active": [True, False, True, True],
        "score": [98.5, 87.0, 92.3, 76.4]
    })
    profile = profile_columns(df)
    by_col = {p["column"]: p for p in profile}
    assert by_col["user_id"]["semantic_type"] == "identifier"
    assert by_col["email"]["semantic_type"] == "email"
    assert by_col["is_active"]["semantic_type"] == "boolean"
    assert by_col["score"]["semantic_type"] == "numeric_continuous"
