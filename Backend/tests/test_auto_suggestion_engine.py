import polars as pl
import pytest
from app.services.auto_suggestion_engine import (
    generate_comprehensive_suggestions,
    detect_masked_null_suggestions,
    detect_dirty_numeric_suggestions,
    detect_constant_column_suggestions,
    detect_text_hygiene_suggestions,
)
from app.services.data_processing import inspect_dataset, clean_data


def test_detect_masked_nulls_and_remediation():
    """Verify masked null strings ('N/A', 'NULL', '-') are detected with high confidence and cleaned to nulls."""
    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
        "country": ["USA", "Canada", "N/A", "Germany", "null", "Japan", "-", "UK"]
    })
    suggestions = detect_masked_null_suggestions(df)
    assert len(suggestions) == 1
    sugg = suggestions[0]
    assert sugg["type"] == "masked_nulls"
    assert sugg["column"] == "country"
    assert sugg["action"] == "clean_masked_nulls"
    assert sugg["confidence_score"] >= 90.0
    assert sugg["target_pillar"] == "Completeness"
    assert "N/A" in sugg["before_sample"] or "null" in sugg["before_sample"]

    # Verify remediation in clean_data
    cleaned_df, report, dag = clean_data(df, rules={"country": {"action": "clean_masked_nulls"}})
    null_count = cleaned_df["country"].null_count()
    assert null_count == 3  # N/A, null, - should all become null
    assert "USA" in cleaned_df["country"].to_list()


def test_detect_dirty_numeric_and_coercion():
    """Verify currency and percentage text are flagged with coerce_numeric and cast to float."""
    df = pl.DataFrame({
        "revenue": ["$1,250.00", "$3,400.50", "€450.00", "$8,900.25", "$250.00", "$15,000.00"]
    })
    suggestions = detect_dirty_numeric_suggestions(df)
    assert len(suggestions) == 1
    sugg = suggestions[0]
    assert sugg["type"] == "dirty_numeric"
    assert sugg["column"] == "revenue"
    assert sugg["action"] == "coerce_numeric"
    assert sugg["target_pillar"] == "Validity"
    assert sugg["confidence_score"] >= 90.0

    # Verify remediation in clean_data
    cleaned_df, report, dag = clean_data(df, rules={"revenue": {"action": "coerce_numeric"}})
    assert cleaned_df["revenue"].dtype == pl.Float64
    assert cleaned_df["revenue"][0] == 1250.0
    assert cleaned_df["revenue"][1] == 3400.5


def test_detect_constant_column_and_drop():
    """Verify zero-variance constant columns are suggested for dropping."""
    df = pl.DataFrame({
        "status": ["Active"] * 25,
        "amount": [10.0, 20.0, 30.0] * 8 + [40.0]
    })
    suggestions = detect_constant_column_suggestions(df)
    assert len(suggestions) == 1
    sugg = suggestions[0]
    assert sugg["type"] == "constant_column"
    assert sugg["column"] == "status"
    assert sugg["action"] == "drop_column"
    assert sugg["confidence_score"] >= 90.0
    assert sugg["target_pillar"] == "Stability"

    # Verify remediation in clean_data
    cleaned_df, report, dag = clean_data(df, rules={"status": {"action": "drop_column"}})
    assert "status" not in cleaned_df.columns
    assert "amount" in cleaned_df.columns


def test_detect_text_hygiene_and_trim():
    """Verify trailing whitespace is flagged and cleaned."""
    df = pl.DataFrame({
        "city": ["  New York ", "London  ", " Tokyo", "Paris", " Berlin "]
    })
    suggestions = detect_text_hygiene_suggestions(df)
    assert len(suggestions) == 1
    sugg = suggestions[0]
    assert sugg["type"] == "text_whitespace"
    assert sugg["column"] == "city"
    assert sugg["action"] == "clean_text"
    assert sugg["target_pillar"] == "Consistency"

    # Verify remediation in clean_data
    cleaned_df, report, dag = clean_data(df, rules={"city": {"action": "clean_text"}})
    assert cleaned_df["city"][0] == "New York"
    assert cleaned_df["city"][1] == "London"
    assert cleaned_df["city"][2] == "Tokyo"


def test_inspect_dataset_returns_suggestion_summary():
    """Verify inspect_dataset populates suggestion_summary and comprehensive issues."""
    df = pl.DataFrame({
        "id": ["ID_001", "ID_002", None, "ID_004", "ID_005"],
        "cost": ["$10.00", "$20.00", "$30.00", "$40.00", "$50.00"],
    })
    res = inspect_dataset(df)
    assert "issues" in res
    assert "suggestion_summary" in res
    assert res["suggestion_summary"]["total"] >= 2
    assert res["suggestion_summary"]["high_confidence_count"] >= 1
