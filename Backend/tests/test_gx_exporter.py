"""
tests/test_gx_exporter.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the GreatExpectationsSuiteExporter service.
Verifies Great Expectations contract structure, column validations, and JSON serialization.
"""
import json
import pytest
import polars as pl
from app.services.gx_exporter import GreatExpectationsSuiteExporter


@pytest.fixture
def clean_dataset():
    return pl.DataFrame({
        "order_id": [f"ORD-{i:04d}" for i in range(1, 51)],
        "amount": [float(i * 10) for i in range(1, 51)],
        "status": ["COMPLETED"] * 30 + ["PENDING"] * 15 + ["REFUNDED"] * 5,
        "customer_rating": [4.5] * 45 + [None] * 5
    })


def test_gx_suite_generation(clean_dataset):
    exporter = GreatExpectationsSuiteExporter(suite_name="orders_quality_contract")
    suite = exporter.generate_suite_from_polars(
        clean_dataset,
        quality_report={"grade": "A+", "score": 98.5}
    )

    assert suite["expectation_suite_name"] == "orders_quality_contract"
    assert suite["data_asset_type"] == "Dataset"
    assert "meta" in suite
    assert suite["meta"]["quality_grade"] == "A+"
    assert suite["meta"]["confidence_score"] == 98.5
    assert suite["meta"]["dataset_rows"] == 50
    assert suite["meta"]["dataset_columns"] == 4

    # Verify expectations list
    expectations = suite["expectations"]
    assert len(expectations) > 0

    exp_types = [e["expectation_type"] for e in expectations]
    assert "expect_table_row_count_to_be_between" in exp_types
    assert "expect_table_columns_to_match_ordered_list" in exp_types
    assert "expect_column_to_exist" in exp_types
    assert "expect_column_values_to_not_be_null" in exp_types
    assert "expect_column_values_to_be_unique" in exp_types


def test_gx_suite_incorporates_approved_issues(clean_dataset):
    approved_rules = [
        {
            "id": "rule_01",
            "column": "customer_rating",
            "issue_type": "missing_values",
            "status": "approved"
        },
        {
            "id": "rule_02",
            "column": "order_id",
            "issue_type": "duplicates",
            "status": "approved"
        }
    ]

    exporter = GreatExpectationsSuiteExporter(suite_name="remediated_orders")
    suite = exporter.generate_suite_from_polars(
        clean_dataset,
        quality_report={"grade": "A", "score": 92.0},
        approved_issues=approved_rules
    )

    # Check that customer_rating has mostly=1.0 expectation added
    rating_non_null = [
        e for e in suite["expectations"]
        if e["expectation_type"] == "expect_column_values_to_not_be_null"
        and e["kwargs"].get("column") == "customer_rating"
        and e["kwargs"].get("mostly") == 1.0
    ]
    assert len(rating_non_null) >= 1


def test_gx_suite_json_serialization(clean_dataset):
    exporter = GreatExpectationsSuiteExporter(suite_name="serialized_suite")
    suite = exporter.generate_suite_from_polars(clean_dataset)
    json_str = exporter.export_json(suite)

    # Verify JSON parses cleanly
    parsed = json.loads(json_str)
    assert parsed["expectation_suite_name"] == "serialized_suite"
    assert isinstance(parsed["expectations"], list)
