import pytest
import json
import polars as pl
from app.services.data_quality_suite import (
    DataQualityTestSuite,
    generate_baseline_test_suite,
    export_openmetadata_json,
    export_great_expectations_json
)
from app.services.pii_guard import PIIGuard, mask_pii_value, verify_luhn_checksum
from app.services.dataset_graph_builder import build_dataset_graph


def test_openmetadata_data_quality_suite_assertions():
    # Construct a sample dataset with known valid and violating records
    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "age": [25, 30, 45, 120, 29],  # 120 violates between(18, 100)
        "status": ["active", "pending", "active", "active", "invalid_status"],  # "invalid_status"
        "score": [85.0, 90.0, None, 75.0, 95.0]  # None violates not_null
    })

    suite = DataQualityTestSuite(name="Customer Quality Suite")
    suite.add_assertion(
        test_type="column_values_to_be_unique",
        column="id",
        description="ID must be unique"
    )
    suite.add_assertion(
        test_type="column_values_to_be_between",
        column="age",
        params={"min_val": 18, "max_val": 100},
        description="Age must be between 18 and 100"
    )
    suite.add_assertion(
        test_type="column_values_to_be_in_set",
        column="status",
        params={"allowed_values": ["active", "pending", "closed"]},
        description="Status must be in active, pending, closed"
    )
    suite.add_assertion(
        test_type="column_values_to_be_not_null",
        column="score",
        description="Score cannot be null"
    )

    results = suite.run(df)
    assert results["total_tests"] == 4
    assert results["passed"] == 1  # only 'id' passed
    assert results["failed"] == 3
    assert results["status"] == "FAILED"

    # Export tests
    om_json = export_openmetadata_json(results)
    parsed_om = json.loads(om_json)
    assert "testCases" in parsed_om
    assert len(parsed_om["testCases"]) == 4

    ge_json = export_great_expectations_json(results)
    parsed_ge = json.loads(ge_json)
    assert "expectations" in parsed_ge
    assert len(parsed_ge["expectations"]) == 4


def test_generate_baseline_test_suite():
    clean_df = pl.DataFrame({
        "user_id": [101, 102, 103, 104],
        "category": ["A", "B", "A", "B"],
        "amount": [10.5, 20.0, 15.2, 33.1]
    })
    suite = generate_baseline_test_suite(clean_df)
    results = suite.run(clean_df)
    assert results["passed"] == results["total_tests"]
    assert results["status"] == "PASSED"


def test_pii_guard_scanner_and_masking():
    # Verify Luhn Checksum
    assert verify_luhn_checksum("4532015112830366") is True  # Valid test Visa
    assert verify_luhn_checksum("4532015112830367") is False  # Invalid checksum

    # Verify Masking
    assert mask_pii_value("john.doe@getreport.ai", "EMAIL") == "j***@getreport.ai"
    assert mask_pii_value("4532-0151-1283-0366", "CREDIT_CARD") == "****-****-****-0366"
    assert mask_pii_value("123-45-6789", "SSN") == "***-**-6789"
    assert mask_pii_value("+1 (555) 234-5678", "PHONE") == "(***) ***-5678"

    # Verify DataFrame PII Scanner
    df_pii = pl.DataFrame({
        "username": ["alice", "bob", "carol", "dan"],
        "contact_email": [
            "alice@example.com",
            "bob@enterprise.org",
            "carol@domain.net",
            "dan@startup.io"
        ],
        "national_id": [
            "123-45-6789",
            "987-65-4321",
            "234-56-7890",
            "345-67-8901"
        ],
        "description": [
            "Software engineer",
            "Product manager",
            "Data scientist",
            "Security analyst"
        ]
    })

    scan_res = PIIGuard.scan_dataframe(df_pii)
    assert scan_res["has_pii"] is True
    assert scan_res["total_pii_columns"] == 2
    assert "contact_email" in scan_res["findings"]
    assert scan_res["findings"]["contact_email"]["pii_type"] == "EMAIL"
    assert "national_id" in scan_res["findings"]
    assert scan_res["findings"]["national_id"]["pii_type"] == "SSN"
    assert "description" not in scan_res["findings"]


def test_dataset_graph_with_dq_and_pii():
    mock_job = {
        "filename": "customers.csv",
        "analysis": {
            "metadata": {"total_rows": 100, "total_columns": 3},
            "summary": {
                "email": {"null_count": 0, "null_percentage": 0.0},
                "age": {"null_count": 5, "null_percentage": 5.0}
            },
            "columns": {
                "email": {"data_type": "String"},
                "age": {"data_type": "Int64"}
            },
            "time_series": {
                "changepoints": {
                    "age": [{"index": 45, "mean_shift": 15.0, "timestamp": "2026-03-01", "pre_mean": 25.0, "post_mean": 40.0, "llr_score": 12.5}]
                }
            }
        },
        "pii": {
            "findings": {
                "email": {"pii_type": "EMAIL", "confidence": "HIGH", "sample_masked": "j***@domain.com"}
            }
        },
        "data_quality": {
            "results": [
                {"column": "age", "test_type": "column_values_to_be_between", "success": False, "description": "Age outside bounds", "violation_count": 5}
            ]
        }
    }

    graph_store = build_dataset_graph("task_phase2_test", mock_job)
    assert graph_store.graph.has_node("col:email")
    assert graph_store.graph.has_node("pii:email")
    assert graph_store.graph.has_node("dq_failure:age:column_values_to_be_between")
    assert graph_store.graph.has_node("changepoint:age:45")

    # Verify edge connections
    assert graph_store.graph.has_edge("col:email", "pii:email")
    assert graph_store.graph.has_edge("col:age", "dq_failure:age:column_values_to_be_between")
    assert graph_store.graph.has_edge("col:age", "changepoint:age:45")
