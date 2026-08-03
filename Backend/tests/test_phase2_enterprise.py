import pytest
import polars as pl
import pandas as pd
from datetime import datetime, timedelta

from app.services.foreign_key_integrity import detect_foreign_key_violations
from app.services.analysis.outliers import detect_outliers, detect_time_series_stl_outliers
from app.services.issue_ledger import (
    detect_issues,
    _detect_masked_null_issues,
    _detect_fuzzy_duplicate_issues,
    _detect_business_rule_violations,
    _detect_summary_rows_and_mid_headers,
    _detect_missing_value_issues,
)
from app.services.visualization import generate_stl_decomposition_chart, generate_er_relationship_diagram


def test_foreign_key_violations_detection():
    customers_df = pl.DataFrame({
        "customer_id": [101, 102, 103, 104],
        "name": ["Alice", "Bob", "Charlie", "Diana"],
    })

    orders_df = pl.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "cust_id": [101, 102, 109, 110, 103],
        "amount": [250.0, 120.0, 450.0, 90.0, 310.0],
    })

    res = detect_foreign_key_violations(
        primary_df=customers_df,
        foreign_df=orders_df,
        pk_col="customer_id",
        fk_col="cust_id",
        primary_name="customers",
        foreign_name="orders",
    )

    assert res["has_issue"] is True
    assert res["orphan_count"] == 2
    assert res["referential_completeness"] == 60.0
    assert 109 in res["sample_orphan_keys"]
    assert "semi" in res["fix_code"]


def test_foreign_key_perfect_integrity():
    customers_df = pl.DataFrame({"customer_id": [101, 102, 103]})
    orders_df = pl.DataFrame({"cust_id": [101, 102, 103, 101]})

    res = detect_foreign_key_violations(
        primary_df=customers_df,
        foreign_df=orders_df,
        pk_col="customer_id",
        fk_col="cust_id",
    )

    assert res["has_issue"] is False
    assert res["orphan_count"] == 0
    assert res["referential_completeness"] == 100.0


def test_time_series_stl_outliers_decomposition():
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(30)]
    values = [float(100 + (i % 7) * 20) for i in range(30)]
    values[15] = 950.0  # Residual anomaly

    df = pl.DataFrame({
        "transaction_date": dates,
        "revenue": values,
    })

    res = detect_time_series_stl_outliers(df, date_col="transaction_date", numeric_col="revenue")

    assert res["has_stl_outliers"] is True
    assert res["stl_outlier_count"] >= 1
    assert res["total_observations"] == 30


def test_time_series_stl_short_series_edge_case():
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(5)]
    df = pl.DataFrame({"transaction_date": dates, "revenue": [10.0, 20.0, 30.0, 40.0, 50.0]})

    res = detect_time_series_stl_outliers(df, date_col="transaction_date", numeric_col="revenue")
    assert res["has_stl_outliers"] is False
    assert "Insufficient" in res["reason"] or "Fewer than" in res["reason"]


def test_multivariate_isolation_forest_and_lof():
    # 25 normal records + 1 multivariate outlier (age 25 with $2M transaction)
    ages = [20 + i for i in range(25)] + [25]
    amounts = [100.0 + i * 10 for i in range(25)] + [2_000_000.0]

    df = pl.DataFrame({"age": ages, "transaction_amount": amounts})
    outliers = detect_outliers(df, ["age", "transaction_amount"])

    assert "_multivariate_summary" in outliers or "transaction_amount" in outliers
    if "_multivariate_summary" in outliers:
        assert outliers["_multivariate_summary"]["isolation_forest_count"] >= 1


def test_log_transformed_iqr_skewed_metrics():
    # Heavy right-skewed distribution (exponential revenue values)
    amounts = [10.0, 12.0, 15.0, 18.0, 22.0, 25.0, 30.0, 35.0, 40.0, 50.0, 1000.0, 5000.0]
    df = pl.DataFrame({"revenue": amounts})

    outliers = detect_outliers(df, ["revenue"])
    assert "revenue" in outliers
    assert outliers["revenue"]["is_heavy_skew"] is True
    assert outliers["revenue"]["log_outlier_count"] >= 1


def test_masked_null_placeholder_scanning():
    df = pl.DataFrame({
        "status": ["active", "N/A", "pending", "null", "?", "missing", "-999", "unknown", "active", "active"],
        "score": [95.0, -999.0, 88.0, 92.0, 9999.0, 85.0, 90.0, 91.0, 89.0, 93.0],
    })

    issues = _detect_masked_null_issues(df)
    assert len(issues) >= 1
    masked_cols = [iss.column for iss in issues]
    assert "status" in masked_cols or "score" in masked_cols


def test_fuzzy_entity_near_duplicates():
    df = pl.DataFrame({
        "company_name": [
            "Reliance Industries Ltd",
            "Reliance Industries Limited",
            "Acme Corp",
            "Acme Corporation",
            "Tata Motors",
            "Tata Motors Ltd",
            "Infosys Tech",
            "Infosys Technologies",
            "Google Inc",
            "Google LLC",
        ]
    })

    issues = _detect_fuzzy_duplicate_issues(df)
    assert len(issues) >= 1
    assert issues[0].issue_type == "duplicates"
    assert "Fuzzy" in issues[0].description


def test_email_syntax_business_rule_validation():
    df = pl.DataFrame({
        "user_email": ["alice@example.com", "bob_at_gmail.com", "charlie@company.org", "invalid-email-address"],
        "user_id": [1, 2, 3, 4],
    })

    issues = _detect_business_rule_violations(df)
    email_issues = [iss for iss in issues if iss.column == "user_email"]
    assert len(email_issues) >= 1
    assert email_issues[0].affected_rows == 2
    assert "invalid email address" in email_issues[0].description


def test_age_out_of_bounds_validation():
    df = pl.DataFrame({
        "customer_age": [25, 34, -5, 45, 214, 52, 60],
    })

    issues = _detect_business_rule_violations(df)
    age_issues = [iss for iss in issues if iss.column == "customer_age"]
    assert len(age_issues) >= 1
    assert age_issues[0].affected_rows == 2
    assert "clip(0, 120)" in age_issues[0].fix_code


def test_future_date_timestamp_validation():
    today = datetime.now()
    future_date = today + timedelta(days=365)
    
    df = pl.DataFrame({
        "created_at": [today - timedelta(days=10), today - timedelta(days=5), future_date],
    })

    issues = _detect_business_rule_violations(df)
    date_issues = [iss for iss in issues if iss.column == "created_at"]
    assert len(date_issues) >= 1
    assert date_issues[0].affected_rows == 1
    assert "future" in date_issues[0].description


def test_cross_field_date_chronology_violations():
    today = datetime.now()
    df = pl.DataFrame({
        "order_date": [today, today - timedelta(days=2), today - timedelta(days=10)],
        "ship_date": [today + timedelta(days=2), today - timedelta(days=5), today - timedelta(days=8)],  # row 1 has ship < order
    })

    issues = _detect_business_rule_violations(df)
    chronology_issues = [iss for iss in issues if "Chronology" in iss.description]
    assert len(chronology_issues) >= 1
    assert chronology_issues[0].affected_rows >= 1


def test_mid_file_headers_and_summary_rows():
    df = pl.DataFrame({
        "category": ["Electronics", "Clothing", "category", "Furniture", "Total", "Grand Total"],
        "sales": ["1000", "500", "sales", "800", "2300", "2300"],
    })

    issues = _detect_summary_rows_and_mid_headers(df)
    assert len(issues) >= 1
    descriptions = " ".join([iss.description for iss in issues])
    assert "repeated header" in descriptions or "summary/total" in descriptions


def test_mcar_mar_mnar_missingness_classification():
    # Sensitive field bias (income) -> MNAR
    df = pl.DataFrame({
        "customer_income": [50000.0, None, 75000.0, None, 120000.0, None, 60000.0, None, 90000.0, None],
    })

    issues = _detect_missing_value_issues(df)
    income_issues = [iss for iss in issues if iss.column == "customer_income"]
    assert len(income_issues) >= 1
    assert "MNAR" in income_issues[0].description
    assert "_is_missing" in income_issues[0].fix_code


def test_phase2_visualizations():
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(20)]
    values = [float(50 + (i % 7) * 10) for i in range(20)]
    df = pl.DataFrame({"dt": dates, "val": values})

    stl_chart = generate_stl_decomposition_chart(df, date_col="dt", numeric_col="val")
    assert stl_chart is None or isinstance(stl_chart, str)

    relationships = [{
        "primary_table": "Customers",
        "foreign_table": "Orders",
        "pk_col": "customer_id",
        "fk_col": "cust_id",
        "orphan_count": 2,
    }]
    er_chart = generate_er_relationship_diagram(relationships)
    assert isinstance(er_chart, str)
    assert len(er_chart) > 100
