import os
import tempfile
import polars as pl
import pytest

from app.services.data_processing import (
    _sanitize_and_coerce_df,
    _is_phone_column,
    clean_data,
    load_dataframe,
    inspect_dataset,
    CleaningReport,
)
from app.services.transformation_dag import create_dag


def test_is_phone_column_detection():
    """Verify phone column recognition and financial exclusion."""
    s_phone = pl.Series("PHONE", ["-512080055", "-9611110285"])
    assert _is_phone_column("PHONE", s_phone) is True
    assert _is_phone_column("mobile_no", s_phone) is True
    assert _is_phone_column("customer_cell", s_phone) is True
    assert _is_phone_column("emergency_contact", s_phone) is True

    s_profit = pl.Series("profit", [-50000, 20000])
    assert _is_phone_column("profit", s_profit) is False
    assert _is_phone_column("account_balance", s_profit) is False
    assert _is_phone_column("revenue", s_profit) is False


def test_user_screenshot_exact_ten_rows():
    """Verify exact 10 phone rows from user screenshot are cleaned of leading '-'."""
    user_rows = [
        -512080055,
        -9611110285,
        -7278532425,
        -2850676250,
        -4754102251,
        -3882211456,
        -184980424,
        -3556434368,
        -1651623197,
        -2004615264,
    ]
    df = pl.DataFrame({"PHONE": user_rows})
    sanitized = _sanitize_and_coerce_df(df)

    expected = [
        "512080055",
        "9611110285",
        "7278532425",
        "2850676250",
        "4754102251",
        "3882211456",
        "184980424",
        "3556434368",
        "1651623197",
        "2004615264",
    ]
    assert sanitized["phone"].to_list() == expected
    assert sanitized["phone"].dtype in (pl.Utf8, pl.String)


def test_string_phone_column_leading_dashes():
    """Verify string phone column strips various leading dash/whitespace patterns."""
    raw_values = [
        "-512080055",
        "--9611110285",
        "- (512) 080-055",
        "-+1 555-1234",
        "-",
        "   -7278532425",
        None,
    ]
    df = pl.DataFrame({"mobile": raw_values})
    sanitized = _sanitize_and_coerce_df(df)

    expected = [
        "512080055",
        "9611110285",
        "(512) 080-055",
        "+1 555-1234",
        None,
        "7278532425",
        None,
    ]
    assert sanitized["mobile"].to_list() == expected
    assert sanitized["mobile"].dtype in (pl.Utf8, pl.String)


def test_phone_internal_hyphens_preserved():
    """Verify internal formatting and hyphens are preserved, only leading dash is stripped."""
    raw_values = [
        "-512-080-0555",
        "512-080-0555",
        "+1-512-080-0555",
        "-+1-512-080-0555",
    ]
    df = pl.DataFrame({"contact_number": raw_values})
    sanitized = _sanitize_and_coerce_df(df)

    expected = [
        "512-080-0555",
        "512-080-0555",
        "+1-512-080-0555",
        "+1-512-080-0555",
    ]
    assert sanitized["contact_number"].to_list() == expected


def test_financial_columns_never_corrupted():
    """Verify financial columns with negative numbers are not misclassified as phones."""
    df = pl.DataFrame({
        "balance": [-1500000, 2000000, -350000],
        "net_profit": [-50000.75, 120000.50, -8200.00],
    })
    sanitized = _sanitize_and_coerce_df(df)

    assert sanitized["balance"].to_list() == [-1500000, 2000000, -350000]
    assert sanitized["balance"].dtype == pl.Int64
    assert sanitized["net_profit"].to_list() == [-50000.75, 120000.50, -8200.00]
    assert sanitized["net_profit"].dtype == pl.Float64


def test_clean_data_audit_and_dag_tracking():
    """Verify clean_data logs phone cleaning in CleaningReport and TransformationDAG."""
    df = pl.DataFrame({
        "PHONE": ["-512080055", "-9611110285", "7278532425"],
        "name": ["Alice", "Bob", "Charlie"],
    })
    dag = create_dag(df, "test_dataset")
    cleaned_df, report, dag = clean_data(df, rules=None, dag=dag, dataset_name="test_dataset")

    assert report.phone_numbers_cleaned == 2
    assert cleaned_df["phone"].to_list() == ["512080055", "9611110285", "7278532425"]

    dag_dict = dag.to_dict()
    operations = [n["operation"] for n in dag_dict.get("nodes", {}).values()]
    assert "clean_phone_numbers" in operations


def test_end_to_end_csv_ingestion_and_preview():
    """Verify end-to-end load_dataframe and inspect_dataset produce clean preview rows."""
    csv_content = "PHONE,NAME\n-512080055,Alice\n-9611110285,Bob\n-7278532425,Charlie\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        temp_path = f.name

    try:
        loaded_df = load_dataframe(temp_path)
        quality = inspect_dataset(loaded_df)

        preview = quality["preview"]
        assert len(preview) == 3
        assert preview[0]["phone"] == "512080055"
        assert preview[1]["phone"] == "9611110285"
        assert preview[2]["phone"] == "7278532425"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
