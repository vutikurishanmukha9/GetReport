import os
import tempfile
import polars as pl
import pytest

from app.services.data_processing import (
    load_dataframe,
    clean_data,
    inspect_dataset,
    _detect_csv_parameters,
    _sanitize_and_coerce_df,
    ParseError
)

def test_messy_csv_metadata_preamble_and_currency_coercion():
    """
    Test loading a messy CSV containing 2 lines of metadata preamble,
    currency strings ($1,250.50), percentages (15%), and string nulls ("N/A", "-").
    """
    messy_content = (
        "CONFIDENTIAL DATA REPORT\n"
        "Generated on 2026-07-27\n"
        "Product ID; Price ; Discount ; Status\n"
        "P101; $1,250.50 ; 10% ; Active\n"
        "P102; $500.00 ; 15% ; N/A\n"
        "P103; -$75.25 ; 0% ; -\n"
    )
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        tmp.write(messy_content)
        tmp_path = tmp.name

    try:
        df = load_dataframe(tmp_path)
        
        # Verify columns were cleaned to snake_case
        assert "product_id" in df.columns
        assert "price" in df.columns
        assert "discount" in df.columns
        assert "status" in df.columns
        
        # Verify Currency string "$1,250.50" was automatically coerced to numeric Float64
        assert df["price"].dtype in (pl.Float64, pl.Float32)
        assert df["price"].to_list() == [1250.50, 500.00, -75.25]
        
        # Verify Percentage string "10%" was coerced to Float64
        assert df["discount"].dtype in (pl.Float64, pl.Float32)
        assert df["discount"].to_list() == [10.0, 15.0, 0.0]
        
        # Verify string nulls ("N/A", "-") were converted to true nulls
        assert df["status"].null_count() == 2
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_iso_8859_1_encoded_csv():
    """
    Test loading a non-UTF8 (ISO-8859-1 / Latin-1) encoded CSV file.
    """
    latin_text = "city;temperature;note\nZürich;22.5;Café\nSão Paulo;28.0;Música\n"
    
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp:
        tmp.write(latin_text.encode("iso-8859-1"))
        tmp_path = tmp.name

    try:
        df = load_dataframe(tmp_path)
        assert df.height == 2
        assert "city" in df.columns
        assert "temperature" in df.columns
        assert df["temperature"].dtype in (pl.Float64, pl.Float32)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_duplicate_and_empty_header_sanitization():
    """
    Test that duplicate or blank column names are sanitized into unique snake_case names.
    """
    df = pl.DataFrame({
        "  Age  ": [25, 30],
        "Age": [100, 200],
        "": ["a", "b"]
    })
    
    sanitized = _sanitize_and_coerce_df(df)
    assert len(set(sanitized.columns)) == 3
    assert "age" in sanitized.columns
    assert "age_1" in sanitized.columns
    assert "column_3" in sanitized.columns


def test_infinity_and_nan_scrubbing():
    """
    Test that float infinity values (inf, -inf) are scrubbed to Null to prevent JSON/stats crashes.
    """
    df = pl.DataFrame({
        "val": [1.0, float("inf"), float("-inf"), 4.5]
    })
    
    sanitized = _sanitize_and_coerce_df(df)
    assert sanitized["val"].null_count() == 2
    assert sanitized["val"].drop_nulls().to_list() == [1.0, 4.5]
