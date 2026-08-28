import pytest
import polars as pl
from app.services.smart_schema import _detect_date_in_numeric

def test_smart_schema_detects_excel_serial_dates():
    """Verify numeric column containing Excel serial dates (~45000) is recognized."""
    # Excel serial days for 2023-2024
    excel_dates = pl.Series("excel_days", [45200, 45201, 45202, 45203, 45204], dtype=pl.Int64)
    is_date, conf = _detect_date_in_numeric(excel_dates)
    assert is_date is True
    assert conf >= 0.80
