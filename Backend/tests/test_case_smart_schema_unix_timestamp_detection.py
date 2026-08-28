import pytest
import polars as pl
from app.services.smart_schema import _detect_date_in_numeric

def test_smart_schema_detects_unix_timestamps():
    """Verify numeric column containing Unix epoch seconds is identified as potential timestamp."""
    # Timestamps around year 2024 (~1.7 billion seconds)
    timestamps = pl.Series("ts", [1704067200, 1704153600, 1704240000, 1704326400, 1704412800], dtype=pl.Int64)
    is_date, conf = _detect_date_in_numeric(timestamps)
    assert is_date is True
    assert conf >= 0.70
