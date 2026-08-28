import pytest
import polars as pl
from app.services.smart_schema import analyze_smart_schema

def test_smart_schema_detects_mixed_data_formats():
    """Verify smart schema analyzes columns with diverse formats."""
    df = pl.DataFrame({
        "mixed_col": ["100", "200.5", "300", "400.99", "500"],
        "label": ["A", "B", "C", "D", "E"]
    })
    
    schema_res = analyze_smart_schema(df)
    assert schema_res is not None
    assert isinstance(schema_res.type_corrections, list)
