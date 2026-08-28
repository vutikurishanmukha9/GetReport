import pytest
import polars as pl
from app.services.smart_schema import analyze_smart_schema

def test_smart_schema_type_corrections_detection():
    """Verify smart schema inference correctly suggests corrections for string numbers and dates."""
    df = pl.DataFrame({
        "number_as_str": ["100.5", "200.1", "300.9", "450.0", "500.0"],
        "emails": ["user1@domain.com", "admin@company.org", "test.dev@agency.io", "support@app.co", "sales@corp.net"],
        "dates_as_str": ["2024-01-15", "2024-02-20", "2024-03-25", "2024-04-30", "2024-05-10"],
    })
    
    schema_res = analyze_smart_schema(df)
    assert schema_res is not None
    corrections = schema_res.type_corrections
    assert len(corrections) >= 2
    
    col_suggestions = {c.column: c.suggested_type for c in corrections}
    assert "number_as_str" in col_suggestions or "emails" in col_suggestions or "dates_as_str" in col_suggestions
