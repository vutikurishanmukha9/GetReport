import pytest
import polars as pl
from app.services.smart_schema import _detect_derived_columns

def test_smart_schema_detects_derived_sum_columns():
    """Verify detection of derived columns where Total = Col_A + Col_B."""
    a = [10.0, 20.0, 30.0, 40.0, 50.0]
    b = [5.0, 15.0, 25.0, 35.0, 45.0]
    total = [15.0, 35.0, 55.0, 75.0, 95.0]
    
    df = pl.DataFrame({
        "item_price": a,
        "tax": b,
        "total_amount": total
    })
    
    derived = _detect_derived_columns(df)
    assert len(derived) >= 1
    assert any(d.relationship_type == "derived" for d in derived)
