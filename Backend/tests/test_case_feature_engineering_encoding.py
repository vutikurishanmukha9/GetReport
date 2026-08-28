import pytest
import polars as pl
from app.services.feature_engineering import _suggest_encoding

def test_feature_engineering_encoding_selection():
    """Verify optimal encoding choices for binary, low-cardinality, and high-cardinality columns."""
    df = pl.DataFrame({
        "binary_flag": (["YES", "NO"] * 50),
        "low_cardinality": (["A", "B", "C", "D", "E"] * 20),
        "high_cardinality": [f"ID_{i}" for i in range(100)]
    })
    
    rec_binary = _suggest_encoding(df, "binary_flag")
    assert rec_binary.recommended_encoding == "binary"
    
    rec_low = _suggest_encoding(df, "low_cardinality")
    assert rec_low.recommended_encoding == "one_hot"
    
    rec_high = _suggest_encoding(df, "high_cardinality")
    assert rec_high.recommended_encoding == "hash_or_embedding"
