import pytest
import polars as pl
from app.services.analysis.missing import analyze_missing_patterns

def test_missingness_complete_data_matrix():
    """Verify that a 100% complete dataset reports no missing patterns cleanly."""
    df = pl.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": ["x", "y", "z", "w", "v"],
        "c": [10.5, 20.2, 30.1, 40.0, 50.9]
    })
    
    missing_res = analyze_missing_patterns(df)
    assert missing_res["has_missing"] is False
    assert "No missing values" in missing_res["message"]
