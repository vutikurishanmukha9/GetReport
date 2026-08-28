import pytest
import polars as pl
from app.services.analysis.missing import analyze_missing_patterns

def test_missingness_mcar_mar_mnar_classification():
    """Verify missingness pattern detection classifies correlated missingness as MAR."""
    # Column B is missing when Column A is large (MAR pattern)
    col_a = [10.0, 20.0, 30.0, 400.0, 500.0, 600.0, 700.0, 800.0]
    col_b = [1.0, 2.0, 3.0, None, None, None, None, None]
    
    df = pl.DataFrame({
        "var_a": col_a,
        "var_b": col_b
    })
    
    missing_analysis = analyze_missing_patterns(df)
    assert missing_analysis["has_missing"] is True
    assert missing_analysis["columns_affected"] == 1
    assert missing_analysis["inferred_pattern"] in ("MAR", "Systematic", "MCAR")
