import pytest
import polars as pl
from app.services.analysis.outliers import detect_outliers

def test_outlier_skewness_adjusted_bounds_and_consensus():
    """Verify detect_outliers calculates skewness-adjusted bounds and consensus agreement rating."""
    df = pl.DataFrame({
        "skewed_salaries": [30000.0, 32000.0, 31000.0, 33000.0, 35000.0, 40000.0, 45000.0, 60000.0, 80000.0, 300000.0]
    })
    
    outliers = detect_outliers(df, ["skewed_salaries"])
    assert "skewed_salaries" in outliers
    res = outliers["skewed_salaries"]
    
    assert "skew_adjusted_lower" in res
    assert "skew_adjusted_upper" in res
    assert "consensus_rating" in res
    # Upper adjusted bound must accommodate the positive skew
    assert res["skew_adjusted_upper"] >= res["upper_bound"]
