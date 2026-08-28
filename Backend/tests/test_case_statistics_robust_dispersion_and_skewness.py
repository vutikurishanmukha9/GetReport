import pytest
import polars as pl
from app.services.analysis.statistics import compute_summary

def test_statistics_robust_dispersion_and_skewness():
    """Verify compute_summary calculates IQR, MAD, Bowley skewness, CV, and 5% trimmed mean."""
    # Right-skewed distribution
    df = pl.DataFrame({
        "skewed_vals": [10.0, 12.0, 11.0, 10.5, 11.5, 13.0, 100.0, 500.0]
    })
    
    summary = compute_summary(df, ["skewed_vals"])
    assert "skewed_vals" in summary
    stats = summary["skewed_vals"]
    
    # Check that all non-parametric metrics are computed
    assert "iqr" in stats
    assert "mad" in stats
    assert "bowley_skewness" in stats
    assert "coefficient_of_variation" in stats
    assert "trimmed_mean_5pct" in stats
    
    assert stats["iqr"] > 0
    assert stats["mad"] > 0
    assert stats["trimmed_mean_5pct"] < stats["mean"] # Trimmed mean resists the 500.0 spike
