import pytest
import polars as pl
from app.services.analysis.statistics import compute_summary

def test_skewness_and_kurtosis_on_uniform_distribution():
    """Verify that a uniform distribution yields skewness near 0 and negative excess kurtosis (platykurtic)."""
    # 100 uniformly spaced points
    vals = [float(i) for i in range(100)]
    df = pl.DataFrame({"uniform_metric": vals})
    
    summary = compute_summary(df, ["uniform_metric"])
    assert "uniform_metric" in summary
    stats = summary["uniform_metric"]
    
    # Skewness of symmetric uniform is ~0
    assert abs(stats["skewness"]) < 0.2
    # Excess kurtosis of uniform is approximately -1.2
    assert stats["kurtosis"] < 0.0
