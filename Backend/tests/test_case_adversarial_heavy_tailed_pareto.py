import pytest
import numpy as np
import polars as pl
from app.services.analysis.outliers import detect_outliers

def test_adversarial_heavy_tailed_pareto_skewness():
    """Verify log-IQR bound calculations on severe Pareto power-law distributions with heavy skew."""
    np.random.seed(42)
    pareto_data = (np.random.pareto(a=1.2, size=500) + 1.0) * 100.0
    
    df = pl.DataFrame({
        "revenue_pareto": pareto_data,
        "normal_metric": np.random.normal(loc=50.0, scale=5.0, size=500)
    })
    
    outliers = detect_outliers(df, ["revenue_pareto", "normal_metric"])
    assert "revenue_pareto" in outliers
    
    pareto_outliers = outliers["revenue_pareto"]
    assert pareto_outliers["is_heavy_skew"] is True
    assert pareto_outliers["skewness"] > 2.0
    assert pareto_outliers["count"] > 0
    assert pareto_outliers["upper_bound"] > 0
