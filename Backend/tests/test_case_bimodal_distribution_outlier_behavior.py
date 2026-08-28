import pytest
import polars as pl
import numpy as np
from app.services.analysis.outliers import detect_outliers

def test_bimodal_distribution_outlier_behavior():
    """Verify that bimodal distributions (two separated clusters) do not treat cluster centers as outliers."""
    np.random.seed(42)
    cluster1 = np.random.normal(loc=10.0, scale=1.0, size=100)
    cluster2 = np.random.normal(loc=100.0, scale=1.0, size=100)
    outlier = [1000.0]
    
    data = np.concatenate([cluster1, cluster2, outlier])
    df = pl.DataFrame({"bimodal": data})
    
    outliers = detect_outliers(df, ["bimodal"])
    assert "bimodal" in outliers
    assert outliers["bimodal"]["count"] >= 1
    assert outliers["bimodal"]["upper_bound"] > 100.0
