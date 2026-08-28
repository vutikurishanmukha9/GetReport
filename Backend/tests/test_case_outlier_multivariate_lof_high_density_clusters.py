import pytest
import polars as pl
import numpy as np
from app.services.analysis.outliers import detect_outliers

def test_outlier_multivariate_density_detection():
    """Verify multivariate outlier detection on 2D correlated features."""
    np.random.seed(42)
    # Dense main cluster
    x = np.random.normal(loc=10.0, scale=1.0, size=100)
    y = 2.0 * x + np.random.normal(loc=0.0, scale=0.5, size=100)
    
    # Severe isolated anomaly
    x = np.append(x, [100.0])
    y = np.append(y, [-50.0])
    
    df = pl.DataFrame({"feature_1": x, "feature_2": y})
    outliers = detect_outliers(df, ["feature_1", "feature_2"])
    
    assert "feature_1" in outliers
    assert "feature_2" in outliers
    assert outliers["feature_1"]["count"] >= 1
    assert outliers["feature_2"]["count"] >= 1
