import pytest
import polars as pl
from app.services.feature_engineering import _compute_feature_importance_proxy

def test_feature_engineering_target_correlation_proxy():
    """Verify feature importance proxy computes correlation against labeled target variable."""
    df = pl.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature_b": [5.0, 4.0, 3.0, 2.0, 1.0],
        "target": [2.0, 4.0, 6.0, 8.0, 10.0]
    })
    
    importance = _compute_feature_importance_proxy(df, ["feature_a", "feature_b"], target_col="target")
    assert len(importance) == 2
    assert importance[0]["method"] == "target_correlation"
    assert abs(importance[0]["importance_score"]) > 0.90
