import pytest
import polars as pl
from app.services.feature_engineering import _compute_feature_importance_proxy

def test_feature_engineering_variance_proxy():
    """Verify unsupervised feature importance proxy uses coefficient of variation when target is absent."""
    df = pl.DataFrame({
        "high_var_col": [1.0, 50.0, 200.0, 1000.0, 5000.0],
        "low_var_col": [10.0, 10.1, 10.0, 10.2, 10.1]
    })
    
    importance = _compute_feature_importance_proxy(df, ["high_var_col", "low_var_col"], target_col=None)
    assert len(importance) == 2
    assert importance[0]["column"] == "high_var_col"
    assert importance[0]["method"] == "variance_proxy"
