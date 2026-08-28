import pytest
import polars as pl
from app.services.feature_engineering import _suggest_numeric_features

def test_feature_engineering_suggests_log_and_sqrt_transforms():
    """Verify numeric feature extractor suggests log and square root transforms on skewed positive values."""
    df = pl.DataFrame({
        "revenue": [10.0, 50.0, 200.0, 1500.0, 50000.0]
    })
    
    sugg = _suggest_numeric_features(df, "revenue")
    assert sugg.category == "numeric"
    names = [f["name"] for f in sugg.suggested_features]
    assert "revenue_log" in names or "revenue_sqrt" in names
