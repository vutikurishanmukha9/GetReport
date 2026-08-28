import pytest
import polars as pl
from app.services.feature_engineering import analyze_feature_engineering

def test_feature_engineering_full_analysis():
    """Verify end-to-end feature engineering analysis generates recommendations across all data types."""
    df = pl.DataFrame({
        "category": ["A", "B", "A", "C", "B"],
        "price": [10.5, 20.0, 15.0, 50.0, 12.0],
        "comments": ["Fast delivery", "Product was okay", "Great support", "Loved the item", "Superb"]
    })
    
    res = analyze_feature_engineering(df)
    assert res is not None
    d = res.to_dict()
    assert "encoding_recommendations" in d
    assert "scaling_recommendations" in d
    assert "feature_extraction" in d
    assert "feature_importance" in d
