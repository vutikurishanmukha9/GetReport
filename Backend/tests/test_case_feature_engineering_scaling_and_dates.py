import pytest
import polars as pl
from app.services.feature_engineering import _suggest_scaling, _suggest_datetime_features

def test_feature_engineering_scaling_and_datetime_features():
    """Verify scaling recommendations and datetime feature extraction recommendations."""
    df = pl.DataFrame({
        "skewed_income": [1000.0, 1200.0, 1500.0, 2000.0, 500000.0],
        "signup_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    })
    
    scaling_rec = _suggest_scaling(df, "skewed_income")
    assert scaling_rec is not None
    assert scaling_rec.recommended_scaler in ("robust", "log_then_standard", "standard", "power_transform")
    
    date_features_sugg = _suggest_datetime_features("signup_date")
    assert date_features_sugg is not None
    feature_names = [f["name"] for f in date_features_sugg.suggested_features]
    assert "signup_date_year" in feature_names
    assert "signup_date_month" in feature_names
    assert "signup_date_day" in feature_names
    assert "signup_date_is_weekend" in feature_names
