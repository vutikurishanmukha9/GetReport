import pytest
import polars as pl
from app.services.dataset_versioning import build_schema_profile, compare_schema_profiles

def test_schema_drift_detects_new_added_column():
    """Verify schema drift correctly reports newly introduced columns in subsequent dataset versions."""
    v1_df = pl.DataFrame({"user_id": [1, 2, 3], "revenue": [10.0, 20.0, 30.0]})
    v2_df = pl.DataFrame({"user_id": [1, 2, 3], "revenue": [10.0, 20.0, 30.0], "country": ["US", "DE", "FR"]})
    
    p1 = build_schema_profile(v1_df)
    p2 = build_schema_profile(v2_df)
    
    drift = compare_schema_profiles(p1, p2)
    assert drift["status"] == "drift_detected"
    assert "country" in drift["added_columns"]
