import pytest
import polars as pl
from app.services.dataset_versioning import build_schema_profile, compare_schema_profiles

def test_schema_drift_detects_removed_columns():
    """Verify schema drift flags when deprecated columns are dropped from data."""
    v1_df = pl.DataFrame({"user_id": [1, 2], "legacy_token": ["tok1", "tok2"], "score": [90, 95]})
    v2_df = pl.DataFrame({"user_id": [1, 2], "score": [90, 95]})
    
    p1 = build_schema_profile(v1_df)
    p2 = build_schema_profile(v2_df)
    
    drift = compare_schema_profiles(p1, p2)
    assert drift["status"] == "drift_detected"
    assert "legacy_token" in drift["removed_columns"]
