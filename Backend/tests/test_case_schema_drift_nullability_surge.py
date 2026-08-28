import pytest
import polars as pl
from app.services.dataset_versioning import build_schema_profile, compare_schema_profiles

def test_schema_drift_detects_nullability_surges():
    """Verify schema drift flags >=5 percentage point surge in missing values."""
    # V1: 0% nulls
    v1_df = pl.DataFrame({"status": ["active", "active", "active", "active", "active"] * 20})
    # V2: 20% nulls
    v2_df = pl.DataFrame({"status": (["active"] * 80) + ([None] * 20)})
    
    p1 = build_schema_profile(v1_df)
    p2 = build_schema_profile(v2_df)
    
    drift = compare_schema_profiles(p1, p2)
    assert drift["status"] == "drift_detected"
    assert len(drift["nullability_changes"]) >= 1
    assert drift["nullability_changes"][0]["column"] == "status"
