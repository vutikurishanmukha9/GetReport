import pytest
import polars as pl
from app.services.dataset_versioning import build_schema_profile, compare_schema_profiles

def test_schema_drift_detects_type_mutation():
    """Verify schema drift flags when an existing column mutated from Int64 to String."""
    v1_df = pl.DataFrame({"user_id": [101, 102, 103], "balance": [500.0, 1200.0, 80.0]})
    v2_df = pl.DataFrame({"user_id": ["U101", "U102", "U103"], "balance": [500.0, 1200.0, 80.0]}) # user_id became string
    
    p1 = build_schema_profile(v1_df)
    p2 = build_schema_profile(v2_df)
    
    drift = compare_schema_profiles(p1, p2)
    assert drift["status"] == "drift_detected"
    assert len(drift["type_changes"]) == 1
    assert drift["type_changes"][0]["column"] == "user_id"
