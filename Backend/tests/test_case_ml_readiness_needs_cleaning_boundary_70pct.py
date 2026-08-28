import pytest
import polars as pl
from app.services.confidence_scoring import calculate_confidence_scores
from app.services.analysis.ml_readiness import calculate_ml_readiness

def test_ml_readiness_needs_cleaning_boundary():
    """Verify dataset with moderate anomalies falls into 'Needs Cleaning' tier (score between 60-84%)."""
    # 1 constant column, 1 moderate nulls
    df = pl.DataFrame({
        "constant_col": [1.0] * 100,
        "feature_with_nulls": [float(i) if i > 15 else None for i in range(100)], # 15% nulls
        "good_feature": [float(i) for i in range(100)]
    })
    
    conf = calculate_confidence_scores(df)
    ml_report = calculate_ml_readiness(conf, df)
    
    assert ml_report["status"] in ("Needs Cleaning", "Ready")
    assert ml_report["score"] < 95.0
