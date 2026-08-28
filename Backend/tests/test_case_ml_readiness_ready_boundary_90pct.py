import pytest
import polars as pl
from app.services.confidence_scoring import calculate_confidence_scores
from app.services.analysis.ml_readiness import calculate_ml_readiness

def test_ml_readiness_ready_boundary_90pct():
    """Verify clean dataset achieves 'Ready' status with score >= 85%."""
    df = pl.DataFrame({
        "feature_1": [float(i) for i in range(100)],
        "feature_2": [float(i * 2 + 5) for i in range(100)],
        "target": [float(i % 2) for i in range(100)]
    })
    
    conf = calculate_confidence_scores(df)
    ml_report = calculate_ml_readiness(conf, df)
    
    assert ml_report["status"] == "Ready"
    assert ml_report["score"] >= 85.0
