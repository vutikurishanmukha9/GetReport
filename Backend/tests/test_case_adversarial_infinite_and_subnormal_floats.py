import pytest
import polars as pl
from app.services.analysis.statistics import compute_summary
from app.services.analysis.outliers import detect_outliers
from app.services.confidence_scoring import calculate_confidence_scores
from app.services.analysis.ml_readiness import calculate_ml_readiness

def test_adversarial_infinite_and_subnormal_floats_resilience():
    """Verify system stability when encountering all-null columns, extreme subnormal and massive floats."""
    df = pl.DataFrame({
        "all_null_numeric": pl.Series([None, None, None, None, None], dtype=pl.Float64),
        "all_null_str": pl.Series([None, None, None, None, None], dtype=pl.String),
        "extreme_floats": pl.Series([1e300, -1e300, 1e-300, 0.0, None], dtype=pl.Float64),
        "single_value": pl.Series([42.0, 42.0, 42.0, 42.0, 42.0], dtype=pl.Float64)
    })
    
    # Must compute summary on extreme float column without unhandled crash
    stats = compute_summary(df, ["extreme_floats", "single_value"])
    assert "extreme_floats" in stats
    assert "single_value" in stats
    
    outliers = detect_outliers(df, ["extreme_floats", "single_value"])
    assert isinstance(outliers, dict)
    
    conf = calculate_confidence_scores(df)
    assert conf.dataset_confidence < 90.0
    
    ml_ready = calculate_ml_readiness(conf, df)
    assert ml_ready["score"] <= 70.0
