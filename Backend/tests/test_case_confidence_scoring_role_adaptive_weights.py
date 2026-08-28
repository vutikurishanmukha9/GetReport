import pytest
import polars as pl
from app.services.confidence_scoring import _get_role_adaptive_weights, calculate_confidence_scores

def test_confidence_scoring_role_adaptive_weights():
    """Verify that role-adaptive calibration adjusts pillar weights according to semantic role and sums to 1.0."""
    # Test ID weights
    w_id = _get_role_adaptive_weights("customer_id", pl.Int64, 100, 100)
    assert sum(w_id) == pytest.approx(1.0, 1e-5)
    assert w_id[2] >= 0.35  # Validity / uniqueness weighted heavily
    
    # Test Date weights
    w_date = _get_role_adaptive_weights("transaction_date", pl.Utf8, 80, 100)
    assert sum(w_date) == pytest.approx(1.0, 1e-5)
    assert w_date[1] >= 0.30  # Consistency weighted heavily
    
    # Test Numeric Metric weights
    w_metric = _get_role_adaptive_weights("revenue", pl.Float64, 90, 100)
    assert sum(w_metric) == pytest.approx(1.0, 1e-5)
    assert w_metric[3] >= 0.20  # Stability weighted heavily
