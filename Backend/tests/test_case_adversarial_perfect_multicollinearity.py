import pytest
import numpy as np
import polars as pl
from app.services.analysis.core import analyze_dataset

def test_adversarial_perfect_multicollinear_correlation_pairs():
    """Verify exact 1.0 and -1.0 linear correlation pairs are identified in analysis."""
    x = [float(i) for i in range(100)]
    y_exact = [2.5 * v + 15.0 for v in x]
    y_inverse = [-1.0 * v for v in x]
    
    df = pl.DataFrame({
        "feature_x": x,
        "feature_y_exact": y_exact,
        "feature_y_inv": y_inverse
    })
    
    analysis = analyze_dataset(df)
    assert "strong_correlations" in analysis
    
    corrs = analysis.get("strong_correlations", [])
    assert len(corrs) >= 2
    
    r_values = [abs(c["r_value"]) for c in corrs]
    assert any(pytest.approx(r, 0.01) == 1.0 for r in r_values)
