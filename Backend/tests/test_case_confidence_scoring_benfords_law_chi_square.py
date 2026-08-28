import pytest
import numpy as np
import polars as pl
from app.services.confidence_scoring import calculate_benfords_law, calculate_confidence_scores

def test_confidence_scoring_benfords_law_natural_and_synthetic_distributions():
    """
    Verify Benford's Law Chi-Square test:
    - Natural logarithmic/exponential power-law distribution conforms (chi-square low, conforms=True).
    - Uniformly distributed synthetic random numbers trigger anomaly flag (conforms=False).
    """
    # 1. Natural power-law distribution (e.g. 10^x)
    np.random.seed(42)
    natural_vals = [float(10 ** (x)) for x in np.random.uniform(1.0, 5.0, 200)]
    df_natural = pl.DataFrame({"natural_amounts": natural_vals})
    
    res_natural = calculate_benfords_law(df_natural["natural_amounts"])
    assert res_natural["applicable"] is True
    assert res_natural["conforms"] is True
    assert res_natural["chi_square"] <= res_natural["critical_value_99"]
    
    # 2. Fabricated Uniform Distribution between 10.0 and 10,000.0 (Equal 11% digit spread)
    fabricated_vals = [float(np.random.uniform(10.0, 10000.0)) for _ in range(200)]
    df_fab = pl.DataFrame({"suspicious_ledger": fabricated_vals})
    
    res_fab = calculate_benfords_law(df_fab["suspicious_ledger"])
    assert res_fab["applicable"] is True
    assert res_fab["conforms"] is False
    assert res_fab["chi_square"] > res_fab["critical_value_99"]
    
    # Verify integration with calculate_confidence_scores
    conf = calculate_confidence_scores(df_fab)
    col_conf = next(c for c in conf.columns if c.column == "suspicious_ledger")
    assert any("Benford" in issue for issue in col_conf.issues)
