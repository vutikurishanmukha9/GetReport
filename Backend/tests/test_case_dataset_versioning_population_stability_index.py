import pytest
import numpy as np
import polars as pl
from app.services.dataset_versioning import calculate_population_stability_index

def test_population_stability_index_detects_distribution_drift():
    """
    Verify Population Stability Index (PSI) calculation:
    - Same distribution gives PSI < 0.10 (stable).
    - Shifted mean/variance distribution gives PSI >= 0.25 (critical drift).
    """
    np.random.seed(42)
    # Baseline: Normal(100, 15)
    base_vals = list(np.random.normal(100.0, 15.0, 500))
    
    # Target 1: Identical distribution -> stable
    target_stable_vals = list(np.random.normal(100.0, 15.0, 500))
    res_stable = calculate_population_stability_index(
        pl.Series("base", base_vals),
        pl.Series("target", target_stable_vals)
    )
    assert res_stable["status"] == "stable"
    assert res_stable["psi"] < 0.10
    
    # Target 2: Significantly shifted distribution Normal(140, 25) -> critical drift
    target_drift_vals = list(np.random.normal(140.0, 25.0, 500))
    res_drift = calculate_population_stability_index(
        pl.Series("base", base_vals),
        pl.Series("target", target_drift_vals)
    )
    assert res_drift["status"] in ("moderate_drift", "critical_drift")
    assert res_drift["psi"] >= 0.20
