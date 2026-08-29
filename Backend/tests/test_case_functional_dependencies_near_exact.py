import polars as pl
import pytest
from app.services.smart_schema import mine_functional_dependencies

def test_functional_dependencies_mining():
    """
    Verify functional dependency mining finds exact mappings (X -> Y):
    e.g., zip_code -> state, department_id -> department_name.
    """
    zip_codes = ["90210", "90210", "10001", "10001", "30301", "30301", "90210", "10001", "30301", "90210", "10001", "30301"]
    states = ["CA", "CA", "NY", "NY", "GA", "GA", "CA", "NY", "GA", "CA", "NY", "GA"]
    users = [f"user_{i}" for i in range(len(zip_codes))]
    
    df = pl.DataFrame({
        "user_id": users,
        "zip_code": zip_codes,
        "state": states
    })
    
    deps = mine_functional_dependencies(df)
    assert isinstance(deps, list)
    assert len(deps) > 0
    
    # zip_code uniquely determines state
    dep_pairs = [(d["determinant_x"], d["dependent_y"]) for d in deps]
    assert ("zip_code", "state") in dep_pairs
