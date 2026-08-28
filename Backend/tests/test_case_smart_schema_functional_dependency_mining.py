import pytest
import polars as pl
from app.services.smart_schema import mine_functional_dependencies, analyze_smart_schema

def test_smart_schema_mines_functional_dependencies():
    """
    Verify mine_functional_dependencies identifies X -> Y determinants (e.g. zip_code uniquely determines state).
    """
    # 10 rows: zip_code -> state
    df = pl.DataFrame({
        "zip_code": ["78701", "78701", "78702", "75001", "75001", "90210", "90210", "90211", "10001", "10001"],
        "state":    ["TX",    "TX",    "TX",    "TX",    "TX",    "CA",    "CA",    "CA",    "NY",    "NY"],
        "random_metric": [10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
    })
    
    deps = mine_functional_dependencies(df)
    assert len(deps) >= 1
    
    zip_state_dep = next((d for d in deps if d["determinant_x"] == "zip_code" and d["dependent_y"] == "state"), None)
    assert zip_state_dep is not None
    assert zip_state_dep["is_exact"] is True
    assert zip_state_dep["strength"] == 1.0
    
    # Test integration in analyze_smart_schema
    res = analyze_smart_schema(df)
    assert len(res.functional_dependencies) >= 1
    assert "functional_dependencies" in res.to_dict()
