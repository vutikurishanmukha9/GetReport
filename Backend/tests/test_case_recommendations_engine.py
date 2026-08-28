import pytest
import polars as pl
from app.services.recommendations import generate_recommendations

def test_recommendations_engine_generation():
    """Verify that rule-based recommendations are generated for anomalous datasets."""
    df = pl.DataFrame({
        "income": [1000.0, 1200.0, 1500.0, 2000.0, 500000.0],
        "age": [25, 30, 35, 40, 45]
    })
    mock_analysis = {
        "metadata": {"total_rows": 1000, "total_columns": 5, "missing_value_pct": 12.5},
        "summary": {
            "income": {"mean": 65000.0, "skew": 3.2, "null_count": 50},
            "age": {"mean": 38.0, "skew": 0.1, "null_count": 0}
        },
        "strong_correlations": [
            {"column_a": "ad_spend", "column_b": "sales", "r_value": 0.94}
        ],
        "outliers": {
            "income": {"count": 45, "percentage": 4.5}
        }
    }
    
    recs = generate_recommendations(df, "finance", mock_analysis)
    assert recs is not None
    assert recs.to_dict()["total_count"] >= 1
