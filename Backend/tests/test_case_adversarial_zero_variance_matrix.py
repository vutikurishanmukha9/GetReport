import pytest
import polars as pl
from app.services.analysis.core import analyze_dataset

def test_adversarial_entirely_zero_variance_matrix():
    """Verify that a matrix where all columns have zero variance completes analysis without mathematical divide-by-zero error."""
    df = pl.DataFrame({
        "const_num1": [10.0] * 50,
        "const_num2": [20.0] * 50,
        "const_str": ["STATIC"] * 50
    })
    
    analysis = analyze_dataset(df)
    assert analysis is not None
    assert analysis["metadata"]["total_rows"] == 50
    assert analysis["metadata"]["total_columns"] == 3
