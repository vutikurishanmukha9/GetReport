import pytest
import polars as pl
from app.services.analysis.core import analyze_dataset

def test_adversarial_control_characters_in_column_names():
    """Verify system handles column names containing spaces, newlines, tabs, and punctuation safely."""
    df = pl.DataFrame({
        "User\tID": [1, 2, 3],
        "Gross Revenue ($)": [100.0, 250.0, 400.0],
        "Line\nBreak": ["A", "B", "C"]
    })
    
    analysis = analyze_dataset(df)
    assert analysis is not None
    assert analysis["metadata"]["total_columns"] == 3
    assert analysis["metadata"]["total_rows"] == 3
