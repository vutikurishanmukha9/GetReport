import pytest
import polars as pl
from app.services.analysis.core import analyze_dataset

def test_adversarial_extreme_aspect_ratio_shapes():
    """Verify performance and stability on 1-row wide data and 5000-row tall data."""
    # 1. Wide data: 1 row with 50 columns
    cols_50 = {f"metric_{i}": [float(i * 10)] for i in range(50)}
    df_wide = pl.DataFrame(cols_50)
    analysis_wide = analyze_dataset(df_wide)
    assert analysis_wide["metadata"]["total_columns"] == 50
    assert analysis_wide["metadata"]["total_rows"] == 1

    # 2. Tall data: 5,000 rows with 1 column
    df_tall = pl.DataFrame({"long_series": list(range(5000))})
    analysis_tall = analyze_dataset(df_tall)
    assert analysis_tall["metadata"]["total_rows"] == 5000
    assert analysis_tall["metadata"]["total_columns"] == 1
