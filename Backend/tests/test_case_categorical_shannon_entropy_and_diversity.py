import pytest
import polars as pl
from app.services.analysis.core import analyze_dataset

def test_categorical_shannon_entropy_and_simpson_diversity():
    """Verify categorical distribution computes Shannon entropy, normalized evenness, and Simpson diversity."""
    # Balanced 4-category distribution
    balanced_cats = ["A", "B", "C", "D"] * 25 # Total 100 rows, each 25%
    df = pl.DataFrame({
        "category": balanced_cats
    })
    
    analysis = analyze_dataset(df)
    cat_res = analysis["categorical_distribution"]["category"]
    
    assert "shannon_entropy" in cat_res
    assert "normalized_evenness" in cat_res
    assert "simpson_diversity" in cat_res
    assert "rare_categories_count" in cat_res
    
    # For a uniform distribution across 4 categories:
    # H = -4 * (0.25 * log2(0.25)) = 2.0
    # Normalized evenness = 2.0 / log2(4) = 1.0
    assert cat_res["shannon_entropy"] == pytest.approx(2.0, 1e-2)
    assert cat_res["normalized_evenness"] == pytest.approx(1.0, 1e-2)
    assert cat_res["simpson_diversity"] == pytest.approx(0.75, 1e-2)
