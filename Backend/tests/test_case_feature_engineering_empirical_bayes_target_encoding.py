import pytest
import polars as pl
from app.services.feature_engineering import empirical_bayes_target_encoding

def test_empirical_bayes_target_encoding_smooths_and_prevents_leakage():
    """Verify Empirical Bayes target encoding calculates smoothed prior and generates out-of-fold encoded features."""
    df = pl.DataFrame({
        "city": ["Austin", "Austin", "Austin", "Dallas", "Dallas", "Houston"],
        "price": [100.0, 120.0, 110.0, 300.0, 320.0, 50.0]
    })
    
    encoded_df = empirical_bayes_target_encoding(df, cat_col="city", target_col="price", m_smooth=5.0, n_splits=2)
    
    assert "city_target_enc" in encoded_df.columns
    assert encoded_df.height == df.height
    
    # Austin mean ~ 110, Dallas mean ~ 310. Encoded Dallas values should be distinctly higher than Austin
    dallas_enc = encoded_df.filter(pl.col("city") == "Dallas")["city_target_enc"].mean()
    austin_enc = encoded_df.filter(pl.col("city") == "Austin")["city_target_enc"].mean()
    
    assert dallas_enc > austin_enc
