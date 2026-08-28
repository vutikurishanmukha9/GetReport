import pytest
import polars as pl
from app.services.feature_engineering import cyclical_fourier_encoding, synthesize_interaction_features

def test_cyclical_fourier_embeddings_and_interaction_synthesis():
    """Verify cyclical trigonometric Fourier embeddings and interaction feature synthesis."""
    # 24-hour cycle (0, 6, 12, 18, 24)
    df = pl.DataFrame({
        "hour": [0.0, 6.0, 12.0, 18.0],
        "temperature": [15.0, 20.0, 28.0, 22.0],
        "humidity": [80.0, 60.0, 40.0, 50.0]
    })
    
    # 1. Fourier encoding for 24-hour cycle
    cyclical_df = cyclical_fourier_encoding(df, "hour", period=24.0)
    assert "hour_sin" in cyclical_df.columns
    assert "hour_cos" in cyclical_df.columns
    
    # Hour 0: sin(0) = 0.0, cos(0) = 1.0
    # Hour 6: sin(pi/2) = 1.0, cos(pi/2) = 0.0
    hour_0 = cyclical_df.filter(pl.col("hour") == 0.0).row(0, named=True)
    assert hour_0["hour_sin"] == pytest.approx(0.0, 1e-3)
    assert hour_0["hour_cos"] == pytest.approx(1.0, 1e-3)
    
    # 2. Interaction synthesis
    interaction_df = synthesize_interaction_features(df, ["temperature", "humidity"])
    assert "temperature_x_humidity" in interaction_df.columns
    assert "temperature_div_humidity" in interaction_df.columns
