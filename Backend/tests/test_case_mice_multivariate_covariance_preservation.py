import polars as pl
import numpy as np
import pytest
from app.services.data_processing import impute_multivariate_mice

def test_mice_multivariate_covariance_preservation():
    """
    Verify MICE multivariate chained regression maintains joint linear covariance
    and correlation between features without variance collapse.
    """
    np.random.seed(42)
    n = 2000
    
    # Generate correlated bivariate normal variables (r ~ 0.85)
    mean = [50, 100]
    cov = [[25, 34], [34, 64]]  # sigma_x=5, sigma_y=8, corr = 34 / (5*8) = 0.85
    data = np.random.multivariate_normal(mean, cov, n)
    
    df = pl.DataFrame({
        "feature_x": data[:, 0],
        "feature_y": data[:, 1]
    })
    
    original_corr = float(np.corrcoef(df["feature_x"].to_numpy(), df["feature_y"].to_numpy())[0, 1])
    assert 0.80 <= original_corr <= 0.90
    
    # Introduce 20% MCAR missingness into feature_y
    mask = np.random.rand(n) < 0.20
    df_missing = df.with_columns([
        pl.when(pl.Series(mask)).then(None).otherwise(pl.col("feature_y")).alias("feature_y")
    ])
    assert df_missing["feature_y"].null_count() > 0
    
    # Run MICE imputation
    imputed_df = impute_multivariate_mice(df_missing, numeric_cols=["feature_x", "feature_y"])
    assert imputed_df["feature_y"].null_count() == 0
    
    # Verify imputed correlation is preserved close to original
    imputed_corr = float(np.corrcoef(imputed_df["feature_x"].to_numpy(), imputed_df["feature_y"].to_numpy())[0, 1])
    assert abs(imputed_corr - original_corr) < 0.08
