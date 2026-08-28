import pytest
import polars as pl
from app.services.data_processing import impute_multivariate_mice

def test_multivariate_mice_imputation_preserves_linear_covariance():
    """
    Verify MICE multivariate chained iterative regression imputation:
    Given strongly correlated variables Y = 2X + 1 with missing values in Y,
    MICE predicts the missing Y values based on X rather than replacing with a flat mean.
    """
    x_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    y_vals = [2.0 * x + 1.0 for x in x_vals]
    
    # Introduce nulls at index 3 (x=4, true y=9) and index 7 (x=8, true y=17)
    y_with_nulls = y_vals.copy()
    y_with_nulls[3] = None
    y_with_nulls[7] = None
    
    df = pl.DataFrame({
        "x": x_vals,
        "y": y_with_nulls
    })
    
    imputed_df = impute_multivariate_mice(df, numeric_cols=["x", "y"], max_iter=5)
    
    assert imputed_df["y"].null_count() == 0
    imputed_y = imputed_df["y"].to_list()
    
    # Check that imputed values closely match the true linear relationship
    assert imputed_y[3] == pytest.approx(9.0, abs=0.5)
    assert imputed_y[7] == pytest.approx(17.0, abs=0.5)
