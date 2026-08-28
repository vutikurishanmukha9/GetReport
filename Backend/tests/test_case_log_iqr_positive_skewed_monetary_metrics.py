import pytest
import polars as pl
from app.services.analysis.outliers import detect_outliers

def test_log_iqr_skewed_monetary_metrics():
    """Verify log-IQR bound calculations for log-normally distributed monetary transaction amounts."""
    amounts = [10.0, 12.0, 15.0, 18.0, 22.0, 25.0, 30.0, 45.0, 80.0, 2000.0]
    df = pl.DataFrame({"transaction_amount": amounts})
    
    outliers = detect_outliers(df, ["transaction_amount"])
    assert "transaction_amount" in outliers
    assert outliers["transaction_amount"]["count"] >= 1
    assert outliers["transaction_amount"]["upper_bound"] > 0
