from __future__ import annotations
import polars as pl
import numpy as np
import logging
from typing import Any

logger = logging.getLogger(__name__)

def detect_time_columns(df: pl.DataFrame) -> list[str]:
    """Identify datetime columns in the DataFrame."""
    return [c for c, t in df.schema.items() if t in (pl.Date, pl.Datetime)]

def detect_trend(df: pl.DataFrame, time_col: str, value_col: str) -> dict[str, Any]:
    """
    Detect trend using linear regression slope.
    Returns trend direction, strength, and p-value approximation.
    """
    try:
        # Sort by time
        sorted_df = df.select([time_col, value_col]).drop_nulls().sort(time_col)
        if sorted_df.height < 10:
            return {"detected": False, "reason": "Insufficient data points"}
        
        # Create numeric time index
        y = sorted_df[value_col].to_numpy()
        x = np.arange(len(y))
        
        # Linear regression
        n = len(x)
        sum_x, sum_y = x.sum(), y.sum()
        sum_xy = (x * y).sum()
        sum_x2 = (x ** 2).sum()
        
        denom = n * sum_x2 - sum_x ** 2
        if denom == 0:
            return {"detected": False, "reason": "Constant values"}
        
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        
        # R-squared
        y_pred = slope * x + intercept
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Trend direction
        if abs(slope) < 1e-10:
            direction = "flat"
        elif slope > 0:
            direction = "upward"
        else:
            direction = "downward"
        
        # Strength classification
        if r_squared >= 0.7:
            strength = "strong"
        elif r_squared >= 0.3:
            strength = "moderate"
        else:
            strength = "weak"
        
        return {
            "detected": True,
            "direction": direction,
            "slope": round(float(slope), 6),
            "r_squared": round(float(r_squared), 4),
            "strength": strength,
            "data_points": n
        }
    except Exception as e:
        logger.warning(f"Trend detection failed: {e}")
        return {"detected": False, "reason": str(e)}

def detect_seasonality(df: pl.DataFrame, time_col: str, value_col: str) -> dict[str, Any]:
    """
    Detect seasonality using autocorrelation at common lags (7=weekly, 30=monthly, 365=yearly).
    """
    try:
        sorted_df = df.select([time_col, value_col]).drop_nulls().sort(time_col)
        y = sorted_df[value_col].to_numpy()
        n = len(y)
        
        if n < 60:  # Need enough data for seasonality
            return {"detected": False, "reason": "Insufficient data for seasonality analysis"}
        
        # Detrend (subtract mean)
        y_detrend = y - y.mean()
        
        # Check common seasonal lags
        seasonal_lags = {7: "weekly", 30: "monthly", 90: "quarterly", 365: "yearly"}
        detected_patterns = []
        
        for lag, period_name in seasonal_lags.items():
            if n < lag * 2:
                continue
            
            # Calculate autocorrelation at this lag
            autocorr = np.corrcoef(y_detrend[:-lag], y_detrend[lag:])[0, 1]
            
            if np.isnan(autocorr):
                continue
            
            # Strong autocorrelation suggests seasonality
            if abs(autocorr) >= 0.3:
                detected_patterns.append({
                    "period": period_name,
                    "lag": lag,
                    "autocorrelation": round(float(autocorr), 4),
                    "strength": "strong" if abs(autocorr) >= 0.6 else "moderate"
                })
        
        if detected_patterns:
            return {
                "detected": True,
                "patterns": detected_patterns,
                "primary_pattern": detected_patterns[0]["period"]
            }
        else:
            return {"detected": False, "reason": "No significant seasonal patterns found"}
            
    except Exception as e:
        logger.warning(f"Seasonality detection failed: {e}")
        return {"detected": False, "reason": str(e)}

def analyze_time_series(df: pl.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
    """
    Full time series analysis: trend + seasonality for each numeric column.
    """
    time_cols = detect_time_columns(df)
    if not time_cols:
        return {"has_time_series": False, "reason": "No datetime columns found"}
    
    time_col = time_cols[0]  # Use first datetime column
    results = {
        "has_time_series": True,
        "time_column": time_col,
        "analyses": {}
    }
    
    # Analyze top 5 numeric columns
    for col in numeric_cols[:5]:
        trend = detect_trend(df, time_col, col)
        seasonality = detect_seasonality(df, time_col, col)
        
        results["analyses"][col] = {
            "trend": trend,
            "seasonality": seasonality
        }
    
    return results
