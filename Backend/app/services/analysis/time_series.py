from __future__ import annotations
import polars as pl
import numpy as np
import logging
import math
from typing import Any

import re

logger = logging.getLogger(__name__)

def detect_time_columns(df: pl.DataFrame) -> list[str]:
    """Identify datetime columns in the DataFrame, including mixed-format string date columns."""
    direct = [c for c, t in df.schema.items() if t in (pl.Date, pl.Datetime)]
    if direct:
        return direct

    # Scan string columns for date headers or patterns
    date_candidates = []
    date_keywords = ["date", "time", "timestamp", "created_at", "order_date", "dob", "period", "day"]
    for c in df.columns:
        if df[c].dtype in (pl.Utf8, pl.String, pl.Object):
            c_lower = c.lower()
            if any(k in c_lower for k in date_keywords):
                sample = df[c].drop_nulls().head(60).to_list()
                if sample:
                    matches = sum(1 for s in sample if re.search(r"\d{1,4}[-/. ]\d{1,2}[-/. ]\d{1,4}|\d{4}[-]\d{2}", str(s)))
                    if matches >= len(sample) * 0.6:
                        date_candidates.append(c)
    return date_candidates

def sens_slope(series: np.ndarray) -> float:
    """Non-parametric Sen's slope estimator (median of pairwise slopes)."""
    n = len(series)
    if n < 2:
        return 0.0
    if n > 1000:
        step = n // 1000 + 1
        series = series[::step]
        n = len(series)
    i, j = np.triu_indices(n, k=1)
    diff = j - i
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = (series[j] - series[i]) / diff
        finite_slopes = slopes[np.isfinite(slopes)]
    return float(np.median(finite_slopes)) if len(finite_slopes) > 0 else 0.0

def mann_kendall_trend_test(series: np.ndarray) -> dict[str, Any]:
    """
    Non-parametric Mann-Kendall test for monotonic trends.
    Invariant to outliers and non-linear monotonic growth/decay.
    """
    n = len(series)
    if n < 4:
        return {"trend": "insufficient_data", "tau": 0.0, "p_value": 1.0}
        
    # Limit sample size for quadratic calculation efficiency if n > 2000
    if n > 2000:
        step = n // 2000 + 1
        series = series[::step]
        n = len(series)
        
    # Calculate Mann-Kendall S statistic
    # S = sum(sgn(x_k - x_j)) for k > j (later observation minus earlier observation)
    diffs = series[None, :] - series[:, None]
    s = int(np.sign(diffs[np.triu_indices(n, k=1)]).sum())
    
    # Kendall's Tau: 2 * S / (n * (n - 1))
    max_pairs = (n * (n - 1)) / 2.0
    tau = s / max_pairs if max_pairs > 0 else 0.0
    
    # Variance of S: n * (n - 1) * (2n + 5) / 18
    var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
    
    # Z-statistic
    if s > 0:
        z = (s - 1.0) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1.0) / math.sqrt(var_s)
    else:
        z = 0.0
        
    p_value = math.erfc(abs(z) / math.sqrt(2))
    
    if p_value < 0.05:
        direction = "upward" if tau > 0 else "downward"
    else:
        direction = "flat_or_no_trend"
        
    return {
        "s_stat": s,
        "kendall_tau": round(float(tau), 4),
        "z_score": round(float(z), 4),
        "p_value": round(float(p_value), 4),
        "statistically_significant": bool(p_value < 0.05),
        "direction": direction
    }

def detect_trend(df: pl.DataFrame, time_col: str, value_col: str) -> dict[str, Any]:
    """
    Detect trend using both Linear Regression and Non-Parametric Mann-Kendall tests.
    Returns trend direction, strength, Sen's slope, and p-value approximation.
    """
    try:
        # Chronological sorting with robust mixed-date parsing if time_col is string
        time_series = df[time_col]
        if time_series.dtype in (pl.Utf8, pl.String, pl.Object):
            # Pure Polars datetime parsing (Rust-accelerated, zero-dependency)
            parsed_df = df.select([
                pl.coalesce([
                    pl.col(time_col).str.to_datetime(strict=False),
                    pl.col(time_col).str.to_date(strict=False).cast(pl.Datetime),
                ]).alias("__parsed_dt"),
                pl.col(value_col).cast(pl.Float64, strict=False).alias("__val")
            ])
            # Fallback for non-standard date strings if standard parsing yields all nulls
            if parsed_df["__parsed_dt"].null_count() == parsed_df.height and df.height > 0:
                try:
                    from dateutil.parser import parse as parse_date
                    dates = []
                    for val in time_series.to_list():
                        try:
                            dates.append(parse_date(str(val)) if val else None)
                        except Exception:
                            dates.append(None)
                    parsed_df = pl.DataFrame([
                        pl.Series("__parsed_dt", dates, dtype=pl.Datetime),
                        df[value_col].cast(pl.Float64, strict=False).alias("__val")
                    ])
                except Exception:
                    pass

            clean_df = parsed_df.filter(
                pl.col("__parsed_dt").is_not_null() &
                pl.col("__val").is_not_null() &
                pl.col("__val").is_finite() &
                (pl.col("__val").abs() < 1e12)
            ).sort("__parsed_dt")
            y = clean_df["__val"].to_numpy()
            y = np.clip(y[np.isfinite(y)], -1e9, 1e9)
        else:
            clean_df = df.filter(pl.col(time_col).is_not_null() & pl.col(value_col).is_not_null() & pl.col(value_col).is_finite() & (pl.col(value_col).abs() < 1e12))
            sorted_df = clean_df.select([time_col, value_col]).sort(time_col)
            y = sorted_df[value_col].to_numpy()
            y = np.clip(y[np.isfinite(y)], -1e9, 1e9)
            
        if len(y) < 10:
            return {"detected": False, "reason": "Insufficient data points"}
        
        # Create numeric time index (float64 prevents int64 overflow on 100k+ series)
        x = np.arange(len(y), dtype=np.float64)
        y = y.astype(np.float64)
        
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
        p_value = _linear_trend_p_value(float(slope), x, y, y_pred)
        
        # Non-parametric Mann-Kendall verification & Sen's slope
        mk_res = mann_kendall_trend_test(y)
        s_slope = sens_slope(y)
        
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
            "sens_slope": round(float(s_slope), 6),
            "r_squared": round(float(r_squared), 4),
            "p_value": round(float(p_value), 4) if p_value is not None else None,
            "statistically_significant": bool(p_value is not None and p_value < 0.05),
            "strength": strength,
            "data_points": n,
            "mann_kendall": mk_res
        }
    except Exception as e:
        logger.warning(f"Trend detection failed: {e}")
        return {"detected": False, "reason": str(e)}


def _linear_trend_p_value(
    slope: float,
    x: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
) -> float | None:
    """Approximate two-sided p-value for the fitted slope."""
    n = len(x)
    if n < 3:
        return None

    x_var = float(((x - x.mean()) ** 2).sum())
    if x_var == 0:
        return None

    residual_ss = float(((y - y_pred) ** 2).sum())
    residual_var = residual_ss / (n - 2)
    if residual_var <= 0:
        return 0.0

    standard_error = math.sqrt(residual_var / x_var)
    if standard_error == 0:
        return 0.0

    t_stat = abs(slope / standard_error)
    return _normal_two_sided_p_value(t_stat)


def _normal_two_sided_p_value(z_score: float) -> float:
    """Normal approximation used to avoid adding a SciPy dependency."""
    return math.erfc(z_score / math.sqrt(2))

def detect_seasonality(df: pl.DataFrame, time_col: str, value_col: str) -> dict[str, Any]:
    """
    Detect seasonality using autocorrelation at common lags (7=weekly, 30=monthly, 90=quarterly, 365=yearly).
    """
    try:
        sorted_df = df.select([time_col, value_col]).drop_nulls().sort(time_col)
        y = sorted_df[value_col].cast(pl.Float64, strict=False).to_numpy()
        y = y[np.isfinite(y) & (np.abs(y) < 1e12)]
        n = len(y)
        
        if n < 60:
            return {"detected": False, "reason": "Insufficient data for seasonality analysis"}
        
        # Detrend using linear regression residuals
        x = np.arange(n, dtype=np.float64)
        with np.errstate(all="ignore"):
            if np.all(y == y[0]):
                y_detrend = y - y[0]
            else:
                slope, intercept = np.polyfit(x, y, 1)
                y_detrend = y - (slope * x + intercept)
            
            residual_std = float(np.std(y_detrend))
        if residual_std < 1e-4:
            return {"detected": False, "reason": "No residual variance after detrending"}

        seasonal_lags = {7: "weekly", 30: "monthly", 90: "quarterly", 365: "yearly"}
        detected_patterns = []
        
        for lag, period_name in seasonal_lags.items():
            if n < lag * 2:
                continue
            
            s1 = y_detrend[:-lag]
            s2 = y_detrend[lag:]
            with np.errstate(all="ignore"):
                if np.std(s1) < 1e-4 or np.std(s2) < 1e-4:
                    continue
                autocorr = np.corrcoef(s1, s2)[0, 1]
            
            if np.isnan(autocorr):
                continue
            
            if autocorr >= 0.3:
                detected_patterns.append({
                    "period": period_name,
                    "lag_days": lag,
                    "autocorrelation": round(float(autocorr), 4),
                    "strength": "strong" if autocorr >= 0.6 else "moderate"
                })
        
        if detected_patterns:
            return {
                "detected": True,
                "patterns": detected_patterns,
                "primary_period": detected_patterns[0]["period"]
            }
        else:
            return {"detected": False, "reason": "No significant autocorrelation at standard lags"}
            
    except Exception as e:
        logger.warning(f"Seasonality detection failed: {e}")
        return {"detected": False, "reason": str(e)}

def analyze_time_series(df: pl.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
    """
    Tier 1: Complete Time Series Analysis for all numeric columns against detected time columns.
    Enhanced with Meta Kats CUSUM Changepoints, Holt-Winters Forecasting, and TSFeatures.
    """
    time_cols = detect_time_columns(df)
    if not time_cols or not numeric_cols:
        return {"has_time_series": False, "message": "No datetime or numeric columns found"}
    
    primary_time_col = time_cols[0]
    
    # Check for mixed date formats
    mixed_formats = False
    detected_formats = []
    if df[primary_time_col].dtype in (pl.Utf8, pl.String, pl.Object):
        sample = df[primary_time_col].drop_nulls().head(100).to_list()
        formats = set()
        for s in sample:
            str_s = str(s).strip()
            if re.match(r"^\d{4}-\d{2}-\d{2}$", str_s):
                formats.add("YYYY-MM-DD")
            elif re.match(r"^\d{4}/\d{2}/\d{2}$", str_s):
                formats.add("YYYY/MM/DD")
            elif re.match(r"^\d{2}-[A-Za-z]{3}-\d{4}$", str_s):
                formats.add("DD-Mon-YYYY")
            elif re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", str_s):
                formats.add("MM/DD/YYYY")
            elif " " in str_s and "-" in str_s:
                formats.add("YYYY-MM-DD HH:MM:SS")
        if len(formats) > 1:
            mixed_formats = True
            detected_formats = sorted(list(formats))

    results = {
        "has_time_series": True,
        "time_column": primary_time_col,
        "mixed_date_formats": mixed_formats,
        "detected_formats": detected_formats,
        "trends": {},
        "seasonality": {},
        "kats_intelligence": {},
        "changepoints": {},
        "forecasts": {}
    }
    
    for num_col in numeric_cols[:5]:
        trend = detect_trend(df, primary_time_col, num_col)
        seasonality = detect_seasonality(df, primary_time_col, num_col)
        
        results["trends"][num_col] = trend
        results["seasonality"][num_col] = seasonality

        # Meta Kats Time Series Intelligence (CUSUM, Holt-Winters, TSFeatures)
        try:
            from app.services.timeseries_kats import run_kats_time_series_intelligence
            kats_res = run_kats_time_series_intelligence(df, primary_time_col, num_col)
            if kats_res.get("success"):
                results["kats_intelligence"][num_col] = kats_res
                results["changepoints"][num_col] = kats_res.get("changepoints", [])
                results["forecasts"][num_col] = kats_res.get("forecast", {})
        except Exception as kats_err:
            logger.warning(f"Kats intelligence failed for {num_col}: {kats_err}")
        
    return results
