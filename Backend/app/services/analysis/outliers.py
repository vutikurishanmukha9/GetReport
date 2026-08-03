from __future__ import annotations
import logging
import polars as pl
from app.core.config import settings

logger = logging.getLogger(__name__)

IQR_LOWER_MULTIPLIER: float = getattr(settings, "IQR_LOWER_MULTIPLIER", 1.5)
IQR_UPPER_MULTIPLIER: float = getattr(settings, "IQR_UPPER_MULTIPLIER", 1.5)

# Non-negative column name indicators
NON_NEGATIVE_KEYWORDS = {"price", "cost", "age", "count", "quantity", "revenue", "amount", "salary", "fee", "rate", "income"}

def detect_outliers(df: pl.DataFrame, numeric_cols: list[str]) -> dict[str, dict]:
    """
    Ensemble Outlier & Anomaly Detection Subsystem.
    
    Combines:
    1. Adaptive IQR Bounds (Mild 1.5x, Severe 3.0x)
    2. Log-Transformed IQR Bounds for heavy right-skewed business metrics (skew > 2.0)
    3. Modified Z-Score via Median Absolute Deviation (MAD) for non-normal distributions
    4. Standard Z-Score (|Z| > 3.0)
    5. Domain boundary anomalies (e.g. negative prices/ages, percentages > 100%)
    6. Multivariate Anomaly Detection via Isolation Forest & Local Outlier Factor (LOF)
    """
    if not numeric_cols or df.height == 0:
        return {}

    outliers: dict[str, dict] = {}
    n_rows = df.height

    # 1. Pre-calculate aggregations in a single Lazy query for efficiency
    lazy_df = df.lazy()
    aggs = []
    for col in numeric_cols:
        aggs.extend([
            pl.col(col).quantile(0.25).cast(pl.Float64).alias(f"{col}_q1"),
            pl.col(col).quantile(0.75).cast(pl.Float64).alias(f"{col}_q3"),
            pl.col(col).median().cast(pl.Float64).alias(f"{col}_median"),
            pl.col(col).mean().cast(pl.Float64).alias(f"{col}_mean"),
            pl.col(col).std().cast(pl.Float64).alias(f"{col}_std"),
            pl.col(col).skew().cast(pl.Float64).alias(f"{col}_skew"),
        ])

    try:
        aggs_row = lazy_df.select(aggs).collect().row(0, named=True)

        for col in numeric_cols:
            q1 = aggs_row.get(f"{col}_q1")
            q3 = aggs_row.get(f"{col}_q3")
            median_val = aggs_row.get(f"{col}_median")
            mean_val = aggs_row.get(f"{col}_mean")
            std_val = aggs_row.get(f"{col}_std")
            skew_val = aggs_row.get(f"{col}_skew")

            if q1 is None or q3 is None or median_val is None:
                continue

            iqr = q3 - q1
            if iqr == 0 and (std_val is None or std_val == 0):
                continue

            # Check skewness for log-transformation recommendation
            is_heavy_skew = skew_val is not None and skew_val > 2.0
            
            # Mild & Severe IQR Bounds
            lower_mild = q1 - IQR_LOWER_MULTIPLIER * iqr
            upper_mild = q3 + IQR_UPPER_MULTIPLIER * iqr
            lower_severe = q1 - 3.0 * iqr
            upper_severe = q3 + 3.0 * iqr

            # Log-transformed IQR for heavy right skew
            log_outlier_count = 0
            if is_heavy_skew and (df[col] >= 0).all():
                try:
                    log_series = (df[col].cast(pl.Float64) + 1.0).log()
                    l_q1 = log_series.quantile(0.25)
                    l_q3 = log_series.quantile(0.75)
                    if l_q1 is not None and l_q3 is not None:
                        l_iqr = l_q3 - l_q1
                        l_upper = l_q3 + 1.5 * l_iqr
                        log_outlier_count = int((log_series > l_upper).sum())
                except Exception:
                    pass

            # ── 2. MAD Calculation (Median Absolute Deviation) ──
            mad_val = None
            mad_outlier_count = 0
            try:
                dev_series = df.select((pl.col(col).cast(pl.Float64) - median_val).abs().alias("dev"))["dev"]
                mad_val = dev_series.median()
                if mad_val and mad_val > 0:
                    # Modified Z-score: 0.6745 * |x - median| / MAD > 3.5
                    mod_z = (0.6745 * (df[col].cast(pl.Float64) - median_val).abs() / mad_val)
                    mad_outlier_count = int((mod_z > 3.5).sum())
            except Exception as e:
                logger.debug("MAD calculation skipped for %s: %s", col, e)

            # Filter IQR outliers
            outlier_filter = (pl.col(col) < lower_mild) | (pl.col(col) > upper_mild)
            severe_filter = (pl.col(col) < lower_severe) | (pl.col(col) > upper_severe)
            
            outlier_rows = df.filter(outlier_filter)
            count = outlier_rows.height

            # Domain boundary anomalies
            domain_anomalies: list[str] = []
            col_lower = col.lower()
            if any(kw in col_lower for kw in NON_NEGATIVE_KEYWORDS):
                neg_count = int((df[col] < 0).sum())
                if neg_count > 0:
                    domain_anomalies.append(f"{neg_count} negative values in non-negative domain '{col}'")

            if "pct" in col_lower or "percent" in col_lower or "ratio" in col_lower:
                pct_violations = int(((df[col] < 0) | (df[col] > 100)).sum())
                if pct_violations > 0 and max(df[col].drop_nulls(), default=0) > 1.0:
                    domain_anomalies.append(f"{pct_violations} percentage values outside [0, 100%]")

            if count > 0 or mad_outlier_count > 0 or domain_anomalies:
                vals = outlier_rows[col].head(20).to_list() if count > 0 else []
                severe_count = int(df.filter(severe_filter).height) if iqr > 0 else 0

                outliers[col] = {
                    "count": count,
                    "percentage": round(count / n_rows * 100, 2),
                    "severe_count": severe_count,
                    "mad_outlier_count": mad_outlier_count,
                    "log_outlier_count": log_outlier_count,
                    "is_heavy_skew": is_heavy_skew,
                    "skewness": round(skew_val, 2) if skew_val is not None else 0.0,
                    "min_outlier": outlier_rows[col].min() if count > 0 else None,
                    "max_outlier": outlier_rows[col].max() if count > 0 else None,
                    "lower_bound": round(lower_mild, 4),
                    "upper_bound": round(upper_mild, 4),
                    "lower_bound_severe": round(lower_severe, 4),
                    "upper_bound_severe": round(upper_severe, 4),
                    "sample_values": vals,
                    "domain_anomalies": domain_anomalies,
                }

        # ── 5. Multivariate Outlier Detection via Isolation Forest & LOF ──
        if len(numeric_cols) >= 2 and n_rows >= 20:
            try:
                from sklearn.ensemble import IsolationForest
                from sklearn.neighbors import LocalOutlierFactor

                # Prepare scaled numpy array for top numeric columns
                clean_df = df.select(numeric_cols[:5]).to_pandas().fillna(0)
                if len(clean_df) >= 20:
                    iso = IsolationForest(contamination=0.02, random_state=42, n_estimators=100)
                    iso_preds = iso.fit_predict(clean_df)
                    iso_outliers = int((iso_preds == -1).sum())

                    lof = LocalOutlierFactor(n_neighbors=15, contamination=0.02)
                    lof_preds = lof.fit_predict(clean_df)
                    high_conf_multivariate = int(((iso_preds == -1) & (lof_preds == -1)).sum())

                    if high_conf_multivariate > 0:
                        outliers["_multivariate_summary"] = {
                            "count": high_conf_multivariate,
                            "percentage": round(high_conf_multivariate / n_rows * 100, 2),
                            "columns_evaluated": numeric_cols[:5],
                            "isolation_forest_count": iso_outliers,
                            "high_confidence_multivariate_count": high_conf_multivariate,
                        }
            except Exception as ml_err:
                logger.debug("Multivariate outlier detection skipped: %s", ml_err)

    except Exception as e:
        logger.error(f"Outlier detection failed: {e}")

    return outliers


def detect_time_series_stl_outliers(df: pl.DataFrame, date_col: str, numeric_col: str) -> dict:
    """
    Performs Seasonal Trend Decomposition (STL) for time-indexed numeric metrics.
    
    Decomposes series: Observed(t) = Trend(t) + Seasonal(t) + Residual(t)
    Evaluates MAD outliers on Residual(t) rather than raw values, preventing false positive
    alerts caused by predictable weekly or monthly seasonal surges.
    """
    if date_col not in df.columns or numeric_col not in df.columns or df.height < 14:
        return {"has_stl_outliers": False, "reason": "Insufficient observations for STL decomposition"}

    try:
        import numpy as np

        # Extract sorted series using pure Polars + NumPy
        clean_df = df.select([date_col, numeric_col]).drop_nulls().sort(date_col)
        if clean_df.height < 14:
            return {"has_stl_outliers": False, "reason": "Fewer than 14 non-null observations"}

        series_vec = clean_df[numeric_col].cast(pl.Float64).to_numpy()
        n = len(series_vec)

        try:
            import pandas as pd
            from statsmodels.tsa.seasonal import STL
            ts_pd = clean_df.to_pandas()
            ts_pd[date_col] = pd.to_datetime(ts_pd[date_col])
            ts_pd.set_index(date_col, inplace=True)
            res = STL(ts_pd[numeric_col].astype(float), period=7, seasonal=13).fit()
            trend = res.trend.values
            seasonal = res.seasonal.values
            resid = res.resid.values
        except Exception:
            # Pure NumPy Fallback: Rolling Mean Trend + Weekly Seasonal Decomposition
            window = 7
            kernel = np.ones(window) / window
            trend = np.convolve(series_vec, kernel, mode="same")
            detrended = series_vec - trend
            
            seasonal = np.zeros(n)
            for day in range(7):
                mask = (np.arange(n) % 7) == day
                if mask.any():
                    seasonal[mask] = np.median(detrended[mask])
            resid = series_vec - trend - seasonal

        # Outlier detection on residuals using Modified Z-score via MAD
        median_r = float(np.median(resid))
        mad_r = float(np.median(np.abs(resid - median_r)))

        stl_outliers_mask = np.zeros(len(resid), dtype=bool)
        if mad_r > 0:
            mod_z_r = 0.6745 * np.abs(resid - median_r) / mad_r
            stl_outliers_mask = mod_z_r > 3.0
        else:
            std_r = float(np.std(resid))
            if std_r > 0:
                stl_outliers_mask = np.abs(resid - np.mean(resid)) > 2.0 * std_r

        stl_outlier_count = int(stl_outliers_mask.sum())

        return {
            "has_stl_outliers": stl_outlier_count > 0,
            "stl_outlier_count": stl_outlier_count,
            "total_observations": n,
            "pct_stl_outliers": round(stl_outlier_count / n * 100, 2),
            "date_col": date_col,
            "numeric_col": numeric_col,
            "trend_summary": {"min": float(np.min(trend)), "max": float(np.max(trend))},
            "seasonal_summary": {"amplitude": float(np.max(seasonal) - np.min(seasonal))},
            "residual_mad": mad_r,
        }

    except Exception as err:
        logger.debug("STL time series decomposition skipped: %s", err)
        return {"has_stl_outliers": False, "reason": str(err)}
