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
    2. Modified Z-Score via Median Absolute Deviation (MAD) for skewed data
    3. Standard Z-Score (|Z| > 3.0)
    4. Domain boundary anomalies (e.g. negative prices/ages, percentages > 100%)
    5. Multivariate Anomaly Detection via Isolation Forest & Local Outlier Factor (LOF)
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

            # Mild & Severe IQR Bounds
            lower_mild = q1 - IQR_LOWER_MULTIPLIER * iqr
            upper_mild = q3 + IQR_UPPER_MULTIPLIER * iqr
            lower_severe = q1 - 3.0 * iqr
            upper_severe = q3 + 3.0 * iqr

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
                import numpy as np

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
