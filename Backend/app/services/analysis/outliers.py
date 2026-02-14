from __future__ import annotations
import polars as pl
from app.core.config import settings

IQR_LOWER_MULTIPLIER: float = settings.IQR_LOWER_MULTIPLIER
IQR_UPPER_MULTIPLIER: float = settings.IQR_UPPER_MULTIPLIER

def detect_outliers(df: pl.DataFrame, numeric_cols: list[str]) -> dict:
    if not numeric_cols: return {}
    
    outliers = {}
    
    # 1. Pre-calculate Q1/Q3 for ALL columns in one lazy query
    # This reduces overhead from 2*N queries to 1 query.
    lazy_df = df.lazy()
    bounds_aggs = []
    for col in numeric_cols:
        bounds_aggs.append(pl.col(col).quantile(0.25).cast(pl.Float64).alias(f"{col}_q1"))
        bounds_aggs.append(pl.col(col).quantile(0.75).cast(pl.Float64).alias(f"{col}_q3"))
        
    try:
        # Collect all bounds at once
        bounds_row = lazy_df.select(bounds_aggs).collect().row(0, named=True)
        
        # 2. Iterate and filter using pre-calculated bounds
        for col in numeric_cols:
            q1 = bounds_row.get(f"{col}_q1")
            q3 = bounds_row.get(f"{col}_q3")
            
            if q1 is None or q3 is None: continue
            
            iqr = q3 - q1
            if iqr == 0: continue
            
            lower = q1 - IQR_LOWER_MULTIPLIER * iqr
            upper = q3 + IQR_UPPER_MULTIPLIER * iqr
            
            # Filter (using eager execution here is fine/necessary to extract specific rows)
            # We use the pre-calculated quantitative bounds
            outlier_rows = df.filter((pl.col(col) < lower) | (pl.col(col) > upper))
            count = outlier_rows.height
            
            if count > 0:
                vals = outlier_rows[col].head(20).to_list()
                outliers[col] = {
                    "count": count,
                    "percentage": round(count / df.height * 100, 2),
                    "min_outlier": outlier_rows[col].min(),
                    "max_outlier": outlier_rows[col].max(),
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "sample_values": vals
                }
                
    except Exception as e:
        # Fallback or log error
        # If the huge query fails (e.g. OOM), we might want to log it
        import logging
        logging.getLogger(__name__).error(f"Outlier detection failed: {e}")
        
    return outliers
