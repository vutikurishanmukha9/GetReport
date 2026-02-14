from __future__ import annotations
import polars as pl
import numpy as np
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

CORRELATION_STRONG_THRESHOLD: float = settings.CORRELATION_STRONG_THRESHOLD
SKEWNESS_THRESHOLD: float = settings.SKEWNESS_THRESHOLD

def compute_summary(df: pl.DataFrame, numeric_cols: list[str]) -> dict[str, dict[str, float]]:
    if not numeric_cols: return {}
    
    # Use LazyFrame to compute all statistics in a single pass
    lazy_df = df.lazy()
    
    # Build aggregation expressions
    aggs = []
    for col in numeric_cols:
        aggs.extend([
            pl.col(col).mean().cast(pl.Float64).alias(f"{col}__mean"),
            pl.col(col).std().cast(pl.Float64).alias(f"{col}__std"),
            pl.col(col).min().cast(pl.Float64).alias(f"{col}__min"),
            pl.col(col).max().cast(pl.Float64).alias(f"{col}__max"),
            pl.col(col).median().cast(pl.Float64).alias(f"{col}__50%"),
            pl.col(col).quantile(0.25).cast(pl.Float64).alias(f"{col}__25%"),
            pl.col(col).quantile(0.75).cast(pl.Float64).alias(f"{col}__75%"),
            pl.col(col).skew().cast(pl.Float64).alias(f"{col}__skewness"),
            pl.col(col).kurtosis().cast(pl.Float64).alias(f"{col}__kurtosis")
        ])
        
    try:
        # Collect result (single row DataFrame)
        stats_row = lazy_df.select(aggs).collect().row(0, named=True)
        
        # Format result
        result = {}
        for col in numeric_cols:
            result[col] = {
                "mean": stats_row.get(f"{col}__mean") or 0.0,
                "std": stats_row.get(f"{col}__std") or 0.0,
                "min": stats_row.get(f"{col}__min") or 0.0,
                "max": stats_row.get(f"{col}__max") or 0.0,
                "50%": stats_row.get(f"{col}__50%") or 0.0,
                "25%": stats_row.get(f"{col}__25%") or 0.0,
                "75%": stats_row.get(f"{col}__75%") or 0.0,
                "skewness": stats_row.get(f"{col}__skewness") or 0.0,
                "kurtosis": stats_row.get(f"{col}__kurtosis") or 0.0,
            }
        return result
        
    except Exception as e:
        logger.error(f"Lazy summary computation failed: {e}")
        return {}

def compute_correlation(df: pl.DataFrame, numeric_cols: list[str]):
    if len(numeric_cols) < 2: return {}, []
    
    # Optimized: Vectorized Correlation Matrix
    if len(numeric_cols) < 2: return {}, []

    try:
        # Convert to numpy (Zero Copy if possible, but drops nulls for safety)
        # We drop rows with nulls in ANY of the target columns to ensure valid correlation
        # This is standard behavior for correlation matrices (listwise deletion)
        
        # Selecting columns and dropping nulls
        clean_df = df.select(numeric_cols).drop_nulls()
        
        if clean_df.height < 2:
            return {}, [] # Not enough data
            
        data_matrix = clean_df.to_numpy().T # Transpose for np.corrcoef (expects variables as rows)
        
        # Compute Matrix
        corr_matrix = np.corrcoef(data_matrix)
        
        # Map back to dictionary
        corr_dict = {c: {} for c in numeric_cols}
        strong_pairs = []
        
        for i, col_a in enumerate(numeric_cols):
            # Self correlation
            corr_dict[col_a][col_a] = 1.0
            
            for j in range(i + 1, len(numeric_cols)):
                col_b = numeric_cols[j]
                val = float(corr_matrix[i, j])
                
                # Handle NaN (constant columns)
                if np.isnan(val): val = 0.0
                
                corr_dict[col_a][col_b] = val
                corr_dict[col_b][col_a] = val
                
                if abs(val) >= CORRELATION_STRONG_THRESHOLD:
                    strong_pairs.append({
                        "column_a": col_a,
                        "column_b": col_b,
                        "r_value": round(val, 4),
                        "direction": "positive" if val > 0 else "negative",
                        "strength": "very strong" if abs(val) >= 0.9 else "strong"
                    })
                    
        return corr_dict, strong_pairs
        
    except Exception as e:
        logger.error(f"Vectorized correlation failed: {e}")
        return {}, []
