from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import polars as pl
import numpy as np

# ─── Logger ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
IQR_LOWER_MULTIPLIER: float = 1.5
IQR_UPPER_MULTIPLIER: float = 1.5
CORRELATION_STRONG_THRESHOLD: float = 0.7

SKEWNESS_THRESHOLD: float = 1.0

# ─── Custom Exceptions ───────────────────────────────────────────────────────
class EmptyDatasetError(ValueError): pass
class InsufficientDataError(ValueError): pass
class AnalysisError(RuntimeError): pass

@dataclass
class AnalysisResult:
    metadata:                  dict[str, Any]                  = field(default_factory=dict)
    summary:                   dict[str, dict[str, float]]     = field(default_factory=dict)
    correlation:               dict[str, dict[str, float]]     = field(default_factory=dict)
    strong_correlations:       list[dict[str, Any]]            = field(default_factory=list)
    outliers:                  dict[str, dict[str, Any]]       = field(default_factory=dict)
    categorical_distribution:  dict[str, dict[str, Any]]       = field(default_factory=dict)
    column_quality_flags:      dict[str, list[str]]            = field(default_factory=dict)
    timing_ms:                 float                           = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata":                  self.metadata,
            "summary":                   self.summary,
            "correlation":               self.correlation,
            "strong_correlations":       self.strong_correlations,
            "outliers":                  self.outliers,
            "categorical_distribution":  self.categorical_distribution,
            "column_quality_flags":      self.column_quality_flags,
            "timing_ms":                 round(self.timing_ms, 2),
        }

def _validate_input(df: pl.DataFrame) -> None:
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected pl.DataFrame, got {type(df)}")
    if df.height == 0 or df.width == 0:
        raise EmptyDatasetError("Empty dataset")

def _compute_summary(df: pl.DataFrame, numeric_cols: list[str]) -> dict[str, dict[str, float]]:
    if not numeric_cols: return {}
    
    # Polars describe() gives stats for all numeric cols
    stats = df.select(numeric_cols).describe()
    # Convert to dict format expected by frontend: {col: {mean: x, std: y...}}
    
    # This is a bit manual in Polars to match Pandas output structure exactly.
    # We will compute aggregations manually for precision and structure.
    
    result = {}
    for col in numeric_cols:
        col_stats = df.select([
            pl.col(col).mean().alias("mean"),
            pl.col(col).std().alias("std"),
            pl.col(col).min().alias("min"),
            pl.col(col).max().alias("max"),
            pl.col(col).median().alias("50%"),
            pl.col(col).quantile(0.25).alias("25%"),
            pl.col(col).quantile(0.75).alias("75%"),
            pl.col(col).skew().alias("skewness"),
            pl.col(col).kurtosis().alias("kurtosis")
        ]).to_dict(as_series=False)
        
        # Unwrap lists
        single_stats = {k: (v[0] if v[0] is not None else 0.0) for k, v in col_stats.items()}
        result[col] = single_stats
        
    return result

def _compute_correlation(df: pl.DataFrame, numeric_cols: list[str]):
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

def _detect_outliers(df: pl.DataFrame, numeric_cols: list[str]) -> dict:
    outliers = {}
    for col in numeric_cols:
        q1 = df.select(pl.col(col).quantile(0.25)).item()
        q3 = df.select(pl.col(col).quantile(0.75)).item()
        
        if q1 is None or q3 is None: continue
        
        iqr = q3 - q1
        if iqr == 0: continue
        
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        # Filter
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
    return outliers

def analyze_dataset(df: pl.DataFrame, top_categories: int = 10) -> dict[str, Any]:
    start = time.perf_counter()
    _validate_input(df)
    
    numeric_cols = [c for c, t in df.schema.items() if t in (pl.Int64, pl.Float64, pl.Int32, pl.Float32)]
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    
    metadata = {
        "total_rows": df.height,
        "total_columns": df.width,
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(cat_cols)
    }
    
    summary_stats = _compute_summary(df, numeric_cols)
    correlation, strong_pairs = _compute_correlation(df, numeric_cols)
    outliers = _detect_outliers(df, numeric_cols)
    
    # Categorical Distrib (Top 10)
    cat_dist = {}
    for c in cat_cols:
        counts = df[c].value_counts(sort=True).head(top_categories)
        # counts is a struct or df with col, count
        # In Polars, value_counts returns struct with columns [col_name, "count"] or similar
        # Need to handle carefully. usually it returns DataFrame(column, count)
        
        # We need to reshape to dict
        cats = {}
        for row in counts.iter_rows():
            val, cnt = row
            cats[str(val)] = {"count": cnt, "percentage": round(cnt/df.height*100, 2)}
            
        cat_dist[c] = {
            "categories": cats,
            "total_unique": df[c].n_unique(),
            "missing_pct": round(df[c].null_count()/df.height*100, 2)
        }

    flags = {} # TODO: port flags logic if needed
    
    elapsed = (time.perf_counter() - start) * 1000
    
    return AnalysisResult(
        metadata=metadata,
        summary=summary_stats,
        correlation=correlation,
        strong_correlations=strong_pairs,
        outliers=outliers,
        categorical_distribution=cat_dist,
        timing_ms=elapsed
    ).to_dict()