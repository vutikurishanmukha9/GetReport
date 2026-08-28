from __future__ import annotations
import polars as pl
import numpy as np
import logging
import math
from app.core.config import settings

logger = logging.getLogger(__name__)

CORRELATION_STRONG_THRESHOLD: float = settings.CORRELATION_STRONG_THRESHOLD
SKEWNESS_THRESHOLD: float = settings.SKEWNESS_THRESHOLD

def compute_summary(df: pl.DataFrame, numeric_cols: list[str]) -> dict[str, dict[str, float]]:
    """
    Compute comprehensive parametric and non-parametric summary statistics for numeric columns.
    
    Includes:
    - Central Tendency: Mean, Median, 5% Trimmed Mean
    - Dispersion: Standard Deviation, IQR (Q3 - Q1), MAD (Median Absolute Deviation), CV
    - Extremes & Quantiles: Min, Max, 25% (Q1), 75% (Q3)
    - Shape & Tail Behavior: Moment Skewness, Kurtosis, Bowley's Resistant Skewness
    """
    if not numeric_cols:
        return {}
    
    lazy_df = df.lazy()
    
    # Build aggregation expressions in a single Polars execution plan
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
        stats_row = lazy_df.select(aggs).collect().row(0, named=True)
        
        result = {}
        for col in numeric_cols:
            mean_val = float(stats_row.get(f"{col}__mean") or 0.0)
            std_val = float(stats_row.get(f"{col}__std") or 0.0)
            min_val = float(stats_row.get(f"{col}__min") or 0.0)
            max_val = float(stats_row.get(f"{col}__max") or 0.0)
            median_val = float(stats_row.get(f"{col}__50%") or 0.0)
            q25 = float(stats_row.get(f"{col}__25%") or 0.0)
            q75 = float(stats_row.get(f"{col}__75%") or 0.0)
            skew_val = float(stats_row.get(f"{col}__skewness") or 0.0)
            kurt_val = float(stats_row.get(f"{col}__kurtosis") or 0.0)
            
            # Non-Parametric & Robust Measures
            iqr = round(q75 - q25, 6)
            
            # Bowley's Resistant Skewness: (Q3 + Q1 - 2*Q2) / (Q3 - Q1)
            bowley_skew = 0.0
            if iqr > 1e-12:
                bowley_skew = round((q75 + q25 - 2.0 * median_val) / iqr, 4)
                
            # Coefficient of Variation: std / |mean|
            cv = 0.0
            if abs(mean_val) > 1e-9:
                cv = round(std_val / abs(mean_val), 4)
                
            # MAD (Median Absolute Deviation) & 5% Trimmed Mean
            mad_val = 0.0
            trimmed_mean = mean_val
            try:
                non_null_col = df[col].drop_nulls()
                if non_null_col.len() > 0:
                    dev_series = (non_null_col.cast(pl.Float64) - median_val).abs()
                    mad_val = float(dev_series.median() or 0.0)
                    
                    # Exact 5% trimmed mean calculation
                    sorted_vals = non_null_col.cast(pl.Float64).sort().to_numpy()
                    n = len(sorted_vals)
                    k = int(math.floor(n * 0.05))
                    if k > 0 and n > 2 * k:
                        trimmed_mean = float(np.mean(sorted_vals[k:-k]))
                    elif n > 2:
                        trimmed_mean = float(np.mean(sorted_vals[1:-1]))
            except Exception as ex:
                logger.debug("Error computing robust stats for %s: %s", col, ex)
            
            result[col] = {
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "min": round(min_val, 4),
                "max": round(max_val, 4),
                "50%": round(median_val, 4),
                "median": round(median_val, 4),
                "25%": round(q25, 4),
                "75%": round(q75, 4),
                "iqr": round(iqr, 4),
                "mad": round(mad_val, 4),
                "skewness": round(skew_val, 4),
                "kurtosis": round(kurt_val, 4),
                "bowley_skewness": bowley_skew,
                "coefficient_of_variation": cv,
                "trimmed_mean_5pct": round(trimmed_mean, 4),
            }
        return result
        
    except Exception as e:
        logger.error(f"Lazy summary computation failed: {e}")
        return {}

def compute_correlation(df: pl.DataFrame, numeric_cols: list[str]):
    """
    Vectorized Correlation Engine with numerical stability, zero-variance protection,
    and Spearman rank-order correlation inference.
    """
    if len(numeric_cols) < 2:
        return {}, []

    try:
        # Cast to Float64 and clean nulls
        clean_df = df.select([pl.col(c).cast(pl.Float64) for c in numeric_cols]).drop_nulls()
        
        if clean_df.height < 2:
            # Fallback to median imputation if listwise deletion dropped all rows
            clean_df = df.select([
                pl.col(c).cast(pl.Float64).fill_null(pl.col(c).median()).fill_null(0.0)
                for c in numeric_cols
            ])
            
        if clean_df.height < 2:
            return {}, []
            
        data_matrix = clean_df.to_numpy().T # Shape: (num_features, num_samples)
        
        # Calculate standard deviations to identify zero-variance constant features
        stds = np.std(data_matrix, axis=1)
        valid_var_mask = stds > 1e-12
        
        # Compute Pearson Matrix with warning suppression
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_matrix = np.corrcoef(data_matrix)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            
        # Zero out rows/cols corresponding to zero-variance vectors
        for idx, has_var in enumerate(valid_var_mask):
            if not has_var:
                corr_matrix[idx, :] = 0.0
                corr_matrix[:, idx] = 0.0
                
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Compute Spearman Rank Correlation matrix for non-linear monotonic relationships
        try:
            ranked_matrix = np.argsort(np.argsort(data_matrix, axis=1), axis=1).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                spearman_matrix = np.nan_to_num(np.corrcoef(ranked_matrix), nan=0.0)
                for idx, has_var in enumerate(valid_var_mask):
                    if not has_var:
                        spearman_matrix[idx, :] = 0.0
                        spearman_matrix[:, idx] = 0.0
                np.fill_diagonal(spearman_matrix, 1.0)
        except Exception:
            spearman_matrix = corr_matrix
        
        corr_dict = {c: {} for c in numeric_cols}
        strong_pairs = []
        
        for i, col_a in enumerate(numeric_cols):
            corr_dict[col_a][col_a] = 1.0
            
            for j in range(i + 1, len(numeric_cols)):
                col_b = numeric_cols[j]
                val = float(corr_matrix[i, j])
                spearman_val = float(spearman_matrix[i, j])
                
                if np.isnan(val):
                    val = 0.0
                if np.isnan(spearman_val):
                    spearman_val = 0.0
                
                corr_dict[col_a][col_b] = round(val, 4)
                corr_dict[col_b][col_a] = round(val, 4)
                
                # Pearson threshold filter for strong correlations
                if abs(val) >= CORRELATION_STRONG_THRESHOLD:
                    is_collinear = abs(val) >= 0.90
                    strong_pairs.append({
                        "column_a": col_a,
                        "column_b": col_b,
                        "r_value": round(val, 4),
                        "spearman_rho": round(spearman_val, 4),
                        "direction": "positive" if val > 0 else "negative",
                        "strength": "very strong" if abs(val) >= 0.9 else "strong",
                        "is_multicollinear": is_collinear,
                    })
                    
        return corr_dict, strong_pairs
        
    except Exception as e:
        logger.error(f"Vectorized correlation failed: {e}")
        return {}, []
