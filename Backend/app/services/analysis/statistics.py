from __future__ import annotations
import polars as pl
import numpy as np
import logging
import math
from app.core.config import settings

logger = logging.getLogger(__name__)

CORRELATION_STRONG_THRESHOLD: float = settings.CORRELATION_STRONG_THRESHOLD
SKEWNESS_THRESHOLD: float = settings.SKEWNESS_THRESHOLD

def compute_summary(df: pl.DataFrame, numeric_cols: list[str]) -> dict[str, dict[str, Any]]:
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
    # Filter out sentinel values (-999, -9999, 9999, 99999, inf, -inf, nan)
    aggs = []
    for col in numeric_cols:
        c_float = pl.col(col).cast(pl.Float64, strict=False)
        clean_c = (
            pl.when(
                c_float.is_in([-999.0, -9999.0, 9999.0, 99999.0])
                | c_float.is_nan()
                | c_float.is_infinite()
            )
            .then(None)
            .otherwise(c_float)
        )
        aggs.extend([
            clean_c.mean().alias(f"{col}__mean"),
            clean_c.std().alias(f"{col}__std"),
            clean_c.min().alias(f"{col}__min"),
            clean_c.max().alias(f"{col}__max"),
            clean_c.median().alias(f"{col}__50%"),
            clean_c.quantile(0.25).alias(f"{col}__25%"),
            clean_c.quantile(0.75).alias(f"{col}__75%"),
            clean_c.skew().alias(f"{col}__skewness"),
            clean_c.kurtosis().alias(f"{col}__kurtosis")
        ])
        
    try:
        stats_row = lazy_df.select(aggs).collect().row(0, named=True)
        
        result = {}
        for col in numeric_cols:
            mean_raw = stats_row.get(f"{col}__mean")
            std_raw = stats_row.get(f"{col}__std")
            min_raw = stats_row.get(f"{col}__min")
            max_raw = stats_row.get(f"{col}__max")
            med_raw = stats_row.get(f"{col}__50%")
            q25_raw = stats_row.get(f"{col}__25%")
            q75_raw = stats_row.get(f"{col}__75%")
            skew_raw = stats_row.get(f"{col}__skewness")
            kurt_raw = stats_row.get(f"{col}__kurtosis")

            def _clean_val(v):
                if v is None:
                    return None
                try:
                    fv = float(v)
                    if math.isnan(fv) or math.isinf(fv):
                        return None
                    return fv
                except Exception:
                    return None

            mean_val = _clean_val(mean_raw) or 0.0
            std_val = _clean_val(std_raw) or 0.0
            min_val = _clean_val(min_raw)
            max_val = _clean_val(max_raw)
            median_val = _clean_val(med_raw) or 0.0
            q25 = _clean_val(q25_raw) or 0.0
            q75 = _clean_val(q75_raw) or 0.0
            skew_val = _clean_val(skew_raw) or 0.0
            kurt_val = _clean_val(kurt_raw) or 0.0
            
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
                c_flt = pl.col(col).cast(pl.Float64, strict=False)
                clean_s = (
                    df.select(
                        pl.when(
                            c_flt.is_in([-999.0, -9999.0, 9999.0, 99999.0])
                            | c_flt.is_nan()
                            | c_flt.is_infinite()
                        )
                        .then(None)
                        .otherwise(c_flt)
                        .alias(col)
                    )
                    .to_series()
                    .drop_nulls()
                )
                if clean_s.len() > 0:
                    dev_series = (clean_s - median_val).abs()
                    mad_val = float(dev_series.median() or 0.0)
                    
                    # Exact 5% trimmed mean calculation
                    sorted_vals = clean_s.sort().to_numpy()
                    sorted_vals = sorted_vals[np.isfinite(sorted_vals) & (np.abs(sorted_vals) < 1e12)]
                    n = len(sorted_vals)
                    k = int(math.floor(n * 0.05))
                    with np.errstate(all="ignore"):
                        if k > 0 and n > 2 * k:
                            trimmed_mean = float(np.mean(sorted_vals[k:-k]))
                        elif n > 2:
                            trimmed_mean = float(np.mean(sorted_vals[1:-1]))
            except Exception as ex:
                logger.debug("Error computing robust stats for %s: %s", col, ex)
            
            result[col] = {
                "mean": round(mean_val, 4) if mean_raw is not None else None,
                "std": round(std_val, 4) if std_raw is not None else None,
                "min": round(min_val, 4) if min_val is not None else None,
                "max": round(max_val, 4) if max_val is not None else None,
                "50%": round(median_val, 4) if med_raw is not None else None,
                "median": round(median_val, 4) if med_raw is not None else None,
                "25%": round(q25, 4) if q25_raw is not None else None,
                "75%": round(q75, 4) if q75_raw is not None else None,
                "iqr": round(iqr, 4),
                "mad": round(mad_val, 4),
                "skewness": round(skew_val, 4) if skew_raw is not None else None,
                "kurtosis": round(kurt_val, 4) if kurt_raw is not None else None,
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
    Pairwise Correlation Engine with sentinel filtering, zero-variance protection,
    and Spearman rank-order correlation inference.
    """
    if len(numeric_cols) < 2:
        return {}, []

    try:
        # Pre-clean each column to float, filtering out sentinels and non-finite numbers
        cleaned_series: dict[str, np.ndarray] = {}
        for c in numeric_cols:
            c_flt = pl.col(c).cast(pl.Float64, strict=False)
            clean_s = (
                df.select(
                    pl.when(
                        c_flt.is_in([-999.0, -9999.0, 9999.0, 99999.0])
                        | c_flt.is_nan()
                        | c_flt.is_infinite()
                    )
                    .then(None)
                    .otherwise(c_flt)
                    .alias(c)
                )
                .to_series()
                .to_numpy()
            )
            cleaned_series[c] = clean_s

        corr_dict = {c: {} for c in numeric_cols}
        strong_pairs = []

        for i, col_a in enumerate(numeric_cols):
            corr_dict[col_a][col_a] = 1.0
            arr_a = cleaned_series[col_a]

            for j in range(i + 1, len(numeric_cols)):
                col_b = numeric_cols[j]
                arr_b = cleaned_series[col_b]

                # Pairwise mask where both a and b are valid, finite, not negative sentinels, and within realistic bounds
                mask = (
                    np.isfinite(arr_a)
                    & np.isfinite(arr_b)
                    & (~np.isnan(arr_a))
                    & (~np.isnan(arr_b))
                    & (arr_a > -900)
                    & (arr_b > -900)
                    & (np.abs(arr_a) < 1e12)
                    & (np.abs(arr_b) < 1e12)
                )

                val = 0.0
                spearman_val = 0.0
                if mask.sum() >= 4:
                    x = arr_a[mask]
                    y = arr_b[mask]
                    with np.errstate(all="ignore"):
                        std_x = float(np.std(x))
                        std_y = float(np.std(y))
                        if std_x > 1e-12 and std_y > 1e-12:
                            r_matrix = np.corrcoef(x, y)
                            v = float(r_matrix[0, 1])
                            val = 0.0 if np.isnan(v) else v
                            # Spearman via ranking
                            try:
                                rx = np.argsort(np.argsort(x)).astype(float)
                                ry = np.argsort(np.argsort(y)).astype(float)
                                sp_matrix = np.corrcoef(rx, ry)
                                sv = float(sp_matrix[0, 1])
                                spearman_val = 0.0 if np.isnan(sv) else sv
                            except Exception:
                                spearman_val = val

                corr_dict[col_a][col_b] = round(val, 4)
                corr_dict[col_b][col_a] = round(val, 4)

                # Pearson or Spearman threshold filter for strong correlations
                eff_corr = max(abs(val), abs(spearman_val))
                if eff_corr >= CORRELATION_STRONG_THRESHOLD:
                    is_collinear = eff_corr >= 0.90
                    strong_pairs.append({
                        "column_a": col_a,
                        "column_b": col_b,
                        "r_value": round(val, 4),
                        "spearman_rho": round(spearman_val, 4),
                        "direction": "positive" if (val > 0 or (val == 0.0 and spearman_val > 0)) else "negative",
                        "strength": "very strong" if eff_corr >= 0.9 else "strong",
                        "is_multicollinear": is_collinear,
                    })

        return corr_dict, strong_pairs

    except Exception as e:
        logger.error(f"Vectorized correlation failed: {e}")
        return {}, []
