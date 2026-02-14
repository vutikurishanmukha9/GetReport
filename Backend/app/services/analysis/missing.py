from __future__ import annotations
import polars as pl
import numpy as np
from typing import Any

def analyze_missing_patterns(df: pl.DataFrame) -> dict[str, Any]:
    """
    Analyze missing value patterns to detect MCAR, MAR, or MNAR.
    
    MCAR (Missing Completely At Random): No pattern - safe to impute/drop
    MAR (Missing At Random): Related to other variables - use conditional imputation
    MNAR (Missing Not At Random): Related to the value itself - complex handling needed
    """
    missing_info = {}
    cols_with_missing = []
    
    # Step 1: Calculate missing rates
    for col in df.columns:
        null_count = df[col].null_count()
        if null_count > 0:
            missing_rate = null_count / df.height
            cols_with_missing.append(col)
            missing_info[col] = {
                "count": null_count,
                "percentage": round(missing_rate * 100, 2),
                "severity": "critical" if missing_rate > 0.5 else ("high" if missing_rate > 0.2 else ("medium" if missing_rate > 0.05 else "low"))
            }
    
    if not cols_with_missing:
        return {"has_missing": False, "message": "No missing values detected"}
    
    # Step 2: Detect missing value correlations (MAR indicator)
    missing_correlations = []
    for col in cols_with_missing[:5]:  # Limit for performance
        # Create binary missing indicator
        missing_mask = df[col].is_null().cast(pl.Int32)
        
        # Check correlation with other numeric columns
        numeric_cols = [c for c, t in df.schema.items() if t in (pl.Int64, pl.Float64, pl.Int32, pl.Float32) and c != col]
        
        for other_col in numeric_cols[:5]:
            try:
                # Correlation between missing indicator and other variable
                corr = df.select(pl.corr(missing_mask.alias("_missing"), pl.col(other_col))).item()
                if corr is not None and not np.isnan(corr) and abs(corr) >= 0.2:
                    missing_correlations.append({
                        "missing_column": col,
                        "correlated_with": other_col,
                        "correlation": round(float(corr), 4),
                        "interpretation": f"Missing values in '{col}' may be related to '{other_col}'"
                    })
            except:
                pass
    
    # Step 3: Detect row patterns (multiple missing in same rows)
    # Count how many columns are missing per row
    missing_per_row = df.select([pl.col(c).is_null().cast(pl.Int32).alias(c) for c in cols_with_missing])
    row_missing_sum = missing_per_row.select(pl.sum_horizontal(pl.all())).to_series()  # Convert to Series
    
    # Distribution of missing counts
    fully_complete = (row_missing_sum == 0).sum()
    partial_missing = ((row_missing_sum > 0) & (row_missing_sum < len(cols_with_missing))).sum()
    fully_missing = (row_missing_sum == len(cols_with_missing)).sum()
    
    row_patterns = {
        "complete_rows": int(fully_complete),
        "partial_missing_rows": int(partial_missing),
        "fully_missing_rows": int(fully_missing)
    }
    
    # Step 4: Infer pattern type
    if missing_correlations:
        pattern_type = "MAR"
        pattern_advice = "Missing values appear related to other variables. Consider multiple imputation or conditional mean imputation."
    elif row_patterns["fully_missing_rows"] > df.height * 0.1:
        pattern_type = "Systematic"
        pattern_advice = "Many rows have all values missing. Consider removing these rows entirely."
    else:
        pattern_type = "MCAR"
        pattern_advice = "Missing values appear random. Safe to use mean/median imputation or listwise deletion."
    
    return {
        "has_missing": True,
        "columns_affected": len(cols_with_missing),
        "column_details": missing_info,
        "missing_correlations": missing_correlations[:5],  # Top 5
        "row_patterns": row_patterns,
        "inferred_pattern": pattern_type,
        "recommendation": pattern_advice
    }
