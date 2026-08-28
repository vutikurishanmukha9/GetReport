from __future__ import annotations
import polars as pl
import numpy as np
from typing import Any

def analyze_missing_patterns(df: pl.DataFrame) -> dict[str, Any]:
    """
    Advanced Missingness Diagnostic Engine (MCAR, MAR, MNAR) with Inter-Feature Phi Correlations
    and Listwise Deletion Impact Assessment.
    
    MCAR (Missing Completely At Random): No pattern - safe to impute/drop
    MAR (Missing At Random): Statistically correlated with other features - requires conditional imputation
    MNAR (Missing Not At Random): High concentration / systematic pattern - requires indicator flags
    """
    missing_info = {}
    cols_with_missing = []
    total_rows = df.height
    
    if total_rows == 0:
        return {"has_missing": False, "message": "Empty dataset"}
    
    # Step 1: Calculate granular missing rates and recommended fix per column
    for col in df.columns:
        null_count = df[col].null_count()
        if null_count > 0:
            missing_rate = null_count / total_rows
            cols_with_missing.append(col)
            dtype = df[col].dtype
            
            # Imputation strategy heuristic
            if dtype in (pl.Int64, pl.Float64, pl.Int32, pl.Float32):
                rec_fix = "median_imputation" if missing_rate <= 0.25 else "flag_and_impute"
            elif dtype == pl.Utf8:
                rec_fix = "mode_or_missing_category"
            elif dtype in (pl.Date, pl.Datetime):
                rec_fix = "forward_fill_or_median_date"
            else:
                rec_fix = "mode_imputation"

            missing_info[col] = {
                "count": null_count,
                "percentage": round(missing_rate * 100, 2),
                "severity": "critical" if missing_rate > 0.5 else ("high" if missing_rate > 0.2 else ("medium" if missing_rate > 0.05 else "low")),
                "recommended_imputation": rec_fix,
            }
    
    if not cols_with_missing:
        return {
            "has_missing": False,
            "message": "No missing values detected",
            "complete_cases_pct": 100.0,
            "data_loss_risk": "None"
        }
    
    # Step 2: Detect missing value correlations against continuous features (MAR indicator)
    missing_correlations = []
    for col in cols_with_missing[:5]:  # Top 5 columns
        missing_mask = df[col].is_null().cast(pl.Int32)
        numeric_cols = [c for c, t in df.schema.items() if t in (pl.Int64, pl.Float64, pl.Int32, pl.Float32) and c != col]
        
        for other_col in numeric_cols[:5]:
            try:
                corr = df.select(pl.corr(missing_mask.alias("_missing"), pl.col(other_col))).item()
                if corr is not None and not np.isnan(corr) and abs(corr) >= 0.2:
                    missing_correlations.append({
                        "missing_column": col,
                        "correlated_with": other_col,
                        "correlation": round(float(corr), 4),
                        "interpretation": f"Missing values in '{col}' correlate with '{other_col}' (|r| >= 0.20)"
                    })
            except Exception:
                pass
    
    # Step 3: Compute Phi-Coefficient Matrix between missingness indicators (Missingness Co-occurrence)
    missing_inter_correlations = []
    if len(cols_with_missing) >= 2:
        for i, c1 in enumerate(cols_with_missing[:6]):
            for j in range(i + 1, min(len(cols_with_missing), 6)):
                c2 = cols_with_missing[j]
                try:
                    m1 = df[c1].is_null().cast(pl.Int32).to_numpy()
                    m2 = df[c2].is_null().cast(pl.Int32).to_numpy()
                    
                    with np.errstate(divide="ignore", invalid="ignore"):
                        phi = np.corrcoef(m1, m2)[0, 1]
                        if not np.isnan(phi) and abs(phi) >= 0.3:
                            missing_inter_correlations.append({
                                "column_a": c1,
                                "column_b": c2,
                                "phi_coefficient": round(float(phi), 4),
                                "co_occurrence": "simultaneous" if phi > 0 else "alternating"
                            })
                except Exception:
                    pass

    # Step 4: Row patterns and Listwise Deletion Impact
    missing_per_row = df.select([pl.col(c).is_null().cast(pl.Int32).alias(c) for c in cols_with_missing])
    row_missing_sum = missing_per_row.select(pl.sum_horizontal(pl.all())).to_series()
    
    fully_complete = int((row_missing_sum == 0).sum())
    partial_missing = int(((row_missing_sum > 0) & (row_missing_sum < len(cols_with_missing))).sum())
    fully_missing = int((row_missing_sum == len(cols_with_missing)).sum())
    
    complete_pct = round(fully_complete / total_rows * 100, 2)
    data_loss_risk = "High" if complete_pct < 60.0 else ("Moderate" if complete_pct < 85.0 else "Low")
    
    row_patterns = {
        "complete_rows": fully_complete,
        "partial_missing_rows": partial_missing,
        "fully_missing_rows": fully_missing,
        "complete_cases_percentage": complete_pct,
        "data_loss_risk_on_drop": data_loss_risk,
    }
    
    # Step 5: Infer Pattern Type
    if missing_correlations:
        pattern_type = "MAR"
        pattern_advice = "Missing values correlate with other features. Use conditional mean, KNN, or iterative imputation."
    elif row_patterns["fully_missing_rows"] > total_rows * 0.1:
        pattern_type = "Systematic"
        pattern_advice = "Systematic missingness detected across full rows. Consider removing empty rows."
    else:
        pattern_type = "MCAR"
        pattern_advice = "Missing values appear randomly distributed. Safe to use median/mode imputation or listwise deletion."
    
    return {
        "has_missing": True,
        "columns_affected": len(cols_with_missing),
        "column_details": missing_info,
        "missing_correlations": missing_correlations[:5],
        "missing_inter_correlations": missing_inter_correlations[:5],
        "row_patterns": row_patterns,
        "inferred_pattern": pattern_type,
        "recommendation": pattern_advice
    }
