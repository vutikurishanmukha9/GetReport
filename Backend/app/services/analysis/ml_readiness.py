from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING
import polars as pl
import numpy as np

if TYPE_CHECKING:
    from app.services.confidence_scoring import ConfidenceReport

logger = logging.getLogger(__name__)

def _score_completeness(df: pl.DataFrame) -> tuple[float, list[str], set[str]]:
    """
    Compute base score as the median column completeness percentage.
    If a column has >50% and <=70% nulls, apply a -15 penalty and warning.
    """
    reasons: list[str] = []
    flagged: set[str] = set()
    total_rows = len(df)
    
    if total_rows == 0:
        return 0.0, [], set()
        
    completeness_list = []
    soft_penalty_count = 0
    
    for col in df.columns:
        series = df[col]
        null_count = series.null_count()
        completeness = ((total_rows - null_count) / total_rows) * 100
        completeness_list.append(completeness)
        
        null_pct = 100.0 - completeness
        if 50.0 < null_pct <= 70.0:
            reasons.append(f"Column '{col}' has a high missing rate: {null_pct:.1f}%")
            flagged.add(col)
            soft_penalty_count += 1
            
    base_score = float(np.median(completeness_list))
    score = base_score - (soft_penalty_count * 15.0)
    return max(0.0, score), reasons, flagged


def _score_constant_columns(df: pl.DataFrame) -> tuple[float, list[str], set[str]]:
    """
    Detect constant or near-zero variance columns (var < 1e-5).
    Penalty is -5 per column, capped at min(25, total_columns * 0.4 * 5).
    """
    reasons: list[str] = []
    flagged: set[str] = set()
    total_cols = len(df.columns)
    
    for col in df.columns:
        series = df[col].drop_nulls()
        if len(series) < 2:
            reasons.append(f"Column '{col}' has insufficient values for training")
            flagged.add(col)
            continue
            
        dtype = df.schema[col]
        if series.n_unique() == 1:
            reasons.append(f"Column '{col}' is constant across all rows")
            flagged.add(col)
        elif dtype.is_numeric():
            try:
                variance = series.var()
                if variance is not None and variance < 1e-5:
                    reasons.append(f"Column '{col}' has near-zero variance ({variance:.2e} < 1e-5)")
                    flagged.add(col)
            except Exception:
                logger.debug("Could not compute variance...", exc_info=True)
                
    cap = min(25.0, total_cols * 0.4 * 5.0)
    penalty = min(cap, len(flagged) * 5.0)
    
    return -penalty, reasons, flagged


def _score_type_inconsistency(report: ConfidenceReport, total_cols: int) -> tuple[float, list[str], set[str]]:
    """
    Check type/format consistency from the existing confidence report.
    Penalty is -3 per column, capped at min(15, total_columns * 0.3 * 3).
    """
    reasons: list[str] = []
    flagged: set[str] = set()
    
    for col_score in report.columns:
        if col_score.consistency < 80:
            reasons.append(f"Column '{col_score.column}' has format/type consistency issues")
            flagged.add(col_score.column)
            
    cap = min(15.0, total_cols * 0.3 * 3.0)
    penalty = min(cap, len(flagged) * 3.0)
    
    return -penalty, reasons, flagged


def _score_outliers(report: ConfidenceReport, total_cols: int) -> tuple[float, list[str], set[str]]:
    """
    Check outlier flags from the existing confidence report.
    Penalty is -3 per column with outlier issues (validity < 80), capped at min(15, total_columns * 0.3 * 3).
    """
    reasons: list[str] = []
    flagged: set[str] = set()
    
    for col_score in report.columns:
        if col_score.validity < 80:
            reasons.append(f"Column '{col_score.column}' has invalid or out-of-range values")
            flagged.add(col_score.column)
            
    cap = min(15.0, total_cols * 0.3 * 3.0)
    penalty = min(cap, len(flagged) * 3.0)
    
    return -penalty, reasons, flagged


def _score_class_imbalance(df: pl.DataFrame) -> tuple[float, list[str], set[str]]:
    """
    Identify potential target/label variables (categorical/string/boolean columns with 2-10 unique values).
    If dominant class >85%, apply -10 penalty. If >95%, apply -20 penalty.
    Capped at min(30, qualifying_categorical_columns * 0.5 * 10).
    """
    reasons: list[str] = []
    flagged: set[str] = set()
    flagged_penalties: list[tuple[str, float]] = []
    
    qualifying_cols = []
    total_rows = len(df)
    if total_rows == 0:
        return 0.0, [], set()
        
    for col in df.columns:
        dtype = df.schema[col]
        if dtype == pl.String or dtype == pl.Categorical:
            non_null_series = df[col].drop_nulls()
            non_null_len = len(non_null_series)
            if non_null_len > 0:
                n_unique = non_null_series.n_unique()
                if 2 <= n_unique <= 10:
                    qualifying_cols.append(col)
                    
                    vc = non_null_series.value_counts(sort=True)
                    if vc.height > 0:
                        top_count = vc["count"][0]
                        ratio = top_count / non_null_len
                        if ratio > 0.95:
                            reasons.append(f"Column '{col}' is heavily imbalanced ({ratio*100:.1f}% single class)")
                            flagged_penalties.append((col, 20.0))
                            flagged.add(col)
                        elif ratio > 0.85:
                            reasons.append(f"Column '{col}' is moderately imbalanced ({ratio*100:.1f}% single class)")
                            flagged_penalties.append((col, 10.0))
                            flagged.add(col)
                            
    total_penalty = sum(penalty for col, penalty in flagged_penalties)
    n_qualifying = len(qualifying_cols)
    cap = min(30.0, n_qualifying * 0.5 * 10.0)
    penalty = min(cap, total_penalty)
    
    return -penalty, reasons, flagged


def calculate_ml_readiness(report: ConfidenceReport, df: pl.DataFrame) -> dict[str, Any]:
    """
    Calculate the overall ML Readiness Score of the dataset.
    """
    total_cols = len(df.columns)
    total_rows = len(df)
    
    if total_cols == 0 or total_rows == 0:
        return {
            "score": 0.0,
            "status": "Not Ready",
            "reasons": ["Empty dataset"],
            "recommendation": "Upload a valid dataset to calculate ML readiness.",
            "column_context": "0 of 0 columns have issues"
        }
        
    critical_cols = [
        col for col in df.columns
        if (df[col].null_count() / total_rows) * 100 > 70.0
    ]
    if critical_cols:
        reasons = [
            f"Column '{c}' has {(df[c].null_count() / total_rows) * 100:.1f}% missing values"
            for c in critical_cols
        ]
        return {
            "score": 0.0,
            "status": "Not Ready",
            "reasons": reasons,
            "recommendation": "Impute or drop columns with critical missingness before attempting ML.",
            "column_context": f"{len(critical_cols)} of {total_cols} columns have issues"
        }
        
    comp_score, comp_reasons, comp_flagged = _score_completeness(df)
    const_penalty, const_reasons, const_flagged = _score_constant_columns(df)
    inc_penalty, inc_reasons, inc_flagged = _score_type_inconsistency(report, total_cols)
    out_penalty, out_reasons, out_flagged = _score_outliers(report, total_cols)
    imb_penalty, imb_reasons, imb_flagged = _score_class_imbalance(df)
    
    final_score = comp_score + const_penalty + inc_penalty + out_penalty + imb_penalty
    final_score = max(0.0, min(100.0, final_score))
    
    all_reasons = comp_reasons + const_reasons + inc_reasons + out_reasons + imb_reasons
    all_flagged = comp_flagged | const_flagged | inc_flagged | out_flagged | imb_flagged
    column_context = f"{len(all_flagged)} of {total_cols} columns have issues"
    
    if final_score >= 90.0:
        status = "Ready"
    elif final_score >= 65.0:
        status = "Needs Cleaning"
    else:
        status = "Not Ready"
        
    has_imb_95 = any("heavily imbalanced" in r for r in imb_reasons)
    has_soft_null = len(comp_flagged) > 0
    has_imb_85 = any("moderately imbalanced" in r for r in imb_reasons)
    has_constant = len(const_flagged) > 0
    has_outliers = len(out_flagged) > 0
    has_type_inc = len(inc_flagged) > 0
    
    if has_imb_95:
        recommendation = "Target columns are heavily imbalanced. Downsample dominant classes or apply SMOTE to avoid skewed predictions."
    elif has_soft_null:
        recommendation = "High null rates detected. Impute missing variables or drop highly incomplete features to prevent training errors."
    elif has_imb_85:
        recommendation = "Moderate class imbalance detected. Consider using class weight adjustments during model compilation."
    elif has_constant:
        recommendation = "Constant or near-constant features present. Drop columns with zero variance to reduce feature noise."
    elif has_outliers:
        recommendation = "Significant outliers present. Apply Winsorization capping or log transforms to normalize metric scaling."
    elif has_type_inc:
        recommendation = "Format inconsistency detected. Standardize category labels or verify type consistency prior to encoding."
    else:
        if status == "Ready":
            recommendation = "Excellent! The dataset is clean, complete, and ready for ML training."
        elif status == "Needs Cleaning":
            recommendation = "Dataset is generally usable. Impute nulls, cap outliers, and drop near-constant columns before training."
        else:
            recommendation = "Dataset requires substantial clean-up. Resolve missing fields and features variance issues before attempting ML."
            
    return {
        "score": round(final_score, 1),
        "status": status,
        "reasons": all_reasons,
        "recommendation": recommendation,
        "column_context": column_context
    }
