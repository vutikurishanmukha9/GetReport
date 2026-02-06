"""
Column Confidence Scoring System

Calculates trust metrics for each column:
- Completeness: % of non-null values
- Consistency: Type uniformity and format consistency
- Validity: Values within expected ranges
- Stability: Variance detection (near-constant flagging)

These scores enable users to understand data quality at a glance.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any

import polars as pl
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ColumnConfidence:
    """Confidence scores for a single column."""
    column: str
    completeness: float  # 0-100: % non-null
    consistency: float   # 0-100: type/format uniformity
    validity: float      # 0-100: values within expected ranges
    stability: float     # 0-100: variance level (100 = good variance, low = near-constant)
    overall: float       # 0-100: weighted average
    issues: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "completeness": round(self.completeness, 1),
            "consistency": round(self.consistency, 1),
            "validity": round(self.validity, 1),
            "stability": round(self.stability, 1),
            "overall": round(self.overall, 1),
            "grade": self._get_grade(),
            "issues": self.issues
        }
    
    def _get_grade(self) -> str:
        if self.overall >= 90:
            return "A"
        elif self.overall >= 75:
            return "B"
        elif self.overall >= 60:
            return "C"
        elif self.overall >= 40:
            return "D"
        return "F"


@dataclass
class ConfidenceReport:
    """Confidence scores for entire dataset."""
    columns: list[ColumnConfidence]
    dataset_confidence: float
    high_confidence_count: int
    low_confidence_count: int
    critical_issues: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": [c.to_dict() for c in self.columns],
            "dataset_confidence": round(self.dataset_confidence, 1),
            "dataset_grade": self._get_dataset_grade(),
            "high_confidence_count": self.high_confidence_count,
            "low_confidence_count": self.low_confidence_count,
            "critical_issues": self.critical_issues
        }
    
    def _get_dataset_grade(self) -> str:
        if self.dataset_confidence >= 90:
            return "A"
        elif self.dataset_confidence >= 75:
            return "B"
        elif self.dataset_confidence >= 60:
            return "C"
        elif self.dataset_confidence >= 40:
            return "D"
        return "F"


# =============================================================================
# Scoring Functions
# =============================================================================

def _calculate_completeness(series: pl.Series) -> tuple[float, list[str]]:
    """Calculate completeness score (% non-null)."""
    issues = []
    total = series.len()
    if total == 0:
        return 0.0, ["Column is empty"]
    
    null_count = series.null_count()
    completeness = ((total - null_count) / total) * 100
    
    if completeness < 50:
        issues.append(f"High missing rate: {100 - completeness:.1f}%")
    elif completeness < 80:
        issues.append(f"Moderate missing rate: {100 - completeness:.1f}%")
    
    return completeness, issues


def _calculate_consistency(series: pl.Series, dtype: pl.DataType) -> tuple[float, list[str]]:
    """Calculate consistency score based on type uniformity and format."""
    issues = []
    score = 100.0
    
    # For string columns, check format consistency
    if dtype == pl.Utf8 or dtype == pl.String:
        non_null = series.drop_nulls()
        if non_null.len() > 0:
            # Check for mixed case patterns
            sample = non_null.head(min(1000, non_null.len())).to_list()
            
            upper_count = sum(1 for s in sample if s and s.isupper())
            lower_count = sum(1 for s in sample if s and s.islower())
            title_count = sum(1 for s in sample if s and s.istitle())
            mixed_count = len(sample) - upper_count - lower_count - title_count
            
            # If mixed case is dominant, reduce score
            if mixed_count > len(sample) * 0.3:
                score -= 15
                issues.append("Inconsistent text casing")
            
            # Check for leading/trailing whitespace
            whitespace_issues = sum(1 for s in sample if s and (s != s.strip()))
            if whitespace_issues > len(sample) * 0.05:
                score -= 10
                issues.append("Whitespace inconsistencies detected")
            
            # Check for mixed formats (e.g., dates)
            date_patterns = sum(1 for s in sample if s and re.match(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', s))
            if 0 < date_patterns < len(sample) * 0.9:
                score -= 20
                issues.append("Mixed date/non-date values")
    
    # For numeric columns, check for unexpected patterns
    elif dtype in (pl.Int64, pl.Float64, pl.Int32, pl.Float32):
        non_null = series.drop_nulls()
        if non_null.len() > 0:
            # Check for negative values in typically positive columns
            neg_count = (non_null < 0).sum()
            pos_count = (non_null > 0).sum()
            
            # If mostly positive with few negatives, flag as suspicious
            if pos_count > 0 and neg_count > 0:
                neg_ratio = neg_count / (neg_count + pos_count)
                if 0 < neg_ratio < 0.05:  # Less than 5% negative
                    score -= 10
                    issues.append("Sparse negative values (possibly errors)")
    
    return max(0, score), issues


def _calculate_validity(series: pl.Series, dtype: pl.DataType) -> tuple[float, list[str]]:
    """Calculate validity score based on value ranges and patterns."""
    issues = []
    score = 100.0
    
    non_null = series.drop_nulls()
    if non_null.len() == 0:
        return 50.0, ["No values to validate"]
    
    # For numeric columns
    if dtype in (pl.Int64, pl.Float64, pl.Int32, pl.Float32):
        try:
            min_val = non_null.min()
            max_val = non_null.max()
            mean_val = non_null.mean()
            std_val = non_null.std()
            
            # Check for extreme outliers using IQR
            q1 = non_null.quantile(0.25)
            q3 = non_null.quantile(0.75)
            iqr = q3 - q1
            
            if iqr > 0:
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                extreme_outliers = ((non_null < lower_bound) | (non_null > upper_bound)).sum()
                outlier_ratio = extreme_outliers / non_null.len()
                
                if outlier_ratio > 0.05:
                    score -= 25
                    issues.append(f"High outlier rate: {outlier_ratio*100:.1f}%")
                elif outlier_ratio > 0.01:
                    score -= 10
                    issues.append(f"Moderate outliers present")
            
            # Check for suspicious value patterns
            if min_val == max_val:
                score -= 40
                issues.append("All values are identical")
        except Exception:
            pass
    
    # For string columns
    elif dtype == pl.Utf8 or dtype == pl.String:
        # Check cardinality vs row count
        n_unique = non_null.n_unique()
        n_rows = non_null.len()
        
        if n_unique == 1:
            score -= 30
            issues.append("Only one unique value")
        elif n_unique == n_rows and n_rows > 100:
            # Might be an ID column
            score -= 10
            issues.append("All values unique (possible ID)")
    
    return max(0, score), issues


def _calculate_stability(series: pl.Series, dtype: pl.DataType) -> tuple[float, list[str]]:
    """Calculate stability score (variance detection)."""
    issues = []
    score = 100.0
    
    non_null = series.drop_nulls()
    if non_null.len() < 2:
        return 50.0, ["Insufficient data for stability check"]
    
    # For numeric columns
    if dtype in (pl.Int64, pl.Float64, pl.Int32, pl.Float32):
        try:
            std_val = non_null.std()
            mean_val = non_null.mean()
            
            if mean_val != 0 and std_val is not None:
                cv = abs(std_val / mean_val)  # Coefficient of variation
                
                if cv < 0.001:
                    score = 20.0
                    issues.append("Near-constant values (very low variance)")
                elif cv < 0.01:
                    score = 50.0
                    issues.append("Low variance detected")
                elif cv > 10:
                    score = 70.0
                    issues.append("Extremely high variance")
            elif std_val == 0:
                score = 10.0
                issues.append("Zero variance (constant column)")
        except Exception:
            pass
    
    # For string columns, check if dominated by single value
    elif dtype == pl.Utf8 or dtype == pl.String:
        try:
            value_counts = non_null.value_counts()
            if value_counts.height > 0:
                top_count = value_counts["count"].max()
                total = non_null.len()
                dominance = top_count / total if total > 0 else 0
                
                if dominance > 0.95:
                    score = 30.0
                    issues.append("Single value dominates (>95%)")
                elif dominance > 0.80:
                    score = 60.0
                    issues.append("Low category diversity")
        except Exception:
            pass
    
    return score, issues


# =============================================================================
# Main Function
# =============================================================================

def calculate_confidence_scores(df: pl.DataFrame) -> ConfidenceReport:
    """
    Calculate confidence scores for all columns in the dataset.
    
    Returns a ConfidenceReport with per-column scores and dataset-level metrics.
    """
    column_scores: list[ColumnConfidence] = []
    critical_issues: list[str] = []
    
    for col in df.columns:
        series = df[col]
        dtype = df.schema[col]
        
        all_issues: list[str] = []
        
        # Calculate individual scores
        completeness, comp_issues = _calculate_completeness(series)
        all_issues.extend(comp_issues)
        
        consistency, cons_issues = _calculate_consistency(series, dtype)
        all_issues.extend(cons_issues)
        
        validity, val_issues = _calculate_validity(series, dtype)
        all_issues.extend(val_issues)
        
        stability, stab_issues = _calculate_stability(series, dtype)
        all_issues.extend(stab_issues)
        
        # Weighted average (completeness matters most)
        overall = (
            completeness * 0.35 +
            consistency * 0.25 +
            validity * 0.25 +
            stability * 0.15
        )
        
        column_score = ColumnConfidence(
            column=col,
            completeness=completeness,
            consistency=consistency,
            validity=validity,
            stability=stability,
            overall=overall,
            issues=all_issues
        )
        column_scores.append(column_score)
        
        # Track critical issues
        if overall < 40:
            critical_issues.append(f"{col}: Overall score {overall:.0f}% (Grade F)")
        elif completeness < 50:
            critical_issues.append(f"{col}: High missing rate ({100-completeness:.0f}%)")
    
    # Dataset-level metrics
    if column_scores:
        dataset_confidence = sum(c.overall for c in column_scores) / len(column_scores)
        high_confidence = sum(1 for c in column_scores if c.overall >= 75)
        low_confidence = sum(1 for c in column_scores if c.overall < 50)
    else:
        dataset_confidence = 0.0
        high_confidence = 0
        low_confidence = 0
    
    logger.info(f"Confidence scoring: {len(column_scores)} columns, "
                f"dataset score={dataset_confidence:.1f}%, "
                f"{len(critical_issues)} critical issues")
    
    return ConfidenceReport(
        columns=column_scores,
        dataset_confidence=dataset_confidence,
        high_confidence_count=high_confidence,
        low_confidence_count=low_confidence,
        critical_issues=critical_issues
    )
