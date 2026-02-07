"""
Comparison Service - Tier 4 Post-Clean Validation Loop

This service calculates the "delta" between the original dataset (before cleaning)
and the final dataset (after cleaning). It provides quantifiable metrics on how much
the data quality has improved.

Key Metrics:
- Completeness Score (100% - % missing)
- Uniqueness Score (100% - % duplicates)
- Value Distribution shifts (mean, std dev changes)
- Schema evolution
"""

import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import polars as pl

logger = logging.getLogger(__name__)

@dataclass
class QualityMetric:
    metric: str         # "missing_count", "unique_count", "mean", "std"
    before: float
    after: float
    delta: float        # after - before
    percent_change: float # (after - before) / before * 100
    improvement: bool   # Did clarity improve?

@dataclass
class ColumnComparison:
    column: str
    dtype_before: str
    dtype_after: str
    metrics: List[QualityMetric]
    warnings_resolved: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "dtype_before": self.dtype_before,
            "dtype_after": self.dtype_after,
            "metrics": [asdict(m) for m in self.metrics],
            "warnings_resolved": self.warnings_resolved
        }

@dataclass
class DatasetComparison:
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    completeness_score_before: float
    completeness_score_after: float
    uniqueness_score_before: float
    uniqueness_score_after: float
    column_changes: Dict[str, ColumnComparison] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "rows": {"before": self.rows_before, "after": self.rows_after, "delta": self.rows_after - self.rows_before},
                "cols": {"before": self.cols_before, "after": self.cols_after, "delta": self.cols_after - self.cols_before},
                "completeness": {
                    "before": round(self.completeness_score_before, 2),
                    "after": round(self.completeness_score_after, 2),
                    "delta": round(self.completeness_score_after - self.completeness_score_before, 2)
                },
                "uniqueness": {
                    "before": round(self.uniqueness_score_before, 2),
                    "after": round(self.uniqueness_score_after, 2),
                    "delta": round(self.uniqueness_score_after - self.uniqueness_score_before, 2)
                }
            },
            "columns": {k: v.to_dict() for k, v in self.column_changes.items()}
        }

class ComparisonService:
    
    def compare(self, df_before: pl.DataFrame, df_after: pl.DataFrame) -> DatasetComparison:
        """
        Compare two dataframes and return a comprehensive comparison report.
        """
        if df_before.is_empty():
            logger.warning("Original dataframe is empty, skipping comparison")
            return self._empty_comparison()
            
        # 1. Dataset Level Stats
        rows_before = df_before.height
        cols_before = df_before.width
        
        rows_after = df_after.height
        cols_after = df_after.width
        
        # Completeness = (Total Cells - Null Cells) / Total Cells * 100
        # Polars: sum_horizontal() sums across columns
        nulls_before = df_before.null_count().sum_horizontal()[0]
        total_cells_before = rows_before * cols_before
        comp_before = ((total_cells_before - nulls_before) / total_cells_before * 100) if total_cells_before > 0 else 0
        
        nulls_after = df_after.null_count().sum_horizontal()[0]
        total_cells_after = rows_after * cols_after
        comp_after = ((total_cells_after - nulls_after) / total_cells_after * 100) if total_cells_after > 0 else 0
        
        # Uniqueness (Approximate for speed if needed, but unique() is fast enough usually)
        # Using row uniqueness
        unique_rows_before = df_before.n_unique()
        unique_score_before = (unique_rows_before / rows_before * 100) if rows_before > 0 else 0
        
        unique_rows_after = df_after.n_unique()
        unique_score_after = (unique_rows_after / rows_after * 100) if rows_after > 0 else 0

        comparison = DatasetComparison(
            rows_before=rows_before,
            rows_after=rows_after,
            cols_before=cols_before,
            cols_after=cols_after,
            completeness_score_before=comp_before,
            completeness_score_after=comp_after,
            uniqueness_score_before=unique_score_before,
            uniqueness_score_after=unique_score_after
        )
        
        # 2. Column Level Stats
        # We map columns by name where possible. 
        # Note: clean_data ensures standard snake_case names. We should try to map original to new if renamed.
        # For simplicity, we assume comparative analysis usually happens on the "snake_cased" version of before vs after
        # IF the before df is the raw load, we might need to normalize its names first to match.
        
        # Normalizing names of df_before for accurate column mapping
        df_before_norm = self._normalize_columns(df_before)

        common_cols = set(df_before_norm.columns) & set(df_after.columns)
        
        for col in common_cols:
            col_comp = self._compare_column(df_before_norm[col], df_after[col], col)
            comparison.column_changes[col] = col_comp
            
        return comparison

    def _normalize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        # Simple local normalization just for comparison key matching
        # Uses the same logic as data_processing._to_snake_case ideally, 
        # but re-implementing simple heavy lifting here to avoid circular imports if strict
        # Assuming we can just use the column names as is if they match
        import re
        def to_snake(name: str) -> str:
            name = name.strip()
            name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
            name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
            name = re.sub(r'_+', '_', name).strip('_').lower()
            return name
        
        new_names = {c: to_snake(c) for c in df.columns}
        return df.rename(new_names)

    def _compare_column(self, s_before: pl.Series, s_after: pl.Series, col_name: str) -> ColumnComparison:
        metrics = []
        
        # Metric: Missing Count
        missing_before = s_before.null_count()
        missing_after = s_after.null_count()
        metrics.append(self._make_metric("missing_count", missing_before, missing_after, lower_is_better=True))
        
        # Metric: Unique Count
        unique_before = s_before.n_unique()
        unique_after = s_after.n_unique()
        # Higher uniqueness isn't always "better" (e.g. categorical), but meaningful change is noted
        metrics.append(self._make_metric("unique_count", unique_before, unique_after, lower_is_better=False)) 
        
        # Metric: Mean (if numeric)
        if s_after.dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64] and \
           s_before.dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
            mean_before = s_before.mean() or 0.0
            mean_after = s_after.mean() or 0.0
            metrics.append(self._make_metric("mean", mean_before, mean_after, lower_is_better=None))
            
            std_before = s_before.std() or 0.0
            std_after = s_after.std() or 0.0
            metrics.append(self._make_metric("std", std_before, std_after, lower_is_better=True)) # Generally less variance *might* imply cleaner data (outliers removed)

        return ColumnComparison(
            column=col_name,
            dtype_before=str(s_before.dtype),
            dtype_after=str(s_after.dtype),
            metrics=metrics,
            warnings_resolved=0 # Placeholder logic, could count resolved rules
        )

    def _make_metric(self, name: str, before: float, after: float, lower_is_better: Optional[bool]) -> QualityMetric:
        # Check for NaN/None and handle gracefully
        import math
        if before is None or math.isnan(before): before = 0.0
        if after is None or math.isnan(after): after = 0.0
        
        delta = after - before
        
        if before != 0:
            pct_change = (delta / before) * 100
        else:
            pct_change = 0.0 if delta == 0 else 100.0 # Arbitrary 100% if starting from 0 to something
            
        # Determine improvement
        improved = False
        if lower_is_better is True:
            improved = after < before
        elif lower_is_better is False:
            improved = after > before
        else:
            # Neutral metric (like Mean), just tracking change
            improved = abs(delta) > 0 
            
        return QualityMetric(
            metric=name,
            before=before,
            after=after,
            delta=delta,
            percent_change=round(pct_change, 2),
            improvement=improved
        )

    def _empty_comparison(self) -> DatasetComparison:
        return DatasetComparison(0,0,0,0,0.0,0.0,0.0,0.0)

# Singleton instance
comparison_service = ComparisonService()
