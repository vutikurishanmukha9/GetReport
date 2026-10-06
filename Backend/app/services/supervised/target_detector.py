import re
from typing import Optional, List, Tuple
import polars as pl
from .schemas import TaskType

TARGET_KEYWORDS = [
    "target", "label", "outcome", "churn", "is_churned", "churned",
    "status", "conversion", "converted", "is_converted", "default", "is_default",
    "fraud", "is_fraud", "revenue", "sales", "profit", "price", "rating",
    "score", "loss", "salary", "amount"
]

class TargetInferenceService:
    """
    Infers candidate target columns and identifies whether a target represents
    Binary Classification, Multiclass Classification, or Regression.
    """

    def infer_task_type(self, df: pl.DataFrame, target_col: str) -> TaskType:
        if target_col not in df.columns:
            return TaskType.UNKNOWN

        series = df[target_col].drop_nulls()
        n_rows = len(series)
        if n_rows < 10:
            return TaskType.UNKNOWN

        n_unique = series.n_unique()
        dtype = series.dtype

        if n_unique <= 1:
            return TaskType.UNKNOWN

        if n_unique == 2:
            return TaskType.BINARY_CLASSIFICATION

        # Check for Multiclass vs Regression
        is_numeric = dtype in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64
        )

        if not is_numeric:
            if n_unique <= 20:
                return TaskType.MULTICLASS_CLASSIFICATION
            return TaskType.UNKNOWN

        # Numeric target
        if n_unique <= 10 and (n_unique / n_rows) < 0.1:
            return TaskType.MULTICLASS_CLASSIFICATION

        if n_unique > 5:
            return TaskType.REGRESSION

        return TaskType.UNKNOWN

    def detect_candidate_target(self, df: pl.DataFrame) -> Optional[str]:
        """
        Scans columns to identify the most probable target outcome variable
        based on keyword relevance, completeness, and non-triviality.
        """
        best_col: Optional[str] = None
        best_score = -1.0
        n_rows = len(df)

        for col in df.columns:
            col_lower = col.lower().strip()
            
            # Skip obvious identifier tokens
            if any(id_token in col_lower for id_token in ["id", "uuid", "hash", "key", "guid", "index"]):
                # If exact word is not just 'target_id'
                if col_lower in ["id", "uuid", "index"]:
                    continue

            # Calculate match priority
            keyword_score = 0.0
            for idx, kw in enumerate(TARGET_KEYWORDS):
                if col_lower == kw:
                    keyword_score = 100.0 - idx
                    break
                elif re.search(rf"\b{re.escape(kw)}\b", col_lower) or kw in col_lower:
                    keyword_score = 50.0 - idx
                    break

            if keyword_score <= 0.0:
                continue

            series = df[col].drop_nulls()
            n_valid = len(series)
            if n_valid < 10 or (n_valid / max(1, n_rows)) < 0.5:
                continue

            n_unique = series.n_unique()
            if n_unique <= 1:
                continue

            is_numeric = series.dtype in (
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64
            )

            # If all unique and non-numeric (or integer row numbers), likely an identifier
            if not is_numeric and n_unique == n_valid:
                continue
            if is_numeric and series.dtype in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64) and n_unique == n_valid:
                # Check if it's a simple auto-increment sequence (1, 2, 3...)
                min_v = series.min()
                max_v = series.max()
                if min_v is not None and max_v is not None and (max_v - min_v + 1) == n_valid:
                    continue

            total_score = keyword_score + (10.0 if n_unique == 2 else 5.0)
            if total_score > best_score:
                best_score = total_score
                best_col = col

        return best_col
