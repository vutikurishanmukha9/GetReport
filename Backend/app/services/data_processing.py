from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any
from io import BytesIO

import polars as pl
import numpy as np
from fastapi import UploadFile, HTTPException

# ─── Logger ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS: set[str]  = {".csv", ".xls", ".xlsx"}
MAX_FILE_SIZE_BYTES: int      = 50 * 1024 * 1024          # 50 MB hard cap (Upload only)
PREVIEW_ROW_COUNT: int        = 10

# ─── Custom Exceptions ───────────────────────────────────────────────────────
class UnsupportedFileTypeError(Exception):
    pass

class FileTooLargeError(Exception):
    pass

class EmptyFileError(Exception):
    pass

class ParseError(Exception):
    pass

class InvalidDataFrameError(TypeError):
    pass

# ─── Cleaning Report ─────────────────────────────────────────────────────────
@dataclass
class CleaningReport:
    """
    Tracks changes made by the Polars cleaning pipeline.
    """
    empty_rows_dropped:       int                              = 0
    empty_columns_dropped:    int                              = 0
    duplicate_rows_removed:   int                              = 0
    columns_renamed:          dict[str, str]                   = field(default_factory=dict)
    type_conversions:         list[dict[str, str]]             = field(default_factory=list)
    numeric_nans_filled:      int                              = 0
    categorical_nans_filled:  int                              = 0
    total_changes:            int                              = 0
    timing_ms:                float                            = 0.0

    def finalize(self) -> None:
        self.total_changes = (
            self.empty_rows_dropped
            + self.empty_columns_dropped
            + self.duplicate_rows_removed
            + len(self.columns_renamed)
            + len(self.type_conversions)
            + self.numeric_nans_filled
            + self.categorical_nans_filled
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "empty_rows_dropped":       self.empty_rows_dropped,
            "empty_columns_dropped":    self.empty_columns_dropped,
            "duplicate_rows_removed":   self.duplicate_rows_removed,
            "columns_renamed":          self.columns_renamed,
            "type_conversions":         self.type_conversions,
            "numeric_nans_filled":      self.numeric_nans_filled,
            "categorical_nans_filled":  self.categorical_nans_filled,
            "total_changes":            self.total_changes,
            "timing_ms":                round(self.timing_ms, 2),
        }

# ─── Utility: Snake Case Conversion ──────────────────────────────────────────
def _to_snake_case(name: str) -> str:
    name = name.strip()
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_").lower()
    return name

# ─── File Validation ─────────────────────────────────────────────────────────
def _validate_upload(file: UploadFile, content_length: int) -> str:
    # Logic preserved for API validation
    if not file.filename or file.filename.strip() == "":
        raise HTTPException(status_code=400, detail="No filename provided.")
    
    dot_index = file.filename.rfind(".")
    extension = file.filename[dot_index:].lower() if dot_index != -1 else ""

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{extension}'")
        
    # Note: With streaming, we might relax strict memory checks here
    # but for now we keep the 50MB check if content-length is provided.
    if content_length > MAX_FILE_SIZE_BYTES:
        size_mb = content_length / (1024 * 1024)
        raise HTTPException(400, f"File too large ({size_mb:.1f} MB). Limit: 50MB")

    return extension

# ─── File Loader (Polars) ────────────────────────────────────────────────────
def load_dataframe(file_path: str) -> pl.DataFrame:
    """
    Load a file from disk into a Polars DataFrame.
    Optimized for performance and memory.
    """
    logger.info("═══ load_dataframe (Polars) started — '%s' ═══", file_path)
    
    try:
        lower_path = file_path.lower()
        if lower_path.endswith(".csv"):
            # Polars read_csv is extremely fast and multi-threaded
            df = pl.read_csv(file_path, ignore_errors=True, n_rows=None)
        elif lower_path.endswith((".xls", ".xlsx")):
             # Polars read_excel uses engine='xlsx2csv' or similar internally via dependencies
            df = pl.read_excel(file_path)
        else:
            raise UnsupportedFileTypeError(f"Unsupported extension for: {file_path}")

        if df.height == 0:
            raise EmptyFileError("File is empty")
            
        logger.info("Loaded DataFrame: %d rows × %d columns", df.height, df.width)
        return df

    except Exception as e:
        logger.error("Parse failed: %s", e)
        raise ParseError(f"Could not parse file: {str(e)}")

# ─── Inspection (Polars) ─────────────────────────────────────────────────────
def inspect_dataset(df: pl.DataFrame) -> dict[str, Any]:
    """
    Polars-optimized dataset inspection.
    """
    quality_report = {
        "total_rows": df.height,
        "columns": [],
        "issues": [],
        "preview": [] 
    }
    
    # Generate Preview (Sanitized for JSON)
    rows = df.head(5)
    for row in rows.iter_rows(named=True):
        clean_row = {}
        for k, v in row.items():
            # Handle NaN/Inf -> None
            if isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')):
                clean_row[k] = None
            else:
                clean_row[k] = v
        quality_report["preview"].append(clean_row)

    # Iterate over columns efficiently
    for col_name in df.columns:
        null_count = df[col_name].null_count()
        dtype_str = str(df[col_name].dtype)
        
        # Inference (simplified)
        inferred = "string"
        if df[col_name].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
            inferred = "numeric"
        elif df[col_name].dtype in [pl.Date, pl.Datetime]:
            inferred = "datetime"
            
        col_info = {
            "name": col_name,
            "dtype": dtype_str,
            "inferred_type": inferred,
            "missing_count": null_count,
            "missing_percentage": round((null_count / df.height) * 100, 1)
        }
        quality_report["columns"].append(col_info)
        
        if null_count > 0:
            quality_report["issues"].append({
                "type": "missing_values",
                "column": col_name,
                "count": null_count,
                "severity": "high",
                "suggestion": "fill_mean" if inferred == "numeric" else "fill_unknown"
            })
            
    return quality_report

# ─── Cleaning Pipeline (Polars) ──────────────────────────────────────────────
def clean_data(
    df: pl.DataFrame, 
    rules: dict[str, Any] | None = None
) -> tuple[pl.DataFrame, CleaningReport]:
    
    if not isinstance(df, pl.DataFrame):
         raise InvalidDataFrameError(f"Expected pl.DataFrame, got {type(df)}")
         
    start_time = time.perf_counter()
    report = CleaningReport()
    
    original_height = df.height
    original_width = df.width
    
    # 1. Standardize Names
    new_cols = {c: _to_snake_case(c) for c in df.columns}
    df = df.rename(new_cols)
    report.columns_renamed = new_cols

    # 2. Apply Rules (Interactive)
    if rules:
        for original_col, rule in rules.items():
            target_col = _to_snake_case(original_col)
            if target_col not in df.columns: continue
            
            action = rule.get("action")
            if action == "drop_rows":
                # Polars: filter is fast
                df = df.filter(pl.col(target_col).is_not_null())
                
            elif action == "fill_mean":
                if df[target_col].dtype in [pl.Int64, pl.Float64]:
                    mean_val = df[target_col].mean()
                    df = df.with_columns(pl.col(target_col).fill_null(mean_val))
                    report.numeric_nans_filled += 1
                    
            elif action == "fill_value":
                 val = rule.get("value")
                 df = df.with_columns(pl.col(target_col).fill_null(val))

    # 3. Drop fully null rows/cols (Not straightforward in Polars, but efficient via expressions)
    # Actually, dropping fully null columns is just `dropna(how='all')` equiv.
    # Polars doesn't have `how='all'`. We must check null counts.
    # For now, we skip "drop fully empty rows" as it requires horizontal scan which is expensive.
    
    # Only drop fully null columns
    # df = df[[s.name for s in df if s.null_count() != df.height]]
    
    # 4. Remove Duplicates
    init_rows = df.height
    df = df.unique()
    report.duplicate_rows_removed = init_rows - df.height
    
    # 5. Type Conversions & Safe Imputation
    # Polars has strict types. We iterate and cast.
    id_patterns = ["id", "code", "sku", "zip", "phone"]
    
    for col in df.columns:
        col_lower = col.lower()
        is_id = any(p in col_lower for p in id_patterns)
        
        # Determine target type logic here...
        # For simplicity in this rough migration, we rely on Polars inference primarily.
        
        # Missing Value Handling from original logic:
        # Numeric -> Leave NaN (Polars uses null)
        # Categorical -> Fill "Unknown"
        dtype = df[col].dtype
        if dtype == pl.Utf8 or dtype == pl.Object:
            null_cnt = df[col].null_count()
            if null_cnt > 0:
                df = df.with_columns(pl.col(col).fill_null("Unknown"))
                report.categorical_nans_filled += null_cnt
                
    report.timing_ms = (time.perf_counter() - start_time) * 1000
    report.finalize()
    
    return df, report

# ─── Dataset Info (Polars) ───────────────────────────────────────────────────
def get_dataset_info(df: pl.DataFrame) -> dict[str, Any]:
    
    # Polars `describe` is different from Pandas.
    # We construct summary manually or via `describe`
    
    summary_df = df.describe() 
    # summary_df has columns: "statistic", col1, col2...
    summary_dict = summary_df.to_dict(as_series=False)
    
    # Convert 'statistic' column to keys for better structure?
    # Or just keep it.
    
    # Convert DataFrame to dictionaries for preview
    # serialize_rows handles NaN -> None conversion which is safer for JSON
    rows = df.head(PREVIEW_ROW_COUNT)
    preview = []
    for row in rows.iter_rows(named=True):
        clean_row = {}
        for k, v in row.items():
            # Handle NaN/Inf for JSON safety
            if isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')):
                clean_row[k] = None
            else:
                clean_row[k] = v
        preview.append(clean_row)
    
    missing_per_col = {}
    for col in df.columns:
        c = df[col].null_count()
        missing_per_col[col] = {
            "count": c,
            "percentage": round(c / df.height * 100 if df.height else 0, 2)
        }

    numeric_cols = [c for c, t in df.schema.items() if t in (pl.Int64, pl.Float64, pl.Int32, pl.Float32)]
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    
    return {
        "rows": df.height,
        "columns": df.columns,
        "dtypes": {c: str(t) for c, t in df.schema.items()},
        "summary": summary_dict,
        "preview": preview,
        "missing_values": missing_per_col,
        "duplicate_rows": 0, # Expensive to check again
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "memory_usage_mb": round(df.estimated_size() / (1024*1024), 2)
    }

# ─── Advanced Analysis ───────────────────────────────────────────────────────
def analyze_dataset(df: pl.DataFrame) -> dict[str, Any]:
    """
    Performs statistical analysis (correlations, distributions).
    """
    numeric_cols = [c for c, t in df.schema.items() if t in (pl.Int64, pl.Float64, pl.Int32, pl.Float32)]
    
    # 1. Summary Stats
    summary = df.describe().to_dict(as_series=False)
    
    # 2. Correlations (only if numeric cols > 1)
    correlations = {}
    if len(numeric_cols) > 1:
        # Polars correlation is simpler to just do pairwise for now or use Pearson
        # Computing full correlation matrix can be expensive.
        # Let's do a limited sample corr if rows > 10000?
        target_df = df.select(numeric_cols)
        if target_df.height > 10000:
            target_df = target_df.sample(10000)
            
        # Pearson correlation matrix
        # Polars doesn't have a direct 'corr()' returning a matrix easily like pandas.
        # We iteration:
        for c1 in numeric_cols:
            correlations[c1] = {}
            for c2 in numeric_cols:
                 if c1 == c2:
                     correlations[c1][c2] = 1.0
                 else:
                     # corr handles nulls?
                     val = target_df.select(pl.corr(c1, c2)).item()
                     # Handle NaN
                     if val is not None and not np.isnan(val):
                         correlations[c1][c2] = round(val, 2)
                     else:
                         correlations[c1][c2] = 0.0

    return {
        "summary": summary,
        "correlations": correlations
    }