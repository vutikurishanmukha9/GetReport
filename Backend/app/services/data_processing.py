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
        elif lower_path.endswith(".parquet"):
            # Parquet: used for intermediate cleaned data files
            df = pl.read_parquet(file_path)
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
                "suggestion": "fill_median" if inferred == "numeric" else "fill_unknown"
            })

        # Detect Outliers (Numeric only)
        if inferred == "numeric" and df.height > 10:
            q1 = df[col_name].quantile(0.25)
            q3 = df[col_name].quantile(0.75)
            if q1 is not None and q3 is not None:
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                
                outlier_count = df.select(
                    pl.col(col_name).is_between(lower, upper, closed="both").not_().sum()
                ).item()
                
                if outlier_count > 0:
                     quality_report["issues"].append({
                        "type": "outliers",
                        "column": col_name,
                        "count": outlier_count,
                        "severity": "medium",
                        "suggestion": "replace_outliers_median"
                    })
            
    # Check for Partial Duplicates (Rule #4)
    # Detect ID columns
    id_patterns = ["id", "code", "sku", "uuid", "pk"]
    potential_ids = [c for c in df.columns if any(p in c.lower() for p in id_patterns)]
    
    if len(potential_ids) > 0 and len(potential_ids) < len(df.columns):
        # Check duplicates on NON-ID columns
        subset_cols = [c for c in df.columns if c not in potential_ids]
        if len(subset_cols) > 0:
            n_dupes = df.select(subset_cols).is_duplicated().sum()
            if n_dupes > 0:
                quality_report["issues"].append({
                    "type": "partial_duplicates",
                    "column": "Multiple",
                    "count": n_dupes,
                    "severity": "medium",
                    "suggestion": "investigate"
                })

    # Calculate Histograms (Mugshots) - Numeric Only
    # LIMIT to first 15 numeric columns to prevent performance bottleneck on wide datasets.
    numeric_cols_for_hist = [c for c, t in df.schema.items() if t in (pl.Int64, pl.Int32, pl.Float64, pl.Float32)]
    
    # Analyze only the first 15 for the preview report
    for col_name in numeric_cols_for_hist[:15]:
        # Skip if mostly null
        if df[col_name].null_count() == df.height:
            continue

        try:
            # Use numpy for histogram if available as fallback
            # We convert to numpy array (zero copy often).
            arr = df[col_name].drop_nulls().to_numpy()
            
            if len(arr) == 0: continue
            
            counts, bin_edges = np.histogram(arr, bins=15)
            
            dist_data = []
            for i in range(len(counts)):
                label = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
                dist_data.append({
                    "label": label, 
                    "count": int(counts[i]), 
                    "min": float(bin_edges[i]), 
                    "max": float(bin_edges[i+1])
                })
                
            for col in quality_report["columns"]:
                if col["name"] == col_name:
                    col["distribution"] = dist_data
                    break

        except Exception as e:
            logger.warning(f"Failed to compute histogram for {col_name}: {e}")

    return quality_report

# ─── Cleaning Pipeline (Polars) ──────────────────────────────────────────────
def clean_data(
    df: pl.DataFrame, 
    rules: dict[str, Any] | None = None,
    dag: "TransformationDAG | None" = None,
    dataset_name: str = "",
) -> tuple[pl.DataFrame, CleaningReport, "TransformationDAG"]:
    """
    Clean the dataframe with optional transformation tracking.
    
    Args:
        df: Input DataFrame
        rules: User-specified cleaning rules
        dag: Optional TransformationDAG for audit tracking
        dataset_name: Name for audit trail
        
    Returns:
        Tuple of (cleaned_df, cleaning_report, transformation_dag)
    """
    from app.services.transformation_dag import TransformationDAG, create_dag
    
    if not isinstance(df, pl.DataFrame):
         raise InvalidDataFrameError(f"Expected pl.DataFrame, got {type(df)}")
         
    start_time = time.perf_counter()
    report = CleaningReport()
    
    # Initialize DAG if not provided
    if dag is None:
        dag = create_dag(df, dataset_name)
    
    original_height = df.height
    original_width = df.width
    
    # ─── Step 1: Standardize Names ───────────────────────────────────────────
    step_start = time.perf_counter()
    df_before = df.clone()
    
    new_cols = {c: _to_snake_case(c) for c in df.columns}
    df = df.rename(new_cols)
    report.columns_renamed = new_cols
    
    # Only add node if columns actually changed
    changed_cols = {k: v for k, v in new_cols.items() if k != v}
    if changed_cols:
        dag.add_node(
            operation="rename_columns",
            df_before=df_before,
            df_after=df,
            parameters={"mappings": changed_cols},
            duration_ms=(time.perf_counter() - step_start) * 1000,
        )

    # ─── Step 2: Apply User Rules (Interactive) ──────────────────────────────
    if rules:
        for original_col, rule in rules.items():
            target_col = _to_snake_case(original_col)
            if target_col not in df.columns: 
                continue
            
            action = rule.get("action")
            step_start = time.perf_counter()
            df_before = df.clone()
            
            if action == "drop_rows":
                df = df.filter(pl.col(target_col).is_not_null())
                dag.add_node(
                    operation="drop_null_rows",
                    df_before=df_before,
                    df_after=df,
                    target_column=target_col,
                    duration_ms=(time.perf_counter() - step_start) * 1000,
                )
                
            elif action == "fill_mean":
                if df[target_col].dtype in [pl.Int64, pl.Float64]:
                    mean_val = df[target_col].mean()
                    null_cnt = df[target_col].null_count()
                    if null_cnt > 0:
                        df = df.with_columns(pl.col(target_col).fill_null(mean_val))
                        report.numeric_nans_filled += null_cnt
                        dag.add_node(
                            operation="fill_null_mean",
                            df_before=df_before,
                            df_after=df,
                            target_column=target_col,
                            parameters={"fill_value": mean_val, "nulls_filled": null_cnt},
                            duration_ms=(time.perf_counter() - step_start) * 1000,
                            values_changed=null_cnt,
                        )

            elif action == "fill_median":
                if df[target_col].dtype in [pl.Int64, pl.Float64, pl.Float32, pl.Int32]:
                    median_val = df[target_col].median()
                    if median_val is not None:
                        null_cnt = df[target_col].null_count()
                        if null_cnt > 0:
                            df = df.with_columns(pl.col(target_col).fill_null(median_val))
                            report.numeric_nans_filled += null_cnt
                            dag.add_node(
                                operation="fill_null_median",
                                df_before=df_before,
                                df_after=df,
                                target_column=target_col,
                                parameters={"fill_value": median_val, "nulls_filled": null_cnt},
                                duration_ms=(time.perf_counter() - step_start) * 1000,
                                values_changed=null_cnt,
                            )
            
            elif action == "fill_mode":
                mode_s = df[target_col].mode()
                if mode_s.len() > 0:
                    mode_val = mode_s[0]
                    if mode_val is not None:
                        null_cnt = df[target_col].null_count()
                        if null_cnt > 0:
                            df = df.with_columns(pl.col(target_col).fill_null(mode_val))
                            report.categorical_nans_filled += null_cnt
                            dag.add_node(
                                operation="fill_null_mode",
                                df_before=df_before,
                                df_after=df,
                                target_column=target_col,
                                parameters={"fill_value": mode_val, "nulls_filled": null_cnt},
                                duration_ms=(time.perf_counter() - step_start) * 1000,
                                values_changed=null_cnt,
                            )

            elif action == "fill_value":
                 val = rule.get("value")
                 if val is not None:
                    null_cnt = df[target_col].null_count()
                    if null_cnt > 0:
                        df = df.with_columns(pl.col(target_col).fill_null(val))
                        dag.add_node(
                            operation="fill_null_value",
                            df_before=df_before,
                            df_after=df,
                            target_column=target_col,
                            parameters={"fill_value": val, "nulls_filled": null_cnt},
                            duration_ms=(time.perf_counter() - step_start) * 1000,
                            values_changed=null_cnt,
                        )

            elif action == "replace_outliers_median":
                if df[target_col].dtype in [pl.Int64, pl.Float64]:
                    q1 = df[target_col].quantile(0.25)
                    q3 = df[target_col].quantile(0.75)
                    if q1 is not None and q3 is not None:
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        median_val = df[target_col].median()
                        
                        # Calculate outlier count for impact
                        outlier_mask = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
                        outliers_replaced = df.filter(outlier_mask).height
                        
                        df = df.with_columns(
                            pl.when(outlier_mask)
                            .then(median_val)
                            .otherwise(pl.col(target_col))
                            .alias(target_col)
                        )
                        dag.add_node(
                            operation="replace_outliers",
                            df_before=df_before,
                            df_after=df,
                            target_column=target_col,
                            parameters={
                                "method": "iqr",
                                "lower_bound": lower_bound,
                                "upper_bound": upper_bound,
                                "replacement": median_val,
                                "outliers_replaced": outliers_replaced,
                            },
                            duration_ms=(time.perf_counter() - step_start) * 1000,
                            values_changed=outliers_replaced,
                        )

    # ─── Step 3: Remove Duplicates ───────────────────────────────────────────
    step_start = time.perf_counter()
    df_before = df.clone()
    init_rows = df.height
    df = df.unique()
    dups_removed = init_rows - df.height
    report.duplicate_rows_removed = dups_removed
    
    if dups_removed > 0:
        dag.add_node(
            operation="remove_duplicates",
            df_before=df_before,
            df_after=df,
            parameters={"duplicates_removed": dups_removed},
            duration_ms=(time.perf_counter() - step_start) * 1000,
        )
    
    # ─── Step 4: Type Conversions & Safe Imputation ──────────────────────────
    id_patterns = ["id", "code", "sku", "zip", "phone"]
    
    for col in df.columns:
        col_lower = col.lower()
        is_id = any(p in col_lower for p in id_patterns)
        
        dtype = df[col].dtype
        if dtype == pl.Utf8 or dtype == pl.Object:
            null_cnt = df[col].null_count()
            if null_cnt > 0:
                step_start = time.perf_counter()
                df_before = df.clone()
                
                df = df.with_columns(pl.col(col).fill_null("Unknown"))
                report.categorical_nans_filled += null_cnt
                
                dag.add_node(
                    operation="fill_null_value",
                    df_before=df_before,
                    df_after=df,
                    target_column=col,
                    parameters={"fill_value": "Unknown", "nulls_filled": null_cnt},
                    duration_ms=(time.perf_counter() - step_start) * 1000,
                    values_changed=null_cnt,
                )
                
    report.timing_ms = (time.perf_counter() - start_time) * 1000
    report.finalize()
    
    return df, report, dag

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