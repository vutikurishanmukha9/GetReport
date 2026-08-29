from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any
from io import BytesIO

import polars as pl
import numpy as np
import zipfile
from fastapi import UploadFile, HTTPException
from app.core.config import settings

# ─── Logger ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS: set[str]  = {".csv", ".xls", ".xlsx", ".parquet", ".json", ".jsonl", ".ndjson", ".tsv", ".feather", ".arrow", ".gz"}
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

def _validate_zip_bomb(file_path: str, max_uncompressed_size_mb: int | None = None, max_ratio: float = 100.0) -> None:
    """
    Validate that a zip file (like .xlsx) is not a zip bomb.
    Checks compression ratio and total uncompressed size.
    """
    if max_uncompressed_size_mb is None:
        max_uncompressed_size_mb = settings.MAX_EXCEL_DECOMPRESSED_SIZE_MB
        
    if not zipfile.is_zipfile(file_path):
        return
        
    total_uncompressed_size = 0
    total_compressed_size = 0
    
    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            infolist = zf.infolist()
            # Check for extreme number of files
            if len(infolist) > 5000:
                raise ParseError("Too many files in archive (potential zip bomb).")
                
            for info in infolist:
                # Prevent directory traversal attacks inside zip extraction
                if "../" in info.filename or info.filename.startswith("/"):
                    raise ParseError("Malicious filename detected in archive.")
                
                total_uncompressed_size += info.file_size
                total_compressed_size += info.compress_size
                    
        # Check total uncompressed size limits
        max_size_bytes = max_uncompressed_size_mb * 1024 * 1024
        if total_uncompressed_size > max_size_bytes:
            raise ParseError(f"Decompressed size ({total_uncompressed_size / (1024*1024):.1f}MB) exceeds safe limit of {max_uncompressed_size_mb}MB.")
            
        # Check compression ratio
        if total_compressed_size > 0:
            ratio = total_uncompressed_size / total_compressed_size
            if ratio > max_ratio and total_uncompressed_size > 10 * 1024 * 1024: # Only enforce ratio if size is > 10MB
                raise ParseError(f"High compression ratio ({ratio:.1f}x) detected (potential zip bomb).")
                
    except zipfile.BadZipFile:
        pass
    except Exception as e:
        if isinstance(e, ParseError):
            raise
        raise ParseError(f"Error checking zip archive safety: {str(e)}")

# Extended Null Values List for Polars CSV parsing and post-processing
EXTENDED_NULL_VALUES: list[str] = [
    "nan", "NaN", "NAN", "N/A", "n/a", "null", "NULL", "None", "none",
    "#N/A", "#REF!", "#VALUE!", "#DIV/0!", "-", "?", "", "  ", "nil",
    "missing", "NA", "N.A.", "<NA>", "undefined", "null_value"
]
_LOWER_NULL_SET: set[str] = {v.lower() for v in EXTENDED_NULL_VALUES}
_CURRENCY_CLEAN_REGEX: str = r"[\$,€,£,₹,¥,\s]"
_PAREN_NEG_REGEX: str = r"^\((.*)\)$"


def _detect_csv_parameters(file_path: str) -> tuple[str, str, int]:
    """
    Auto-detect encoding, separator, and header row offset for messy CSV files.
    Optimized buffer sampling.
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1", "utf-16"]
    best_encoding = "utf-8"
    sample_text = ""
    
    # 1. Detect Encoding using fast 16KB chunk read
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc, errors="replace") as f:
                sample_text = "".join([f.readline() for _ in range(30)])
                best_encoding = enc
                break
        except Exception:
            continue

    if not sample_text:
        return "utf-8", ",", 0

    lines = [line.strip() for line in sample_text.splitlines() if line.strip()]
    if not lines:
        return best_encoding, ",", 0

    # 2. Detect Separator
    possible_separators = [",", ";", "\t", "|", ":"]
    sep_counts: dict[str, tuple[int, int]] = {}
    for sep in possible_separators:
        counts = [line.count(sep) for line in lines[:10]]
        if counts and max(counts) > 0:
            mode_count = max(set(counts), key=counts.count)
            sep_counts[sep] = (mode_count, counts.count(mode_count))
            
    best_sep = ","
    if sep_counts:
        best_sep = max(sep_counts.keys(), key=lambda s: (sep_counts[s][0], sep_counts[s][1]))

    # 3. Detect Skip Rows (Header Offset for Metadata text rows)
    skip_rows = 0
    if len(lines) > 2 and best_sep:
        counts = [line.count(best_sep) for line in lines]
        max_sep_in_file = max(counts) if counts else 0
        for idx, cnt in enumerate(counts[:10]):
            if max_sep_in_file > 2 and cnt < (max_sep_in_file * 0.5):
                skip_rows = idx + 1
            else:
                break

    return best_encoding, best_sep, skip_rows


def _sanitize_and_coerce_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Sanitize dataframe columns:
    1. Replace string null variants with true Null.
    2. Coerce string columns with currency/symbols/percentages into numeric floats.
    3. Ensure clean, unique snake_case column names.
    4. Replace infinity / -infinity float values with Null.
    """
    if df.height == 0 or df.width == 0:
        return df

    # Step A: Unique snake_case column names
    seen_names: set[str] = set()
    new_column_names: list[str] = []
    for idx, col in enumerate(df.columns):
        clean_name = _to_snake_case(str(col)) if col and str(col).strip() else f"column_{idx+1}"
        base_name = clean_name
        counter = 1
        while clean_name in seen_names:
            clean_name = f"{base_name}_{counter}"
            counter += 1
        seen_names.add(clean_name)
        new_column_names.append(clean_name)

    df = df.rename(dict(zip(df.columns, new_column_names)))

    # Step B: Column-level cleaning & vectorized auto coercion
    exprs = []
    null_list = list(_LOWER_NULL_SET)
    for col_name in df.columns:
        col_expr = pl.col(col_name)
        dtype = df[col_name].dtype

        if dtype in (pl.Utf8, pl.Object):
            trimmed = col_expr.str.strip_chars()
            null_handled = (
                pl.when(trimmed.str.to_lowercase().is_in(null_list) | (trimmed == ""))
                .then(None)
                .otherwise(trimmed)
            )

            # Fast sampling: Check if string column is actually numeric (currency $, €, £, %, commas)
            sample = df[col_name].drop_nulls().head(100)
            if sample.len() > 0:
                clean_sample = (
                    sample.str.strip_chars()
                    .str.replace_all(_CURRENCY_CLEAN_REGEX, "")
                    .str.replace_all(r"%", "")
                    .str.replace_all(_PAREN_NEG_REGEX, r"-\1")
                )
                numeric_parsed = clean_sample.cast(pl.Float64, strict=False)
                valid_num_count = numeric_parsed.drop_nulls().len()

                if valid_num_count / float(sample.len()) >= 0.7:
                    coerced = (
                        null_handled
                        .str.replace_all(_CURRENCY_CLEAN_REGEX, "")
                        .str.replace_all(r"%", "")
                        .str.replace_all(_PAREN_NEG_REGEX, r"-\1")
                        .cast(pl.Float64, strict=False)
                    )
                    exprs.append(coerced.alias(col_name))
                    continue

            exprs.append(null_handled.alias(col_name))

        elif dtype in (pl.Float64, pl.Float32):
            clean_float = (
                pl.when(col_expr.is_infinite() | col_expr.is_nan())
                .then(None)
                .otherwise(col_expr)
            )
            exprs.append(clean_float.alias(col_name))
        else:
            exprs.append(col_expr)

    return df.with_columns(exprs)


# ─── File Loader (Polars) ────────────────────────────────────────────────────
def load_dataframe(file_path: str) -> pl.DataFrame:
    """
    Load a file from disk into a Polars DataFrame with multi-encoding fallback,
    auto-delimiter detection, dirty string auto-coercion, and zip bomb validation.
    """
    logger.info("═══ load_dataframe (Ultra-Robust) started — '%s' ═══", file_path)
    
    # Pre-validate zip files (Excel) against Zip Bombs
    lower_path = file_path.lower()
    if lower_path.endswith((".xls", ".xlsx")):
        _validate_zip_bomb(file_path, max_uncompressed_size_mb=settings.MAX_EXCEL_DECOMPRESSED_SIZE_MB)
    
    try:
        lower_path = file_path.lower()
        if lower_path.endswith(".csv") or lower_path.endswith(".txt"):
            encoding, sep, skip_rows = _detect_csv_parameters(file_path)
            try:
                df = pl.read_csv(
                    file_path,
                    separator=sep,
                    skip_rows=skip_rows,
                    encoding=encoding,
                    ignore_errors=True,
                    null_values=EXTENDED_NULL_VALUES,
                    truncate_ragged_lines=True,
                )
            except Exception as csv_err:
                logger.warning("Primary CSV read failed (%s). Falling back to Latin-1 lenient parse with sep='%s'.", csv_err, sep)
                df = pl.read_csv(
                    file_path,
                    separator=sep,
                    skip_rows=skip_rows,
                    encoding="latin-1",
                    ignore_errors=True,
                    null_values=EXTENDED_NULL_VALUES,
                    truncate_ragged_lines=True,
                )
        elif lower_path.endswith(".tsv"):
            df = pl.read_csv(
                file_path,
                separator="\t",
                ignore_errors=True,
                null_values=EXTENDED_NULL_VALUES,
                truncate_ragged_lines=True,
            )
        elif lower_path.endswith((".xls", ".xlsx")):
            df = pl.read_excel(file_path)
        elif lower_path.endswith(".parquet"):
            df = pl.read_parquet(file_path)
        elif lower_path.endswith((".jsonl", ".ndjson")):
            df = pl.read_ndjson(file_path)
        elif lower_path.endswith(".json"):
            df = pl.read_json(file_path)
        elif lower_path.endswith((".feather", ".arrow")):
            df = pl.read_ipc(file_path)
        elif lower_path.endswith(".gz"):
            df = pl.read_csv(file_path, ignore_errors=True, null_values=EXTENDED_NULL_VALUES)
        else:
            raise UnsupportedFileTypeError(f"Unsupported extension for: {file_path}")

        if df.height == 0:
            raise EmptyFileError("File is empty")

        # Sanitize column names, coerce currency/percent strings, and clean nulls
        df = _sanitize_and_coerce_df(df)
            
        logger.info("Loaded DataFrame: %d rows × %d columns", df.height, df.width)
        return df

    except Exception as e:
        logger.error("Parse failed: %s", e)
        raise ParseError(f"Could not parse file: {str(e)}")


def join_datasets(
    dfs_dict: dict[str, pl.DataFrame],
    join_key: str,
    how: str = "inner"
) -> pl.DataFrame:
    """
    Polars-optimized multi-dataset join engine.
    Joins multiple datasets on a common key column (e.g., 'id', 'user_id', 'date').
    Appends suffix to overlapping non-key columns.
    """
    if not dfs_dict:
        raise ValueError("No datasets provided for join.")
    
    file_names = list(dfs_dict.keys())
    if len(file_names) == 1:
        return dfs_dict[file_names[0]]

    # Standardize join_key to snake_case
    clean_key = _to_snake_case(join_key)
    
    base_name = file_names[0]
    base_df = dfs_dict[base_name]
    if clean_key not in base_df.columns:
        matched_cols = [c for c in base_df.columns if c.lower() == clean_key.lower()]
        if matched_cols:
            clean_key = matched_cols[0]
        else:
            raise KeyError(f"Join key '{join_key}' not found in primary dataset '{base_name}'. Available: {base_df.columns}")

    result_df = base_df

    for idx, fname in enumerate(file_names[1:], start=1):
        other_df = dfs_dict[fname]
        
        other_key = clean_key
        if other_key not in other_df.columns:
            matched_cols = [c for c in other_df.columns if c.lower() == clean_key.lower()]
            if matched_cols:
                other_key = matched_cols[0]
            else:
                raise KeyError(f"Join key '{join_key}' not found in dataset '{fname}'. Available: {other_df.columns}")

        col_rename = {}
        if other_key != clean_key:
            col_rename[other_key] = clean_key

        for c in other_df.columns:
            if c != other_key and c in result_df.columns:
                safe_name = f"{c}_{idx+1}"
                col_rename[c] = safe_name
        
        if col_rename:
            other_df = other_df.rename(col_rename)

        how_mode = how.lower()
        if how_mode == "outer":
            how_mode = "full"
        if how_mode not in ("inner", "left", "full"):
            how_mode = "inner"

        result_df = result_df.join(other_df, on=clean_key, how=how_mode, coalesce=True)

    return _sanitize_and_coerce_df(result_df)


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
                if df[target_col].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]:
                    q1 = df[target_col].quantile(0.25)
                    q3 = df[target_col].quantile(0.75)
                    if q1 is not None and q3 is not None:
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        
                        # Calculate outlier count for impact
                        outlier_mask = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
                        outliers_replaced = df.filter(outlier_mask).height
                        
                        df = df.with_columns(
                            pl.when(pl.col(target_col) > upper_bound)
                            .then(pl.lit(upper_bound, dtype=df[target_col].dtype))
                            .when(pl.col(target_col) < lower_bound)
                            .then(pl.lit(lower_bound, dtype=df[target_col].dtype))
                            .otherwise(pl.col(target_col))
                            .alias(target_col)
                        )
                        dag.add_node(
                            operation="replace_outliers",
                            df_before=df_before,
                            df_after=df,
                            target_column=target_col,
                            parameters={
                                "method": "winsorization_iqr",
                                "lower_bound": lower_bound,
                                "upper_bound": upper_bound,
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
    
    # ─── Step 4: Automated Smart Cleaning & Safe Imputation ─────────────────────
    masked_placeholders = ["n/a", "na", "-999", "null", "?", "-", "missing", "unknown", "none"]
    id_patterns = ["id", "code", "sku", "zip", "phone"]
    
    for col in df.columns:
        col_lower = col.lower()
        is_id = any(p in col_lower for p in id_patterns)
        dtype = df[col].dtype
        
        # 4a. Strip masked null string placeholders in Utf8/Object columns & fill categorical
        if dtype == pl.Utf8 or dtype == pl.Object:
            step_start = time.perf_counter()
            df_before = df.clone()
            
            # Convert masked string placeholders to null
            df = df.with_columns(
                pl.when(pl.col(col).str.strip_chars().str.to_lowercase().is_in(masked_placeholders))
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
            
            null_cnt = df[col].null_count()
            if null_cnt > 0 and not is_id:
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

        # 4b. Auto-impute numeric missing values with median for non-ID numeric columns
        elif dtype.is_numeric() and not is_id:
            null_cnt = df[col].null_count()
            if null_cnt > 0:
                step_start = time.perf_counter()
                df_before = df.clone()
                med_val = df[col].median()
                if med_val is not None:
                    df = df.with_columns(pl.col(col).fill_null(med_val))
                    report.numeric_nans_filled += null_cnt
                    dag.add_node(
                        operation="fill_null_median",
                        df_before=df_before,
                        df_after=df,
                        target_column=col,
                        parameters={"fill_value": med_val, "nulls_filled": null_cnt},
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


def impute_multivariate_mice(
    df: pl.DataFrame,
    numeric_cols: list[str] | None = None,
    max_iter: int = 5,
    ridge_alpha: float = 1e-3
) -> pl.DataFrame:
    """
    Multivariate Imputation by Chained Equations (MICE).
    Performs iterative linear/ridge regression to impute missing values in numeric columns,
    preserving joint feature covariances and correlations without variance deflation.
    """
    if numeric_cols is None:
        numeric_cols = [c for c, t in df.schema.items() if t in (pl.Int64, pl.Float64, pl.Int32, pl.Float32, pl.UInt64, pl.UInt32)]
        
    if len(numeric_cols) < 2 or df.height < 3:
        return df
        
    has_nulls = any(df[c].null_count() > 0 for c in numeric_cols)
    if not has_nulls:
        return df
        
    num_df = df.select([pl.col(c).cast(pl.Float64) for c in numeric_cols])
    data = num_df.to_numpy().copy()
    
    missing_masks = np.isnan(data)
    if not missing_masks.any():
        return df
        
    # Initial median imputation
    for j in range(data.shape[1]):
        col_vals = data[:, j]
        nan_mask = missing_masks[:, j]
        if nan_mask.any():
            valid_vals = col_vals[~nan_mask]
            med_val = float(np.median(valid_vals)) if len(valid_vals) > 0 else 0.0
            data[nan_mask, j] = med_val
            
    # Iterative MICE chained regression
    n_samples, n_features = data.shape
    for _ in range(max_iter):
        for j in range(n_features):
            nan_mask = missing_masks[:, j]
            if not nan_mask.any():
                continue
                
            obs_mask = ~nan_mask
            obs_count = int(obs_mask.sum())
            if obs_count < 2:
                continue
                
            feature_indices = [idx for idx in range(n_features) if idx != j]
            X_obs = data[obs_mask][:, feature_indices]
            y_obs = data[obs_mask, j]
            X_miss = data[nan_mask][:, feature_indices]
            
            # Subsample training data if very large to prevent memory ballooning
            if obs_count > 10000:
                sample_idx = np.random.choice(obs_count, size=10000, replace=False)
                X_obs = X_obs[sample_idx]
                y_obs = y_obs[sample_idx]
            
            X_obs_bias = np.c_[np.ones(X_obs.shape[0]), X_obs]
            X_miss_bias = np.c_[np.ones(X_miss.shape[0]), X_miss]
            
            XtX = X_obs_bias.T @ X_obs_bias
            reg = ridge_alpha * np.eye(XtX.shape[0])
            try:
                beta = np.linalg.solve(XtX + reg, X_obs_bias.T @ y_obs)
                y_pred = X_miss_bias @ beta
                data[nan_mask, j] = y_pred
            except Exception:
                pass
                
    imputed_exprs = [
        pl.Series(col_name, data[:, idx]).alias(col_name)
        for idx, col_name in enumerate(numeric_cols)
    ]
    del data, missing_masks
    return df.with_columns(imputed_exprs)