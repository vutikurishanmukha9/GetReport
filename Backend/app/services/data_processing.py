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
    # Use Polars hist() or dynamic binning
    for col_name in df.columns:
        if df[col_name].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
            # Skip if mostly null
            if df[col_name].null_count() == df.height:
                continue

            try:
                # Polars hist returns a PL Series of Type Struct.
                # structure: break_point (f64), category (cat/str), count (u32, etc) depending on version.
                # Safest way in recent Polars:
                # df.select(pl.col(c).hist(bin_count=15)).unnest(c)
                # If unnest fails, it means it's not a struct?
                
                # Let's try separate binning to be safe and version-agnostic.
                # We need min/max.
                
                min_v = df[col_name].min()
                max_v = df[col_name].max()
                
                if min_v is None or max_v is None or min_v == max_v:
                     continue

                # Use numpy for histogram if available as fallback, OR simple polars cut/group
                # Let's stick to Polars but use `hist` carefully.
                # If unnest failed, maybe it returned a Series named different?
                # Actually, `unnest(col_name)` expects the column `col_name` to be Struct.
                # df.select(pl.col(col).hist()) returns a DF with column `col` which IS the struct.
                # BUT if user has old polars, it might be different.
                
                # Let's try native numpy to avoid Polars version hell for this visual feature.
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

            elif action == "fill_median":
                if df[target_col].dtype in [pl.Int64, pl.Float64, pl.Float32, pl.Int32]:
                    median_val = df[target_col].median()
                    if median_val is not None:
                        df = df.with_columns(pl.col(target_col).fill_null(median_val))
                        report.numeric_nans_filled += 1
            
            elif action == "fill_mode":
                # Mode in Polars returns a Series
                mode_s = df[target_col].mode()
                if mode_s.len() > 0:
                    mode_val = mode_s[0]
                    if mode_val is not None:
                        df = df.with_columns(pl.col(target_col).fill_null(mode_val))
                        report.categorical_nans_filled += 1

            elif action == "fill_value":
                 val = rule.get("value")
                 if val is not None:
                    df = df.with_columns(pl.col(target_col).fill_null(val))

            elif action == "replace_outliers_median":
                # IQR Method
                if df[target_col].dtype in [pl.Int64, pl.Float64]:
                    q1 = df[target_col].quantile(0.25)
                    q3 = df[target_col].quantile(0.75)
                    if q1 is not None and q3 is not None:
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        median_val = df[target_col].median()
                        
                        # Replace outliers with median
                        df = df.with_columns(
                            pl.when((pl.col(target_col) < lower_bound) | (pl.col(target_col) > upper_bound))
                            .then(median_val)
                            .otherwise(pl.col(target_col))
                            .alias(target_col)
                        )

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

    # 3. Advanced Statistics (Skewness, Kurtosis)
    advanced_stats = {}
    for c in numeric_cols:
        skew = df[c].skew()
        kurt = df[c].kurtosis()
        advanced_stats[c] = {
            "skewness": round(skew, 2) if skew is not None else None,
            "kurtosis": round(kurt, 2) if kurt is not None else None
        }

    # 4. Multicollinearity Flags (VIF Proxy)
    # True VIF needs OLS. Here we flag columns with pairwise correlation > 0.9 or < -0.9
    multicollinearity = []
    seen_pairs = set()
    for c1, matrix in correlations.items():
        for c2, val in matrix.items():
            if c1 == c2: continue
            if abs(val) > 0.9:
                pair = tuple(sorted((c1, c2)))
                if pair not in seen_pairs:
                    multicollinearity.append({
                        "features": pair,
                        "correlation": val,
                        "severity": "high" if abs(val) > 0.95 else "medium"
                    })
                    seen_pairs.add(pair)

    return {
        "summary": summary,
        "correlations": correlations,
        "advanced_stats": advanced_stats,
        "multicollinearity": multicollinearity,
        "time_series_analysis": _analyze_time_series(df)
    }

def _analyze_time_series(df: pl.DataFrame) -> dict[str, Any] | None:
    """
    Rule #13: Check sort order, drift, and gaps.
    """
    # 1. Find Datetime Column
    time_cols = [c for c, t in df.schema.items() if t in (pl.Date, pl.Datetime)]
    if not time_cols:
        return None
    
    # Take the first one as primary for now
    time_col = time_cols[0]
    
    # 2. Check Sort Order
    is_sorted = df[time_col].is_sorted()
    
    # 3. Check for Drift (Concept Drift)
    # Split data chronologically (if sorted) or just index-based (assuming implicit time)?
    # Rule #13 says "Always sort". So we sort internally for the check.
    
    drift_flags = []
    
    if df.height > 50:
        # Sort for analysis
        df_sorted = df.sort(time_col)
        midpoint = df.height // 2
        
        part1 = df_sorted.slice(0, midpoint)
        part2 = df_sorted.slice(midpoint, df.height - midpoint)
        
        numeric_cols = [c for c, t in df.schema.items() if t in (pl.Int64, pl.Float64, pl.Int32, pl.Float32)]
        
        for col in numeric_cols:
            m1 = part1[col].mean()
            m2 = part2[col].mean()
            
            if m1 is not None and m2 is not None and m1 != 0:
                # Calculate % change
                pct_change = abs((m2 - m1) / m1)
                
                # Blunt Threshold: 30% shift in mean suggests drift/seasonality shift
                if pct_change > 0.30:
                    drift_flags.append({
                        "column": col,
                        "shift_pct": round(pct_change * 100, 1),
                        "mean_p1": round(m1, 2),
                        "mean_p2": round(m2, 2)
                    })

    return {
        "primary_time_col": time_col,
        "is_sorted": is_sorted,
        "drift_detected": drift_flags
    }