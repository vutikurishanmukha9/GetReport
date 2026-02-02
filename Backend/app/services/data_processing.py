from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from fastapi import UploadFile, HTTPException
from io import BytesIO

# ─── Logger ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS: set[str]  = {".csv", ".xls", ".xlsx"}
MAX_FILE_SIZE_BYTES: int      = 50 * 1024 * 1024          # 50 MB hard cap
PREVIEW_ROW_COUNT: int        = 10                         # rows returned in preview


# ─── Custom Exceptions ───────────────────────────────────────────────────────
class UnsupportedFileTypeError(Exception):
    """Raised when the uploaded file extension is not in the allowed set."""


class FileTooLargeError(Exception):
    """Raised when the uploaded file exceeds MAX_FILE_SIZE_BYTES."""


class EmptyFileError(Exception):
    """Raised when the file contains no data or only a header row."""


class ParseError(Exception):
    """Raised when pandas cannot parse the file contents."""


class InvalidDataFrameError(TypeError):
    """Raised when a function receives something that is not a DataFrame."""


# ─── Cleaning Report ─────────────────────────────────────────────────────────
@dataclass
class CleaningReport:
    """
    Tracks every change the cleaning pipeline makes.
    Returned alongside the cleaned DataFrame so the PDF report
    can tell the user exactly what happened to their data.

    Attributes:
        empty_rows_dropped:       How many fully-empty rows were removed.
        empty_columns_dropped:    How many fully-empty columns were removed.
        duplicate_rows_removed:   How many duplicate rows were removed.
        columns_renamed:          Mapping of original name → new snake_case name.
        type_conversions:         List of {column, original_type, new_type}.
        numeric_nans_filled:      Total numeric NaNs filled with 0.
        categorical_nans_filled:  Total non-numeric NaNs filled with "Unknown".
        total_changes:            Sum of all changes made.
        timing_ms:                How long cleaning took (milliseconds).
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
        """Compute total_changes from all individual counts."""
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
        """Serialize to a plain dictionary (JSON-ready)."""
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
    """
    Convert a column name to clean snake_case.

    Steps:
        1. Strip leading/trailing whitespace          (original logic)
        2. Insert underscore before uppercase runs     (CamelCase → camel_case)
        3. Replace any non-alphanumeric char with _
        4. Collapse multiple underscores into one
        5. Strip leading/trailing underscores
        6. Lowercase everything

    Examples:
        "First Name"      → "first_name"
        "totalRevenue$"   → "total_revenue"
        "dateOfBirth"     → "date_of_birth"
        " User ID "       → "user_id"
    """
    name = name.strip()                                          # original logic
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)         # CamelCase split
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)                  # non-alphanumeric → _
    name = re.sub(r"_+", "_", name)                              # collapse multiples
    name = name.strip("_").lower()                               # final cleanup
    return name


# ─── File Validation ─────────────────────────────────────────────────────────
def _validate_upload(file: UploadFile, content_length: int) -> str:
    """
    Run all pre-flight checks on the uploaded file before reading it.

    Checks:
        1. Filename exists and is not empty
        2. Extension is in the allowed whitelist
        3. File size does not exceed the hard cap

    Args:
        file:           The FastAPI UploadFile object.
        content_length: The size of the file in bytes (from file.size or headers).

    Returns:
        The validated file extension (e.g. ".csv", ".xlsx").

    Raises:
        HTTPException (400): Wraps the specific custom exception with a clear message.
    """
    # ── Filename check ──────────────────────────────────────────────────────
    if not file.filename or file.filename.strip() == "":
        logger.warning("Upload rejected — filename is missing or empty.")
        raise HTTPException(status_code=400, detail="No filename provided.")

    # ── Extension check ─────────────────────────────────────────────────────
    # Extract extension safely (handles files with no extension)
    dot_index = file.filename.rfind(".")
    extension = file.filename[dot_index:].lower() if dot_index != -1 else ""

    if extension not in ALLOWED_EXTENSIONS:
        logger.warning(
            "Upload rejected — unsupported extension '%s' for file '%s'.",
            extension, file.filename
        )
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{extension}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # ── Size check ──────────────────────────────────────────────────────────
    size_mb = content_length / (1024 * 1024)
    if content_length > MAX_FILE_SIZE_BYTES:
        logger.warning(
            "Upload rejected — file '%s' is %.1f MB (limit: %d MB).",
            file.filename, size_mb, MAX_FILE_SIZE_BYTES // (1024 * 1024)
        )
        raise HTTPException(
            status_code=400,
            detail=f"File is {size_mb:.1f} MB. Maximum allowed size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
        )

    logger.info("Upload validated — '%s' (%.2f MB, extension: %s).", file.filename, size_mb, extension)
    return extension


# ─── File Loader ─────────────────────────────────────────────────────────────
async def load_dataframe(file: UploadFile) -> pd.DataFrame:
    """
    Load a CSV or Excel file into a pandas DataFrame.

    Preserves original logic:
        - await file.read() → BytesIO buffer
        - Extension-based routing: .csv → read_csv, .xls/.xlsx → read_excel
        - HTTPException on unsupported format or parse failure

    Enhanced with:
        - Pre-read validation (extension whitelist, file size cap)
        - Post-parse validation (empty file, header-only file)
        - Structured logging at every stage
        - Specific error messages tied to the actual failure

    Args:
        file: The FastAPI UploadFile from the upload endpoint.

    Returns:
        A pandas DataFrame containing the parsed file data.

    Raises:
        HTTPException (400): For any validation or parsing failure.
    """
    logger.info("═══ load_dataframe started — '%s' ═══", file.filename)

    # ── 1. Validate before reading ──────────────────────────────────────────
    file_size   = file.size if file.size else 0
    extension   = _validate_upload(file, file_size)

    # ── 2. Read file into memory (original logic) ──────────────────────────
    contents = await file.read()
    buffer   = BytesIO(contents)
    logger.debug("File read into memory — %d bytes.", len(contents))

    # ── 3. Parse based on extension (original logic preserved) ──────────────
    try:
        if extension == ".csv":
            df = pd.read_csv(buffer)
            logger.info("Parsed as CSV successfully.")
        else:
            # .xls or .xlsx (original logic)
            df = pd.read_excel(buffer)
            logger.info("Parsed as Excel successfully.")
    except Exception as e:
        logger.error("Parse failed for '%s': %s", file.filename, str(e))
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse file: {str(e)}"
        )

    # ── 4. Post-parse validation ────────────────────────────────────────────
    if df.empty and len(df.columns) == 0:
        logger.warning("File '%s' is completely empty — no rows or columns.", file.filename)
        raise HTTPException(status_code=400, detail="The file is completely empty.")

    if len(df) == 0:
        logger.warning("File '%s' contains only a header row — no data rows.", file.filename)
        raise HTTPException(status_code=400, detail="The file contains only headers — no data rows found.")

    logger.info(
        "═══ load_dataframe complete — %d rows × %d columns ═══",
        len(df), len(df.columns)
    )
    return df


# ─── INSPECTION & INTERACTIVE CLEANING ──────────────────────────────────────

def inspect_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyzes the raw dataset to find quality issues.
    Returns a report suitable for the "Health Check" UI.
    """
    quality_report = {
        "total_rows": len(df),
        "columns": [],
        "issues": []
    }
    
    for col in df.columns:
        # 1. NaN Detection
        nan_count = int(df[col].isnull().sum())
        dtype = str(df[col].dtype)
        
        # 2. Type Inference (Simpler version for UI)
        inferred_type = "string"
        if pd.api.types.is_numeric_dtype(df[col]):
            inferred_type = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            inferred_type = "datetime"
            
        col_info = {
            "name": col,
            "dtype": dtype,
            "inferred_type": inferred_type,
            "missing_count": nan_count,
            "missing_percentage": round((nan_count / len(df)) * 100, 1)
        }
        quality_report["columns"].append(col_info)
        
        # 3. Issue Flagging
        if nan_count > 0:
            quality_report["issues"].append({
                "type": "missing_values",
                "column": col,
                "count": nan_count,
                "severity": "high" if nan_count > 0 else "none",
                "suggestion": "fill_mean" if inferred_type == "numeric" else "fill_unknown"
            })
            
    return quality_report


# ─── Cleaning Pipeline ───────────────────────────────────────────────────────
def clean_data(
    df: pd.DataFrame, 
    rules: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Run the data cleaning pipeline.
    If 'rules' is provided, applies specific user actions (e.g. drop vs fill).
    Otherwise, applies safe defaults.
    """
    # ── Input validation ────────────────────────────────────────────────────
    if not isinstance(df, pd.DataFrame):
        raise InvalidDataFrameError(
            f"clean_data expects a pandas DataFrame, got {type(df).__name__}."
        )

    start_time = time.perf_counter()
    report     = CleaningReport()
    logger.info("═══ Cleaning Pipeline Started (Rules: %s) ═══", rules)

    # ── Snapshot original state ─────────────────────────────────────────────
    original_rows = len(df)
    original_cols = len(df.columns)
    
    # Standardize column names FIRST to handle rules mapping easily
    # (Original: it was Step 2, but moving it to 0 helps matching rules)
    df.columns = [_to_snake_case(c) for c in df.columns]

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 0 — User Defined Rules (Interactive Mode)
    # ═════════════════════════════════════════════════════════════════════════
    if rules:
        for original_col, rule in rules.items():
            target_col = _to_snake_case(original_col)
            if target_col not in df.columns:
                continue
                
            action = rule.get("action")
            
            if action == "drop_rows":
                initial = len(df)
                df.dropna(subset=[target_col], inplace=True)
                dropped = initial - len(df)
                if dropped > 0:
                    logger.info(f"Rule: Dropped {dropped} rows for '{target_col}'")
            
            elif action == "fill_mean":
                if pd.api.types.is_numeric_dtype(df[target_col]):
                    mean_val = df[target_col].mean()
                    df[target_col].fillna(mean_val, inplace=True)
                    report.numeric_nans_filled += 1 # Rough count
                    
            elif action == "fill_value":
                val = rule.get("value")
                df[target_col].fillna(val, inplace=True)

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 1 — Drop fully empty rows and columns
    # ═════════════════════════════════════════════════════════════════════════
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    report.empty_rows_dropped    = original_rows - len(df)
    report.empty_columns_dropped = original_cols - len(df.columns)

    if report.empty_rows_dropped > 0:
        logger.info("Step 1 — Dropped %d fully empty row(s).", report.empty_rows_dropped)

    # ── Enhanced: Duplicate row removal ─────────────────────────────────────
    duplicates_before = df.duplicated().sum()
    if duplicates_before > 0:
        df.drop_duplicates(inplace=True)
        report.duplicate_rows_removed = int(duplicates_before)
        logger.info("Step 1 (Enhanced) — Removed %d duplicate row(s).", duplicates_before)

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2 — Standardize column names (Already done at Step 0, just track)
    # ═════════════════════════════════════════════════════════════════════════
    # We did it at start, so just rely on that.

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 3 — Type inference with ID detection
    # ═════════════════════════════════════════════════════════════════════════
    logger.debug("Step 3 — Starting smart type inference.")
    id_patterns = {"id", "code", "sku", "zip", "phone", "year", "date", "day"}
    
    for col in df.columns:
        original_dtype = str(df[col].dtype)
        col_lower = col.lower()
        
        # Check if column likely represents an ID/Label rather than a measurement
        is_id_like = any(p in col_lower for p in id_patterns) or col_lower.endswith("_id")
        
        # If it looks like an ID, prefer string (object) unless it's clearly a date
        if is_id_like:
            date_terms = {"date", "time", "year", "day", "dob", "created", "updated", "at"}
            if any(t in col_lower for t in date_terms):
                try:
                    df[col] = pd.to_datetime(df[col], format="mixed")
                    new_dtype = str(df[col].dtype)
                    if new_dtype != original_dtype:
                        report.type_conversions.append({
                            "column": col, "original_type": original_dtype, "new_type": new_dtype
                        })
                    continue
                except (ValueError, TypeError):
                    pass 

            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(str)
                report.type_conversions.append({
                    "column": col, "original_type": original_dtype, "new_type": "object (ID)"
                })
            continue

        try:
            df[col] = pd.to_numeric(df[col])
            new_dtype = str(df[col].dtype)
            if new_dtype != original_dtype:
                report.type_conversions.append({
                    "column": col, "original_type": original_dtype, "new_type": new_dtype
                })
            continue
        except (ValueError, TypeError):
            pass

        try:
            df[col] = pd.to_datetime(df[col], format="mixed")
            new_dtype = str(df[col].dtype)
            if new_dtype != original_dtype:
                report.type_conversions.append({
                    "column": col, "original_type": original_dtype, "new_type": new_dtype
                })
            continue
        except (ValueError, TypeError):
            pass

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 4 — Handling Missing Values (CORRECTED)
    # ═════════════════════════════════════════════════════════════════════════
    # Safe Imputation: Only fill Categorical. Leave Numeric NaNs unless rule applied.
    
    for col in df.columns:
        nan_count = int(df[col].isnull().sum())
        if nan_count > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Log it but DO NOT FILL with 0
                logger.info("Step 4 — '%s': has %d missing values (left as NaN for accurate stats).", col, nan_count)
            else:
                # Fill categorical missing values to prevent errors in grouping/charts
                df[col] = df[col].fillna("Unknown")
                report.categorical_nans_filled += nan_count
                logger.info("Step 4 — '%s': filled %d NaN(s) with 'Unknown'.", col, nan_count)

    # ── Finalize report ─────────────────────────────────────────────────────
    report.timing_ms = (time.perf_counter() - start_time) * 1000
    report.finalize()

    logger.info(
        "═══ Cleaning Pipeline Complete — %d total change(s) in %.2f ms ═══",
        report.total_changes, report.timing_ms
    )
    return df, report


# ─── Dataset Info ────────────────────────────────────────────────────────────
def get_dataset_info(df: pd.DataFrame) -> dict[str, Any]:
    """
    Build a rich metadata dictionary about the dataset for the frontend
    and for inclusion in the PDF report.

    Preserves original logic:
        - rows count
        - columns list
        - dtypes mapping
        - summary via df.describe(include='all')
        - preview via df.head(10)

    Enhanced with:
        - Input validation
        - Missing value counts and percentages per column
        - Duplicate row count
        - Numeric vs categorical column breakdown
        - Memory usage of the DataFrame
        - Preview NaN → None replacement is now explicit per-cell

    Args:
        df: The cleaned pandas DataFrame.

    Returns:
        A plain dictionary with full dataset metadata (JSON-serialisable).

    Raises:
        InvalidDataFrameError: If input is not a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise InvalidDataFrameError(
            f"get_dataset_info expects a pandas DataFrame, got {type(df).__name__}."
        )

    logger.info("Building dataset info — %d rows × %d columns.", len(df), len(df.columns))

    # ── Original fields (logic preserved) ──────────────────────────────────
    rows    = len(df)
    columns = list(df.columns)
    dtypes  = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Original: describe(include='all') with NaN → empty string
    summary = df.describe(include="all").fillna("").to_dict()

    # Original: head(10) with NaN → None
    preview = df.head(PREVIEW_ROW_COUNT).replace({np.nan: None}).to_dict(orient="records")

    # ── Enhanced fields ─────────────────────────────────────────────────────
    # Missing values per column
    missing_per_column: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        count = int(df[col].isnull().sum())
        missing_per_column[col] = {
            "count":      count,
            "percentage": round((count / rows) * 100, 2) if rows > 0 else 0.0,
        }

    # Column type breakdown
    numeric_columns     = list(df.select_dtypes(include=[np.number]).columns)
    categorical_columns = list(df.select_dtypes(exclude=[np.number]).columns)

    # Duplicate rows
    duplicate_row_count = int(df.duplicated().sum())

    # Memory usage
    memory_bytes = int(df.memory_usage(deep=True).sum())
    memory_mb    = round(memory_bytes / (1024 * 1024), 2)

    # ── Assemble final output ───────────────────────────────────────────────
    info: dict[str, Any] = {
        # Original fields
        "rows":     rows,
        "columns":  columns,
        "dtypes":   dtypes,
        "summary":  summary,
        "preview":  preview,
        # Enhanced fields
        "missing_values":        missing_per_column,
        "duplicate_rows":        duplicate_row_count,
        "numeric_columns":       numeric_columns,
        "categorical_columns":   categorical_columns,
        "memory_usage_mb":       memory_mb,
    }

    logger.info(
        "Dataset info built — %d numeric, %d categorical, %d duplicates, %.2f MB.",
        len(numeric_columns), len(categorical_columns), duplicate_row_count, memory_mb
    )
    return info