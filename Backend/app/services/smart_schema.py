"""
Smart Schema Inference Service

Provides intelligent schema analysis and correction suggestions:
1. Detect mistyped columns (numbers stored as strings, dates as numbers)
2. Suggest proper data types for each column
3. Detect relationships between columns (foreign keys, parent-child)
4. Identify schema inconsistencies and quality issues
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA TYPE DETECTION PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

DATE_PATTERNS = [
    r'^\d{4}-\d{2}-\d{2}$',                    # 2024-01-15
    r'^\d{2}/\d{2}/\d{4}$',                    # 01/15/2024
    r'^\d{2}-\d{2}-\d{4}$',                    # 15-01-2024
    r'^\d{4}/\d{2}/\d{2}$',                    # 2024/01/15
    r'^\d{1,2}\s+\w{3}\s+\d{4}$',              # 15 Jan 2024
    r'^\w{3}\s+\d{1,2},?\s+\d{4}$',            # Jan 15, 2024
]

DATETIME_PATTERNS = [
    r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',  # ISO datetime
    r'^\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}',         # MM/DD/YYYY HH:MM
]

NUMERIC_PATTERNS = [
    r'^-?\d+\.?\d*$',                          # Simple number
    r'^-?\d{1,3}(,\d{3})*(\.\d+)?$',           # Comma-separated
    r'^\$-?\d+\.?\d*$',                        # Currency with $
    r'^-?\d+\.?\d*%$',                         # Percentage
]

EMAIL_PATTERN = r'^[\w\.-]+@[\w\.-]+\.\w+$'
URL_PATTERN = r'^https?://[\w\.-]+(/[\w\.-]*)*$'
PHONE_PATTERN = r'^[\d\s\-\+\(\)]{7,20}$'


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA INFERENCE RESULT CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TypeCorrection:
    """Suggested data type correction for a column."""
    column: str
    current_type: str
    suggested_type: str
    confidence: float  # 0.0 to 1.0
    reason: str
    sample_values: list[str] = field(default_factory=list)
    conversion_code: str = ""


@dataclass
class RelationshipDetection:
    """Detected relationship between columns."""
    source_column: str
    target_column: str
    relationship_type: str  # foreign_key, parent_child, derived, correlated
    confidence: float
    details: str


@dataclass
class SchemaIssue:
    """Detected schema quality issue."""
    column: str
    issue_type: str  # mixed_types, inconsistent_format, encoding_issue
    severity: str  # high, medium, low
    description: str
    affected_rows: int
    suggestion: str


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE DETECTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_numeric_in_string(series: pl.Series) -> tuple[bool, float, str]:
    """Check if a string column contains mostly numeric values."""
    if series.dtype != pl.Utf8:
        return False, 0.0, ""
    
    sample = series.drop_nulls().head(1000)
    if sample.len() == 0:
        return False, 0.0, ""
    
    # Try to cast to numeric
    numeric_count = 0
    decimal_count = 0
    
    for val in sample.to_list():
        if val is None:
            continue
        clean = str(val).strip().replace(',', '').replace('$', '').replace('%', '')
        try:
            float(clean)
            numeric_count += 1
            if '.' in clean:
                decimal_count += 1
        except:
            pass
    
    confidence = numeric_count / sample.len() if sample.len() > 0 else 0
    
    if confidence >= 0.9:
        suggested = "float" if decimal_count > numeric_count * 0.1 else "integer"
        return True, confidence, suggested
    
    return False, confidence, ""


def _detect_date_in_string(series: pl.Series) -> tuple[bool, float, str]:
    """Check if a string column contains date values."""
    if series.dtype != pl.Utf8:
        return False, 0.0, ""
    
    sample = series.drop_nulls().head(100)
    if sample.len() == 0:
        return False, 0.0, ""
    
    date_count = 0
    datetime_count = 0
    
    for val in sample.to_list():
        if val is None:
            continue
        val_str = str(val).strip()
        
        for pattern in DATETIME_PATTERNS:
            if re.match(pattern, val_str):
                datetime_count += 1
                break
        else:
            for pattern in DATE_PATTERNS:
                if re.match(pattern, val_str):
                    date_count += 1
                    break
    
    total_matched = date_count + datetime_count
    confidence = total_matched / sample.len() if sample.len() > 0 else 0
    
    if confidence >= 0.8:
        suggested = "datetime" if datetime_count > date_count else "date"
        return True, confidence, suggested
    
    return False, confidence, ""


def _detect_date_in_numeric(series: pl.Series) -> tuple[bool, float]:
    """Check if a numeric column contains Excel serial date values."""
    if series.dtype not in (pl.Int64, pl.Float64, pl.Int32, pl.Float32):
        return False, 0.0
    
    sample = series.drop_nulls()
    if sample.len() == 0:
        return False, 0.0
    
    min_val = sample.min()
    max_val = sample.max()
    
    # Excel serial date range: ~25569 (1970-01-01) to ~47847 (2030-12-31)
    if 25569 <= min_val <= 73050 and 25569 <= max_val <= 73050:
        # Check if mostly integers
        int_ratio = (sample == sample.round(0)).sum() / sample.len()
        if int_ratio > 0.9:
            return True, 0.85
    
    # Unix timestamp range (seconds since 1970)
    if 0 <= min_val <= 2000000000 and 0 <= max_val <= 2000000000:
        # Check if values are large enough to be timestamps
        if min_val > 1000000000:  # After ~2001
            return True, 0.75
    
    return False, 0.0


def _detect_categorical_in_numeric(series: pl.Series) -> tuple[bool, float]:
    """Check if a numeric column is actually categorical (few unique values)."""
    if series.dtype not in (pl.Int64, pl.Float64, pl.Int32, pl.Float32):
        return False, 0.0
    
    n_unique = series.n_unique()
    total = series.len()
    
    # If very few unique values relative to total
    if n_unique <= 10 and total > 100:
        return True, 0.9
    if n_unique <= 20 and total > 1000:
        return True, 0.8
    
    return False, 0.0


def _detect_special_string_type(series: pl.Series) -> tuple[str | None, float]:
    """Detect if string column contains emails, URLs, phone numbers, etc."""
    if series.dtype != pl.Utf8:
        return None, 0.0
    
    sample = series.drop_nulls().head(100)
    if sample.len() == 0:
        return None, 0.0
    
    email_count = 0
    url_count = 0
    phone_count = 0
    
    for val in sample.to_list():
        if val is None:
            continue
        val_str = str(val).strip()
        
        if re.match(EMAIL_PATTERN, val_str, re.IGNORECASE):
            email_count += 1
        elif re.match(URL_PATTERN, val_str, re.IGNORECASE):
            url_count += 1
        elif re.match(PHONE_PATTERN, val_str):
            phone_count += 1
    
    total = sample.len()
    
    if email_count / total > 0.8:
        return "email", email_count / total
    if url_count / total > 0.8:
        return "url", url_count / total
    if phone_count / total > 0.8:
        return "phone", phone_count / total
    
    return None, 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# RELATIONSHIP DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_foreign_key_relationships(df: pl.DataFrame) -> list[RelationshipDetection]:
    """Detect potential foreign key relationships between columns."""
    relationships = []
    
    # Get columns by type
    int_cols = [c for c, t in df.schema.items() if t in (pl.Int64, pl.Int32)]
    
    for i, col1 in enumerate(int_cols):
        for col2 in int_cols[i+1:]:
            # Check if one column's values are a subset of another
            unique1 = df[col1].unique()
            unique2 = df[col2].unique()
            
            # Skip if both have same number of unique values (likely not FK)
            if unique1.len() == unique2.len():
                continue
            
            # Check if smaller set is subset of larger
            if unique1.len() < unique2.len():
                potential_fk, potential_pk = col1, col2
                fk_vals, pk_vals = set(unique1.to_list()), set(unique2.to_list())
            else:
                potential_fk, potential_pk = col2, col1
                fk_vals, pk_vals = set(unique2.to_list()), set(unique1.to_list())
            
            # Remove nulls
            fk_vals.discard(None)
            pk_vals.discard(None)
            
            # Check containment
            if fk_vals and fk_vals.issubset(pk_vals):
                containment = len(fk_vals) / len(pk_vals) if pk_vals else 0
                if containment < 0.9:  # FK should have fewer values
                    relationships.append(RelationshipDetection(
                        source_column=potential_fk,
                        target_column=potential_pk,
                        relationship_type="foreign_key",
                        confidence=0.7,
                        details=f"{potential_fk} values are subset of {potential_pk}"
                    ))
    
    return relationships[:5]  # Limit to top 5


def _detect_derived_columns(df: pl.DataFrame) -> list[RelationshipDetection]:
    """Detect columns that appear to be derived from others."""
    relationships = []
    numeric_cols = [c for c, t in df.schema.items() if t in (pl.Int64, pl.Float64)]
    
    if len(numeric_cols) < 2:
        return []
    
    # Check for sum/product relationships
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols[i+1:], i+1):
            for k, col3 in enumerate(numeric_cols[j+1:], j+1):
                try:
                    # Check if col3 = col1 + col2
                    diff = (df[col3] - (df[col1] + df[col2])).abs()
                    if diff.mean() < 0.01:
                        relationships.append(RelationshipDetection(
                            source_column=col3,
                            target_column=f"{col1} + {col2}",
                            relationship_type="derived",
                            confidence=0.85,
                            details=f"{col3} appears to be sum of {col1} and {col2}"
                        ))
                    
                    # Check if col3 = col1 * col2
                    product_diff = (df[col3] - (df[col1] * df[col2])).abs()
                    if product_diff.mean() < 0.01:
                        relationships.append(RelationshipDetection(
                            source_column=col3,
                            target_column=f"{col1} * {col2}",
                            relationship_type="derived",
                            confidence=0.85,
                            details=f"{col3} appears to be product of {col1} and {col2}"
                        ))
                except:
                    pass
    
    return relationships[:5]


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA ISSUE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_schema_issues(df: pl.DataFrame) -> list[SchemaIssue]:
    """Detect schema quality issues like mixed types, encoding problems."""
    issues = []
    
    for col in df.columns:
        series = df[col]
        
        # Check for mixed content in string columns
        if series.dtype == pl.Utf8:
            sample = series.drop_nulls().head(1000)
            if sample.len() > 0:
                # Check for leading/trailing whitespace
                trimmed = sample.str.strip_chars()
                whitespace_count = (sample != trimmed).sum()
                if whitespace_count > sample.len() * 0.1:
                    issues.append(SchemaIssue(
                        column=col,
                        issue_type="whitespace",
                        severity="low",
                        description=f"{whitespace_count} values have leading/trailing whitespace",
                        affected_rows=whitespace_count,
                        suggestion=f"Apply: df['{col}'].str.strip()"
                    ))
                
                # Check for inconsistent casing
                lower = sample.str.to_lowercase()
                upper = sample.str.to_uppercase()
                has_mixed = (sample != lower).sum() > 0 and (sample != upper).sum() > 0
                unique_lower = lower.n_unique()
                unique_orig = sample.n_unique()
                
                if has_mixed and unique_lower < unique_orig * 0.9:
                    issues.append(SchemaIssue(
                        column=col,
                        issue_type="inconsistent_casing",
                        severity="medium",
                        description=f"Inconsistent casing may cause {unique_orig - unique_lower} duplicate categories",
                        affected_rows=int(unique_orig - unique_lower),
                        suggestion=f"Standardize with: df['{col}'].str.to_lowercase()"
                    ))
        
        # Check for constant columns
        n_unique = series.n_unique()
        if n_unique == 1:
            issues.append(SchemaIssue(
                column=col,
                issue_type="constant_column",
                severity="medium",
                description="Column has only one unique value",
                affected_rows=series.len(),
                suggestion=f"Consider dropping: df.drop('{col}')"
            ))
        
        # Check for high null percentage
        null_pct = series.null_count() / series.len() * 100
        if null_pct > 50:
            issues.append(SchemaIssue(
                column=col,
                issue_type="high_nulls",
                severity="high" if null_pct > 80 else "medium",
                description=f"{null_pct:.1f}% of values are null",
                affected_rows=series.null_count(),
                suggestion="Consider imputation or dropping column"
            ))
    
    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SmartSchemaResult:
    """Complete schema analysis result."""
    type_corrections: list[TypeCorrection]
    relationships: list[RelationshipDetection]
    schema_issues: list[SchemaIssue]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type_corrections": [
                {
                    "column": t.column,
                    "current_type": t.current_type,
                    "suggested_type": t.suggested_type,
                    "confidence": round(t.confidence * 100),
                    "reason": t.reason,
                    "samples": t.sample_values[:3]
                }
                for t in self.type_corrections
            ],
            "relationships": [
                {
                    "source": r.source_column,
                    "target": r.target_column,
                    "type": r.relationship_type,
                    "confidence": round(r.confidence * 100),
                    "details": r.details
                }
                for r in self.relationships
            ],
            "schema_issues": [
                {
                    "column": s.column,
                    "issue": s.issue_type,
                    "severity": s.severity,
                    "description": s.description,
                    "suggestion": s.suggestion
                }
                for s in self.schema_issues
            ]
        }


def analyze_smart_schema(df: pl.DataFrame) -> SmartSchemaResult:
    """
    Perform smart schema inference on a DataFrame.
    
    Returns:
        SmartSchemaResult with type corrections, relationships, and issues
    """
    logger.info("Starting smart schema analysis...")
    
    type_corrections = []
    
    for col in df.columns:
        series = df[col]
        current_type = str(series.dtype)
        
        # Check string columns for hidden types
        if series.dtype == pl.Utf8:
            # Check for numeric
            is_numeric, conf, suggested = _detect_numeric_in_string(series)
            if is_numeric:
                samples = series.drop_nulls().head(3).to_list()
                type_corrections.append(TypeCorrection(
                    column=col,
                    current_type=current_type,
                    suggested_type=suggested,
                    confidence=conf,
                    reason=f"String column contains {conf*100:.0f}% numeric values",
                    sample_values=[str(s) for s in samples],
                    conversion_code=f"df['{col}'].cast(pl.Float64)" if suggested == "float" else f"df['{col}'].cast(pl.Int64)"
                ))
                continue
            
            # Check for dates
            is_date, conf, suggested = _detect_date_in_string(series)
            if is_date:
                samples = series.drop_nulls().head(3).to_list()
                type_corrections.append(TypeCorrection(
                    column=col,
                    current_type=current_type,
                    suggested_type=suggested,
                    confidence=conf,
                    reason=f"String column contains {conf*100:.0f}% date values",
                    sample_values=[str(s) for s in samples],
                    conversion_code=f"df['{col}'].str.to_date()"
                ))
                continue
            
            # Check for special types
            special_type, conf = _detect_special_string_type(series)
            if special_type:
                samples = series.drop_nulls().head(3).to_list()
                type_corrections.append(TypeCorrection(
                    column=col,
                    current_type=current_type,
                    suggested_type=special_type,
                    confidence=conf,
                    reason=f"Column contains {special_type} values",
                    sample_values=[str(s) for s in samples],
                    conversion_code=""  # Keep as string but mark semantic type
                ))
        
        # Check numeric columns for hidden types
        elif series.dtype in (pl.Int64, pl.Float64, pl.Int32, pl.Float32):
            # Check for dates
            is_date, conf = _detect_date_in_numeric(series)
            if is_date:
                samples = series.drop_nulls().head(3).to_list()
                type_corrections.append(TypeCorrection(
                    column=col,
                    current_type=current_type,
                    suggested_type="date",
                    confidence=conf,
                    reason="Numeric values appear to be Excel serial dates",
                    sample_values=[str(int(s)) for s in samples],
                    conversion_code=f"pl.from_epoch(df['{col}'], unit='d')"
                ))
                continue
            
            # Check for categorical
            is_cat, conf = _detect_categorical_in_numeric(series)
            if is_cat:
                samples = series.drop_nulls().unique().head(5).to_list()
                type_corrections.append(TypeCorrection(
                    column=col,
                    current_type=current_type,
                    suggested_type="categorical",
                    confidence=conf,
                    reason=f"Only {series.n_unique()} unique values - likely categorical",
                    sample_values=[str(int(s)) if s == int(s) else str(s) for s in samples],
                    conversion_code=f"df['{col}'].cast(pl.Categorical)"
                ))
    
    # Detect relationships
    relationships = []
    relationships.extend(_detect_foreign_key_relationships(df))
    relationships.extend(_detect_derived_columns(df))
    
    # Detect schema issues
    schema_issues = _detect_schema_issues(df)
    
    logger.info(f"Schema analysis complete: {len(type_corrections)} corrections, "
                f"{len(relationships)} relationships, {len(schema_issues)} issues")
    
    return SmartSchemaResult(
        type_corrections=type_corrections,
        relationships=relationships,
        schema_issues=schema_issues
    )
