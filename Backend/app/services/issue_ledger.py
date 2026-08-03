"""
Issue Ledger System

"Jira for Dirty Data" - Tracks all detected data quality issues and allows
users to approve, reject, or modify suggested fixes before cleaning.

Features:
1. Issue detection from multiple sources (confidence, schema, outliers)
2. Approve/reject/modify workflow
3. Lock mechanism before execution
4. Audit trail of decisions
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import polars as pl

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ISSUE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

IssueType = Literal[
    "missing_values",
    "duplicates", 
    "type_mismatch",
    "outliers",
    "format_issue",
    "high_cardinality",
    "empty_column",
    "constant_column",
    "encoding_issue",
]

Severity = Literal["critical", "high", "medium", "low"]
IssueStatus = Literal["pending", "approved", "rejected", "modified"]


@dataclass
class Issue:
    """A single data quality issue with suggested fix."""
    id: str                           # Unique issue ID
    issue_type: IssueType             # Category of issue
    severity: Severity                # Impact level
    column: str | None                # Affected column (None for row-level)
    affected_rows: int                # Number of rows affected
    affected_pct: float               # Percentage of data affected
    description: str                  # Human-readable description
    suggested_fix: str                # What we propose to do
    fix_code: str                     # Polars code to execute
    status: IssueStatus = "pending"   # Current approval status
    user_note: str = ""               # Optional user comment
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "column": self.column,
            "affected_rows": self.affected_rows,
            "affected_pct": round(self.affected_pct, 2),
            "description": self.description,
            "suggested_fix": self.suggested_fix,
            "fix_code": self.fix_code,
            "status": self.status,
            "user_note": self.user_note,
        }


@dataclass
class IssueLedger:
    """Complete issue ledger for a dataset."""
    issues: list[Issue] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    locked: bool = False
    locked_at: datetime | None = None
    
    def add_issue(self, issue: Issue) -> None:
        """Add an issue to the ledger."""
        if self.locked:
            raise ValueError("Cannot add issues to a locked ledger")
        self.issues.append(issue)
    
    def approve(self, issue_id: str) -> bool:
        """Approve an issue for execution."""
        if self.locked:
            raise ValueError("Cannot modify a locked ledger")
        for issue in self.issues:
            if issue.id == issue_id:
                issue.status = "approved"
                return True
        return False
    
    def reject(self, issue_id: str, note: str = "") -> bool:
        """Reject an issue - fix will not be applied."""
        if self.locked:
            raise ValueError("Cannot modify a locked ledger")
        for issue in self.issues:
            if issue.id == issue_id:
                issue.status = "rejected"
                issue.user_note = note
                return True
        return False
    
    def modify(self, issue_id: str, new_fix_code: str, note: str = "") -> bool:
        """Modify the suggested fix code."""
        if self.locked:
            raise ValueError("Cannot modify a locked ledger")
        for issue in self.issues:
            if issue.id == issue_id:
                issue.status = "modified"
                issue.fix_code = new_fix_code
                issue.user_note = note
                return True
        return False
    
    def approve_all(self) -> int:
        """Approve all pending issues."""
        if self.locked:
            raise ValueError("Cannot modify a locked ledger")
        count = 0
        for issue in self.issues:
            if issue.status == "pending":
                issue.status = "approved"
                count += 1
        return count
    
    def reject_all(self) -> int:
        """Reject all pending issues."""
        if self.locked:
            raise ValueError("Cannot modify a locked ledger")
        count = 0
        for issue in self.issues:
            if issue.status == "pending":
                issue.status = "rejected"
                count += 1
        return count
    
    def lock(self) -> None:
        """Lock the ledger - no more changes allowed."""
        self.locked = True
        self.locked_at = datetime.now()
    
    def get_approved_issues(self) -> list[Issue]:
        """Get all approved or modified issues for execution."""
        return [i for i in self.issues if i.status in ("approved", "modified")]
    
    def get_summary(self) -> dict[str, int]:
        """Get count summary by status."""
        summary = {"pending": 0, "approved": 0, "rejected": 0, "modified": 0, "total": 0}
        for issue in self.issues:
            summary[issue.status] += 1
            summary["total"] += 1
        return summary
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "issues": [i.to_dict() for i in self.issues],
            "summary": self.get_summary(),
            "locked": self.locked,
            "locked_at": self.locked_at.isoformat() if self.locked_at else None,
            "created_at": self.created_at.isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ISSUE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_id() -> str:
    """Generate a short unique ID."""
    return str(uuid.uuid4())[:8]


def _detect_missing_value_issues(df: pl.DataFrame) -> list[Issue]:
    """Detect columns with significant missing values.
    
    Enhancements over baseline:
    - Threshold lowered from 5% to 1% so low-level gaps are surfaced.
    - Four severity tiers: critical (≥30%), high (≥15%), medium (≥5%), low (≥1%).
    - Head/tail null concentration analysis for time-series bias detection.
    - Numeric columns get median fill; string columns get mode fill.
    """
    issues = []
    n_rows = df.height
    if n_rows == 0:
        return issues

    for col in df.columns:
        null_count = df[col].null_count()
        null_pct = (null_count / n_rows * 100)

        if null_pct >= 30:
            severity: Severity = "critical"
        elif null_pct >= 15:
            severity = "high"
        elif null_pct >= 5:
            severity = "medium"
        elif null_pct >= 1:
            severity = "low"
        else:
            continue  # < 1% — negligible

        # ── Null concentration pattern (head / tail / scattered) ──
        pattern = "scattered"
        try:
            sample_size = min(n_rows, 100)
            head_nulls = df[col].head(sample_size).null_count()
            tail_nulls = df[col].tail(sample_size).null_count()
            head_ratio = head_nulls / sample_size if sample_size > 0 else 0
            tail_ratio = tail_nulls / sample_size if sample_size > 0 else 0
            if tail_ratio > 0.5 and tail_ratio > head_ratio * 2:
                pattern = "concentrated at tail (recent records may be incomplete)"
            elif head_ratio > 0.5 and head_ratio > tail_ratio * 2:
                pattern = "concentrated at head (early records may be incomplete)"
        except Exception:
            pass

        # ── Fix suggestion based on dtype ──
        dtype = df[col].dtype
        if dtype in (pl.Int64, pl.Float64, pl.Int32, pl.Float32, pl.Int16, pl.Int8, pl.UInt32, pl.UInt64):
            fix_code = f"df = df.with_columns(pl.col('{col}').fill_null(pl.col('{col}').median()))"
            suggested_fix = "Fill with median value"
        else:
            fix_code = f"df = df.with_columns(pl.col('{col}').fill_null(pl.col('{col}').mode().first()))"
            suggested_fix = "Fill with most common value"

        desc = f"{null_pct:.1f}% missing values ({null_count:,} rows) — {pattern}"

        issues.append(Issue(
            id=_generate_id(),
            issue_type="missing_values",
            severity=severity,
            column=col,
            affected_rows=null_count,
            affected_pct=null_pct,
            description=desc,
            suggested_fix=suggested_fix,
            fix_code=fix_code,
        ))

    return issues


def _detect_duplicate_issues(df: pl.DataFrame) -> list[Issue]:
    """Detect duplicate rows with enhanced diagnostics.
    
    Enhancements:
    - Critical severity tier for ≥40% duplicates.
    - Near-duplicate column-subset scan on top 5 lowest-cardinality columns.
    - Reports which columns contribute most to duplication.
    """
    issues = []
    n_rows = df.height
    if n_rows == 0:
        return issues

    # ── Exact duplicate detection ──
    n_unique = df.n_unique()
    dup_count = n_rows - n_unique

    if dup_count > 0:
        dup_pct = (dup_count / n_rows * 100)

        if dup_pct >= 40:
            severity: Severity = "critical"
        elif dup_pct >= 20:
            severity = "high"
        elif dup_pct >= 5:
            severity = "medium"
        else:
            severity = "low"

        # ── Identify columns contributing most to duplication ──
        contributing_cols: list[str] = []
        try:
            for col in df.columns[:10]:  # Check first 10 columns
                col_unique = df[col].n_unique()
                col_ratio = col_unique / n_rows if n_rows > 0 else 1
                if col_ratio < 0.5:  # Low cardinality = likely contributor
                    contributing_cols.append(col)
        except Exception:
            pass

        desc = f"{dup_count:,} exact duplicate rows ({dup_pct:.1f}%)"
        if contributing_cols:
            desc += f" — low-cardinality columns: {', '.join(contributing_cols[:5])}"

        issues.append(Issue(
            id=_generate_id(),
            issue_type="duplicates",
            severity=severity,
            column=None,
            affected_rows=dup_count,
            affected_pct=dup_pct,
            description=desc,
            suggested_fix="Remove duplicate rows",
            fix_code="df = df.unique()",
        ))

    # ── Near-duplicate detection (column-subset scan) ──
    try:
        # Pick top 5 lowest-cardinality non-empty columns for subset check
        col_cardinalities = []
        for col in df.columns:
            if df[col].null_count() < n_rows:
                col_cardinalities.append((col, df[col].n_unique()))
        col_cardinalities.sort(key=lambda x: x[1])
        subset_cols = [c[0] for c in col_cardinalities[:5]]

        if len(subset_cols) >= 2 and len(subset_cols) < len(df.columns):
            subset_unique = df.select(subset_cols).n_unique()
            near_dup_count = n_rows - subset_unique
            near_dup_pct = (near_dup_count / n_rows * 100) if n_rows > 0 else 0

            # Only report if near-dupes exceed exact dupes by a meaningful margin
            if near_dup_count > dup_count and near_dup_pct >= 3:
                extra = near_dup_count - dup_count
                issues.append(Issue(
                    id=_generate_id(),
                    issue_type="duplicates",
                    severity="medium",
                    column=None,
                    affected_rows=extra,
                    affected_pct=(extra / n_rows * 100) if n_rows > 0 else 0,
                    description=f"{extra:,} near-duplicate rows detected on columns [{', '.join(subset_cols)}]",
                    suggested_fix="Review rows that differ only in minor fields",
                    fix_code=f"# Near-duplicates on subset: df.unique(subset={subset_cols})",
                ))
    except Exception:
        pass

    # ── Primary Key Collision Detection ──
    try:
        id_cols = [c for c in df.columns if c.lower() in ("id", "code", "key") or c.lower().endswith("_id") or c.lower().endswith("_code")]
        for key_col in id_cols:
            if df[key_col].null_count() < n_rows:
                key_dupes = df[key_col].is_duplicated().sum()
                if key_dupes > dup_count and key_dupes > 0:
                    conflicts = key_dupes - dup_count
                    pct = (conflicts / n_rows * 100)
                    issues.append(Issue(
                        id=_generate_id(),
                        issue_type="duplicates",
                        severity="high",
                        column=key_col,
                        affected_rows=conflicts,
                        affected_pct=pct,
                        description=f"Primary key conflict: {conflicts:,} duplicate IDs in '{key_col}' with non-identical attribute values",
                        suggested_fix=f"Deduplicate on key column '{key_col}'",
                        fix_code=f"df = df.unique(subset=['{key_col}'], keep='first')",
                    ))
    except Exception:
        pass

    return issues


def _detect_empty_column_issues(df: pl.DataFrame) -> list[Issue]:
    """Detect columns that are entirely empty."""
    issues = []
    n_rows = df.height
    
    for col in df.columns:
        null_count = df[col].null_count()
        if null_count == n_rows:
            issues.append(Issue(
                id=_generate_id(),
                issue_type="empty_column",
                severity="high",
                column=col,
                affected_rows=n_rows,
                affected_pct=100.0,
                description=f"Column is entirely empty",
                suggested_fix="Drop empty column",
                fix_code=f"df = df.drop('{col}')",
            ))
    
    return issues


def _detect_constant_column_issues(df: pl.DataFrame) -> list[Issue]:
    """Detect columns with only one unique value."""
    issues = []
    n_rows = df.height
    
    for col in df.columns:
        n_unique = df[col].n_unique()
        if n_unique == 1 and df[col].null_count() < n_rows:
            issues.append(Issue(
                id=_generate_id(),
                issue_type="constant_column",
                severity="low",
                column=col,
                affected_rows=n_rows,
                affected_pct=100.0,
                description=f"Column has only one unique value",
                suggested_fix="Consider dropping (no predictive value)",
                fix_code=f"df = df.drop('{col}')",
            ))
    
    return issues

import re as _re

_NUMERIC_PATTERN = _re.compile(r'^[\s$€£₹¥]?-?[\d,]+\.?\d*\s*%?$')
_BOOL_LOWER_SET = frozenset({"true", "false", "yes", "no", "0", "1", "t", "f", "y", "n"})


def _detect_type_mismatch_issues(
    df: pl.DataFrame,
    smart_schema: dict[str, Any] | None = None
) -> list[Issue]:
    """Detect type mismatches with enhanced auto-detection.
    
    Enhancements:
    - Auto-detects numeric strings in Utf8 columns even when smart_schema is None.
    - Detects boolean-like columns (true/false, yes/no, 0/1).
    - Includes sample values in descriptions for user clarity.
    """
    issues = []
    already_flagged: set[str] = set()

    # ── 1. Smart-schema driven corrections (if available) ──
    if smart_schema:
        corrections = smart_schema.get("type_corrections", [])
        for corr in corrections:
            col = corr.get("column", "")
            current = corr.get("current_type", "")
            suggested = corr.get("suggested_type", "")
            reason = corr.get("reason", "")
            code = corr.get("conversion_code", "")

            if not code:
                if suggested == "datetime":
                    code = f"df = df.with_columns(pl.col('{col}').str.to_datetime())"
                elif suggested == "integer":
                    code = f"df = df.with_columns(pl.col('{col}').cast(pl.Int64))"
                elif suggested == "float":
                    code = f"df = df.with_columns(pl.col('{col}').cast(pl.Float64))"

            # Include sample values for clarity
            samples_str = ""
            try:
                if col in df.columns:
                    samples = df[col].drop_nulls().head(3).to_list()
                    if samples:
                        samples_str = f" — samples: {samples[:3]}"
            except Exception:
                pass

            issues.append(Issue(
                id=_generate_id(),
                issue_type="type_mismatch",
                severity="medium",
                column=col,
                affected_rows=df.height,
                affected_pct=100.0,
                description=f"Currently {current}, should be {suggested}. {reason}{samples_str}",
                suggested_fix=f"Convert to {suggested}",
                fix_code=code if code else f"# Manual conversion needed for {col}",
            ))
            already_flagged.add(col)

    # ── 2. Auto-detect numeric strings in Utf8 columns ──
    n_rows = df.height
    if n_rows == 0:
        return issues

    for col in df.columns:
        if col in already_flagged:
            continue
        if df[col].dtype != pl.Utf8:
            continue

        non_null = df[col].drop_nulls()
        sample_size = min(non_null.len(), 100)
        if sample_size < 5:
            continue

        sample = non_null.head(sample_size).to_list()

        # ── Boolean detection ──
        lower_sample = [str(v).strip().lower() for v in sample if v is not None]
        if lower_sample and all(v in _BOOL_LOWER_SET for v in lower_sample):
            unique_vals = set(lower_sample)
            if len(unique_vals) <= 4:  # e.g. {"true", "false"} or {"yes", "no"}
                issues.append(Issue(
                    id=_generate_id(),
                    issue_type="type_mismatch",
                    severity="low",
                    column=col,
                    affected_rows=n_rows,
                    affected_pct=100.0,
                    description=f"Column contains boolean-like values ({', '.join(sorted(unique_vals))}) stored as strings — samples: {sample[:3]}",
                    suggested_fix="Convert to Boolean",
                    fix_code=f"df = df.with_columns(pl.col('{col}').str.to_lowercase().is_in(['true','yes','1','t','y']).alias('{col}'))",
                ))
                already_flagged.add(col)
                continue

        # ── Numeric string detection ──
        numeric_matches = sum(1 for v in sample if v is not None and _NUMERIC_PATTERN.match(str(v).strip()))
        match_ratio = numeric_matches / sample_size if sample_size > 0 else 0

        if match_ratio >= 0.80:
            # Determine if int or float
            has_decimal = any("." in str(v) for v in sample if v is not None)
            target_type = "Float64" if has_decimal else "Int64"

            issues.append(Issue(
                id=_generate_id(),
                issue_type="type_mismatch",
                severity="medium",
                column=col,
                affected_rows=n_rows,
                affected_pct=100.0,
                description=f"String column appears numeric ({match_ratio:.0%} of sampled values parse as numbers) — samples: {sample[:3]}",
                suggested_fix=f"Convert to {target_type}",
                fix_code=f"df = df.with_columns(pl.col('{col}').cast(pl.{target_type}, strict=False))",
            ))
            already_flagged.add(col)

    return issues


def _detect_outlier_issues(
    df: pl.DataFrame,
    outliers: dict[str, Any] | None = None
) -> list[Issue]:
    """Detect outlier issues with enhanced diagnostics.
    
    Enhancements:
    - Self-computes IQR outliers when no pre-computed outlier data is passed.
    - Adds max |Z-score| annotation for context.
    - Critical severity tier for ≥20% outliers.
    - Differentiated fix suggestions: flag (< 5%), Winsorize (5–15%), clip (> 15%).
    """
    issues = []
    n_rows = df.height
    if n_rows == 0:
        return issues

    # ── Build outlier data: use provided dict or self-compute via IQR ──
    outlier_data: dict[str, dict[str, Any]] = {}

    if outliers:
        outlier_data = outliers
    else:
        # Self-compute IQR outliers for all numeric columns
        numeric_dtypes = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64, pl.UInt32, pl.UInt64)
        for col in df.columns:
            if df[col].dtype not in numeric_dtypes:
                continue
            series = df[col].drop_nulls()
            if series.len() < 10:  # Skip tiny series
                continue
            try:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                if q1 is None or q3 is None:
                    continue
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                count = series.filter((series < lower) | (series > upper)).len()
                if count > 0:
                    outlier_data[col] = {"count": count, "q1": q1, "q3": q3}
            except Exception:
                continue

    for col, outlier_info in outlier_data.items():
        count = outlier_info.get("count", 0)
        if count <= 0:
            continue

        pct = (count / n_rows * 100) if n_rows > 0 else 0

        if pct >= 20:
            severity: Severity = "critical"
        elif pct >= 10:
            severity = "high"
        elif pct >= 5:
            severity = "medium"
        else:
            severity = "low"

        q1 = outlier_info.get("q1", 0)
        q3 = outlier_info.get("q3", 0)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # ── Z-score annotation (max |z|) ──
        z_note = ""
        try:
            if col in df.columns:
                series = df[col].drop_nulls().cast(pl.Float64)
                mean_val = series.mean()
                std_val = series.std()
                if std_val and std_val > 0 and mean_val is not None:
                    max_z = ((series - mean_val) / std_val).abs().max()
                    if max_z is not None:
                        z_note = f", max |z| = {max_z:.1f}"
        except Exception:
            pass

        desc = f"{count:,} outliers detected ({pct:.1f}%{z_note})"

        # ── Differentiated fix suggestions ──
        if pct < 5:
            suggested_fix = "Flag for manual review"
            fix_code = f"# Review outliers in '{col}' — values outside [{lower:.2f}, {upper:.2f}]"
        elif pct < 15:
            suggested_fix = f"Winsorize to IQR bounds [{lower:.2f}, {upper:.2f}]"
            fix_code = f"df = df.with_columns(pl.col('{col}').clip({lower:.2f}, {upper:.2f}))"
        else:
            suggested_fix = f"Clip to range [{lower:.2f}, {upper:.2f}]"
            fix_code = f"df = df.with_columns(pl.col('{col}').clip({lower:.2f}, {upper:.2f}))"

        issues.append(Issue(
            id=_generate_id(),
            issue_type="outliers",
            severity=severity,
            column=col,
            affected_rows=count,
            affected_pct=pct,
            description=desc,
            suggested_fix=suggested_fix,
            fix_code=fix_code,
        ))

    return issues


def _detect_high_cardinality_issues(df: pl.DataFrame) -> list[Issue]:
    """Detect high cardinality categorical columns."""
    issues = []
    n_rows = df.height
    
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            n_unique = df[col].n_unique()
            unique_ratio = n_unique / n_rows if n_rows > 0 else 0
            
            if n_unique > 100 and unique_ratio > 0.5:
                issues.append(Issue(
                    id=_generate_id(),
                    issue_type="high_cardinality",
                    severity="medium",
                    column=col,
                    affected_rows=n_rows,
                    affected_pct=100.0,
                    description=f"{n_unique:,} unique values ({unique_ratio:.1%} of rows)",
                    suggested_fix="Consider grouping rare categories or using embeddings",
                    fix_code=f"# Manual review needed: consider df['{col}'].value_counts()",
                ))
    
    return issues


import hashlib

def compute_dataset_fingerprint(df: pl.DataFrame) -> str:
    """Compute normalized semantic fingerprint for cross-format duplicate dataset detection."""
    sorted_cols = ",".join(sorted(df.columns))
    row_count = str(df.height)
    null_total = str(sum(df[c].null_count() for c in df.columns))
    
    # Sample head & tail representations
    sample_text = str(df.head(5).to_dicts()) + str(df.tail(5).to_dicts())
    fingerprint_raw = f"{sorted_cols}|{row_count}|{null_total}|{sample_text}"
    return hashlib.sha256(fingerprint_raw.encode("utf-8")).hexdigest()


MASKED_NULL_PATTERNS = frozenset({
    "n/a", "na", "null", "none", "-", "--", "?", "missing", "unknown",
    "#n/a", "-999", "9999", "00/00/0000", "undefined", "blank", "n.a.", "nil"
})


def _detect_masked_null_issues(df: pl.DataFrame) -> list[Issue]:
    """Detect hidden/masked null placeholders in string and numeric columns."""
    issues = []
    n_rows = df.height
    if n_rows == 0:
        return issues

    for col in df.columns:
        dtype = df[col].dtype
        masked_count = 0

        if dtype == pl.Utf8:
            non_null = df[col].drop_nulls()
            if non_null.len() == 0:
                continue
            masked_mask = non_null.str.to_lowercase().str.strip_chars().is_in(list(MASKED_NULL_PATTERNS))
            masked_count = int(masked_mask.sum())
        elif dtype in (pl.Int64, pl.Float64, pl.Int32, pl.Float32):
            non_null = df[col].drop_nulls()
            if non_null.len() == 0:
                continue
            masked_count = int(non_null.is_in([-999, -9999, 9999, 99999]).sum())

        if masked_count > 0:
            masked_pct = (masked_count / n_rows * 100)
            severity: Severity = "high" if masked_pct >= 10 else ("medium" if masked_pct >= 3 else "low")
            
            if dtype == pl.Utf8:
                fix_code = f"df = df.with_columns(pl.when(pl.col('{col}').str.to_lowercase().str.strip_chars().is_in({list(MASKED_NULL_PATTERNS)})).then(None).otherwise(pl.col('{col}')).alias('{col}'))"
            else:
                fix_code = f"df = df.with_columns(pl.when(pl.col('{col}').is_in([-999, -9999, 9999, 99999])).then(None).otherwise(pl.col('{col}')).alias('{col}'))"

            issues.append(Issue(
                id=_generate_id(),
                issue_type="missing_values",
                severity=severity,
                column=col,
                affected_rows=masked_count,
                affected_pct=masked_pct,
                description=f"{masked_count:,} masked null placeholders detected (e.g. 'N/A', '-999', '?')",
                suggested_fix="Convert masked null strings to native nulls",
                fix_code=fix_code,
            ))

    return issues


def _detect_whitespace_and_case_issues(df: pl.DataFrame) -> list[Issue]:
    """Detect leading/trailing whitespace pollution and case fragmentation in string columns."""
    issues = []
    n_rows = df.height
    if n_rows == 0:
        return issues

    for col in df.columns:
        if df[col].dtype != pl.Utf8:
            continue

        non_null = df[col].drop_nulls()
        if non_null.len() == 0:
            continue

        # ── 1. Whitespace pollution ──
        trimmed = non_null.str.strip_chars()
        space_diff = int((non_null != trimmed).sum())

        if space_diff > 0:
            pct = (space_diff / n_rows * 100)
            issues.append(Issue(
                id=_generate_id(),
                issue_type="format_issue",
                severity="low" if pct < 5 else "medium",
                column=col,
                affected_rows=space_diff,
                affected_pct=pct,
                description=f"{space_diff:,} values have leading/trailing whitespace pollution",
                suggested_fix="Trim leading and trailing whitespace",
                fix_code=f"df = df.with_columns(pl.col('{col}').str.strip_chars().alias('{col}'))",
            ))

        # ── 2. Case fragmentation in categorical columns ──
        n_unique_raw = non_null.n_unique()
        n_unique_lower = non_null.str.to_lowercase().n_unique()

        if n_unique_raw > n_unique_lower and n_unique_raw <= 100:
            diff = n_unique_raw - n_unique_lower
            issues.append(Issue(
                id=_generate_id(),
                issue_type="format_issue",
                severity="medium",
                column=col,
                affected_rows=n_rows,
                affected_pct=100.0,
                description=f"Case fragmentation: {n_unique_raw} raw distinct values reduce to {n_unique_lower} when lowercased ({diff} casing variations)",
                suggested_fix="Normalize text to title case",
                fix_code=f"df = df.with_columns(pl.col('{col}').str.to_titlecase().alias('{col}'))",
            ))

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def detect_issues(
    df: pl.DataFrame,
    smart_schema: dict[str, Any] | None = None,
    outliers: dict[str, Any] | None = None,
) -> IssueLedger:
    """
    Detect all data quality issues and return an IssueLedger.
    
    Args:
        df: Polars DataFrame to analyze
        smart_schema: Optional smart schema analysis results
        outliers: Optional outlier detection results
        
    Returns:
        IssueLedger with all detected issues
    """
    logger.info(f"Detecting issues for DataFrame with {df.height} rows, {df.width} columns")
    
    ledger = IssueLedger()
    
    # Detect various issue types
    for issue in _detect_empty_column_issues(df):
        ledger.add_issue(issue)
    
    for issue in _detect_missing_value_issues(df):
        ledger.add_issue(issue)
    
    for issue in _detect_masked_null_issues(df):
        ledger.add_issue(issue)
    
    for issue in _detect_duplicate_issues(df):
        ledger.add_issue(issue)

    for issue in _detect_whitespace_and_case_issues(df):
        ledger.add_issue(issue)
    
    for issue in _detect_constant_column_issues(df):
        ledger.add_issue(issue)
    
    for issue in _detect_type_mismatch_issues(df, smart_schema):
        ledger.add_issue(issue)
    
    for issue in _detect_outlier_issues(df, outliers):
        ledger.add_issue(issue)
    
    for issue in _detect_high_cardinality_issues(df):
        ledger.add_issue(issue)
    
    # Sort by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    ledger.issues.sort(key=lambda x: severity_order.get(x.severity, 4))
    
    logger.info(f"Detected {len(ledger.issues)} issues")
    return ledger


def apply_remediation(
    df: pl.DataFrame,
    ledger: IssueLedger,
    selected_ids: list[str] | None = None
) -> pl.DataFrame:
    """
    Applies approved or selected quality remediation fixes to a Polars DataFrame safely.
    
    Args:
        df: Input Polars DataFrame
        ledger: IssueLedger containing quality issues
        selected_ids: Optional specific issue IDs to execute (if None, applies all approved/modified)
        
    Returns:
        Remediated Polars DataFrame
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected pl.DataFrame, got {type(df)}")

    issues_to_apply = []
    for issue in ledger.issues:
        if selected_ids is not None:
            if issue.id in selected_ids:
                issues_to_apply.append(issue)
        elif issue.status in ("approved", "modified"):
            issues_to_apply.append(issue)

    if not issues_to_apply:
        return df

    loc_env = {"df": df, "pl": pl}
    for issue in issues_to_apply:
        try:
            if issue.fix_code and not issue.fix_code.startswith("#"):
                exec(issue.fix_code, globals(), loc_env)
                df = loc_env["df"]
                logger.info("Applied remediation fix for issue %s (%s: %s)", issue.id, issue.issue_type, issue.column)
        except Exception as e:
            logger.warning("Failed to apply remediation fix for issue %s: %s", issue.id, e)

    return df

