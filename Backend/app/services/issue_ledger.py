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
    """Detect columns with significant missing values."""
    issues = []
    n_rows = df.height
    
    for col in df.columns:
        null_count = df[col].null_count()
        null_pct = (null_count / n_rows * 100) if n_rows > 0 else 0
        
        if null_pct >= 30:
            severity = "critical"
        elif null_pct >= 15:
            severity = "high"
        elif null_pct >= 5:
            severity = "medium"
        else:
            continue  # Skip low missing %
        
        # Determine fix based on column type
        dtype = df[col].dtype
        if dtype in (pl.Int64, pl.Float64):
            fix_code = f"df = df.with_columns(pl.col('{col}').fill_null(pl.col('{col}').median()))"
            suggested_fix = f"Fill with median value"
        else:
            fix_code = f"df = df.with_columns(pl.col('{col}').fill_null(pl.col('{col}').mode().first()))"
            suggested_fix = f"Fill with most common value"
        
        issues.append(Issue(
            id=_generate_id(),
            issue_type="missing_values",
            severity=severity,
            column=col,
            affected_rows=null_count,
            affected_pct=null_pct,
            description=f"{null_pct:.1f}% missing values ({null_count:,} rows)",
            suggested_fix=suggested_fix,
            fix_code=fix_code,
        ))
    
    return issues


def _detect_duplicate_issues(df: pl.DataFrame) -> list[Issue]:
    """Detect duplicate rows."""
    issues = []
    n_rows = df.height
    n_unique = df.n_unique()
    dup_count = n_rows - n_unique
    
    if dup_count > 0:
        dup_pct = (dup_count / n_rows * 100) if n_rows > 0 else 0
        
        if dup_pct >= 20:
            severity = "high"
        elif dup_pct >= 5:
            severity = "medium"
        else:
            severity = "low"
        
        issues.append(Issue(
            id=_generate_id(),
            issue_type="duplicates",
            severity=severity,
            column=None,
            affected_rows=dup_count,
            affected_pct=dup_pct,
            description=f"{dup_count:,} duplicate rows ({dup_pct:.1f}%)",
            suggested_fix="Remove duplicate rows",
            fix_code="df = df.unique()",
        ))
    
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


def _detect_type_mismatch_issues(
    df: pl.DataFrame,
    smart_schema: dict[str, Any] | None = None
) -> list[Issue]:
    """Detect type mismatches from smart schema analysis."""
    issues = []
    
    if not smart_schema:
        return issues
    
    corrections = smart_schema.get("type_corrections", [])
    for corr in corrections:
        col = corr.get("column", "")
        current = corr.get("current_type", "")
        suggested = corr.get("suggested_type", "")
        reason = corr.get("reason", "")
        code = corr.get("conversion_code", "")
        
        if not code:
            # Generate default conversion code
            if suggested == "datetime":
                code = f"df = df.with_columns(pl.col('{col}').str.to_datetime())"
            elif suggested == "integer":
                code = f"df = df.with_columns(pl.col('{col}').cast(pl.Int64))"
            elif suggested == "float":
                code = f"df = df.with_columns(pl.col('{col}').cast(pl.Float64))"
        
        issues.append(Issue(
            id=_generate_id(),
            issue_type="type_mismatch",
            severity="medium",
            column=col,
            affected_rows=df.height,
            affected_pct=100.0,
            description=f"Currently {current}, should be {suggested}. {reason}",
            suggested_fix=f"Convert to {suggested}",
            fix_code=code if code else f"# Manual conversion needed for {col}",
        ))
    
    return issues


def _detect_outlier_issues(
    df: pl.DataFrame,
    outliers: dict[str, Any] | None = None
) -> list[Issue]:
    """Detect outlier issues from analysis results."""
    issues = []
    
    if not outliers:
        return issues
    
    n_rows = df.height
    
    for col, outlier_info in outliers.items():
        count = outlier_info.get("count", 0)
        if count > 0:
            pct = (count / n_rows * 100) if n_rows > 0 else 0
            
            if pct >= 10:
                severity = "high"
            elif pct >= 5:
                severity = "medium"
            else:
                severity = "low"
            
            # Clip to IQR bounds
            q1 = outlier_info.get("q1", 0)
            q3 = outlier_info.get("q3", 0)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            
            issues.append(Issue(
                id=_generate_id(),
                issue_type="outliers",
                severity=severity,
                column=col,
                affected_rows=count,
                affected_pct=pct,
                description=f"{count:,} outliers detected ({pct:.1f}%)",
                suggested_fix=f"Clip to range [{lower:.2f}, {upper:.2f}]",
                fix_code=f"df = df.with_columns(pl.col('{col}').clip({lower:.2f}, {upper:.2f}))",
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
    
    for issue in _detect_duplicate_issues(df):
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
