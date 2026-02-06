"""
Analysis Decisions Tracker

Provides transparency into why each analysis step was run or skipped.
This builds trust by making automation decisions explainable.

Every decision includes:
- Action taken (ran/skipped)
- Reason why
- Data evidence that drove the decision
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal
from enum import Enum

import polars as pl

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    RAN = "ran"
    SKIPPED = "skipped"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class AnalysisDecision:
    """A single analysis decision with explanation."""
    analysis_name: str
    decision: DecisionType
    reason: str
    evidence: list[str] = field(default_factory=list)
    impact: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis_name,
            "decision": self.decision.value,
            "reason": self.reason,
            "evidence": self.evidence,
            "impact": self.impact
        }


@dataclass
class DecisionLog:
    """Complete log of all analysis decisions."""
    decisions: list[AnalysisDecision] = field(default_factory=list)
    
    def add(self, decision: AnalysisDecision):
        self.decisions.append(decision)
        logger.info(f"Decision: {decision.analysis_name} -> {decision.decision.value}: {decision.reason}")
    
    def ran(self, name: str, reason: str, evidence: list[str] = None, impact: str = ""):
        self.add(AnalysisDecision(
            analysis_name=name,
            decision=DecisionType.RAN,
            reason=reason,
            evidence=evidence or [],
            impact=impact
        ))
    
    def skipped(self, name: str, reason: str, evidence: list[str] = None):
        self.add(AnalysisDecision(
            analysis_name=name,
            decision=DecisionType.SKIPPED,
            reason=reason,
            evidence=evidence or []
        ))
    
    def partial(self, name: str, reason: str, evidence: list[str] = None):
        self.add(AnalysisDecision(
            analysis_name=name,
            decision=DecisionType.PARTIAL,
            reason=reason,
            evidence=evidence or []
        ))
    
    def failed(self, name: str, reason: str):
        self.add(AnalysisDecision(
            analysis_name=name,
            decision=DecisionType.FAILED,
            reason=reason
        ))
    
    def to_dict(self) -> dict[str, Any]:
        ran_count = sum(1 for d in self.decisions if d.decision == DecisionType.RAN)
        skipped_count = sum(1 for d in self.decisions if d.decision == DecisionType.SKIPPED)
        
        return {
            "decisions": [d.to_dict() for d in self.decisions],
            "summary": {
                "total": len(self.decisions),
                "ran": ran_count,
                "skipped": skipped_count,
                "partial": sum(1 for d in self.decisions if d.decision == DecisionType.PARTIAL),
                "failed": sum(1 for d in self.decisions if d.decision == DecisionType.FAILED)
            }
        }
    
    def get_skipped(self) -> list[AnalysisDecision]:
        return [d for d in self.decisions if d.decision == DecisionType.SKIPPED]
    
    def get_ran(self) -> list[AnalysisDecision]:
        return [d for d in self.decisions if d.decision == DecisionType.RAN]


def evaluate_analysis_decisions(df: pl.DataFrame) -> DecisionLog:
    """
    Pre-evaluate which analyses should run and why.
    Returns a decision log with explanations for each analysis type.
    """
    log = DecisionLog()
    
    # Get column info
    numeric_cols = [c for c, t in df.schema.items() 
                    if t in (pl.Int64, pl.Float64, pl.Int32, pl.Float32)]
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    date_cols = [c for c, t in df.schema.items() 
                 if t in (pl.Date, pl.Datetime)]
    
    n_rows = df.height
    n_cols = df.width
    
    # 1. Correlation Analysis
    if len(numeric_cols) >= 2:
        log.ran(
            "Correlation Analysis",
            f"Found {len(numeric_cols)} numeric columns suitable for correlation",
            [f"Numeric columns: {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}"],
            "Identifies relationships between numeric variables"
        )
    else:
        log.skipped(
            "Correlation Analysis",
            f"Requires at least 2 numeric columns (found {len(numeric_cols)})",
            [f"Available columns: {', '.join(numeric_cols) if numeric_cols else 'None'}"]
        )
    
    # 2. Distribution Analysis
    if numeric_cols:
        log.ran(
            "Distribution Analysis",
            f"Analyzing distributions for {min(len(numeric_cols), 10)} numeric columns",
            [f"Columns: {', '.join(numeric_cols[:5])}"],
            "Detects skewness, kurtosis, and outliers"
        )
    else:
        log.skipped(
            "Distribution Analysis",
            "No numeric columns found for distribution analysis"
        )
    
    # 3. Categorical Analysis
    low_cardinality_cats = [c for c in cat_cols if df[c].n_unique() <= 50]
    if low_cardinality_cats:
        log.ran(
            "Categorical Analysis",
            f"Found {len(low_cardinality_cats)} categorical columns with manageable cardinality",
            [f"Columns: {', '.join(low_cardinality_cats[:5])}"],
            "Frequency distributions and mode analysis"
        )
    elif cat_cols:
        log.partial(
            "Categorical Analysis",
            f"High cardinality in all {len(cat_cols)} categorical columns",
            ["All columns have >50 unique values - limited analysis possible"]
        )
    else:
        log.skipped(
            "Categorical Analysis",
            "No categorical columns found"
        )
    
    # 4. Time-Series Analysis
    if date_cols:
        log.ran(
            "Time-Series Analysis",
            f"Found {len(date_cols)} date/datetime columns",
            [f"Date columns: {', '.join(date_cols)}"],
            "Trend detection, seasonality, and temporal patterns"
        )
    else:
        log.skipped(
            "Time-Series Analysis",
            "No date or datetime columns detected",
            ["Consider converting date strings to proper date types"]
        )
    
    # 5. Outlier Detection
    if numeric_cols:
        log.ran(
            "Outlier Detection",
            f"IQR-based outlier detection on {len(numeric_cols)} numeric columns",
            ["Using 1.5x IQR threshold"],
            "Flags extreme values for review"
        )
    else:
        log.skipped(
            "Outlier Detection",
            "No numeric columns for outlier analysis"
        )
    
    # 6. Multicollinearity (VIF)
    if len(numeric_cols) >= 3 and n_rows >= 10:
        log.ran(
            "Multicollinearity Analysis (VIF)",
            f"Checking variance inflation for {len(numeric_cols)} numeric features",
            [f"Row count ({n_rows}) sufficient for VIF calculation"],
            "Detects redundant features that may cause model issues"
        )
    elif len(numeric_cols) < 3:
        log.skipped(
            "Multicollinearity Analysis (VIF)",
            f"Requires at least 3 numeric columns (found {len(numeric_cols)})"
        )
    else:
        log.skipped(
            "Multicollinearity Analysis (VIF)",
            f"Insufficient rows ({n_rows}) for reliable VIF calculation"
        )
    
    # 7. Skewness/Kurtosis
    if numeric_cols and n_rows >= 20:
        log.ran(
            "Distribution Shape Analysis",
            f"Computing skewness and kurtosis for {len(numeric_cols)} columns",
            [f"Sample size ({n_rows}) adequate for shape metrics"],
            "Identifies non-normal distributions"
        )
    elif n_rows < 20:
        log.skipped(
            "Distribution Shape Analysis",
            f"Insufficient rows ({n_rows}) for reliable shape metrics"
        )
    else:
        log.skipped(
            "Distribution Shape Analysis",
            "No numeric columns available"
        )
    
    # 8. Missing Pattern Analysis
    null_cols = [c for c in df.columns if df[c].null_count() > 0]
    if null_cols:
        log.ran(
            "Missing Value Pattern Analysis",
            f"Analyzing patterns in {len(null_cols)} columns with missing values",
            [f"Columns with nulls: {', '.join(null_cols[:5])}{'...' if len(null_cols) > 5 else ''}"],
            "Determines if missing data is random or systematic"
        )
    else:
        log.skipped(
            "Missing Value Pattern Analysis",
            "No missing values detected in dataset"
        )
    
    # 9. Feature Engineering Recommendations
    if numeric_cols or cat_cols:
        log.ran(
            "Feature Engineering Analysis",
            f"Generating ML-prep suggestions for {len(numeric_cols)} numeric + {len(cat_cols)} categorical columns",
            impact="Encoding, scaling, and feature extraction recommendations"
        )
    
    # 10. Smart Schema Inference
    log.ran(
        "Smart Schema Analysis",
        f"Inferring data types and relationships for {n_cols} columns",
        impact="Type corrections, relationship detection, quality issues"
    )
    
    return log
