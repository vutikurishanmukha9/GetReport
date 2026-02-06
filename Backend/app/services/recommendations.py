"""
Actionable Recommendations Service

Provides intelligent, domain-aware recommendations based on data analysis:
1. Domain-specific recommendations (education, sales, healthcare, etc.)
2. Data quality recommendations
3. Analysis recommendations (what to explore next)
4. Visualization recommendations
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Recommendation:
    """A single actionable recommendation."""
    category: str  # data_quality, analysis, visualization, domain_specific
    priority: str  # high, medium, low
    title: str
    description: str
    action: str  # What to do
    impact: str  # Expected benefit
    code_hint: str = ""  # Optional code snippet


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN-SPECIFIC RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

DOMAIN_RECOMMENDATIONS = {
    "education": [
        Recommendation(
            category="domain_specific",
            priority="high",
            title="Student Performance Segmentation",
            description="Segment students by performance grades to identify at-risk groups",
            action="Create performance quartiles and analyze characteristics of bottom 25%",
            impact="Enable targeted interventions for struggling students"
        ),
        Recommendation(
            category="domain_specific",
            priority="medium",
            title="Attendance-Performance Correlation",
            description="Analyze relationship between attendance and academic outcomes",
            action="Generate scatter plot of attendance vs. grades with regression line",
            impact="Quantify attendance impact for policy decisions"
        ),
        Recommendation(
            category="domain_specific",
            priority="medium",
            title="Subject-wise Analysis",
            description="Compare performance across different subjects",
            action="Create box plots for each subject to identify difficult areas",
            impact="Focus curriculum improvements on challenging subjects"
        ),
    ],
    "sales": [
        Recommendation(
            category="domain_specific",
            priority="high",
            title="Revenue Driver Analysis",
            description="Identify top factors driving revenue",
            action="Analyze correlation between product features and sales amounts",
            impact="Focus marketing on high-impact drivers"
        ),
        Recommendation(
            category="domain_specific",
            priority="high",
            title="Customer Segmentation",
            description="Segment customers by purchase behavior",
            action="Apply RFM (Recency, Frequency, Monetary) analysis",
            impact="Enable targeted marketing campaigns"
        ),
        Recommendation(
            category="domain_specific",
            priority="medium",
            title="Seasonal Trend Detection",
            description="Identify seasonal patterns in sales data",
            action="Create time-series decomposition if date columns exist",
            impact="Optimize inventory and marketing timing"
        ),
    ],
    "healthcare": [
        Recommendation(
            category="domain_specific",
            priority="high",
            title="Risk Factor Identification",
            description="Identify key predictors of health outcomes",
            action="Perform correlation analysis with outcome variables",
            impact="Enable preventive care targeting"
        ),
        Recommendation(
            category="domain_specific",
            priority="high",
            title="Anomaly Detection for Critical Values",
            description="Flag patients with out-of-range measurements",
            action="Set up alerts for values outside clinical norms",
            impact="Enable early intervention for at-risk patients"
        ),
    ],
    "hr": [
        Recommendation(
            category="domain_specific",
            priority="high",
            title="Attrition Risk Analysis",
            description="Identify factors correlated with employee turnover",
            action="Analyze differences between departed and retained employees",
            impact="Improve retention strategies"
        ),
        Recommendation(
            category="domain_specific",
            priority="medium",
            title="Compensation Equity Analysis",
            description="Check for pay disparities across demographics",
            action="Compare salary distributions by department, role, tenure",
            impact="Ensure fair compensation practices"
        ),
    ],
    "finance": [
        Recommendation(
            category="domain_specific",
            priority="high",
            title="Fraud Detection Setup",
            description="Identify unusual transaction patterns",
            action="Flag transactions > 3 standard deviations from mean",
            impact="Early fraud detection and prevention"
        ),
        Recommendation(
            category="domain_specific",
            priority="medium",
            title="Cash Flow Analysis",
            description="Analyze inflow/outflow patterns over time",
            action="Create time-series of net balance changes",
            impact="Improve financial planning"
        ),
    ],
    "logistics": [
        Recommendation(
            category="domain_specific",
            priority="high",
            title="Delivery Performance Analysis",
            description="Analyze on-time vs late deliveries",
            action="Calculate delivery time distributions by route/carrier",
            impact="Optimize delivery operations"
        ),
    ],
    "iot": [
        Recommendation(
            category="domain_specific",
            priority="high",
            title="Anomaly Detection",
            description="Detect abnormal sensor readings",
            action="Set up threshold-based alerts for sensor values",
            impact="Predictive maintenance and early warning"
        ),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_quality_recommendations(
    df: pl.DataFrame,
    analysis_results: dict[str, Any] | None = None
) -> list[Recommendation]:
    """Generate recommendations based on data quality issues."""
    recommendations = []
    
    # Check for missing values
    for col in df.columns:
        null_pct = df[col].null_count() / df.height * 100
        
        if null_pct > 30:
            recommendations.append(Recommendation(
                category="data_quality",
                priority="high",
                title=f"Address Missing Values in '{col}'",
                description=f"{null_pct:.1f}% of values are missing",
                action=f"Consider imputation (mean/median/mode) or dropping rows/column",
                impact="Improve model accuracy and analysis reliability",
                code_hint=f"df['{col}'].fill_null(df['{col}'].median())"
            ))
        elif null_pct > 5:
            recommendations.append(Recommendation(
                category="data_quality",
                priority="medium",
                title=f"Handle Missing Values in '{col}'",
                description=f"{null_pct:.1f}% of values are missing",
                action="Apply appropriate imputation strategy",
                impact="Avoid data loss in analysis",
                code_hint=f"df['{col}'].fill_null(strategy='forward')"
            ))
    
    # Check for potential duplicates
    unique_ratio = df.n_unique() / df.height if df.height > 0 else 1
    if unique_ratio < 0.95:
        dup_count = df.height - df.n_unique()
        recommendations.append(Recommendation(
            category="data_quality",
            priority="medium",
            title="Investigate Duplicate Rows",
            description=f"~{dup_count} potential duplicate rows detected",
            action="Review and remove exact duplicates if confirmed",
            impact="Prevent bias from repeated records",
            code_hint="df.unique()"
        ))
    
    # Check for high cardinality in categorical columns
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            n_unique = df[col].n_unique()
            if n_unique > 100 and n_unique > df.height * 0.5:
                recommendations.append(Recommendation(
                    category="data_quality",
                    priority="medium",
                    title=f"High Cardinality in '{col}'",
                    description=f"{n_unique} unique values - may need grouping",
                    action="Consider grouping rare categories or using embeddings",
                    impact="Improve model performance with categorical features"
                ))
    
    # Limit recommendations
    return recommendations[:8]


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_analysis_recommendations(
    df: pl.DataFrame,
    analysis_results: dict[str, Any] | None = None
) -> list[Recommendation]:
    """Generate recommendations for further analysis."""
    recommendations = []
    
    # Get column types
    numeric_cols = [c for c, t in df.schema.items() if t in (pl.Int64, pl.Float64)]
    cat_cols = [c for c, t in df.schema.items() if t == pl.Utf8]
    date_cols = [c for c, t in df.schema.items() if t in (pl.Date, pl.Datetime)]
    
    # Recommend correlation analysis if multiple numeric columns
    if len(numeric_cols) >= 3:
        recommendations.append(Recommendation(
            category="analysis",
            priority="high",
            title="Deep Correlation Analysis",
            description=f"{len(numeric_cols)} numeric columns available for correlation study",
            action="Identify the strongest correlations and investigate causal relationships",
            impact="Discover hidden relationships in data"
        ))
    
    # Recommend time-series if date columns present
    if date_cols:
        recommendations.append(Recommendation(
            category="analysis",
            priority="high",
            title="Time-Series Investigation",
            description=f"Date column '{date_cols[0]}' enables temporal analysis",
            action="Analyze trends, seasonality, and period-over-period changes",
            impact="Understand how metrics evolve over time"
        ))
    
    # Recommend segmentation if categorical columns present
    if cat_cols and numeric_cols:
        recommendations.append(Recommendation(
            category="analysis",
            priority="medium",
            title="Categorical Segmentation",
            description=f"Compare metrics across '{cat_cols[0]}' categories",
            action=f"Group by '{cat_cols[0]}' and aggregate numeric columns",
            impact="Identify best/worst performing segments",
            code_hint=f"df.group_by('{cat_cols[0]}').agg([pl.mean('{numeric_cols[0]}')])"
        ))
    
    # Recommend outlier investigation if outliers detected
    if analysis_results and analysis_results.get("outliers"):
        outlier_cols = list(analysis_results["outliers"].keys())[:3]
        recommendations.append(Recommendation(
            category="analysis",
            priority="medium",
            title="Outlier Investigation",
            description=f"Outliers detected in: {', '.join(outlier_cols)}",
            action="Investigate if outliers are errors or genuine extreme values",
            impact="Decide on data cleaning approach"
        ))
    
    return recommendations[:6]


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_viz_recommendations(
    df: pl.DataFrame,
    domain: str = "unknown"
) -> list[Recommendation]:
    """Generate visualization recommendations."""
    recommendations = []
    
    numeric_cols = [c for c, t in df.schema.items() if t in (pl.Int64, pl.Float64)]
    cat_cols = [c for c, t in df.schema.items() if t == pl.Utf8]
    date_cols = [c for c, t in df.schema.items() if t in (pl.Date, pl.Datetime)]
    
    if len(numeric_cols) >= 2:
        recommendations.append(Recommendation(
            category="visualization",
            priority="medium",
            title="Scatter Plot Matrix",
            description="Visualize relationships between numeric variables",
            action=f"Create pairplot of top 5 numeric columns",
            impact="Quick overview of all variable relationships"
        ))
    
    if date_cols and numeric_cols:
        recommendations.append(Recommendation(
            category="visualization",
            priority="high",
            title="Time-Series Line Charts",
            description="Visualize trends over time",
            action=f"Plot '{numeric_cols[0]}' over '{date_cols[0]}'",
            impact="Identify trends and patterns"
        ))
    
    if cat_cols and numeric_cols:
        recommendations.append(Recommendation(
            category="visualization",
            priority="medium",
            title="Box Plots by Category",
            description="Compare distributions across categories",
            action=f"Create box plot of '{numeric_cols[0]}' grouped by '{cat_cols[0]}'",
            impact="Identify category-level differences"
        ))
    
    return recommendations[:4]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RecommendationsResult:
    """Complete recommendations analysis."""
    domain_recommendations: list[Recommendation]
    quality_recommendations: list[Recommendation]
    analysis_recommendations: list[Recommendation]
    visualization_recommendations: list[Recommendation]
    
    def to_dict(self) -> dict[str, Any]:
        def rec_to_dict(r: Recommendation) -> dict:
            return {
                "category": r.category,
                "priority": r.priority,
                "title": r.title,
                "description": r.description,
                "action": r.action,
                "impact": r.impact,
                "code_hint": r.code_hint if r.code_hint else None
            }
        
        return {
            "domain_specific": [rec_to_dict(r) for r in self.domain_recommendations],
            "data_quality": [rec_to_dict(r) for r in self.quality_recommendations],
            "analysis": [rec_to_dict(r) for r in self.analysis_recommendations],
            "visualization": [rec_to_dict(r) for r in self.visualization_recommendations],
            "total_count": (
                len(self.domain_recommendations) +
                len(self.quality_recommendations) +
                len(self.analysis_recommendations) +
                len(self.visualization_recommendations)
            )
        }
    
    def get_high_priority(self) -> list[Recommendation]:
        """Get all high-priority recommendations."""
        all_recs = (
            self.domain_recommendations +
            self.quality_recommendations +
            self.analysis_recommendations +
            self.visualization_recommendations
        )
        return [r for r in all_recs if r.priority == "high"]


def generate_recommendations(
    df: pl.DataFrame,
    domain: str = "unknown",
    analysis_results: dict[str, Any] | None = None
) -> RecommendationsResult:
    """
    Generate actionable recommendations based on data and analysis.
    
    Args:
        df: Polars DataFrame
        domain: Detected data domain (education, sales, etc.)
        analysis_results: Optional results from analyze_dataset()
        
    Returns:
        RecommendationsResult with categorized recommendations
    """
    logger.info(f"Generating recommendations for domain: {domain}")
    
    # Get domain-specific recommendations
    domain_recs = DOMAIN_RECOMMENDATIONS.get(domain, [])
    
    # Generate quality recommendations
    quality_recs = _generate_quality_recommendations(df, analysis_results)
    
    # Generate analysis recommendations
    analysis_recs = _generate_analysis_recommendations(df, analysis_results)
    
    # Generate visualization recommendations
    viz_recs = _generate_viz_recommendations(df, domain)
    
    result = RecommendationsResult(
        domain_recommendations=domain_recs[:3],  # Top 3 per category
        quality_recommendations=quality_recs[:5],
        analysis_recommendations=analysis_recs[:4],
        visualization_recommendations=viz_recs[:3]
    )
    
    logger.info(f"Generated {result.to_dict()['total_count']} total recommendations")
    
    return result
