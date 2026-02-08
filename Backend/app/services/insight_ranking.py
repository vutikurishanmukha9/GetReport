from dataclasses import dataclass, field
from typing import Any, List, Dict
import logging

logger = logging.getLogger(__name__)

@dataclass
class RankedInsight:
    """
    A normalized, scored finding from the analysis.
    Used to surface the most important "signals" to the user/AI.
    """
    type: str          # e.g., 'correlation', 'outlier', 'missing_pattern', 'trend'
    variable: str      # The primary column involved
    description: str   # Human-readable summary
    score: float       # 0.0 to 1.0 (1.0 = most critical)
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "variable": self.variable,
            "description": self.description,
            "score": round(self.score, 2),
            "evidence": self.evidence
        }

def rank_insights(analysis_results: Dict[str, Any]) -> List[RankedInsight]:
    """
    Extracts findings from raw analysis results, scores them, and returns a sorted list.
    """
    insights: List[RankedInsight] = []
    
    # 1. Processing Strong Correlations
    # -------------------------------------------------------
    correlations = analysis_results.get("strong_correlations", [])
    if correlations:
        for corr in correlations:
            r_val = abs(corr.get("r_value", 0))
            col_a = corr.get("column_a")
            col_b = corr.get("column_b")
            
            # Score: direct mapping of r_value (0.7 to 1.0)
            score = r_val
            
            insights.append(RankedInsight(
                type="correlation",
                variable=f"{col_a} & {col_b}",
                description=f"Strong {corr.get('direction')} correlation ({r_val}) between {col_a} and {col_b}",
                score=score,
                evidence=corr
            ))

    # 2. Processing Missing Value Patterns
    # -------------------------------------------------------
    missing_data = analysis_results.get("missing_patterns", {})
    if missing_data and missing_data.get("has_missing"):
        # Column-level missing
        for col, details in missing_data.get("column_details", {}).items():
            pct = details.get("percentage", 0)
            severity = details.get("severity", "low")
            
            # Score: High missing % is bad, but 100% missing is critical schema issue
            # 50% missing is very high impact for analysis
            if pct > 99:
                score = 0.95 # Almost empty column
            elif pct > 50:
                score = 0.85
            elif pct > 20:
                score = 0.70
            else:
                score = 0.40
                
            insights.append(RankedInsight(
                type="data_quality",
                variable=col,
                description=f"{pct}% missing values ({severity} severity)",
                score=score,
                evidence=details
            ))
            
        # Systematic patterns (MAR)
        if missing_data.get("inferred_pattern") == "MAR":
             insights.append(RankedInsight(
                type="missing_pattern",
                variable="Dataset",
                description="Missing Not Random (MAR) detected - missingness correlates with other variables",
                score=0.88, # High importance warning
                evidence={"correlations": missing_data.get("missing_correlations")}
            ))

    # 3. Processing Outliers
    # -------------------------------------------------------
    outliers = analysis_results.get("outliers", {})
    if outliers:
        for col, details in outliers.items():
            count = details.get("count", 0)
            pct = details.get("percentage", 0)
            
            # Outliers are interesting if they are rare but present (1-5%)
            # If 40% of data is "outliers", the distribution is just skewed/heavy-tailed (lower score)
            if 0.1 < pct < 5.0: 
                score = 0.80 # True anomalies
            elif pct >= 5.0:
                score = 0.60 # Skewed distribution
            else:
                score = 0.50 # Negligible
                
            insights.append(RankedInsight(
                type="outlier",
                variable=col,
                description=f"{count} outliers detected ({pct}%)",
                score=score,
                evidence=details
            ))

    # 4. Processing Time Series Trends
    # -------------------------------------------------------
    ts_data = analysis_results.get("time_series_analysis", {})
    if ts_data and ts_data.get("has_time_series"):
        for col, analysis in ts_data.get("analyses", {}).items():
            trend = analysis.get("trend", {})
            if trend.get("detected"):
                strength = trend.get("strength_score", 0.5) # Assuming strength score exists or default
                direction = trend.get("direction")
                
                insights.append(RankedInsight(
                    type="trend",
                    variable=col,
                    description=f"{direction.title()} trend detected over time",
                    score=0.75 + (strength * 0.2), # Boost score by strength
                    evidence=trend
                ))

    # Sort by score descending
    insights.sort(key=lambda x: x.score, reverse=True)
    
    return insights
