from dataclasses import dataclass, field
from typing import Any, List, Dict
import logging

logger = logging.getLogger(__name__)


def _humanize_col(col_name: str) -> str:
    """Convert snake_case column names to readable Title Case."""
    if not col_name:
        return "Unknown"
    return col_name.replace('_', ' ').replace('-', ' ').title()


@dataclass
class RankedInsight:
    """
    A normalized, scored finding from the analysis.
    Used to surface the most important "signals" to the user/AI.
    """
    type: str          # e.g., 'correlation', 'outlier', 'missing_pattern', 'trend'
    title: str         # Bold executive heading (e.g. "Revenue & Profit Alignment")
    variable: str      # The primary column involved
    description: str   # Executive-readable business narrative
    score: float       # 0.0 to 1.0 (1.0 = most critical)
    actionable_recommendation: str = ""  # Strategic recommendation for leadership
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "title": self.title,
            "variable": self.variable,
            "description": self.description,
            "score": round(self.score, 2),
            "actionable_recommendation": self.actionable_recommendation,
            "evidence": self.evidence
        }

def rank_insights(analysis_results: Dict[str, Any]) -> List[RankedInsight]:
    """
    Extracts findings from raw analysis results, scores them, and returns a sorted list.
    All descriptions are written for non-technical business stakeholders.
    """
    insights: List[RankedInsight] = []
    
    # 1. Processing Strong Correlations
    # -------------------------------------------------------
    correlations = analysis_results.get("strong_correlations", [])
    if correlations:
        for corr in correlations:
            r_val = abs(corr.get("r_value", 0))
            col_a = corr.get("column_a", "")
            col_b = corr.get("column_b", "")
            direction = corr.get("direction", "positive" if corr.get("r_value", 0) > 0 else "negative")
            strength = corr.get("strength", "strong")
            
            col_a_clean = _humanize_col(col_a)
            col_b_clean = _humanize_col(col_b)
            
            # Score: direct mapping of r_value (0.7 to 1.0)
            score = r_val
            
            # Executive narrative
            if direction == "positive":
                desc = (f"{col_a_clean} and {col_b_clean} move closely together — "
                        f"when one increases, the other follows proportionally. "
                        f"This {strength} alignment suggests they are operationally linked.")
                rec = (f"Leverage the synergy between {col_a_clean} and {col_b_clean} "
                       f"in forecasting and resource allocation.")
                title = f"{col_a_clean} & {col_b_clean} Synergy"
            else:
                desc = (f"{col_a_clean} and {col_b_clean} move in opposite directions — "
                        f"as one grows, the other tends to decline. "
                        f"This inverse relationship may indicate a trade-off or constraint.")
                rec = (f"Investigate whether the trade-off between {col_a_clean} and "
                       f"{col_b_clean} can be optimized.")
                title = f"{col_a_clean} vs {col_b_clean} Trade-off"
            
            insights.append(RankedInsight(
                type="correlation",
                title=title,
                variable=f"{col_a} & {col_b}",
                description=desc,
                score=score,
                actionable_recommendation=rec,
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
            col_clean = _humanize_col(col)
            
            # Score
            if pct > 99:
                score = 0.95
            elif pct > 50:
                score = 0.85
            elif pct > 20:
                score = 0.70
            else:
                score = 0.40
            
            # Executive narrative
            if pct > 50:
                desc = (f"More than half ({pct:.0f}%) of values are missing in '{col_clean}'. "
                        f"This significantly limits the reliability of any analysis involving this field.")
                rec = f"Evaluate whether '{col_clean}' should be excluded or requires upstream data collection improvements."
                title = f"Critical Data Gap in {col_clean}"
            elif pct > 20:
                desc = (f"'{col_clean}' has {pct:.0f}% missing values, which may introduce bias "
                        f"into reports and dashboards that rely on this field.")
                rec = f"Consider imputation strategies or flag this column in downstream reporting."
                title = f"Notable Missing Data in {col_clean}"
            else:
                desc = (f"'{col_clean}' has a minor data gap ({pct:.1f}% missing), "
                        f"unlikely to materially affect analysis but worth monitoring.")
                rec = ""
                title = f"Minor Data Gap in {col_clean}"
                
            insights.append(RankedInsight(
                type="data_quality",
                title=title,
                variable=col,
                description=desc,
                score=score,
                actionable_recommendation=rec,
                evidence=details
            ))
            
        # Systematic patterns (MAR)
        if missing_data.get("inferred_pattern") == "MAR":
             insights.append(RankedInsight(
                type="missing_pattern",
                title="Systematic Data Collection Gap",
                variable="Dataset",
                description=("Missing values in this dataset are not random — they correlate "
                             "with other variables, suggesting a systematic data collection "
                             "or reporting gap that could skew business conclusions."),
                score=0.88,
                actionable_recommendation="Audit the data pipeline to identify why certain records are systematically incomplete.",
                evidence={"correlations": missing_data.get("missing_correlations")}
            ))

    # 3. Processing Outliers
    # -------------------------------------------------------
    outliers = analysis_results.get("outliers", {})
    if outliers:
        for col, details in outliers.items():
            count = details.get("count", 0)
            pct = details.get("percentage", 0)
            col_clean = _humanize_col(col)
            
            # Outliers are interesting if they are rare but present (1-5%)
            if 0.1 < pct < 5.0: 
                score = 0.80
                desc = (f"{count} unusual data points flagged in '{col_clean}' ({pct:.1f}% of records). "
                        f"These may represent exceptional transactions, data entry errors, "
                        f"or genuinely high/low-value events worth investigating.")
                rec = f"Review the flagged records in '{col_clean}' to determine if they are valid edge cases or errors."
                title = f"Anomalies Detected in {col_clean}"
            elif pct >= 5.0:
                score = 0.60
                desc = (f"'{col_clean}' shows a wide value spread with {count} data points "
                        f"({pct:.1f}%) falling outside the typical range. This suggests a "
                        f"naturally skewed distribution rather than data errors.")
                rec = f"Use robust statistical measures (median, IQR) instead of averages when reporting on '{col_clean}'."
                title = f"Wide Distribution in {col_clean}"
            else:
                score = 0.50
                desc = (f"'{col_clean}' has minimal variance outliers ({count} points, {pct:.1f}%), "
                        f"indicating a stable and well-bounded data distribution.")
                rec = ""
                title = f"Stable Distribution in {col_clean}"
                
            insights.append(RankedInsight(
                type="outlier",
                title=title,
                variable=col,
                description=desc,
                score=score,
                actionable_recommendation=rec,
                evidence=details
            ))

    # 4. Processing Time Series Trends
    # -------------------------------------------------------
    ts_data = analysis_results.get("time_series_analysis", {})
    if ts_data and ts_data.get("has_time_series"):
        for col, analysis in ts_data.get("analyses", {}).items():
            trend = analysis.get("trend", {})
            if trend.get("detected"):
                strength = trend.get("strength_score", 0.5)
                direction = trend.get("direction", "upward")
                p_value = trend.get("p_value")
                col_clean = _humanize_col(col)
                
                is_significant = p_value is not None and p_value < 0.05
                
                if direction and direction.lower() in ("upward", "up", "increasing"):
                    desc = (f"{col_clean} shows a consistent growth trajectory over the evaluated period. "
                            f"This upward movement {'is statistically confirmed' if is_significant else 'warrants continued monitoring'}.")
                    rec = (f"Capitalize on the upward momentum in {col_clean} — consider scaling operations "
                           f"or adjusting targets to align with this growth.")
                    title = f"Growth Trajectory in {col_clean}"
                else:
                    desc = (f"{col_clean} is trending downward over the evaluated period. "
                            f"This decline {'is statistically confirmed' if is_significant else 'warrants attention'}.")
                    rec = (f"Investigate root causes of the declining {col_clean} and consider "
                           f"corrective interventions before the trend deepens.")
                    title = f"Declining Trend in {col_clean}"
                
                insights.append(RankedInsight(
                    type="trend",
                    title=title,
                    variable=col,
                    description=desc,
                    score=0.75 + (strength * 0.2),
                    actionable_recommendation=rec,
                    evidence=trend
                ))

    # Sort by score descending
    insights.sort(key=lambda x: x.score, reverse=True)
    
    return insights
