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
    type: str          # e.g., 'driver', 'cohort', 'correlation', 'outlier', 'missing_pattern', 'trend'
    title: str         # Bold executive heading (e.g. "Primary Business Driver")
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
    Invisible ML algorithms power the statistical findings without exposing technical jargon.
    """
    insights: List[RankedInsight] = []

    # 1. Processing Key Business Drivers (Invisible Supervised ML)
    # -------------------------------------------------------
    supervised = analysis_results.get("supervised_learning")
    if supervised and isinstance(supervised, dict) and supervised.get("ran"):
        target_col = supervised.get("target_column", "")
        key_drivers = supervised.get("key_drivers", [])
        target_clean = _humanize_col(target_col)

        for i, driver in enumerate(key_drivers[:3]):
            feat = driver.get("feature", "")
            imp_pct = driver.get("importance_pct", 0.0)
            direction = driver.get("direction", "positive")
            feat_clean = _humanize_col(feat)

            # High importance threshold
            if imp_pct >= 8.0:
                score = min(0.98, 0.90 + (imp_pct / 100.0) * 0.10 - (i * 0.03))
                if direction == "negative":
                    impact_text = f"an inverse relationship: increases in {feat_clean} are associated with lower {target_clean}"
                else:
                    impact_text = f"a positive relationship: higher {feat_clean} strongly amplifies {target_clean}"

                desc = (
                    f"Multivariate outcome analysis identifies '{feat_clean}' as a primary business driver for '{target_clean}', "
                    f"accounting for {imp_pct:.1f}% of overall outcome variance with {impact_text}."
                )
                rec = f"Prioritize operational initiatives that optimize '{feat_clean}' to systematically improve '{target_clean}'."
                title = f"Primary Driver: {feat_clean} Directly Impacts {target_clean}"

                insights.append(RankedInsight(
                    type="driver",
                    title=title,
                    variable=f"{target_col} driven by {feat}",
                    description=desc,
                    score=score,
                    actionable_recommendation=rec,
                    evidence={"target": target_col, "driver": feat, "importance_pct": imp_pct, "direction": direction}
                ))

    # 2. Processing Operational Cohorts (Invisible Unsupervised ML)
    # -------------------------------------------------------
    unsupervised = analysis_results.get("unsupervised_learning")
    if unsupervised and isinstance(unsupervised, dict) and unsupervised.get("ran"):
        k = unsupervised.get("optimal_k", 0)
        var_pct = unsupervised.get("explained_variance_pct", 0.0)
        personas = unsupervised.get("personas", [])

        if personas and k >= 2:
            top_persona = personas[0]
            desc = (
                f"Records naturally segment into {k} cohesive operational cohorts capturing {var_pct:.0f}% of cross-variable variance. "
                f"The leading group ('{top_persona.get('name', 'Tier 1')}') represents {top_persona.get('share_pct', 0.0):.1f}% of total data: "
                f"{top_persona.get('summary', '')}."
            )
            rec = "Calibrate resource allocation and service tiers to align with these distinct operational cohorts rather than applying blanket policies."
            title = f"Natural Cohort Structure: {k} Distinct Operational Segments Identified"

            insights.append(RankedInsight(
                type="cohort",
                title=title,
                variable="Dataset Population",
                description=desc,
                score=0.88,
                actionable_recommendation=rec,
                evidence={"optimal_k": k, "explained_variance_pct": var_pct, "personas_count": len(personas)}
            ))

    # 3. Processing Strong Correlations
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
            
            score = r_val
            
            if direction == "positive":
                desc = (f"{col_a_clean} and {col_b_clean} move closely together: "
                        f"when one increases, the other follows proportionally. "
                        f"This {strength} alignment suggests they are operationally linked.")
                rec = (f"Leverage the synergy between {col_a_clean} and {col_b_clean} "
                       f"in forecasting and resource allocation.")
                title = f"{col_a_clean} & {col_b_clean} Synergy"
            else:
                desc = (f"{col_a_clean} and {col_b_clean} move in opposite directions: "
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

    # 4. Processing Missing Value Patterns
    # -------------------------------------------------------
    missing_data = analysis_results.get("missing_patterns", {})
    if missing_data and missing_data.get("has_missing"):
        for col, details in missing_data.get("column_details", {}).items():
            pct = details.get("percentage", 0)
            col_clean = _humanize_col(col)
            
            if pct > 99:
                score = 0.95
            elif pct > 50:
                score = 0.85
            elif pct > 20:
                score = 0.70
            else:
                score = 0.40
            
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
            
        if missing_data.get("inferred_pattern") == "MAR":
             insights.append(RankedInsight(
                type="missing_pattern",
                title="Systematic Data Collection Gap",
                variable="Dataset",
                description=("Missing values in this dataset are not random: they correlate "
                             "with other variables, suggesting a systematic data collection "
                             "or reporting gap that could skew business conclusions."),
                score=0.88,
                actionable_recommendation="Audit the data pipeline to identify why certain records are systematically incomplete.",
                evidence={"correlations": missing_data.get("missing_correlations")}
            ))

    # 5. Processing Outliers
    # -------------------------------------------------------
    outliers = analysis_results.get("outliers", {})
    if outliers:
        for col, details in outliers.items():
            count = details.get("count", 0)
            pct = details.get("percentage", 0)
            col_clean = _humanize_col(col)
            
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

    # 6. Processing Time Series Trends & Structural Changepoints (Invisible Kats ML)
    # -------------------------------------------------------
    ts_data = analysis_results.get("time_series_analysis", {})
    if ts_data and ts_data.get("has_time_series"):
        # Check for structural shift / drift
        drift_events = ts_data.get("drift_detected", [])
        for drift in drift_events[:2]:
            col = drift.get("column", "")
            shift_pct = drift.get("shift_pct", 0.0)
            col_clean = _humanize_col(col)
            if abs(shift_pct) >= 10.0:
                direction_word = "increased" if shift_pct > 0 else "decreased"
                desc = (
                    f"Statistical changepoint testing pinpoints a structural regime shift in '{col_clean}': "
                    f"historical average {direction_word} by {abs(shift_pct):.1f}% across partitions and stabilized."
                )
                rec = f"Align quarterly reviews with this transition point to verify underlying business catalysts for {col_clean}."
                title = f"Structural Trend Shift Detected in {col_clean}"
                insights.append(RankedInsight(
                    type="trend",
                    title=title,
                    variable=col,
                    description=desc,
                    score=0.91,
                    actionable_recommendation=rec,
                    evidence=drift
                ))

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
                    rec = (f"Capitalize on the upward momentum in {col_clean}: consider scaling operations "
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
