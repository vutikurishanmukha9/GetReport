from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import polars as pl
from app.services.analysis_config import AnalysisConfig
from app.services.insight_ranking import rank_insights

# Import modularized components
from app.services.analysis.validation import validate_input, EmptyDatasetError, InsufficientDataError, AnalysisError
from app.services.analysis.classification import classify_numeric_columns
from app.services.analysis.statistics import compute_summary, compute_correlation
from app.services.analysis.outliers import detect_outliers
from app.services.analysis.time_series import analyze_time_series
from app.services.analysis.missing import analyze_missing_patterns

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    metadata:                  dict[str, Any]                  = field(default_factory=dict)
    summary:                   dict[str, dict[str, float]]     = field(default_factory=dict)
    correlation:               dict[str, dict[str, float]]     = field(default_factory=dict)
    strong_correlations:       list[dict[str, Any]]            = field(default_factory=list)
    outliers:                  dict[str, dict[str, Any]]       = field(default_factory=dict)
    categorical_distribution:  dict[str, dict[str, Any]]       = field(default_factory=dict)
    column_quality_flags:      dict[str, list[str]]            = field(default_factory=dict)
    ranked_insights:           list[dict[str, Any]]            = field(default_factory=list)
    timing_ms:                 float                           = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata":                  self.metadata,
            "summary":                   self.summary,
            "correlation":               self.correlation,
            "strong_correlations":       self.strong_correlations,
            "outliers":                  self.outliers,
            "categorical_distribution":  self.categorical_distribution,
            "column_quality_flags":      self.column_quality_flags,
            "ranked_insights":           self.ranked_insights,
            "timing_ms":                 round(self.timing_ms, 2),
        }

def analyze_dataset(df: pl.DataFrame, top_categories: int = 10, config: AnalysisConfig | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    validate_input(df)
    
    # Initialize Config if not provided
    if config is None:
        config = AnalysisConfig.default()
        logger.info("Using default AnalysisConfig")
    else:
        logger.info(f"Using provided AnalysisConfig for domain: {config.domain}")
    
    # Get all numeric columns
    all_numeric_cols = [c for c, t in df.schema.items() if t in (pl.Int64, pl.Float64, pl.Int32, pl.Float32)]
    cat_cols = [c for c in df.columns if c not in all_numeric_cols]
    
    # Semantic column classification - filter out IDs, dates, low-variance
    column_classification = classify_numeric_columns(df, all_numeric_cols)
    analytical_cols = column_classification["analytical"]
    excluded_cols = column_classification["excluded"]
    
    logger.info(f"Column classification: {len(analytical_cols)} analytical, {len(excluded_cols)} excluded")
    if excluded_cols:
        logger.info(f"Excluded from analysis: {excluded_cols} - Reasons: {column_classification['exclusion_reasons']}")
    
    metadata = {
        "total_rows": df.height,
        "total_columns": df.width,
        "numeric_columns": len(all_numeric_cols),
        "categorical_columns": len(cat_cols),
        "analytical_numeric_columns": len(analytical_cols),
        "excluded_columns": excluded_cols,
        "exclusion_reasons": column_classification["exclusion_reasons"]
    }
    
    # Use ONLY analytical columns for meaningful analysis
    summary_stats = compute_summary(df, analytical_cols)
    
    correlation = {}
    strong_pairs = []
    if config.enable_correlation:
        correlation, strong_pairs = compute_correlation(df, analytical_cols)
    
    outliers = {}
    if config.enable_outliers:
        outliers = detect_outliers(df, analytical_cols)
    
    # Tier 1: Time Series Analysis (still use all numeric for now, since it looks for datetime cols)
    time_series_analysis = {}
    if config.enable_time_series:
        time_series_analysis = analyze_time_series(df, analytical_cols)
    
    # Tier 1: Missing Value Patterns
    missing_patterns = analyze_missing_patterns(df)
    
    # Categorical Distrib (Top 10)
    cat_dist = {}
    for c in cat_cols:
        counts = df[c].value_counts(sort=True).head(top_categories)
        cats = {}
        for row in counts.iter_rows():
            val, cnt = row
            cats[str(val)] = {"count": cnt, "percentage": round(cnt/df.height*100, 2)}
            
        cat_dist[c] = {
            "categories": cats,
            "total_unique": df[c].n_unique(),
            "missing_pct": round(df[c].null_count()/df.height*100, 2)
        }
        
    # Tier 5: Insight Ranking
    # Aggregating all partial results to pass to ranking engine
    partial_results_for_ranking = {
        "strong_correlations": strong_pairs,
        "outliers": outliers,
        "missing_patterns": missing_patterns,
        "time_series_analysis": time_series_analysis
    }
    ranked_insights = rank_insights(partial_results_for_ranking)
    logger.info(f"Insight Ranking: Generated {len(ranked_insights)} insights")

    elapsed = (time.perf_counter() - start) * 1000
    
    # Build result with new fields
    result = AnalysisResult(
        metadata=metadata,
        summary=summary_stats,
        correlation=correlation,
        strong_correlations=strong_pairs,
        outliers=outliers,
        categorical_distribution=cat_dist,
        ranked_insights=[i.to_dict() for i in ranked_insights],
        timing_ms=elapsed
    ).to_dict()
    
    # Add Tier 1 enhancements
    result["time_series_analysis"] = time_series_analysis
    result["missing_patterns"] = missing_patterns
    
    # ─── Tier 1 + Tier 2: Pipeline Steps ─────────────────────────────────────────
    # Each step enriches `result` in-place. Only data errors are caught;
    # code bugs (ImportError, AttributeError) propagate immediately.
    from app.services.analysis_pipeline import run_pipeline
    pipeline_outcome = run_pipeline(df, result)
    if pipeline_outcome.failed:
        logger.warning(
            "Pipeline: %d step(s) failed: %s",
            len(pipeline_outcome.failed),
            [s.name for s in pipeline_outcome.failed],
        )
    
    return result
