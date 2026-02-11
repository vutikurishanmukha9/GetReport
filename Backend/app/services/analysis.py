from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import polars as pl
import numpy as np

# ─── Semantic Inference ───────────────────────────────────────────────────────
from app.services.semantic_inference import analyze_semantic_structure

# ─── Tier 1: Trust Foundation ─────────────────────────────────────────────────
from app.services.confidence_scoring import calculate_confidence_scores
from app.services.analysis_decisions import evaluate_analysis_decisions

# ─── Tier 2: Advanced Intelligence ────────────────────────────────────────────
from app.services.feature_engineering import analyze_feature_engineering
from app.services.smart_schema import analyze_smart_schema
from app.services.recommendations import generate_recommendations

# ─── Tier 5: Analysis Selection Engine ────────────────────────────────────────
from app.services.analysis_config import AnalysisConfig
from app.services.insight_ranking import rank_insights

# ─── Logger ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Constants (from centralized config — overridable via env vars) ──────────
from app.core.config import settings

IQR_LOWER_MULTIPLIER: float = settings.IQR_LOWER_MULTIPLIER
IQR_UPPER_MULTIPLIER: float = settings.IQR_UPPER_MULTIPLIER
CORRELATION_STRONG_THRESHOLD: float = settings.CORRELATION_STRONG_THRESHOLD

SKEWNESS_THRESHOLD: float = settings.SKEWNESS_THRESHOLD

# ─── Semantic Column Detection Thresholds ─────────────────────────────────────
ID_UNIQUENESS_THRESHOLD: float = settings.ID_UNIQUENESS_THRESHOLD  # >98% unique values AND name suggests ID = likely ID
EXCEL_DATE_RANGE = (25569, 73050)  # Excel serial dates: 1970-2100
# Only very strong ID patterns - avoid false positives like "student_number" which is valid data
ID_COLUMN_PATTERNS = ['_id', 'uuid', 'guid', 'pk', 'primary_key', 'row_id', 'record_id']
DATE_COLUMN_PATTERNS = ['date', 'time', 'timestamp', 'created_at', 'updated_at', 'modified_at']

# ─── Custom Exceptions ───────────────────────────────────────────────────────
class EmptyDatasetError(ValueError): pass
class InsufficientDataError(ValueError): pass
class AnalysisError(RuntimeError): pass

# ─── Semantic Column Classifier ──────────────────────────────────────────────
def _classify_numeric_columns(df: pl.DataFrame, numeric_cols: list[str]) -> dict[str, list[str]]:
    """
    Classify numeric columns into semantic categories:
    - analytical: Real numeric data suitable for correlation/distribution (e.g., sales, price)
    - id_like: Likely identifiers (high uniqueness, sequential) - exclude from analysis
    - date_like: Likely Excel serial dates (values in 25569-73050 range) - exclude from analysis
    - low_variance: Near-constant values - exclude from correlation
    
    Returns dict with 'analytical', 'excluded', 'exclusion_reasons'
    """
    analytical = []
    excluded = []
    exclusion_reasons = {}
    
    for col in numeric_cols:
        col_lower = col.lower()
        reasons = []
        
        # Check 1: Column name patterns
        is_id_name = any(pattern in col_lower for pattern in ID_COLUMN_PATTERNS)
        is_date_name = any(pattern in col_lower for pattern in DATE_COLUMN_PATTERNS)
        
        if is_id_name:
            reasons.append("name_suggests_id")
        if is_date_name:
            reasons.append("name_suggests_date")
        
        # Check 2: High uniqueness = likely ID
        try:
            n_unique = df[col].n_unique()
            uniqueness_ratio = n_unique / df.height if df.height > 0 else 0
            
            if uniqueness_ratio >= ID_UNIQUENESS_THRESHOLD:
                reasons.append(f"high_uniqueness_{round(uniqueness_ratio*100)}%")
        except:
            pass
        
        # Check 3: Excel serial date range detection
        try:
            non_null = df[col].drop_nulls()
            if non_null.len() > 0:
                min_val = non_null.min()
                max_val = non_null.max()
                
                # Excel serial date range check (1970-2100)
                if EXCEL_DATE_RANGE[0] <= min_val <= EXCEL_DATE_RANGE[1] and \
                   EXCEL_DATE_RANGE[0] <= max_val <= EXCEL_DATE_RANGE[1]:
                    # Additional check: values are mostly integers and within date range spread
                    int_ratio = (non_null == non_null.round(0)).sum() / non_null.len()
                    if int_ratio > 0.9:  # 90% integer values
                        reasons.append("excel_serial_date_range")
        except:
            pass
        
        # Check 4: Low variance (near-constant)
        try:
            std = df[col].std()
            mean = df[col].mean()
            if std is not None and mean is not None and mean != 0:
                cv = abs(std / mean)  # Coefficient of variation
                if cv < 0.01:  # <1% variation
                    reasons.append("low_variance")
        except:
            pass
        
        # Decision: Be conservative - only exclude if:
        # 1. Low variance (always exclude near-constants), OR
        # 2. BOTH name pattern AND data evidence (high uniqueness or date range)
        should_exclude = False
        
        if "low_variance" in reasons:
            should_exclude = True  # Low variance is always excluded
        elif (is_id_name or is_date_name):
            # Name suggests ID/date - need additional data evidence
            has_data_evidence = any(
                r for r in reasons 
                if "high_uniqueness" in r or "excel_serial_date" in r
            )
            if has_data_evidence:
                should_exclude = True
        
        if should_exclude:
            excluded.append(col)
            exclusion_reasons[col] = reasons
        else:
            analytical.append(col)
    
    return {
        "analytical": analytical,
        "excluded": excluded,
        "exclusion_reasons": exclusion_reasons
    }

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

def _validate_input(df: pl.DataFrame) -> None:
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected pl.DataFrame, got {type(df)}")
    if df.height == 0 or df.width == 0:
        raise EmptyDatasetError("Empty dataset")

def _compute_summary(df: pl.DataFrame, numeric_cols: list[str]) -> dict[str, dict[str, float]]:
    if not numeric_cols: return {}
    
    # Polars describe() gives stats for all numeric cols
    stats = df.select(numeric_cols).describe()
    # Convert to dict format expected by frontend: {col: {mean: x, std: y...}}
    
    # This is a bit manual in Polars to match Pandas output structure exactly.
    # We will compute aggregations manually for precision and structure.
    
    result = {}
    for col in numeric_cols:
        col_stats = df.select([
            pl.col(col).mean().alias("mean"),
            pl.col(col).std().alias("std"),
            pl.col(col).min().alias("min"),
            pl.col(col).max().alias("max"),
            pl.col(col).median().alias("50%"),
            pl.col(col).quantile(0.25).alias("25%"),
            pl.col(col).quantile(0.75).alias("75%"),
            pl.col(col).skew().alias("skewness"),
            pl.col(col).kurtosis().alias("kurtosis")
        ]).to_dict(as_series=False)
        
        # Unwrap lists
        single_stats = {k: (v[0] if v[0] is not None else 0.0) for k, v in col_stats.items()}
        result[col] = single_stats
        
    return result

def _compute_correlation(df: pl.DataFrame, numeric_cols: list[str]):
    if len(numeric_cols) < 2: return {}, []
    
    # Optimized: Vectorized Correlation Matrix
    if len(numeric_cols) < 2: return {}, []

    try:
        # Convert to numpy (Zero Copy if possible, but drops nulls for safety)
        # We drop rows with nulls in ANY of the target columns to ensure valid correlation
        # This is standard behavior for correlation matrices (listwise deletion)
        
        # Selecting columns and dropping nulls
        clean_df = df.select(numeric_cols).drop_nulls()
        
        if clean_df.height < 2:
            return {}, [] # Not enough data
            
        data_matrix = clean_df.to_numpy().T # Transpose for np.corrcoef (expects variables as rows)
        
        # Compute Matrix
        corr_matrix = np.corrcoef(data_matrix)
        
        # Map back to dictionary
        corr_dict = {c: {} for c in numeric_cols}
        strong_pairs = []
        
        for i, col_a in enumerate(numeric_cols):
            # Self correlation
            corr_dict[col_a][col_a] = 1.0
            
            for j in range(i + 1, len(numeric_cols)):
                col_b = numeric_cols[j]
                val = float(corr_matrix[i, j])
                
                # Handle NaN (constant columns)
                if np.isnan(val): val = 0.0
                
                corr_dict[col_a][col_b] = val
                corr_dict[col_b][col_a] = val
                
                if abs(val) >= CORRELATION_STRONG_THRESHOLD:
                    strong_pairs.append({
                        "column_a": col_a,
                        "column_b": col_b,
                        "r_value": round(val, 4),
                        "direction": "positive" if val > 0 else "negative",
                        "strength": "very strong" if abs(val) >= 0.9 else "strong"
                    })
                    
        return corr_dict, strong_pairs
        
    except Exception as e:
        logger.error(f"Vectorized correlation failed: {e}")
        return {}, []

def _detect_outliers(df: pl.DataFrame, numeric_cols: list[str]) -> dict:
    outliers = {}
    for col in numeric_cols:
        q1 = df.select(pl.col(col).quantile(0.25)).item()
        q3 = df.select(pl.col(col).quantile(0.75)).item()
        
        if q1 is None or q3 is None: continue
        
        iqr = q3 - q1
        if iqr == 0: continue
        
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        # Filter
        outlier_rows = df.filter((pl.col(col) < lower) | (pl.col(col) > upper))
        count = outlier_rows.height
        
        if count > 0:
            vals = outlier_rows[col].head(20).to_list()
            outliers[col] = {
                "count": count,
                "percentage": round(count / df.height * 100, 2),
                "min_outlier": outlier_rows[col].min(),
                "max_outlier": outlier_rows[col].max(),
                "lower_bound": lower,
                "upper_bound": upper,
                "sample_values": vals
            }
    return outliers

# ─── Tier 1 Enhancement: Time Series Detection ───────────────────────────────
def _detect_time_columns(df: pl.DataFrame) -> list[str]:
    """Identify datetime columns in the DataFrame."""
    return [c for c, t in df.schema.items() if t in (pl.Date, pl.Datetime)]

def _detect_trend(df: pl.DataFrame, time_col: str, value_col: str) -> dict[str, Any]:
    """
    Detect trend using linear regression slope.
    Returns trend direction, strength, and p-value approximation.
    """
    try:
        # Sort by time
        sorted_df = df.select([time_col, value_col]).drop_nulls().sort(time_col)
        if sorted_df.height < 10:
            return {"detected": False, "reason": "Insufficient data points"}
        
        # Create numeric time index
        y = sorted_df[value_col].to_numpy()
        x = np.arange(len(y))
        
        # Linear regression
        n = len(x)
        sum_x, sum_y = x.sum(), y.sum()
        sum_xy = (x * y).sum()
        sum_x2 = (x ** 2).sum()
        
        denom = n * sum_x2 - sum_x ** 2
        if denom == 0:
            return {"detected": False, "reason": "Constant values"}
        
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        
        # R-squared
        y_pred = slope * x + intercept
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Trend direction
        if abs(slope) < 1e-10:
            direction = "flat"
        elif slope > 0:
            direction = "upward"
        else:
            direction = "downward"
        
        # Strength classification
        if r_squared >= 0.7:
            strength = "strong"
        elif r_squared >= 0.3:
            strength = "moderate"
        else:
            strength = "weak"
        
        return {
            "detected": True,
            "direction": direction,
            "slope": round(float(slope), 6),
            "r_squared": round(float(r_squared), 4),
            "strength": strength,
            "data_points": n
        }
    except Exception as e:
        logger.warning(f"Trend detection failed: {e}")
        return {"detected": False, "reason": str(e)}

def _detect_seasonality(df: pl.DataFrame, time_col: str, value_col: str) -> dict[str, Any]:
    """
    Detect seasonality using autocorrelation at common lags (7=weekly, 30=monthly, 365=yearly).
    """
    try:
        sorted_df = df.select([time_col, value_col]).drop_nulls().sort(time_col)
        y = sorted_df[value_col].to_numpy()
        n = len(y)
        
        if n < 60:  # Need enough data for seasonality
            return {"detected": False, "reason": "Insufficient data for seasonality analysis"}
        
        # Detrend (subtract mean)
        y_detrend = y - y.mean()
        
        # Check common seasonal lags
        seasonal_lags = {7: "weekly", 30: "monthly", 90: "quarterly", 365: "yearly"}
        detected_patterns = []
        
        for lag, period_name in seasonal_lags.items():
            if n < lag * 2:
                continue
            
            # Calculate autocorrelation at this lag
            autocorr = np.corrcoef(y_detrend[:-lag], y_detrend[lag:])[0, 1]
            
            if np.isnan(autocorr):
                continue
            
            # Strong autocorrelation suggests seasonality
            if abs(autocorr) >= 0.3:
                detected_patterns.append({
                    "period": period_name,
                    "lag": lag,
                    "autocorrelation": round(float(autocorr), 4),
                    "strength": "strong" if abs(autocorr) >= 0.6 else "moderate"
                })
        
        if detected_patterns:
            return {
                "detected": True,
                "patterns": detected_patterns,
                "primary_pattern": detected_patterns[0]["period"]
            }
        else:
            return {"detected": False, "reason": "No significant seasonal patterns found"}
            
    except Exception as e:
        logger.warning(f"Seasonality detection failed: {e}")
        return {"detected": False, "reason": str(e)}

def _analyze_time_series(df: pl.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
    """
    Full time series analysis: trend + seasonality for each numeric column.
    """
    time_cols = _detect_time_columns(df)
    if not time_cols:
        return {"has_time_series": False, "reason": "No datetime columns found"}
    
    time_col = time_cols[0]  # Use first datetime column
    results = {
        "has_time_series": True,
        "time_column": time_col,
        "analyses": {}
    }
    
    # Analyze top 5 numeric columns
    for col in numeric_cols[:5]:
        trend = _detect_trend(df, time_col, col)
        seasonality = _detect_seasonality(df, time_col, col)
        
        results["analyses"][col] = {
            "trend": trend,
            "seasonality": seasonality
        }
    
    return results

# ─── Tier 1 Enhancement: Missing Value Pattern Analysis ─────────────────────
def _analyze_missing_patterns(df: pl.DataFrame) -> dict[str, Any]:
    """
    Analyze missing value patterns to detect MCAR, MAR, or MNAR.
    
    MCAR (Missing Completely At Random): No pattern - safe to impute/drop
    MAR (Missing At Random): Related to other variables - use conditional imputation
    MNAR (Missing Not At Random): Related to the value itself - complex handling needed
    """
    missing_info = {}
    cols_with_missing = []
    
    # Step 1: Calculate missing rates
    for col in df.columns:
        null_count = df[col].null_count()
        if null_count > 0:
            missing_rate = null_count / df.height
            cols_with_missing.append(col)
            missing_info[col] = {
                "count": null_count,
                "percentage": round(missing_rate * 100, 2),
                "severity": "critical" if missing_rate > 0.5 else ("high" if missing_rate > 0.2 else ("medium" if missing_rate > 0.05 else "low"))
            }
    
    if not cols_with_missing:
        return {"has_missing": False, "message": "No missing values detected"}
    
    # Step 2: Detect missing value correlations (MAR indicator)
    missing_correlations = []
    for col in cols_with_missing[:5]:  # Limit for performance
        # Create binary missing indicator
        missing_mask = df[col].is_null().cast(pl.Int32)
        
        # Check correlation with other numeric columns
        numeric_cols = [c for c, t in df.schema.items() if t in (pl.Int64, pl.Float64, pl.Int32, pl.Float32) and c != col]
        
        for other_col in numeric_cols[:5]:
            try:
                # Correlation between missing indicator and other variable
                corr = df.select(pl.corr(missing_mask.alias("_missing"), pl.col(other_col))).item()
                if corr is not None and not np.isnan(corr) and abs(corr) >= 0.2:
                    missing_correlations.append({
                        "missing_column": col,
                        "correlated_with": other_col,
                        "correlation": round(float(corr), 4),
                        "interpretation": f"Missing values in '{col}' may be related to '{other_col}'"
                    })
            except:
                pass
    
    # Step 3: Detect row patterns (multiple missing in same rows)
    # Count how many columns are missing per row
    missing_per_row = df.select([pl.col(c).is_null().cast(pl.Int32).alias(c) for c in cols_with_missing])
    row_missing_sum = missing_per_row.select(pl.sum_horizontal(pl.all())).to_series()  # Convert to Series
    
    # Distribution of missing counts
    fully_complete = (row_missing_sum == 0).sum()
    partial_missing = ((row_missing_sum > 0) & (row_missing_sum < len(cols_with_missing))).sum()
    fully_missing = (row_missing_sum == len(cols_with_missing)).sum()
    
    row_patterns = {
        "complete_rows": int(fully_complete),
        "partial_missing_rows": int(partial_missing),
        "fully_missing_rows": int(fully_missing)
    }
    
    # Step 4: Infer pattern type
    if missing_correlations:
        pattern_type = "MAR"
        pattern_advice = "Missing values appear related to other variables. Consider multiple imputation or conditional mean imputation."
    elif row_patterns["fully_missing_rows"] > df.height * 0.1:
        pattern_type = "Systematic"
        pattern_advice = "Many rows have all values missing. Consider removing these rows entirely."
    else:
        pattern_type = "MCAR"
        pattern_advice = "Missing values appear random. Safe to use mean/median imputation or listwise deletion."
    
    return {
        "has_missing": True,
        "columns_affected": len(cols_with_missing),
        "column_details": missing_info,
        "missing_correlations": missing_correlations[:5],  # Top 5
        "row_patterns": row_patterns,
        "inferred_pattern": pattern_type,
        "recommendation": pattern_advice
    }

def analyze_dataset(df: pl.DataFrame, top_categories: int = 10, config: AnalysisConfig | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    _validate_input(df)
    
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
    column_classification = _classify_numeric_columns(df, all_numeric_cols)
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
    summary_stats = _compute_summary(df, analytical_cols)
    
    correlation = {}
    strong_pairs = []
    if config.enable_correlation:
        correlation, strong_pairs = _compute_correlation(df, analytical_cols)
    
    outliers = {}
    if config.enable_outliers:
        outliers = _detect_outliers(df, analytical_cols)
    
    # Tier 1: Time Series Analysis (still use all numeric for now, since it looks for datetime cols)
    time_series_analysis = {}
    if config.enable_time_series:
        time_series_analysis = _analyze_time_series(df, analytical_cols)
    
    # Tier 1: Missing Value Patterns
    missing_patterns = _analyze_missing_patterns(df)
    
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
    
    # ─── Tier 1: Column Confidence Scores ─────────────────────────────────────
    try:
        confidence_report = calculate_confidence_scores(df)
        result["confidence_scores"] = confidence_report.to_dict()
        logger.info(f"Confidence scoring: Dataset grade={confidence_report._get_dataset_grade()}, "
                    f"{confidence_report.high_confidence_count} high/{confidence_report.low_confidence_count} low confidence columns")
    except Exception as e:
        logger.warning(f"Confidence scoring failed: {e}")
        result["confidence_scores"] = None
    
    # ─── Tier 1: Analysis Decisions (Why I Did X) ─────────────────────────────
    try:
        decision_log = evaluate_analysis_decisions(df)
        result["analysis_decisions"] = decision_log.to_dict()
        ran_count = decision_log.to_dict()["summary"]["ran"]
        skipped_count = decision_log.to_dict()["summary"]["skipped"]
        logger.info(f"Analysis decisions: {ran_count} ran, {skipped_count} skipped")
    except Exception as e:
        logger.warning(f"Analysis decisions failed: {e}")
        result["analysis_decisions"] = None
    
    # Semantic Column Intelligence
    try:
        semantic_analysis = analyze_semantic_structure(df)
        result["semantic_analysis"] = semantic_analysis.to_dict()
        logger.info(f"Semantic analysis: Domain={semantic_analysis.domain.primary_domain} "
                    f"({semantic_analysis.domain.confidence*100:.0f}% confidence), "
                    f"{len(semantic_analysis.analytical_columns)} analytical cols, "
                    f"{len(semantic_analysis.suggested_pairs)} suggestions")
    except Exception as e:
        logger.warning(f"Semantic analysis failed: {e}")
        result["semantic_analysis"] = None
    
    # ─── Tier 2: Advanced Intelligence ────────────────────────────────────────
    
    # Feature Engineering Recommendations
    try:
        column_roles = {}
        if result.get("semantic_analysis") and result["semantic_analysis"].get("column_roles"):
            column_roles = {k: v.get("role", "") for k, v in result["semantic_analysis"]["column_roles"].items()}
        
        fe_result = analyze_feature_engineering(df, column_roles)
        result["feature_engineering"] = fe_result.to_dict()
        logger.info(f"Feature engineering: {len(fe_result.encoding_recommendations)} encoding, "
                    f"{len(fe_result.scaling_recommendations)} scaling suggestions")
    except Exception as e:
        logger.warning(f"Feature engineering analysis failed: {e}")
        result["feature_engineering"] = None
    
    # Smart Schema Inference
    try:
        schema_result = analyze_smart_schema(df)
        result["smart_schema"] = schema_result.to_dict()
        logger.info(f"Smart schema: {len(schema_result.type_corrections)} corrections, "
                    f"{len(schema_result.relationships)} relationships")
    except Exception as e:
        logger.warning(f"Smart schema analysis failed: {e}")
        result["smart_schema"] = None
    
    # Actionable Recommendations
    try:
        domain = "unknown"
        if result.get("semantic_analysis") and result["semantic_analysis"].get("domain"):
            domain = result["semantic_analysis"]["domain"].get("primary", "unknown")
        
        rec_result = generate_recommendations(df, domain, result)
        result["recommendations"] = rec_result.to_dict()
        high_priority = rec_result.get_high_priority()
        logger.info(f"Recommendations: {rec_result.to_dict()['total_count']} total, "
                    f"{len(high_priority)} high priority")
    except Exception as e:
        logger.warning(f"Recommendations generation failed: {e}")
        result["recommendations"] = None
    
    return result