"""
ML-Ready Feature Engineering Service

Provides intelligent suggestions for preparing data for machine learning:
1. Encoding recommendations (one-hot, label, target encoding)
2. Scaling suggestions (Standard, MinMax, Robust)
3. Feature extraction ideas (date parts, text features, binning)
4. Feature importance proxy using correlation analysis
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import polars as pl
import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENCODING RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EncodingRecommendation:
    """Encoding suggestion for a categorical column."""
    column: str
    recommended_encoding: str  # one_hot, label, target, binary, hash
    reason: str
    n_unique: int
    considerations: list[str] = field(default_factory=list)


def _suggest_encoding(df: pl.DataFrame, col: str) -> EncodingRecommendation:
    """
    Suggest the best encoding strategy for a categorical column.
    
    Decision logic:
    - 2 categories → Binary encoding
    - 3-10 categories → One-hot encoding
    - 11-50 categories → Label encoding (ordinal) or Target encoding
    - 50+ categories → Hash encoding or Embeddings
    """
    n_unique = df[col].n_unique()
    null_pct = df[col].null_count() / df.height * 100
    
    considerations = []
    
    if null_pct > 0:
        considerations.append(f"Handle {null_pct:.1f}% missing values before encoding")
    
    if n_unique == 2:
        return EncodingRecommendation(
            column=col,
            recommended_encoding="binary",
            reason="Only 2 unique values - simple 0/1 encoding is optimal",
            n_unique=n_unique,
            considerations=considerations
        )
    
    if n_unique <= 10:
        considerations.append("Creates sparse features - may increase dimensionality")
        return EncodingRecommendation(
            column=col,
            recommended_encoding="one_hot",
            reason=f"Low cardinality ({n_unique} categories) - one-hot encoding preserves information",
            n_unique=n_unique,
            considerations=considerations
        )
    
    if n_unique <= 50:
        considerations.append("Consider target encoding if column correlates with target variable")
        considerations.append("Label encoding assumes ordinal relationship between categories")
        return EncodingRecommendation(
            column=col,
            recommended_encoding="label_or_target",
            reason=f"Medium cardinality ({n_unique} categories) - label or target encoding recommended",
            n_unique=n_unique,
            considerations=considerations
        )
    
    # High cardinality
    considerations.append("High cardinality may cause overfitting with one-hot")
    considerations.append("Consider embeddings for deep learning models")
    return EncodingRecommendation(
        column=col,
        recommended_encoding="hash_or_embedding",
        reason=f"High cardinality ({n_unique} categories) - use hash encoding or embeddings",
        n_unique=n_unique,
        considerations=considerations
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SCALING RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScalingRecommendation:
    """Scaling suggestion for a numeric column."""
    column: str
    recommended_scaler: str  # standard, minmax, robust, log, none
    reason: str
    stats: dict[str, float] = field(default_factory=dict)
    considerations: list[str] = field(default_factory=list)


def _suggest_scaling(df: pl.DataFrame, col: str) -> ScalingRecommendation:
    """
    Suggest the best scaling strategy for a numeric column.
    
    Decision logic:
    - High outliers (zscore > 3) → Robust Scaler
    - Right-skewed (skewness > 1) → Log transform then Standard
    - Bounded range (0-1, 0-100) → MinMax or None
    - Normal-ish → Standard Scaler
    """
    series = df[col].drop_nulls()
    
    if series.len() == 0:
        return ScalingRecommendation(
            column=col,
            recommended_scaler="none",
            reason="Column is empty or all nulls",
            stats={}
        )
    
    mean = series.mean()
    std = series.std()
    min_val = series.min()
    max_val = series.max()
    
    stats = {
        "mean": round(mean, 2) if mean else 0,
        "std": round(std, 2) if std else 0,
        "min": round(min_val, 2) if min_val else 0,
        "max": round(max_val, 2) if max_val else 0
    }
    
    considerations = []
    
    # Check for outliers using IQR
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    has_outliers = (series < (q1 - 1.5 * iqr)).sum() > 0 or (series > (q3 + 1.5 * iqr)).sum() > 0
    
    # Check skewness
    if std and std > 0:
        skewness = ((series - mean) ** 3).mean() / (std ** 3)
    else:
        skewness = 0
    
    # Check if already bounded
    is_percentage = min_val >= 0 and max_val <= 100
    is_normalized = min_val >= 0 and max_val <= 1
    
    if is_normalized:
        considerations.append("Data already appears normalized (0-1 range)")
        return ScalingRecommendation(
            column=col,
            recommended_scaler="none",
            reason="Data already in 0-1 range - no scaling needed",
            stats=stats,
            considerations=considerations
        )
    
    if has_outliers:
        considerations.append("Outliers detected - RobustScaler uses median and IQR")
        considerations.append("Alternative: clip outliers before StandardScaler")
        return ScalingRecommendation(
            column=col,
            recommended_scaler="robust",
            reason="Outliers present - RobustScaler recommended to reduce their influence",
            stats=stats,
            considerations=considerations
        )
    
    if skewness > 1:
        if min_val > 0:
            considerations.append("Apply log1p() then StandardScaler for best results")
            return ScalingRecommendation(
                column=col,
                recommended_scaler="log_then_standard",
                reason=f"Right-skewed (skewness={skewness:.2f}) - log transform recommended",
                stats=stats,
                considerations=considerations
            )
        else:
            considerations.append("Negative values prevent log transform - use Yeo-Johnson")
            return ScalingRecommendation(
                column=col,
                recommended_scaler="power_transform",
                reason=f"Skewed with negative values - PowerTransformer (Yeo-Johnson) recommended",
                stats=stats,
                considerations=considerations
            )
    
    if is_percentage:
        considerations.append("Divide by 100 to get 0-1 range if needed")
        return ScalingRecommendation(
            column=col,
            recommended_scaler="minmax",
            reason="Percentage data (0-100) - MinMaxScaler to normalize to 0-1",
            stats=stats,
            considerations=considerations
        )
    
    # Default: Standard Scaler
    considerations.append("Produces mean=0, std=1 distribution")
    return ScalingRecommendation(
        column=col,
        recommended_scaler="standard",
        reason="Approximately normal distribution - StandardScaler recommended",
        stats=stats,
        considerations=considerations
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION SUGGESTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureExtractionSuggestion:
    """Suggested new features to extract from existing columns."""
    source_column: str
    suggested_features: list[dict[str, str]]  # [{name, description, code_hint}]
    category: str  # datetime, text, numeric, interaction


def _suggest_datetime_features(col: str) -> FeatureExtractionSuggestion:
    """Suggest features to extract from datetime columns."""
    return FeatureExtractionSuggestion(
        source_column=col,
        suggested_features=[
            {"name": f"{col}_year", "description": "Year component", "code": f"df['{col}'].dt.year"},
            {"name": f"{col}_month", "description": "Month (1-12)", "code": f"df['{col}'].dt.month"},
            {"name": f"{col}_day", "description": "Day of month", "code": f"df['{col}'].dt.day"},
            {"name": f"{col}_dayofweek", "description": "Day of week (0=Mon)", "code": f"df['{col}'].dt.weekday"},
            {"name": f"{col}_is_weekend", "description": "Weekend flag", "code": f"df['{col}'].dt.weekday >= 5"},
            {"name": f"{col}_quarter", "description": "Quarter (1-4)", "code": f"df['{col}'].dt.quarter"},
            {"name": f"{col}_hour", "description": "Hour (if time present)", "code": f"df['{col}'].dt.hour"},
        ],
        category="datetime"
    )


def _suggest_text_features(df: pl.DataFrame, col: str) -> FeatureExtractionSuggestion:
    """Suggest features to extract from text columns."""
    sample = df[col].drop_nulls().head(100)
    
    features = [
        {"name": f"{col}_length", "description": "Character count", "code": f"df['{col}'].str.len()"},
        {"name": f"{col}_word_count", "description": "Word count", "code": f"df['{col}'].str.split(' ').list.len()"},
    ]
    
    # Check if contains numbers
    if sample.str.contains(r'\d').any():
        features.append({"name": f"{col}_has_number", "description": "Contains digits", "code": f"df['{col}'].str.contains(r'\\d')"})
    
    # Check if email-like
    if sample.str.contains('@').any():
        features.append({"name": f"{col}_domain", "description": "Email domain", "code": f"df['{col}'].str.extract(r'@(.+)')"})
    
    return FeatureExtractionSuggestion(
        source_column=col,
        suggested_features=features,
        category="text"
    )


def _suggest_numeric_features(df: pl.DataFrame, col: str) -> FeatureExtractionSuggestion:
    """Suggest features to extract/transform from numeric columns."""
    series = df[col].drop_nulls()
    features = []
    
    # Binning suggestion if wide range
    if series.len() > 0:
        range_val = series.max() - series.min()
        if range_val > 100:
            features.append({
                "name": f"{col}_binned", 
                "description": "Quantile bins (5 groups)", 
                "code": f"pd.qcut(df['{col}'], q=5, labels=False)"
            })
    
    # Log transform if positive and skewed
    if series.min() > 0:
        features.append({
            "name": f"{col}_log", 
            "description": "Log transform for skewed data", 
            "code": f"np.log1p(df['{col}'])"
        })
    
    # Square root if all positive
    if series.min() >= 0:
        features.append({
            "name": f"{col}_sqrt", 
            "description": "Square root transform", 
            "code": f"np.sqrt(df['{col}'])"
        })
    
    return FeatureExtractionSuggestion(
        source_column=col,
        suggested_features=features,
        category="numeric"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE PROXY
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_feature_importance_proxy(
    df: pl.DataFrame, 
    numeric_cols: list[str],
    target_col: str | None = None
) -> list[dict[str, Any]]:
    """
    Compute a proxy for feature importance without training a model.
    
    Methods:
    1. If target provided: correlation with target
    2. Otherwise: variance and correlation with other high-variance columns
    """
    importance = []
    
    if target_col and target_col in df.columns:
        # Correlation with target
        for col in numeric_cols:
            if col == target_col:
                continue
            try:
                corr = df.select(pl.corr(col, target_col)).item()
                if corr is not None:
                    importance.append({
                        "column": col,
                        "importance_score": abs(corr),
                        "method": "target_correlation",
                        "detail": f"Correlation with {target_col}: {corr:.3f}"
                    })
            except:
                pass
    else:
        # Use variance as proxy
        for col in numeric_cols:
            try:
                var = df[col].var()
                std = df[col].std()
                mean = abs(df[col].mean()) if df[col].mean() else 1
                cv = std / mean if mean > 0 else 0  # Coefficient of variation
                
                importance.append({
                    "column": col,
                    "importance_score": min(cv, 1.0),  # Cap at 1
                    "method": "variance_proxy",
                    "detail": f"Coefficient of variation: {cv:.3f}"
                })
            except:
                pass
    
    # Sort by importance
    importance.sort(key=lambda x: x["importance_score"], reverse=True)
    return importance


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureEngineeringResult:
    """Complete feature engineering analysis."""
    encoding_recommendations: list[EncodingRecommendation]
    scaling_recommendations: list[ScalingRecommendation]
    feature_extraction_suggestions: list[FeatureExtractionSuggestion]
    feature_importance: list[dict[str, Any]]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "encoding_recommendations": [
                {
                    "column": e.column,
                    "encoding": e.recommended_encoding,
                    "reason": e.reason,
                    "n_unique": e.n_unique,
                    "considerations": e.considerations
                }
                for e in self.encoding_recommendations
            ],
            "scaling_recommendations": [
                {
                    "column": s.column,
                    "scaler": s.recommended_scaler,
                    "reason": s.reason,
                    "stats": s.stats,
                    "considerations": s.considerations
                }
                for s in self.scaling_recommendations
            ],
            "feature_extraction": [
                {
                    "source": f.source_column,
                    "category": f.category,
                    "suggestions": f.suggested_features[:3]  # Top 3
                }
                for f in self.feature_extraction_suggestions
            ],
            "feature_importance": self.feature_importance[:10]  # Top 10
        }


def analyze_feature_engineering(
    df: pl.DataFrame,
    column_roles: dict[str, str] | None = None
) -> FeatureEngineeringResult:
    """
    Analyze dataset and provide ML-ready feature engineering recommendations.
    
    Args:
        df: Polars DataFrame to analyze
        column_roles: Optional dict mapping column names to semantic roles
        
    Returns:
        FeatureEngineeringResult with encoding, scaling, and extraction suggestions
    """
    logger.info("Starting feature engineering analysis...")
    
    # Identify column types
    numeric_cols = [c for c, t in df.schema.items() if t in (pl.Int64, pl.Float64, pl.Int32, pl.Float32)]
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    datetime_cols = [c for c, t in df.schema.items() if t in (pl.Date, pl.Datetime)]
    text_cols = [c for c in cat_cols if df[c].dtype == pl.Utf8]
    
    # Filter out identifiers if roles provided
    if column_roles:
        numeric_cols = [c for c in numeric_cols if column_roles.get(c, "") not in ("identifier", "name_label")]
        cat_cols = [c for c in cat_cols if column_roles.get(c, "") not in ("identifier", "name_label")]
    
    # 1. Encoding recommendations for categorical columns
    encoding_recs = []
    for col in cat_cols:
        if col not in datetime_cols:
            try:
                encoding_recs.append(_suggest_encoding(df, col))
            except Exception as e:
                logger.debug(f"Encoding analysis failed for {col}: {e}")
    
    # 2. Scaling recommendations for numeric columns
    scaling_recs = []
    for col in numeric_cols:
        try:
            scaling_recs.append(_suggest_scaling(df, col))
        except Exception as e:
            logger.debug(f"Scaling analysis failed for {col}: {e}")
    
    # 3. Feature extraction suggestions
    extraction_suggs = []
    
    # Datetime features
    for col in datetime_cols:
        extraction_suggs.append(_suggest_datetime_features(col))
    
    # Text features (limit to first 5)
    for col in text_cols[:5]:
        try:
            extraction_suggs.append(_suggest_text_features(df, col))
        except:
            pass
    
    # Numeric features (limit to first 5)
    for col in numeric_cols[:5]:
        try:
            sugg = _suggest_numeric_features(df, col)
            if sugg.suggested_features:
                extraction_suggs.append(sugg)
        except:
            pass
    
    # 4. Feature importance proxy
    feature_importance = _compute_feature_importance_proxy(df, numeric_cols)
    
    logger.info(f"Feature engineering complete: {len(encoding_recs)} encoding, "
                f"{len(scaling_recs)} scaling, {len(extraction_suggs)} extraction suggestions")
    
    return FeatureEngineeringResult(
        encoding_recommendations=encoding_recs,
        scaling_recommendations=scaling_recs,
        feature_extraction_suggestions=extraction_suggs,
        feature_importance=feature_importance
    )
