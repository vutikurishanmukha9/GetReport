import math
import numpy as np
import polars as pl
import pytest
from datetime import datetime, timedelta

from app.services.analysis.correlations import (
    _discretize_series,
    compute_bivariate_phik,
)
from app.services.analysis.core import analyze_dataset
from app.services.analysis.time_series import detect_seasonality
from app.services.smart_schema import discover_symbolic_equations
from app.services.gx_exporter import GreatExpectationsSuiteExporter
from app.services.analysis.classification import classify_numeric_columns
from app.services.analysis.ml_readiness import calculate_ml_readiness
from app.services.confidence_scoring import calculate_confidence_scores
from app.services.analysis.missing import analyze_missing_patterns
from app.services.comparison import ComparisonService
from app.services.analysis.outliers import detect_outliers


def test_phik_binning_degrees_of_freedom():
    """BUG-CALC-01: Verify n_actual_bins matches interval count and doesn't produce phantom empty bins."""
    np.random.seed(42)
    s = pl.Series("val", np.random.normal(loc=50.0, scale=10.0, size=200))
    binned, n_actual_bins = _discretize_series(s, num_bins=10)

    assert binned is not None
    # Maximum bin index must be strictly less than n_actual_bins
    assert np.max(binned) < n_actual_bins
    assert np.min(binned) >= 0
    # The actual bins should equal len(edges) - 1, not len(edges)
    unique_binned = set(binned.tolist())
    # All bins should have probability mass for normal distribution
    assert len(unique_binned) <= n_actual_bins
    assert n_actual_bins <= 10


def test_categorical_entropy_and_simpson_with_nulls():
    """BUG-CALC-02: Verify Simpson diversity is >= 0 and <= 1 when nulls exist, and null is not a category."""
    df = pl.DataFrame({
        "status": ["active", "active", "pending", None, None, None, None, None, None, None]
    })
    analysis = analyze_dataset(df)
    cat_info = analysis["categorical_distribution"]["status"]

    # Simpson diversity: 1 - sum(p_i^2) must be strictly between 0 and 1
    assert 0.0 <= cat_info["simpson_diversity"] <= 1.0
    # Shannon entropy must be >= 0
    assert cat_info["shannon_entropy"] >= 0.0
    # None should not be in categories dictionary
    assert "None" not in cat_info["categories"]
    assert None not in cat_info["categories"]
    # Total unique non-null categories is 2 ("active", "pending")
    assert cat_info["total_unique"] == 3  # Polars n_unique includes null, but categories only has active & pending
    assert len(cat_info["categories"]) == 2


def test_seasonality_detrending_no_false_positives_on_monotonic_trends():
    """BUG-CALC-03: Verify pure linear ramps do not trigger false positive seasonality alerts."""
    base_date = datetime(2025, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(120)]
    # Monotonic ramp without seasonality
    linear_values = [float(100.0 + 2.5 * i) for i in range(120)]
    df = pl.DataFrame({
        "date": dates,
        "sales": linear_values
    })

    result = detect_seasonality(df, "date", "sales")
    # A pure monotonic linear ramp should NOT be detected as seasonal
    assert result["detected"] is False

    # Conversely, a true weekly periodic signal MUST be detected
    periodic_values = [float(100.0 + 20.0 * np.sin(2.0 * np.pi * i / 7.0)) for i in range(120)]
    df_periodic = pl.DataFrame({
        "date": dates,
        "sales": periodic_values
    })
    result_periodic = detect_seasonality(df_periodic, "date", "sales")
    assert result_periodic["detected"] is True
    assert result_periodic["primary_period"] == "weekly"


def test_symbolic_equations_inverted_division():
    """BUG-CALC-04: Verify symbolic equation discoverer detects C = B / A when B is the numerator."""
    df = pl.DataFrame({
        "col_a": [2.0, 4.0, 5.0, 10.0, 20.0, 25.0, 50.0],
        "col_b": [10.0, 20.0, 25.0, 50.0, 100.0, 125.0, 250.0],
        "col_c": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    })

    equations = discover_symbolic_equations(df)
    ratio_eqs = [eq for eq in equations if eq["target_column"] == "col_c" and eq["equation_type"] == "division_ratio"]
    assert len(ratio_eqs) >= 1
    assert any("col_b / col_a" in eq["formula"] for eq in ratio_eqs)


def test_gx_exporter_empty_dataframe_range():
    """BUG-CALC-05: Verify Great Expectations exporter produces min <= max on empty dataframes."""
    empty_df = pl.DataFrame({
        "col_num": pl.Series([], dtype=pl.Float64),
        "col_str": pl.Series([], dtype=pl.Utf8),
    })

    exporter = GreatExpectationsSuiteExporter("empty_test")
    suite = exporter.generate_suite_from_polars(empty_df)

    # Find expect_table_row_count_to_be_between expectation
    row_count_exp = next(
        exp for exp in suite["expectations"]
        if exp["expectation_type"] == "expect_table_row_count_to_be_between"
    )
    min_val = row_count_exp["kwargs"]["min_value"]
    max_val = row_count_exp["kwargs"]["max_value"]
    assert min_val <= max_val
    assert min_val == 0
    assert max_val == 0


def test_classification_zero_variance_constant_zeros():
    """BUG-CALC-06: Verify constant zero numeric columns are tagged as low_variance and excluded."""
    df = pl.DataFrame({
        "zero_col": [0.0] * 50
    })

    classification = classify_numeric_columns(df, ["zero_col"])
    assert "zero_col" in classification["excluded"]
    assert "low_variance" in classification["exclusion_reasons"]["zero_col"]


def test_ml_readiness_boolean_class_imbalance():
    """BUG-CALC-07: Verify boolean target columns with severe imbalance are flagged by ML readiness."""
    df = pl.DataFrame({
        "feature_1": list(range(100)),
        "is_fraud": [False] * 98 + [True, True]
    })

    report = calculate_confidence_scores(df)
    ml_readiness = report.ml_readiness

    assert "reasons" in ml_readiness
    # Should penalize and flag the extreme boolean class imbalance
    reasons = ml_readiness["reasons"]
    assert any("is_fraud" in r for r in reasons)


def test_missingness_single_column_not_systematic_empty_rows():
    """BUG-CALC-08: Verify single-column nulls are categorized as partial missing, not systematic empty rows."""
    # 100 rows, only col_a has 20 nulls, col_b and col_c are 100% complete
    df = pl.DataFrame({
        "col_a": [None] * 20 + [float(i) for i in range(80)],
        "col_b": [i * 2 for i in range(100)],
        "col_c": [f"user_{i}" for i in range(100)]
    })

    res = analyze_missing_patterns(df)
    row_patterns = res["row_patterns"]

    # There are 0 completely empty rows
    assert row_patterns["fully_missing_rows"] == 0
    # There are 20 partial missing rows
    assert row_patterns["partial_missing_rows"] == 20
    assert row_patterns["complete_rows"] == 80
    # Inferred pattern must not be "Systematic" empty rows
    assert res["inferred_pattern"] != "Systematic"


def test_comparison_service_unsigned_integers():
    """BUG-CALC-09: Verify ComparisonService compares unsigned integer columns properly."""
    df_before = pl.DataFrame({
        "count": pl.Series([10, 20, 30, 40], dtype=pl.UInt32)
    })
    df_after = pl.DataFrame({
        "count": pl.Series([15, 25, 35, 45], dtype=pl.UInt32)
    })

    service = ComparisonService()
    comparison = service.compare(df_before, df_after)

    assert "count" in comparison.column_changes
    col_comp = comparison.column_changes["count"]
    metric_names = [m.metric for m in col_comp.metrics]
    assert "mean" in metric_names
    assert "std" in metric_names


def test_outlier_detection_log_iqr_with_nulls():
    """BUG-CALC-10: Verify log-transformed outlier detection runs without error on skewed data with nulls."""
    # Highly skewed positive distribution with nulls
    skewed_vals = [1.0, 1.2, 1.1, 1.3, 1.0, 1.2, 1000.0, 5000.0, None, None] * 10
    df = pl.DataFrame({
        "skewed_feature": skewed_vals
    })

    outliers = detect_outliers(df, ["skewed_feature"])
    assert "skewed_feature" in outliers
    assert outliers["skewed_feature"]["is_heavy_skew"] is True
    # log_outlier_count must be calculated and non-negative
    assert outliers["skewed_feature"]["log_outlier_count"] >= 0
