"""
test_analysis_properties.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hypothesis property-based tests for analysis.py.
These generate random DataFrames and verify that the analysis pipeline
always obeys its invariants, no matter what data is thrown at it.
"""
from __future__ import annotations

import polars as pl
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from app.services.analysis import (
    analyze_dataset,
    IQR_LOWER_MULTIPLIER,
    IQR_UPPER_MULTIPLIER,
    CORRELATION_STRONG_THRESHOLD,
    SKEWNESS_THRESHOLD,
    ID_UNIQUENESS_THRESHOLD,
    EmptyDatasetError,
    InsufficientDataError,
)


# ─── Strategies ──────────────────────────────────────────────────────────────

def numeric_columns(min_cols: int = 1, max_cols: int = 5):
    """Generate a dict of numeric column name → list[float]."""
    return st.dictionaries(
        keys=st.text(
            alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz_"),
            min_size=2, max_size=8,
        ).filter(lambda s: s[0].isalpha()),
        values=st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=5, max_size=50,
        ),
        min_size=min_cols,
        max_size=max_cols,
    )


def categorical_columns(min_cols: int = 0, max_cols: int = 3):
    """Generate a dict of categorical column name → list[str]."""
    return st.dictionaries(
        keys=st.text(
            alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz_"),
            min_size=2, max_size=8,
        ).filter(lambda s: s[0].isalpha()),
        values=st.lists(
            st.text(min_size=1, max_size=10),
            min_size=5, max_size=50,
        ),
        min_size=min_cols,
        max_size=max_cols,
    )


# ─── Property Tests ─────────────────────────────────────────────────────────

class TestAnalyzeDatasetProperties:
    """Property-based tests for analyze_dataset."""

    @given(data=numeric_columns(min_cols=2, max_cols=4))
    @settings(max_examples=20, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
    def test_always_returns_dict(self, data: dict[str, list[float]]):
        """analyze_dataset must always return a dict, never crash."""
        # Equalize row counts (Polars requires uniform lengths)
        min_len = min(len(v) for v in data.values())
        assume(min_len >= 5)
        trimmed = {k: v[:min_len] for k, v in data.items()}

        df = pl.DataFrame(trimmed)
        result = analyze_dataset(df)

        assert isinstance(result, dict), "analyze_dataset must return a dict"

    @given(data=numeric_columns(min_cols=2, max_cols=4))
    @settings(max_examples=20, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
    def test_summary_keys_match_columns(self, data: dict[str, list[float]]):
        """Summary section must contain an entry for every numeric column."""
        min_len = min(len(v) for v in data.values())
        assume(min_len >= 5)
        trimmed = {k: v[:min_len] for k, v in data.items()}

        df = pl.DataFrame(trimmed)
        result = analyze_dataset(df)

        if "summary" in result and result["summary"]:
            for col in trimmed.keys():
                assert col in result["summary"], (
                    f"Column '{col}' missing from summary"
                )

    @given(data=numeric_columns(min_cols=2, max_cols=4))
    @settings(max_examples=15, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
    def test_outlier_bounds_are_ordered(self, data: dict[str, list[float]]):
        """For every outlier entry, lower_bound must be ≤ upper_bound."""
        min_len = min(len(v) for v in data.values())
        assume(min_len >= 5)
        trimmed = {k: v[:min_len] for k, v in data.items()}

        df = pl.DataFrame(trimmed)
        result = analyze_dataset(df)

        outliers = result.get("outliers", {})
        for col, info in outliers.items():
            lb = info.get("lower_bound")
            ub = info.get("upper_bound")
            if lb is not None and ub is not None:
                assert lb <= ub, (
                    f"Outlier bounds inverted for '{col}': {lb} > {ub}"
                )

    @given(data=numeric_columns(min_cols=2, max_cols=4))
    @settings(max_examples=15, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
    def test_correlation_values_in_range(self, data: dict[str, list[float]]):
        """All reported correlation r-values must be in [-1, 1]."""
        min_len = min(len(v) for v in data.values())
        assume(min_len >= 5)
        trimmed = {k: v[:min_len] for k, v in data.items()}

        df = pl.DataFrame(trimmed)
        result = analyze_dataset(df)

        for pair in result.get("strong_correlations", []):
            r = pair.get("r_value")
            if r is not None:
                r_float = float(r)
                assert -1.0 <= r_float <= 1.0, (
                    f"Correlation r={r_float} out of [-1, 1]"
                )

    @given(data=numeric_columns(min_cols=2, max_cols=4))
    @settings(max_examples=15, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
    def test_confidence_scores_are_valid(self, data: dict[str, list[float]]):
        """All confidence percentages must be in [0, 100]."""
        min_len = min(len(v) for v in data.values())
        assume(min_len >= 5)
        trimmed = {k: v[:min_len] for k, v in data.items()}

        df = pl.DataFrame(trimmed)
        result = analyze_dataset(df)

        conf = result.get("confidence_scores")
        if conf and "columns" in conf:
            for col_info in conf["columns"]:
                for metric in ("completeness", "consistency", "validity", "stability"):
                    val = col_info.get(metric, 0)
                    assert 0 <= val <= 100, (
                        f"{metric} = {val} out of [0, 100] for '{col_info.get('column')}'"
                    )

    @given(data=numeric_columns(min_cols=2, max_cols=4))
    @settings(max_examples=15, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
    def test_metadata_row_count_matches(self, data: dict[str, list[float]]):
        """metadata.total_rows must match the actual DataFrame row count."""
        min_len = min(len(v) for v in data.values())
        assume(min_len >= 5)
        trimmed = {k: v[:min_len] for k, v in data.items()}

        df = pl.DataFrame(trimmed)
        result = analyze_dataset(df)

        meta = result.get("metadata", {})
        if "total_rows" in meta:
            assert meta["total_rows"] == min_len, (
                f"metadata.total_rows={meta['total_rows']} != actual={min_len}"
            )

    @given(data=numeric_columns(min_cols=3, max_cols=5))
    @settings(max_examples=10, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
    def test_strong_correlations_exceed_threshold(self, data: dict[str, list[float]]):
        """Every strong correlation must have |r| >= CORRELATION_STRONG_THRESHOLD."""
        min_len = min(len(v) for v in data.values())
        assume(min_len >= 5)
        trimmed = {k: v[:min_len] for k, v in data.items()}

        df = pl.DataFrame(trimmed)
        result = analyze_dataset(df)

        for pair in result.get("strong_correlations", []):
            r = pair.get("r_value")
            if r is not None:
                assert abs(float(r)) >= CORRELATION_STRONG_THRESHOLD, (
                    f"|r|={abs(float(r))} below threshold {CORRELATION_STRONG_THRESHOLD}"
                )

    @given(data=numeric_columns(min_cols=2, max_cols=3))
    @settings(max_examples=10, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
    def test_outlier_percentage_in_range(self, data: dict[str, list[float]]):
        """Outlier percentage must be in [0, 100]."""
        min_len = min(len(v) for v in data.values())
        assume(min_len >= 5)
        trimmed = {k: v[:min_len] for k, v in data.items()}

        df = pl.DataFrame(trimmed)
        result = analyze_dataset(df)

        for col, info in result.get("outliers", {}).items():
            pct = info.get("percentage", 0)
            assert 0 <= pct <= 100, (
                f"Outlier percentage={pct} for '{col}' out of [0, 100]"
            )
