"""
test_analysis_edge_cases.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Edge-case unit tests for analyze_dataset — covering inputs
that commonly crash data pipelines:

  • Empty DataFrame
  • Single-row DataFrame (no variance)
  • All-null columns
  • Single numeric column (no correlations possible)
  • Mixed-type / zero-numeric DataFrames
"""
from __future__ import annotations

import polars as pl
import pytest

from app.services.analysis import (
    analyze_dataset,
    EmptyDatasetError,
)


class TestEmptyDataFrame:
    """Empty input must raise EmptyDatasetError, not crash silently."""

    def test_zero_rows(self):
        df = pl.DataFrame({"a": [], "b": []}).cast({"a": pl.Float64, "b": pl.Float64})
        with pytest.raises(EmptyDatasetError):
            analyze_dataset(df)

    def test_zero_columns(self):
        df = pl.DataFrame()
        with pytest.raises(EmptyDatasetError):
            analyze_dataset(df)


class TestSingleRow:
    """One row means no variance — analysis should never crash."""

    def test_single_row_numeric(self):
        df = pl.DataFrame({"price": [42.0], "quantity": [7.0]})
        result = analyze_dataset(df)
        assert isinstance(result, dict)
        assert result["metadata"]["total_rows"] == 1

    def test_single_row_mixed(self):
        df = pl.DataFrame({"name": ["Alice"], "score": [95.0]})
        result = analyze_dataset(df)
        assert isinstance(result, dict)


class TestAllNullColumns:
    """Columns full of nulls shouldn't crash outlier or correlation logic."""

    def test_all_null_numeric(self):
        df = pl.DataFrame({
            "a": [None, None, None, None, None],
            "b": [None, None, None, None, None],
        }).cast({"a": pl.Float64, "b": pl.Float64})
        result = analyze_dataset(df)
        assert isinstance(result, dict)
        # Should not have outliers for all-null cols
        outliers = result.get("outliers", {})
        for col_info in outliers.values():
            assert col_info.get("count", 0) == 0

    def test_mixed_null_and_real(self):
        df = pl.DataFrame({
            "good": [1.0, 2.0, 3.0, 4.0, 5.0],
            "empty": [None, None, None, None, None],
        }).cast({"empty": pl.Float64})
        result = analyze_dataset(df)
        assert isinstance(result, dict)
        assert "good" in result.get("summary", {})


class TestSingleColumn:
    """Only one numeric column — correlation should return empty, not error."""

    def test_single_numeric_col(self):
        df = pl.DataFrame({"only": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = analyze_dataset(df)
        assert isinstance(result, dict)
        # Cannot compute correlation with only one column
        strong = result.get("strong_correlations", [])
        assert len(strong) == 0

    def test_single_categorical_col(self):
        df = pl.DataFrame({"category": ["a", "b", "c", "a", "b"]})
        result = analyze_dataset(df)
        assert isinstance(result, dict)
        assert result["metadata"]["numeric_columns"] == 0


class TestZeroNumericColumns:
    """DataFrames with only string/categorical columns must not crash."""

    def test_all_categorical(self):
        df = pl.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "city": ["NYC", "LA", "SF", "NYC", "LA"],
        })
        result = analyze_dataset(df)
        assert isinstance(result, dict)
        assert result["metadata"]["numeric_columns"] == 0
        assert result["metadata"]["categorical_columns"] == 2


class TestConstantValues:
    """Constant columns (zero variance) must not divide by zero."""

    def test_constant_numeric(self):
        df = pl.DataFrame({
            "constant": [42.0, 42.0, 42.0, 42.0, 42.0],
            "varies": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = analyze_dataset(df)
        assert isinstance(result, dict)
        # Constant column should have zero outliers
        outliers = result.get("outliers", {})
        if "constant" in outliers:
            assert outliers["constant"].get("count", 0) == 0
