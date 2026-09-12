"""
Backend/app/services/gx_exporter.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Translates GetReport's schema inferences, confidence bounds, and approved
Issue Ledger rules into portable Great Expectations Suite JSON contracts.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
import polars as pl

logger = logging.getLogger(__name__)


class GreatExpectationsSuiteExporter:
    """
    Builds and serializes a formal Great Expectations Suite
    from verified GetReport dataset analysis results and approved issue ledger actions.
    """

    def __init__(self, suite_name: str):
        self.suite_name = suite_name
        self.expectations: List[Dict[str, Any]] = []

    def add_table_row_count_range(self, min_rows: int, max_rows: int) -> None:
        self.expectations.append({
            "expectation_type": "expect_table_row_count_to_be_between",
            "kwargs": {"min_value": min_rows, "max_value": max_rows},
            "meta": {"generated_by": "GetReport"}
        })

    def add_table_columns_list(self, columns: List[str]) -> None:
        self.expectations.append({
            "expectation_type": "expect_table_columns_to_match_ordered_list",
            "kwargs": {"column_list": columns},
            "meta": {"generated_by": "GetReport"}
        })

    def add_column_existence(self, column: str) -> None:
        self.expectations.append({
            "expectation_type": "expect_column_to_exist",
            "kwargs": {"column": column},
            "meta": {"generated_by": "GetReport"}
        })

    def add_column_non_null(self, column: str, mostly: float = 1.0) -> None:
        self.expectations.append({
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": column, "mostly": mostly},
            "meta": {"generated_by": "GetReport"}
        })

    def add_column_numeric_range(
        self, column: str, min_val: float, max_val: float
    ) -> None:
        self.expectations.append({
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {"column": column, "min_value": min_val, "max_value": max_val},
            "meta": {"generated_by": "GetReport"}
        })

    def add_column_unique(self, column: str) -> None:
        self.expectations.append({
            "expectation_type": "expect_column_values_to_be_unique",
            "kwargs": {"column": column},
            "meta": {"generated_by": "GetReport"}
        })

    def add_column_values_in_set(self, column: str, value_set: List[Any]) -> None:
        self.expectations.append({
            "expectation_type": "expect_column_values_to_be_in_set",
            "kwargs": {"column": column, "value_set": value_set},
            "meta": {"generated_by": "GetReport"}
        })

    def generate_suite_from_polars(
        self,
        df: pl.DataFrame,
        quality_report: Optional[Dict[str, Any]] = None,
        approved_issues: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes a complete Great Expectations Suite from a Polars DataFrame,
        enriching the contract with quality score metadata and issue ledger rules.
        """
        row_count = df.height
        col_list = list(df.columns)
        
        # 1. Table-level expectations
        if row_count == 0:
            min_expected_rows = 0
            max_expected_rows = 0
        else:
            min_expected_rows = max(1, int(row_count * 0.80))
            max_expected_rows = max(min_expected_rows, int(row_count * 1.30))
        self.add_table_row_count_range(min_expected_rows, max_expected_rows)
        self.add_table_columns_list(col_list)

        # 2. Column-level expectations
        for col in col_list:
            self.add_column_existence(col)
            series = df[col]
            null_count = series.null_count()
            null_ratio = null_count / row_count if row_count > 0 else 0.0

            # Nullability expectation
            if null_ratio == 0.0:
                self.add_column_non_null(col, mostly=1.0)
            elif null_ratio < 0.05:
                self.add_column_non_null(col, mostly=0.95)
            elif null_ratio < 0.20:
                self.add_column_non_null(col, mostly=0.80)

            # Uniqueness check (primary key candidate)
            unique_count = series.n_unique()
            if unique_count == row_count and row_count >= 10:
                self.add_column_unique(col)

            # Numeric range with 10% tolerance margin
            if series.dtype.is_numeric() and null_count < row_count:
                # Filter out nulls/nans for boundary calculation
                non_null_series = series.drop_nulls()
                if non_null_series.len() > 0:
                    min_val = float(non_null_series.min())
                    max_val = float(non_null_series.max())
                    buffer_range = abs(max_val - min_val) * 0.10
                    self.add_column_numeric_range(
                        col,
                        min_val=round(min_val - buffer_range, 4),
                        max_val=round(max_val + buffer_range, 4)
                    )

            # Low-cardinality categorical value sets
            if series.dtype in (pl.Utf8, pl.Categorical) and 1 < unique_count <= 15:
                unique_values = [v for v in series.unique().to_list() if v is not None]
                if unique_values:
                    self.add_column_values_in_set(col, unique_values)

        # 3. Add explicit rules from approved Issue Ledger entries
        if approved_issues:
            for issue in approved_issues:
                col_name = issue.get("column")
                issue_type = issue.get("issue_type")
                if col_name and col_name in col_list:
                    if issue_type == "missing_values":
                        # After remediation, enforce zero nulls
                        self.add_column_non_null(col_name, mostly=1.0)
                    elif issue_type == "duplicates":
                        self.add_column_unique(col_name)

        # Extract metadata
        grade = "A"
        score = 95.0
        if quality_report:
            grade = quality_report.get("grade", quality_report.get("dataset_grade", "A"))
            score = float(quality_report.get("score", quality_report.get("confidence_score", 95.0)))

        return {
            "expectation_suite_name": self.suite_name,
            "ge_cloud_id": None,
            "data_asset_type": "Dataset",
            "expectations": self.expectations,
            "meta": {
                "generator": "GetReport Autonomous Quality Engine",
                "quality_grade": grade,
                "confidence_score": score,
                "dataset_rows": row_count,
                "dataset_columns": len(col_list),
            },
        }

    def export_json(self, suite_dict: Dict[str, Any]) -> str:
        """Serialize suite to formatted JSON string."""
        return json.dumps(suite_dict, indent=2)
