"""
OpenMetadata-Inspired Data Quality Assertion Engine for GetReport.
Vectorized, high-speed test suite execution natively using Polars expressions.
Includes bidirectional export to OpenMetadata and Great Expectations schemas.
"""

from __future__ import annotations
import json
import re
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import polars as pl

logger = logging.getLogger(__name__)


class DataQualityAssertion:
    """Represents a single executable data quality rule."""
    def __init__(
        self,
        test_type: str,
        column: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        description: str = ""
    ):
        self.test_type = test_type
        self.column = column
        self.params = params or {}
        self.description = description or f"Test {test_type} on {column or 'Table'}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_type": self.test_type,
            "column": self.column,
            "params": self.params,
            "description": self.description
        }


class DataQualityTestSuite:
    """
    Executes a collection of data quality assertions against a Polars DataFrame.
    """
    def __init__(self, name: str = "Dataset Quality Suite"):
        self.name = name
        self.tests: List[DataQualityAssertion] = []

    def add_assertion(
        self,
        test_type: str,
        column: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        description: str = ""
    ) -> DataQualityTestSuite:
        self.tests.append(DataQualityAssertion(test_type, column, params, description))
        return self

    def run(self, df: pl.DataFrame) -> Dict[str, Any]:
        total = len(self.tests)
        passed = 0
        failed = 0
        results = []

        total_rows = df.height
        total_cols = df.width

        for test in self.tests:
            t_type = test.test_type
            col = test.column
            params = test.params

            success = True
            details = {}
            violation_count = 0

            try:
                # 1. Table-level tests
                if t_type == "table_row_count_to_be_between":
                    min_r = params.get("min_rows", 0)
                    max_r = params.get("max_rows", float("inf"))
                    success = min_r <= total_rows <= max_r
                    details = {"actual_rows": total_rows, "expected_min": min_r, "expected_max": max_r}

                elif t_type == "table_column_count_to_be_between":
                    min_c = params.get("min_cols", 0)
                    max_c = params.get("max_cols", float("inf"))
                    success = min_c <= total_cols <= max_c
                    details = {"actual_cols": total_cols, "expected_min": min_c, "expected_max": max_c}

                # 2. Column-level tests
                elif col and col in df.columns:
                    series = df[col]

                    if t_type == "column_values_to_be_not_null":
                        null_c = series.null_count()
                        success = null_c == 0
                        violation_count = null_c
                        details = {"null_count": null_c, "null_percentage": round(null_c / max(total_rows, 1) * 100, 2)}

                    elif t_type == "column_values_missing_percentage_to_be_below":
                        null_pct = (series.null_count() / max(total_rows, 1)) * 100.0
                        threshold = params.get("threshold_percent", 5.0)
                        success = null_pct <= threshold
                        violation_count = series.null_count()
                        details = {"actual_percent": round(null_pct, 2), "max_allowed_percent": threshold}

                    elif t_type == "column_values_to_be_unique":
                        non_nulls = series.drop_nulls()
                        dup_count = non_nulls.len() - non_nulls.n_unique()
                        success = dup_count == 0
                        violation_count = dup_count
                        details = {"duplicate_count": dup_count}

                    elif t_type == "column_values_to_be_between":
                        min_v = params.get("min_val", -float("inf"))
                        max_v = params.get("max_val", float("inf"))
                        non_nulls = series.drop_nulls()
                        if non_nulls.len() > 0:
                            violators = df.filter(~(pl.col(col).is_between(min_v, max_v)) & pl.col(col).is_not_null())
                            violation_count = violators.height
                            success = violation_count == 0
                        details = {"min_val": min_v, "max_val": max_v, "violations": violation_count}

                    elif t_type == "column_values_to_be_in_set":
                        allowed = set(params.get("allowed_values", []))
                        non_nulls = series.drop_nulls()
                        if non_nulls.len() > 0 and allowed:
                            violators = df.filter(~(pl.col(col).is_in(list(allowed))) & pl.col(col).is_not_null())
                            violation_count = violators.height
                            success = violation_count == 0
                        details = {"allowed_set_size": len(allowed), "violations": violation_count}

                    elif t_type == "column_values_to_match_regex":
                        pattern = params.get("pattern", ".*")
                        non_nulls = series.drop_nulls().cast(pl.String)
                        if non_nulls.len() > 0:
                            violators = df.filter(~(pl.col(col).cast(pl.String).str.contains(pattern)) & pl.col(col).is_not_null())
                            violation_count = violators.height
                            success = violation_count == 0
                        details = {"pattern": pattern, "violations": violation_count}

                    elif t_type == "column_value_mean_to_be_between":
                        min_mean = params.get("min_mean", -float("inf"))
                        max_mean = params.get("max_mean", float("inf"))
                        actual_mean = series.mean()
                        if actual_mean is not None:
                            success = min_mean <= actual_mean <= max_mean
                            details = {"actual_mean": round(float(actual_mean), 4), "min_mean": min_mean, "max_mean": max_mean}
                        else:
                            success = False
                            details = {"actual_mean": None}

                    else:
                        success = False
                        details = {"error": f"Unknown assertion type: {t_type}"}

                else:
                    success = False
                    details = {"error": f"Column '{col}' not found in dataframe"}

            except Exception as ex:
                success = False
                details = {"error": str(ex)}

            if success:
                passed += 1
            else:
                failed += 1

            results.append({
                "test_type": t_type,
                "column": col,
                "description": test.description,
                "success": success,
                "violation_count": violation_count,
                "details": details
            })

        pass_rate = round((passed / max(total, 1)) * 100, 2)

        return {
            "suite_name": self.name,
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": pass_rate,
            "status": "PASSED" if failed == 0 else "FAILED",
            "results": results
        }


def generate_baseline_test_suite(
    df: pl.DataFrame,
    summary_stats: Optional[Dict[str, Any]] = None
) -> DataQualityTestSuite:
    """
    Auto-generates an OpenMetadata-style baseline data quality test suite based on dataset profiling.
    """
    suite = DataQualityTestSuite(name="Automated Baseline Data Quality Suite")
    
    # Table dimensions
    suite.add_assertion(
        test_type="table_row_count_to_be_between",
        params={"min_rows": max(1, int(df.height * 0.8)), "max_rows": int(df.height * 1.5)},
        description=f"Table row count should be approximately {df.height} rows"
    )
    suite.add_assertion(
        test_type="table_column_count_to_be_between",
        params={"min_cols": df.width, "max_cols": df.width},
        description=f"Table must retain all {df.width} schema columns"
    )

    for col in df.columns:
        s = df[col]
        null_c = s.null_count()

        # Non-null assertion if column is currently clean
        if null_c == 0:
            suite.add_assertion(
                test_type="column_values_to_be_not_null",
                column=col,
                description=f"Column '{col}' must not contain null values"
            )
        else:
            pct = (null_c / df.height) * 100
            suite.add_assertion(
                test_type="column_values_missing_percentage_to_be_below",
                column=col,
                params={"threshold_percent": min(100.0, round(pct * 1.25, 1))},
                description=f"Column '{col}' missing rate must stay below {round(pct * 1.25, 1)}%"
            )

        # Unique constraint if n_unique == total rows
        if s.n_unique() == df.height:
            suite.add_assertion(
                test_type="column_values_to_be_unique",
                column=col,
                description=f"Column '{col}' must remain unique (Primary Key candidate)"
            )

        # Value bounds for numeric types
        if s.dtype.is_numeric() and s.len() > 0:
            s_clean = s.drop_nulls()
            if s_clean.len() > 0:
                min_v = float(s_clean.min())
                max_v = float(s_clean.max())
                suite.add_assertion(
                    test_type="column_values_to_be_between",
                    column=col,
                    params={"min_val": min_v, "max_val": max_v},
                    description=f"Column '{col}' values must stay between {min_v} and {max_v}"
                )

        # Allowed set for low-cardinality categorical columns
        if s.dtype in (pl.String, pl.Categorical) and 1 <= s.n_unique() <= 10:
            allowed = [str(x) for x in s.drop_nulls().unique().to_list()]
            suite.add_assertion(
                test_type="column_values_to_be_in_set",
                column=col,
                params={"allowed_values": allowed},
                description=f"Column '{col}' must only contain known categories: {allowed}"
            )

    return suite


def export_openmetadata_json(suite_results: Dict[str, Any]) -> str:
    """Exports test results in OpenMetadata TestSuite schema format."""
    om_tests = []
    for r in suite_results.get("results", []):
        om_tests.append({
            "name": f"{r['test_type']}.{r.get('column') or 'table'}",
            "testDefinition": {
                "name": r["test_type"],
                "displayName": r["description"]
            },
            "testSuite": {"name": suite_results.get("suite_name", "OpenMetadataSuite")},
            "testCaseResult": {
                "executionStatus": "Success" if r["success"] else "Failed",
                "testResultValue": [
                    {"name": k, "value": str(v)} for k, v in r.get("details", {}).items()
                ],
                "result": f"Violations: {r.get('violation_count', 0)}"
            }
        })
    return json.dumps({
        "version": "1.3.0",
        "service": "OpenMetadata-GetReport",
        "testCases": om_tests
    }, indent=2)


def export_great_expectations_json(suite_results: Dict[str, Any]) -> str:
    """Exports test definitions in Great Expectations JSON format."""
    expectations = []
    for r in suite_results.get("results", []):
        t_type = r["test_type"]
        col = r.get("column")
        kwargs = {"column": col} if col else {}
        kwargs.update(r.get("details", {}))
        
        # Map to GE names
        ge_name = f"expect_{t_type}"
        expectations.append({
            "expectation_type": ge_name,
            "kwargs": kwargs,
            "meta": {"status": "PASSED" if r["success"] else "FAILED"}
        })

    return json.dumps({
        "expectation_suite_name": suite_results.get("suite_name", "GetReportExpectations"),
        "ge_cloud_id": None,
        "expectations": expectations,
        "data_asset_type": "Dataset"
    }, indent=2)
