"""Additive schema-drift and historical-trend helpers for completed datasets."""
from __future__ import annotations

from typing import Any
import polars as pl


def build_schema_profile(df: pl.DataFrame) -> dict[str, Any]:
    columns = {}
    for name, dtype in df.schema.items():
        columns[name] = {
            "dtype": str(dtype),
            "null_count": df[name].null_count(),
            "null_percentage": round(df[name].null_count() / df.height * 100, 2) if df.height else 0.0,
            "unique_count": df[name].n_unique(),
        }
    return {"row_count": df.height, "column_count": df.width, "columns": columns}


def compare_schema_profiles(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    if not previous:
        return {"baseline_available": False, "status": "no_baseline"}

    old_columns = previous.get("columns", {})
    new_columns = current.get("columns", {})
    added = sorted(set(new_columns) - set(old_columns))
    removed = sorted(set(old_columns) - set(new_columns))
    type_changes = []
    nullability_changes = []

    for name in sorted(set(old_columns) & set(new_columns)):
        if old_columns[name].get("dtype") != new_columns[name].get("dtype"):
            type_changes.append({"column": name, "before": old_columns[name].get("dtype"), "after": new_columns[name].get("dtype")})
        null_delta = round(new_columns[name].get("null_percentage", 0) - old_columns[name].get("null_percentage", 0), 2)
        if abs(null_delta) >= 5:
            nullability_changes.append({"column": name, "percentage_point_delta": null_delta})

    has_drift = bool(added or removed or type_changes or nullability_changes)
    return {
        "baseline_available": True,
        "status": "drift_detected" if has_drift else "compatible",
        "added_columns": added,
        "removed_columns": removed,
        "type_changes": type_changes,
        "nullability_changes": nullability_changes,
        "row_count_delta": current.get("row_count", 0) - previous.get("row_count", 0),
    }


def build_historical_comparison(previous_job_id: str | None, previous_result: dict[str, Any] | None, current_profile: dict[str, Any]) -> dict[str, Any]:
    previous_result = previous_result or {}
    previous_profile = previous_result.get("schema_profile")
    return {
        "previous_task_id": previous_job_id,
        "schema_drift": compare_schema_profiles(previous_profile, current_profile),
        "trend": {
            "previous_rows": (previous_profile or {}).get("row_count"),
            "current_rows": current_profile.get("row_count"),
            "previous_columns": (previous_profile or {}).get("column_count"),
            "current_columns": current_profile.get("column_count"),
        },
    }
