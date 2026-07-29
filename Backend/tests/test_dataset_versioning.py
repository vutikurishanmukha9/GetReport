import polars as pl

from app.services.analysis_config import AnalysisConfig
from app.services.dataset_versioning import (
    build_historical_comparison,
    build_schema_profile,
    compare_schema_profiles,
)


def test_config_snapshot_is_stable_and_versioned():
    config = AnalysisConfig(enable_outliers=False)
    snapshot = config.snapshot()

    assert snapshot["version"] == "1.0"
    assert snapshot["values"]["enable_outliers"] is False
    assert len(snapshot["fingerprint"]) == 64
    assert snapshot == config.snapshot()


def test_schema_drift_detects_added_columns_types_and_nullability():
    previous = build_schema_profile(pl.DataFrame({"id": [1, 2], "amount": [10, 20]}))
    current = build_schema_profile(
        pl.DataFrame({"id": ["1", "2"], "amount": [None, 20], "region": ["APAC", "EMEA"]})
    )

    drift = compare_schema_profiles(previous, current)

    assert drift["status"] == "drift_detected"
    assert drift["added_columns"] == ["region"]
    assert drift["type_changes"] == [{"column": "id", "before": "Int64", "after": "String"}]
    assert drift["nullability_changes"] == [{"column": "amount", "percentage_point_delta": 50.0}]


def test_historical_comparison_handles_first_dataset_without_a_baseline():
    current = build_schema_profile(pl.DataFrame({"id": [1, 2, 3]}))
    comparison = build_historical_comparison(None, None, current)

    assert comparison["previous_task_id"] is None
    assert comparison["schema_drift"] == {"baseline_available": False, "status": "no_baseline"}
    assert comparison["trend"]["current_rows"] == 3
