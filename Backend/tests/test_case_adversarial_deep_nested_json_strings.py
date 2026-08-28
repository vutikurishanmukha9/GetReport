import pytest
import json
import polars as pl
from app.services.analysis.core import analyze_dataset

def test_adversarial_deep_nested_json_in_string_column():
    """Verify system handles string columns storing deeply nested JSON payloads without crashing."""
    nested_records = [
        json.dumps({"user": {"id": 1, "metadata": {"tier": "gold", "tags": ["vip", "early_adopter"]}}}),
        json.dumps({"user": {"id": 2, "metadata": {"tier": "silver", "tags": []}}}),
        json.dumps({"user": {"id": 3, "metadata": {"tier": "bronze", "tags": ["churn_risk"]}}})
    ]
    
    df = pl.DataFrame({
        "order_id": [101, 102, 103],
        "raw_json_payload": nested_records
    })
    
    analysis = analyze_dataset(df)
    assert analysis is not None
    assert analysis["metadata"]["total_rows"] == 3
