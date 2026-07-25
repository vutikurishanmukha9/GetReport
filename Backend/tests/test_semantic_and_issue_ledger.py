import pytest
import polars as pl
from app.services.semantic_inference import _fuzzy_match, map_column_role, detect_domain
from app.services.issue_ledger import detect_issues
from app.services.feature_engineering import _suggest_scaling

def test_fuzzy_match_word_boundary_demographics():
    # Word boundary matching prevents 'gender' matching 'end'
    assert bool(_fuzzy_match("gender", ["end"])) is False
    assert bool(_fuzzy_match("gender", ["gender", "sex"])) is True
    assert bool(_fuzzy_match("user_sex", ["sex"])) is True

def test_semantic_role_inference():
    df = pl.DataFrame({
        "gender": ["male", "female", "male", "female"],
        "age": [25, 34, 45, 29],
        "screen_time_before_sleep": [1.5, 2.0, 0.5, 3.0],
        "depression_label": [0, 1, 0, 1]
    })
    role_gender = map_column_role(df, "gender")
    role_age = map_column_role(df, "age")
    assert role_gender.role in ["demographic", "categorical", "dimension"]
    assert role_age.role in ["numeric", "metric", "dimension", "predictor_metric"]

def test_issue_ledger_detection():
    df = pl.DataFrame({
        "age": [25, None, 30, 200, 28], # Has null and outlier
        "city": ["NYC", "LA", "NYC", "NYC", "LA"]
    })
    ledger = detect_issues(df)
    assert len(ledger.issues) > 0
    issue_types = [i.issue_type for i in ledger.issues]
    assert "missing_values" in issue_types or "outliers" in issue_types

def test_feature_scaling_recommendations():
    df = pl.DataFrame({
        "completion_pct": [0.0, 50.0, 75.0, 100.0],
        "age": [20.0, 30.0, 40.0, 50.0]
    })
    rec_pct = _suggest_scaling(df, "completion_pct")
    assert rec_pct.column == "completion_pct"
    assert rec_pct.recommended_scaler in ["minmax", "none", "standard", "robust"]

    rec_bounded = _suggest_scaling(df, "age")
    assert rec_bounded.column == "age"
    assert rec_bounded.recommended_scaler in ["standard", "minmax", "robust", "none"]
