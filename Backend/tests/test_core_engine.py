import pytest
import pandas as pd
import numpy as np
from app.services.data_processing import clean_data
from app.services.analysis import analyze_dataset
from app.services.llm_insight import _build_prompt
from app.services.task_manager import TaskManager, TaskStatus

# ─── Data Processing Tests ───────────────────────────────────────────────────

def test_no_silent_corruption_numeric_nan():
    """
    Verify that clean_data does NOT fill numeric NaNs with 0.
    """
    df = pd.DataFrame({
        "A": [1.0, 2.0, np.nan, 4.0],
        "B": ["x", np.nan, "y", "z"],
        "C": [1, 1, 1, 1]  # Prevent row drop
    })
    
    cleaned_df, report = clean_data(df)
    
    # Numeric column 'A' -> 'a' should still have NaN at index 2
    assert np.isnan(cleaned_df["a"].iloc[2]), "Numeric NaN was incorrectly filled!"
    
    # Categorical column 'B' -> 'b' should be filled with "Unknown" at index 1
    assert cleaned_df["b"].iloc[1] == "Unknown", "Categorical NaN was not filled with 'Unknown'"

def test_id_column_detection():
    """
    Verify that 'User ID' or 'Zip Code' are NOT treated as numeric stats.
    """
    df = pd.DataFrame({
        "User ID": [101, 102, 103],
        "Zip Code": [90210, 10001, 20005],
        "Revenue": [500.0, 600.0, 700.0]
    })
    
    cleaned_df, report = clean_data(df)
    
    # "user_id" should be converted to object/string
    # Use is_string_dtype to support both 'object' and 'string' dtypes
    assert pd.api.types.is_string_dtype(cleaned_df["user_id"]), f"User ID should be string-like, got {cleaned_df['user_id'].dtype}"
    
    # "zip_code" should be converted to object/string
    assert pd.api.types.is_string_dtype(cleaned_df["zip_code"]), f"Zip Code should be string-like, got {cleaned_df['zip_code'].dtype}"

    # "revenue" should remain numeric
    assert pd.api.types.is_numeric_dtype(cleaned_df["revenue"]), "Revenue should remain numeric"

# ─── Analysis Tests ──────────────────────────────────────────────────────────

def test_analysis_ignores_nans():
    """
    Verify that mean/std calculations ignore NaNs (don't treat them as 0).
    """
    # [10, 20, NaN] -> Mean should be 15 (if ignored) or 10 (if filled with 0)
    df = pd.DataFrame({"Score": [10.0, 20.0, np.nan]})
    
    result = analyze_dataset(df)
    summary = result["summary"]["Score"]
    
    assert summary["mean"] == 15.0, f"Mean should be 15.0, got {summary['mean']}"
    assert summary["count"] == 2.0, "Count should be 2 (non-null values)"

# ─── LLM Insight Tests ───────────────────────────────────────────────────────

def test_prompt_includes_context():
    """
    Verify that _build_prompt includes sample data and column types.
    """
    analysis_data = {
        "metadata": {
            "preview": [{"col1": 1, "col2": "A"}],
            "dtypes": {"col1": "int64", "col2": "object"}
        },
        "summary": "Fake Summary"
    }
    
    system_prompt, user_prompt = _build_prompt(analysis_data)
    
    assert "--- SAMPLE DATA" in user_prompt, "User prompt missing Sample Data section"
    assert "--- COLUMN DATA TYPES" in user_prompt, "User prompt missing Column Types section"
    assert "col1" in user_prompt, "User prompt missing actual column names"

# ─── Task Manager Tests ──────────────────────────────────────────────────────

def test_task_manager_flow():
    """
    Verify basic create -> update -> complete flow.
    """
    tm = TaskManager()
    task_id = tm.create_job()
    
    # Check initial state
    job = tm.get_job(task_id)
    assert job.status == TaskStatus.PENDING
    
    # Update progress
    tm.update_progress(task_id, 50, "Halfway")
    job = tm.get_job(task_id)
    assert job.status == TaskStatus.PROCESSING
    assert job.progress == 50
    
    # Complete
    tm.complete_job(task_id, {"done": True})
    job = tm.get_job(task_id)
    assert job.status == TaskStatus.COMPLETED
    assert job.result == {"done": True}
