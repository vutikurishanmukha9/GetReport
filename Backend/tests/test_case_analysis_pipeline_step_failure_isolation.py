import pytest
import polars as pl
from app.services.analysis_pipeline import run_pipeline, StepResult

def test_analysis_pipeline_isolates_and_reports_step_results():
    """Verify run_pipeline executes steps and records step execution outcomes."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    result_dict = {}
    
    pipeline_outcome = run_pipeline(df, result_dict)
    assert pipeline_outcome is not None
    assert len(pipeline_outcome.steps) >= 1
    # Verify every executed step has a success boolean
    for step in pipeline_outcome.steps:
        assert isinstance(step.success, bool)
