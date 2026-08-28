import pytest
import polars as pl
from app.services.confidence_scoring import calculate_confidence_scores

def test_confidence_scoring_clean_dataset_high_scores():
    """Verify confidence scoring on clean data yields high scores and robust metrics across 4 pillars."""
    df = pl.DataFrame({
        "user_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "salary": [50000.0, 55000.0, 60000.0, 65000.0, 70000.0, 75000.0, 80000.0, 85000.0, 90000.0, 95000.0],
        "department": ["Eng", "Sales", "Eng", "HR", "Sales", "Eng", "HR", "Sales", "Eng", "HR"]
    })
    
    report = calculate_confidence_scores(df)
    assert report.dataset_confidence >= 85.0
    assert len(report.columns) == 4
    
    for col in report.columns:
        assert col.completeness == 100.0
        assert col.consistency >= 85.0
        assert col.validity >= 85.0
        assert col.stability >= 80.0
