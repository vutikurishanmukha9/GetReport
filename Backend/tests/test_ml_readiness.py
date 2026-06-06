import pytest
import polars as pl
from app.services.confidence_scoring import ConfidenceReport, ColumnConfidence
from app.services.analysis.ml_readiness import calculate_ml_readiness

def make_mock_report(columns_data):
    """
    Helper to generate a mock ConfidenceReport.
    columns_data: list of tuples (column_name, completeness, consistency, validity, stability)
    """
    col_scores = []
    for name, comp, cons, val, stab in columns_data:
        col_scores.append(ColumnConfidence(
            column=name,
            completeness=comp,
            consistency=cons,
            validity=val,
            stability=stab,
            overall=(comp * 0.35 + cons * 0.25 + val * 0.25 + stab * 0.15),
            issues=[]
        ))
    overall_score = sum(c.overall for c in col_scores) / len(col_scores) if col_scores else 100.0
    return ConfidenceReport(
        columns=col_scores,
        dataset_confidence=overall_score,
        high_confidence_count=len(col_scores),
        low_confidence_count=0,
        critical_issues=[]
    )

def test_clean_dataset_scores_ready():
    df = pl.DataFrame({
        "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "col2": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]
    })
    report = make_mock_report([
        ("col1", 100.0, 100.0, 100.0, 100.0),
        ("col2", 100.0, 100.0, 100.0, 100.0)
    ])
    result = calculate_ml_readiness(report, df)
    assert result["score"] >= 90.0
    assert result["status"] == "Ready"
    assert len(result["reasons"]) == 0
    assert "0 of 2 columns have issues" in result["column_context"]

def test_missing_values_reduce_score():
    # 30% missing values in col1
    df = pl.DataFrame({
        "col1": [1.0, 2.0, None, None, None, 6.0, 7.0, 8.0, 9.0, 10.0],
        "col2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    # Median completeness will be (70 + 100)/2 = 85.0
    report = make_mock_report([
        ("col1", 70.0, 100.0, 100.0, 100.0),
        ("col2", 100.0, 100.0, 100.0, 100.0)
    ])
    result = calculate_ml_readiness(report, df)
    # Score should be base 85.0 since no warnings/penalties apply below 50% null rate
    assert result["score"] == 85.0
    assert result["status"] == "Needs Cleaning"

    # Now let's trigger the soft warning penalty: missing pct is 60% (completeness 40%)
    df_soft = pl.DataFrame({
        "col1": [1.0, 2.0, 3.0, 4.0, None, None, None, None, None, None],
        "col2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    report_soft = make_mock_report([
        ("col1", 40.0, 100.0, 100.0, 100.0),
        ("col2", 100.0, 100.0, 100.0, 100.0)
    ])
    result_soft = calculate_ml_readiness(report_soft, df_soft)
    # Base median completeness = (40 + 100)/2 = 70.0
    # Soft warning penalty = -15
    # Result score = 70 - 15 = 55.0
    assert result_soft["score"] == 55.0
    assert result_soft["status"] == "Not Ready"
    assert any("has a high missing rate" in r for r in result_soft["reasons"])

def test_constant_column_penalized():
    df = pl.DataFrame({
        "col1": [1, 2, 3, 4, 5],
        "col2": ["A", "A", "A", "A", "A"] # Constant
    })
    report = make_mock_report([
        ("col1", 100.0, 100.0, 100.0, 100.0),
        ("col2", 100.0, 100.0, 100.0, 100.0)
    ])
    result = calculate_ml_readiness(report, df)
    # Base completeness = 100
    # Penalty: constant col2 = -5.0 (cap is min(25, 2 * 0.4 * 5) = 4.0)
    # Result = 100 - 4 = 96.0
    assert result["score"] == 96.0
    assert any("is constant across all rows" in r for r in result["reasons"])

def test_near_zero_variance_penalized():
    df = pl.DataFrame({
        "col1": [1.000001, 1.0, 1.0, 1.000002, 1.0], # near zero variance (var = 9.0e-13 < 1e-5)
        "col2": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    report = make_mock_report([
        ("col1", 100.0, 100.0, 100.0, 100.0),
        ("col2", 100.0, 100.0, 100.0, 100.0)
    ])
    result = calculate_ml_readiness(report, df)
    # Base completeness = 100
    # Penalty: near-zero variance col1 = -5.0 (cap is min(25, 2 * 0.4 * 5) = 4.0)
    # Result = 100 - 4 = 96.0
    assert result["score"] == 96.0
    assert any("has near-zero variance" in r for r in result["reasons"])

def test_class_imbalance_detected():
    # 9 rows: 8 'A' and 1 'B' -> dominant class ratio = 8/9 = 88.8%
    df = pl.DataFrame({
        "col1": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "col2": ["A", "A", "A", "A", "A", "A", "A", "A", "B"]
    })
    report = make_mock_report([
        ("col1", 100.0, 100.0, 100.0, 100.0),
        ("col2", 100.0, 100.0, 100.0, 100.0)
    ])
    result = calculate_ml_readiness(report, df)
    # Base completeness = 100
    # Imbalance penalty: col2 has 88.8% dominant class -> -10 penalty.
    # Cap = min(30, 1 * 0.5 * 10) = 5.0
    # Result = 100 - 5 = 95.0
    assert result["score"] == 95.0
    assert any("is moderately imbalanced" in r for r in result["reasons"])

def test_hard_failure_gate():
    # 80% null in col1
    df = pl.DataFrame({
        "col1": [1.0, 2.0, None, None, None, None, None, None, None, None],
        "col2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    report = make_mock_report([
        ("col1", 20.0, 100.0, 100.0, 100.0),
        ("col2", 100.0, 100.0, 100.0, 100.0)
    ])
    result = calculate_ml_readiness(report, df)
    # Hard gate triggers immediately
    assert result["score"] == 0.0
    assert result["status"] == "Not Ready"
    assert len(result["reasons"]) == 1
    assert "missing values" in result["reasons"][0]

def test_boundary_needs_cleaning_upper():
    # Engineer score = exactly 89.0
    df = pl.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["A", "A", "A"], # constant (-5)
        "col3": [1, 2, 3],
        "col4": [1, 2, 3],
        "col5": [1, 2, 3],
        "col6": [1, 2, 3],
        "col7": [1, 2, 3],
        "col8": [1, 2, 3],
        "col9": [1, 2, 3],
        "col10": [1, 2, 3]
    })
    report = make_mock_report([
        ("col1", 100.0, 75.0, 100.0, 100.0), # consistency < 80 (-3)
        ("col2", 100.0, 100.0, 100.0, 100.0),
        ("col3", 100.0, 75.0, 100.0, 100.0), # consistency < 80 (-3)
        ("col4", 100.0, 100.0, 100.0, 100.0),
        ("col5", 100.0, 100.0, 100.0, 100.0),
        ("col6", 100.0, 100.0, 100.0, 100.0),
        ("col7", 100.0, 100.0, 100.0, 100.0),
        ("col8", 100.0, 100.0, 100.0, 100.0),
        ("col9", 100.0, 100.0, 100.0, 100.0),
        ("col10", 100.0, 100.0, 100.0, 100.0)
    ])
    result = calculate_ml_readiness(report, df)
    assert result["score"] == 89.0
    assert result["status"] == "Needs Cleaning"

def test_boundary_not_ready_upper():
    # Engineer score = exactly 64.0
    df = pl.DataFrame({
        "col1": list(range(90)) + [None] * 10,
        "col2": ["A"] * 88 + ["B"] * 2 + [None] * 10,
        "col3": ["X"] * 88 + ["Y"] * 2 + [None] * 10,
        "col4": list(range(90)) + [None] * 10,
        "col5": list(range(90)) + [None] * 10,
        "col6": list(range(90)) + [None] * 10,
        "col7": list(range(100)),
        "col8": list(range(100)),
        "col9": [1.0] * 100, # constant column (-5)
        "col10": [2.0] * 100 # constant column (-5)
    })
    
    report = make_mock_report([
        ("col1", 90.0, 100.0, 100.0, 100.0),
        ("col2", 90.0, 100.0, 100.0, 100.0),
        ("col3", 90.0, 100.0, 100.0, 100.0),
        ("col4", 90.0, 100.0, 100.0, 100.0),
        ("col5", 90.0, 75.0, 100.0, 100.0), # consistency < 80 (-3)
        ("col6", 90.0, 75.0, 100.0, 100.0), # consistency < 80 (-3)
        ("col7", 100.0, 100.0, 100.0, 100.0),
        ("col8", 100.0, 100.0, 100.0, 100.0),
        ("col9", 100.0, 100.0, 100.0, 100.0),
        ("col10", 100.0, 100.0, 100.0, 100.0)
    ])
    
    result = calculate_ml_readiness(report, df)
    assert result["score"] == 64.0
    assert result["status"] == "Not Ready"

def test_column_context_string():
    df = pl.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["A", "A", "A"], # constant
        "col3": [1, 2, 3],
        "col4": [1, 2, 3]
    })
    report = make_mock_report([
        ("col1", 100.0, 100.0, 100.0, 100.0),
        ("col2", 100.0, 100.0, 100.0, 100.0),
        ("col3", 100.0, 70.0, 100.0, 100.0), # inconsistent
        ("col4", 100.0, 100.0, 70.0, 100.0)  # outliers/invalid
    ])
    result = calculate_ml_readiness(report, df)
    assert result["column_context"] == "3 of 4 columns have issues"

def test_pdf_template_renders_ml_readiness():
    from app.services.report_generator import generate_pdf_report
    analysis_results = {
        "metadata": {
            "total_rows": 100,
            "total_columns": 5,
            "numeric_columns": 3,
            "categorical_columns": 2,
            "total_missing_values": 50,
            "missing_value_pct": 10.0
        },
        "confidence_scores": {
            "columns": [
                {
                    "column": "col1",
                    "completeness": 90.0,
                    "consistency": 100.0,
                    "validity": 100.0,
                    "stability": 100.0,
                    "overall": 95.0,
                    "grade": "A",
                    "issues": []
                }
            ],
            "dataset_confidence": 95.0,
            "dataset_grade": "A",
            "high_confidence_count": 1,
            "low_confidence_count": 0,
            "critical_issues": [],
            "ml_readiness": {
                "score": 85.0,
                "status": "Needs Cleaning",
                "reasons": ["Column 'col1' has a high missing rate: 10.0%"],
                "recommendation": "Impute missing values.",
                "column_context": "1 of 5 columns have issues"
            }
        }
    }
    charts = {
        "correlation_heatmap": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiXXXXAAEAAN0AyCj2j1sAAAAASUVORK5CYII=",
        "distributions": [],
        "bar_charts": [],
        "trend_charts": [],
        "pie_charts": []
    }
    # Wait, check if generate_pdf_report takes 3 arguments
    pdf_buffer, metadata = generate_pdf_report(analysis_results, charts, "test.csv")
    assert pdf_buffer is not None
    assert len(pdf_buffer.getvalue()) > 0
