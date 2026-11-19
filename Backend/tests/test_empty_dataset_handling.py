import pytest
import polars as pl
from app.services.comparison import comparison_service
from app.services.data_processing import clean_data
from app.services.issue_ledger import detect_issues, _detect_empty_column_issues

def test_comparison_service_safe_with_zero_columns_or_rows():
    """Verify comparison service never throws InvalidOperationError on 0-width or 0-height DataFrames."""
    empty_df = pl.DataFrame()
    normal_df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    
    # 0 columns before and after
    comp1 = comparison_service.compare(empty_df, empty_df)
    assert comp1.rows_before == 0
    assert comp1.cols_before == 0
    assert comp1.completeness_score_before == 0.0

    # Normal before, 0 columns after (all columns dropped)
    comp2 = comparison_service.compare(normal_df, empty_df)
    assert comp2.rows_before == 3
    assert comp2.cols_before == 2
    assert comp2.cols_after == 0
    assert comp2.completeness_score_after == 0.0

    # 0 columns before, normal after
    comp3 = comparison_service.compare(empty_df, normal_df)
    assert comp3.cols_before == 0
    assert comp3.cols_after == 2


def test_clean_data_prevents_dropping_last_remaining_column():
    """Verify clean_data does not drop the last column when drop_column is applied."""
    df = pl.DataFrame({"only_col": [None, None, None]})
    cleaned_df, report, dag = clean_data(df, {"only_col": {"action": "drop_column"}})
    # Must preserve the column rather than leaving 0 columns
    assert cleaned_df.width == 1
    assert "only_col" in cleaned_df.columns


def test_detect_empty_column_issues_all_empty():
    """Verify issue detector handles 100% empty datasets without destructive mass-drop recommendations."""
    df = pl.DataFrame({
        "col1": [None, None, None],
        "col2": [None, None, None]
    })
    issues = _detect_empty_column_issues(df)
    assert len(issues) == 2
    for iss in issues:
        assert iss.severity == "critical"
        assert "cannot drop all columns" in iss.suggested_fix.lower()


def test_detect_and_clean_blank_rows():
    """Verify issue ledger detects blank rows and clean_data removes them while preserving valid records."""
    df = pl.DataFrame({
        "id": ["1", "2", None, "4", ""],
        "name": ["Alice", "Bob", None, "Dave", "   "],
        "score": [95.0, 88.0, None, 72.0, None]
    })
    ledger = detect_issues(df)
    blank_issues = [i for i in ledger.issues if i.description and "blank row" in i.description.lower()]
    assert len(blank_issues) == 1
    assert blank_issues[0].affected_rows == 2
    assert "automatically remove" in blank_issues[0].suggested_fix.lower()

    # Verify clean_data automatically drops the 2 blank rows and retains the 3 valid rows
    cleaned_df, report, _ = clean_data(df, row_intelligence=True)
    assert cleaned_df.height == 3
    assert report.non_data_rows_dropped == 2
    assert cleaned_df["id"].to_list() == ["1", "2", "4"]


def test_detect_all_blank_rows_critical():
    """Verify 100% blank dataset raises critical issue explaining dataset has 0 data records."""
    df = pl.DataFrame({
        "id": [None, "", "  "],
        "name": [None, None, ""],
    })
    ledger = detect_issues(df)
    blank_issues = [i for i in ledger.issues if i.description and "blank" in i.description.lower()]
    assert len(blank_issues) == 1
    assert blank_issues[0].severity == "critical"
    assert blank_issues[0].affected_rows == 3
    assert "upload a dataset with populated data records" in blank_issues[0].suggested_fix.lower()


def test_pipeline_processes_completely_blank_dataset():
    """Verify that even if all rows are blank, the entire pipeline processes from end to end without failing."""
    from app.services.data_processing import inspect_dataset, get_dataset_info
    from app.services.visualization import generate_charts
    from app.services.analysis import build_empty_dataset_analysis, EmptyDatasetError, analyze_dataset
    from app.services.report_generator import generate_pdf_report
    from app.tasks_arq import build_schema_profile

    # 1. Dataset with headers and only empty rows (matching user's scenario)
    raw_df = pl.DataFrame({
        "order_id": [None, "", "  "],
        "customer": [None, None, ""],
        "sales": [None, None, None]
    })
    assert raw_df.height == 3
    assert raw_df.width == 3

    # 2. Inspection works and detects columns & total rows
    inspection = inspect_dataset(raw_df)
    assert inspection["total_rows"] == 3
    assert len(inspection["columns"]) == 3

    # 3. Issue detection works
    ledger = detect_issues(raw_df)
    assert len(ledger.issues) > 0

    # 4. Cleaning drops non-data rows
    cleaned_df, cl_rep, dag = clean_data(raw_df, row_intelligence=True)
    assert cleaned_df.height == 0
    assert cl_rep.non_data_rows_dropped == 3

    # 5. Comparison service works
    comp = comparison_service.compare(raw_df, cleaned_df)
    assert comp.rows_before == 3
    assert comp.rows_after == 0

    # 6. Analysis worker handles EmptyDatasetError gracefully and produces valid profile
    try:
        analysis_res = analyze_dataset(cleaned_df)
    except EmptyDatasetError:
        analysis_res = build_empty_dataset_analysis(cleaned_df)
    assert analysis_res["metadata"]["status"] == "empty_dataset"
    assert analysis_res["metadata"]["total_rows"] == 0

    # 7. Dataset info, schema profile, charts, and PDF report succeed without crashing
    ds_info = get_dataset_info(cleaned_df)
    assert ds_info["rows"] == 0
    sch = build_schema_profile(cleaned_df)
    assert sch["row_count"] == 0
    charts, _ = generate_charts(cleaned_df)
    assert charts == {}
    pdf_buf, _ = generate_pdf_report(analysis_res, charts, "empty_dataset.csv")
    assert len(pdf_buf.getvalue()) > 0


def test_load_dataframe_on_header_only_and_empty_files(tmp_path):
    """Verify load_dataframe loads 0-row CSVs and 0-byte files without raising EmptyFileError."""
    from app.services.data_processing import load_dataframe

    # Case 1: Headers only
    header_file = tmp_path / "headers_only.csv"
    header_file.write_text("user_id,username,email\n", encoding="utf-8")
    df_headers = load_dataframe(str(header_file))
    assert df_headers.width == 3
    assert df_headers.height == 0
    assert df_headers.columns == ["user_id", "username", "email"]

    # Case 2: Completely 0-byte file
    zero_byte_file = tmp_path / "zero_bytes.csv"
    zero_byte_file.write_bytes(b"")
    df_zero = load_dataframe(str(zero_byte_file))
    assert df_zero.width >= 1
    assert df_zero.height == 0



