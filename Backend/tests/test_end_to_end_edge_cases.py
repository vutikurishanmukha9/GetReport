import sys
import polars as pl
import numpy as np
from datetime import date, datetime

from app.services.data_processing import inspect_dataset, clean_data, get_dataset_info
from app.services.issue_ledger import detect_issues
from app.services.comparison import comparison_service
from app.services.analysis import analyze_dataset, build_empty_dataset_analysis, EmptyDatasetError
from app.services.analysis_config import AnalysisConfig
from app.services.visualization import generate_charts
from app.services.llm_insight import _build_fallback
from app.services.report_generator import generate_pdf_report

def run_all_edge_cases():
    test_cases = {
        "1_empty_headers_only": pl.DataFrame({"col_a": [], "col_b": []}),
        "2_single_cell_numeric": pl.DataFrame({"x": [42.0]}),
        "3_single_cell_text": pl.DataFrame({"city": ["London"]}),
        "4_all_nulls": pl.DataFrame({
            "n1": [None, None, None],
            "s1": [None, None, None],
            "d1": [None, None, None]
        }),
        "5_all_constant_numeric": pl.DataFrame({
            "const1": [10.0, 10.0, 10.0, 10.0],
            "const2": [5.0, 5.0, 5.0, 5.0]
        }),
        "6_all_constant_categorical": pl.DataFrame({
            "cat1": ["Fixed", "Fixed", "Fixed"],
            "cat2": ["Always", "Always", "Always"]
        }),
        "7_pure_categorical": pl.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "dept": ["HR", "IT", "HR", "Sales", "IT"],
            "country": ["US", "UK", "US", "DE", "FR"]
        }),
        "8_pure_numeric": pl.DataFrame({
            "score1": [10.5, 20.2, 30.1, 40.8, 50.0],
            "score2": [100.0, 150.0, 120.0, 200.0, 250.0],
            "score3": [5.0, 4.0, 3.0, 2.0, 1.0]
        }),
        "9_high_cardinality_ids": pl.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "uuid": [f"uuid_{i}" for i in range(8)],
            "timestamp": [datetime(2026, 1, i + 1, 10, 0) for i in range(8)]
        }),
        "10_mixed_types_with_booleans": pl.DataFrame({
            "is_active": [True, False, True, True, False],
            "is_verified": [False, False, True, False, True],
            "amount": [100.0, None, 250.0, 300.0, 50.0]
        }),
        "11_wide_dataset": pl.DataFrame({
            f"num_col_{i}": [float(i + j) for j in range(5)]
            for i in range(25)
        }),
        "12_bimodal_with_heavy_skew": pl.DataFrame({
            "val": [1.0, 1.1, 1.2, 1000.0, 1000.1, 1000.2, 50000.0]
        }),
        "13_zero_rows_multi_columns": pl.DataFrame({
            "user_id": [],
            "category": [],
            "revenue": []
        })
    }

    print(f"Testing {len(test_cases)} edge cases end-to-end...")

    failures = []
    for name, df in test_cases.items():
        print(f"\n--- Running case: {name} ({df.height} rows x {df.width} cols) ---")
        try:
            # Step 1: inspect
            info = get_dataset_info(df)
            inspection = inspect_dataset(df)
            print("  [✓] inspect_dataset passed")
            
            # Step 2: issues
            ledger = detect_issues(df)
            print(f"  [✓] detect_issues passed ({len(ledger.issues)} issues)")
            
            # Step 3: clean
            rules = {"impute_missing": True, "remove_duplicates": True}
            cleaned_df, cleaning_rep, dag = clean_data(df, rules, None, f"{name}.csv")
            print(f"  [✓] clean_data passed ({cleaned_df.height} rows x {cleaned_df.width} cols)")
            
            # Step 4: compare
            comp = comparison_service.compare(df, cleaned_df)
            print("  [✓] comparison passed")
            
            # Step 5: analysis
            config = AnalysisConfig.default()
            try:
                analysis_res = analyze_dataset(cleaned_df, 10, config)
            except EmptyDatasetError:
                analysis_res = build_empty_dataset_analysis(cleaned_df)
            print("  [✓] analyze_dataset passed")
            
            # Step 6: charts
            charts, warnings = generate_charts(cleaned_df)
            print(f"  [✓] generate_charts passed ({len(charts)} chart keys)")
            
            # Step 7: offline insights fallback
            insights = _build_fallback("no_api_key", analysis_res)
            print("  [✓] insights passed")
            
            # Step 8: PDF Report compilation
            analysis_data = analysis_res.copy()
            analysis_data["insights"] = insights.to_dict()
            analysis_data["cleaning_report"] = cleaning_rep.to_dict() if hasattr(cleaning_rep, "to_dict") else cleaning_rep
            analysis_data["comparison_report"] = comp.to_dict() if hasattr(comp, "to_dict") else comp
            analysis_data["issue_ledger"] = ledger.to_dict() if hasattr(ledger, "to_dict") else ledger
            
            pdf_buf, meta = generate_pdf_report(analysis_data, charts, f"{name}.csv")
            pdf_len = len(pdf_buf.getvalue())
            print(f"  [✓] generate_pdf_report passed ({pdf_len} bytes, file: {meta.filename})")
            
        except Exception as e:
            print(f"  [X] FAILED: {e}")
            import traceback
            traceback.print_exc()
            failures.append((name, str(e)))

    print("\n" + "="*50)
    if failures:
        print(f"SUMMARY: {len(failures)} failures encountered:")
        for f_name, f_err in failures:
            print(f"  - {f_name}: {f_err}")
        return False
    else:
        print("SUCCESS: All edge cases passed end-to-end!")
        return True

def test_all_edge_cases():
    assert run_all_edge_cases() is True
