import sys
import os
import base64
from io import BytesIO

# Ensure app module is found
sys.path.append(os.getcwd())

from app.services.report_generator import generate_pdf_report, ReportMetadata

def create_mock_base64_image():
    # A tiny 1x1 white pixel PNG
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiXXXXAAEAAN0AyCj2j1sAAAAASUVORK5CYII="

def test_report_generation():
    print("Preparing mock data...")
    
    # Mock Analysis Results
    analysis_results = {
        "metadata": {
            "total_rows": 100,
            "total_columns": 5,
            "numeric_columns": 3,
            "categorical_columns": 2,
            "total_missing_values": 50,
            "missing_value_pct": 10.0
        },
        "cleaning_report": {
            "empty_rows_dropped": 5,
            "empty_columns_dropped": 1,
            "duplicate_rows_removed": 2,
            "numeric_nans_filled": 10,
            "categorical_nans_filled": 0,
            "columns_renamed": {"Old Name": "new_name"}
        },
        "summary": {
            "revenue": {"mean": 1000, "min": 100, "max": 5000},
            "age": {"mean": 30, "min": 18, "max": 65}
        },
        "insights": "1. Revenue is high.\n2. Age is distributed normally.",
        "strong_correlations": [
            {"column_a": "revenue", "column_b": "age", "r_value": 0.85, "direction": "positive", "strength": "strong"}
        ],
        "outliers": {
            "revenue": {"count": 5, "percentage": 5.0, "lower_bound": 0, "upper_bound": 4000}
        },
        "categorical_distribution": {
            "category": {
                "categories": {
                    "A": {"count": 50, "percentage": 50.0},
                    "B": {"count": 50, "percentage": 50.0}
                }
            }
        },
        "column_quality_flags": {
            "age": ["Missing values detected"]
        }
    }
    
    # Mock Charts
    charts = {
        "correlation_heatmap": create_mock_base64_image(),
        "distributions": [{"column": "revenue", "image": create_mock_base64_image()}],
        "bar_charts": [{"column": "category", "image": create_mock_base64_image()}],
        "trend_charts": [],
        "pie_charts": []
    }
    
    filename = "test_dataset.csv"
    
    print("Testing generate_pdf_report...")
    try:
        pdf_buffer, metadata = generate_pdf_report(analysis_results, charts, filename)
        
        print("\nReport Generation Successful!")
        print(f"PDF Size: {len(pdf_buffer.getvalue())} bytes")
        print("\nMetadata:")
        print(metadata.to_dict())
        
        # Verify specific sections were included
        expected_sections = [
            "Dataset Overview", "Cleaning Summary", "Executive Summary", 
            "AI Insights", "Strong Correlations", "Outlier Detection", 
            "Categorical Distribution", "Data Quality Flags", "Visualizations"
        ]
        
        missing = [s for s in expected_sections if s not in metadata.sections_included]
        if missing:
            print(f"\nWARNING: The following sections were unexpectedly skipped: {missing}")
            print(f"Skipped Details: {metadata.sections_skipped}")
        else:
            print("\nAll expected sections were included.")
        
        # Save to file for manual inspection
        output_filename = "sample_report.pdf"
        with open(output_filename, "wb") as f:
            f.write(pdf_buffer.getvalue())
        print(f"\nPDF saved to: {os.path.abspath(output_filename)}")
            
    except Exception as e:
        print(f"\nFATAL: Report generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_report_generation()
