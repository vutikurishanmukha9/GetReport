import os
import sys
import time

sys.path.insert(0, "Backend")

from app.services.data_processing import load_dataframe, clean_data
from app.services.analysis import analyze_dataset
from app.services.issue_ledger import detect_issues
from app.services.visualization import generate_charts
from app.services.report_generator import generate_pdf_report

def main():
    csv_path = "Backend/temp_uploads/d5c49343-5a97-4e5e-b9d3-aa74fbb8c78c.csv"
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return

    print("1. Loading raw dataset...")
    t0 = time.time()
    df = load_dataframe(csv_path)
    print(f"   Loaded: {df.height} rows x {df.width} cols in {time.time()-t0:.2f}s")

    print("2. Cleaning dataset with row triage & deduplication...")
    t0 = time.time()
    cleaned_df, cleaning_report, dag = clean_data(df, {}, None, "getreport_messy_master_100k.csv")
    print(f"   Cleaned: {cleaned_df.height} rows x {cleaned_df.width} cols in {time.time()-t0:.2f}s")
    print(f"   Duplicate rows removed: {cleaning_report.duplicate_rows_removed}")
    print(f"   Non-data rows dropped: {cleaning_report.non_data_rows_dropped}")

    print("3. Executing deep statistical & pipeline analysis...")
    t0 = time.time()
    analysis_res = analyze_dataset(cleaned_df, top_categories=10)
    print(f"   Analysis completed in {time.time()-t0:.2f}s")

    print("4. Detecting issue ledger anomalies...")
    t0 = time.time()
    issue_ledger = detect_issues(cleaned_df)
    print(f"   Issues detected: {len(issue_ledger.issues)} issues in {time.time()-t0:.2f}s")

    print("5. Generating visualization charts...")
    t0 = time.time()
    try:
        charts, _ = generate_charts(cleaned_df)
        print(f"   Generated {len(charts)} chart sets in {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"   Charts failed: {e}")
        charts = {}

    analysis_data = analysis_res.copy()
    if cleaning_report:
        analysis_data["cleaning_report"] = cleaning_report.to_dict()
    if issue_ledger:
        analysis_data["issue_ledger"] = issue_ledger.to_dict()

    print("6. Compiling PDF report with Typst engine...")
    t0 = time.time()
    pdf_buf, meta = generate_pdf_report(analysis_data, charts, "getreport_messy_master_100k.csv")
    out_pdf = "Backend/tests/getreport_messy_master_100k_Report.pdf"
    with open(out_pdf, "wb") as f:
        f.write(pdf_buf.getbuffer())
    print(f"   PDF generated: {out_pdf} ({len(pdf_buf.getvalue()):,} bytes) in {time.time()-t0:.2f}s")
    print("Done!")

if __name__ == "__main__":
    main()
