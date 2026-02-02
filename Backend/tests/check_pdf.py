
import sys
import os
import logging

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CheckPDF")

def test_pdf_generation():
    print("--- STARTING PDF GENERATOR CHECK ---")
    
    try:
        print("1. Attempting Import...")
        from app.services.report_renderer import generate_pdf_report
        print("   [PASS] Import successful.")
        
        print("2. Generating Dummy Report...")
        analysis_data = {
            "metadata": {"total_rows": 100, "total_columns": 5},
            "summary": "Dummy Summary",
            "insights": "Dummy Insights"
        }
        charts = {}
        
        buffer, meta = generate_pdf_report(analysis_data, charts, "test_file.csv")
        print(f"   [PASS] Generation successful. Size: {meta['size_bytes']} bytes")
        
        print("--- PDF GENERATOR CHECK PASSED ---")
        return True
        
    except ImportError as e:
        print(f"!!! FATAL: Import Failed: {e}")
        return False
    except Exception as e:
        print(f"!!! FATAL: Runtime Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_pdf_generation()
