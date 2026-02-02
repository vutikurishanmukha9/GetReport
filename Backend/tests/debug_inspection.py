
import sys
import os
import asyncio
import json
import logging
import pandas as pd # used for creating dummy csv quickly or just write text
import polars as pl

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugInspection")

def create_dummy_csv():
    path = "debug_data.csv"
    with open(path, "w") as f:
        f.write("id,name,value,category\n")
        f.write("1,Alice,10.5,A\n")
        f.write("2,Bob,,B\n") # Missing value
        f.write("3,Charlie,NaN,A\n") # NaN string
        f.write("4,Dave,Infinity,C\n") # Infinity
    return path

async def test_inspection():
    print("--- STARTING INSPECTION DEBUG ---")
    
    file_path = create_dummy_csv()
    print(f"1. Created dummy CSV: {file_path}")
    
    try:
        from app.services.data_processing import load_dataframe, inspect_dataset
        
        print("2. Loading DataFrame...")
        df = load_dataframe(file_path)
        print(f"   [PASS] Loaded {df.height} rows.")
        
        print("3. Inspecting Dataset...")
        report = inspect_dataset(df)
        print("   [PASS] Inspection complete.")
        
        print("4. Testing JSON Serialization (The suspect)...")
        # Try to dump to JSON and see if it fails or produces NaN
        try:
            json_str = json.dumps(report)
            print("   [PASS] report -> json.dumps() worked.")
        except Exception as e:
            print(f"   !!! FAIL: report -> json.dumps() FAILED: {e}")
            return
            
        print("5. Checking for Invalid JSON values (NaN/Infinity) in output...")
        if "NaN" in json_str or "Infinity" in json_str:
             print("   !!! WARNING: JSON contains 'NaN' or 'Infinity'. This breaks JS JSON.parse!")
             # However, standard json.dumps DOES output NaN. 
             # We need to verify if our CLEANING logic in endpoints logic (which uses inspect_dataset)
             # actually produces clean output? 
             # Wait, inspect_dataset returns a dict.
             # The preview part in get_dataset_info handles cleaning.
             # inspect_dataset itself relies on `inspect_dataset` returning safe values?
             # report['columns'][x]['missing_count'] is int.
             # report['issues']...
             pass
        else:
             print("   [PASS] JSON string looks safe (no NaNs found).")

        # Also checking the preview logic which was the fix in 2397
        print("6. content of inspection report Preview (if any)...")
        # inspect_dataset doesn't return preview anymore?
        # endpoints.py:
        # partial_result = { ... "quality_report": quality_report ... }
        # quality_report is checking 'inspect_dataset'.
        
        # Let's inspect what inspect_dataset returns
        print(json.dumps(report, indent=2))
        
        print("--- DEBUG COMPLETE ---")
        
    except Exception as e:
        print(f"!!! FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    asyncio.run(test_inspection())
