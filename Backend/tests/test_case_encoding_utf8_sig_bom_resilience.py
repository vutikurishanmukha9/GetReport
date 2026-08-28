import os
import tempfile
import pytest
import polars as pl
from app.services.data_processing import load_dataframe

def test_utf8_bom_encoded_csv_loading():
    """Verify load_dataframe strips UTF-8 Byte Order Mark (BOM) without leaving artifacts in header names."""
    content = "user_id,amount\n1,100\n2,200\n"
    
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp:
        tmp.write(content.encode("utf-8-sig"))
        tmp_path = tmp.name
        
    try:
        df = load_dataframe(tmp_path)
        assert df.height == 2
        # Ensure BOM '\ufeff' was stripped from first column
        assert df.columns[0] == "user_id"
        assert not df.columns[0].startswith("\ufeff")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
