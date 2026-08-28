import os
import tempfile
import pytest
import polars as pl
from app.services.data_processing import load_dataframe

def test_windows_1252_encoded_csv_loading():
    """Verify load_dataframe seamlessly handles Windows-1252 / CP1252 encoded files with special accents."""
    content = "customer_name,city,balance\nRené,München,500.50\nZoë,Zürich,1200.00\n"
    
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp:
        tmp.write(content.encode("latin-1"))
        tmp_path = tmp.name
        
    try:
        df = load_dataframe(tmp_path)
        assert df.height == 2
        assert "customer_name" in df.columns
        assert "balance" in df.columns
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
